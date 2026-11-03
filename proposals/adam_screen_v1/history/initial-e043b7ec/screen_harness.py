"""Inject a policy into new isolated screenshot-only fixtures; no provider code.

An unknown policy/native timeout latches the entire ScreenSession. Cancellation
of an await does not stop a model thread/service. Its owner must terminate and
reap before another session; this module never retries the pending operation.
Browser video is actual capture, not synthesized actions or another actor.
"""
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import shutil
import time

from screen_cases import check_receipt
from screen_protocol import KEYS, SYSTEM_PROMPT, Limits, Reply, SessionAborted, parse_actions

MAX_PNG = 1024 * 1024
MAX_VIDEO = 64 * 1024 * 1024
RECEIPT_JS = "() => window.__screenReceipt"
FOCUSED_EDITABLE_JS = "() => {const e=document.activeElement;return !!e&&((e.tagName==='INPUT'&&['text','search'].includes(e.type))||e.tagName==='TEXTAREA')&&!e.disabled&&!e.readOnly;}"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value, *, exclusive=False):
    encoded = canonical(value) + b"\n"
    if len(encoded) > 2 * 1024 * 1024:
        raise ValueError("report exceeds2MiB")
    path = Path(path)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(encoded)
    else:
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(encoded)
        temporary.replace(path)


def policy_messages(goal, png, viewport, sequence, remaining, history):
    """Only human goal, pixels, geometry/budget and already-issued action feedback."""
    payload = {"goal": goal, "viewport": viewport, "sequence": sequence,
               "remaining_actions": remaining,
               "coordinates": "integer CSS pixels; screenshot equals viewport; device_scale_factor=1"}
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": [
                {"type": "text", "text": canonical(payload).decode()},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}]


async def execute_action(page, action):
    kind, args = next(iter(action.items()))
    if kind == "click":
        await page.mouse.click(args["x"], args["y"])
    elif kind == "type":
        # This boolean only validates safe input execution. It is never sent to
        # the policy and does not select a field or read any answer.
        if not await page.evaluate(FOCUSED_EDITABLE_JS):
            raise ValueError("type requires a currently focused editable text field")
        await page.keyboard.insert_text(args["text"])
    elif kind == "key":
        if args["name"] == "SELECT_ALL" and not await page.evaluate(FOCUSED_EDITABLE_JS):
            raise ValueError("SELECT_ALL requires a focused editable text field")
        await page.keyboard.press(KEYS[args["name"]])
    elif kind == "scroll":
        await page.mouse.move(args["x"], args["y"])
        await page.mouse.wheel(0, args["dy"])
    elif kind != "done":
        raise ValueError("unknown action")


class ScreenSession:
    def __init__(self, policy, *, actor_label, limits=None, actor_identity=None):
        if not callable(policy) or not isinstance(actor_label, str) or not 1 <= len(actor_label) <= 100:
            raise ValueError("a policy and bounded truthful actor label are required")
        self.policy = policy
        self.actor_label = actor_label
        self.actor_identity = actor_identity  # supplied provenance, not self-certified here
        self.limits = limits or Limits()
        self.aborted = False
        self.pending = None
        self.sequence = 0

    async def _infer(self, messages, result, remaining, save=lambda: None):
        if self.aborted or self.pending is not None:
            raise SessionAborted(result)
        self.pending = asyncio.create_task(self.policy(messages, self.limits.max_output_tokens))
        result["policy_calls"] += 1
        result["policy_pending"] = True
        save()  # Publish entered policy state before yielding to its task.
        try:
            done, _ = await asyncio.wait({self.pending}, timeout=min(self.limits.call_timeout_s, remaining))
            if not done:
                raise TimeoutError("policy returned no receipt within its admitted window")
            reply = self.pending.result()
        except BaseException as exc:
            self.aborted = True
            self.pending.cancel()
            result.update(stop_reason="policy_disposition_unknown", abort_session=True,
                          error_type=type(exc).__name__)
            raise SessionAborted(result) from exc
        self.pending = None
        result["policy_pending"] = False
        if not isinstance(reply, Reply) or not isinstance(reply.text, str):
            raise ValueError("policy must return screen_protocol.Reply")
        for key in ("prompt_tokens", "completion_tokens"):
            value = getattr(reply, key)
            if type(value) is int and value >= 0:
                result[key] += value
            else:
                result["token_usage_complete"] = False
        return reply

    async def run_case(self, browser, case, artifact_dir, *, demonstration=False, publish=None):
        if self.aborted or self.pending is not None:
            raise SessionAborted({"stop_reason": "previous_session_disposition_unknown", "abort_session": True})
        directory = Path(artifact_dir)
        if shutil.disk_usage(directory.parent).free < 128 * 1024 * 1024 + MAX_VIDEO + 28 * MAX_PNG:
            raise ValueError("fresh disk floor must cover128MiB plus per-case video, frames, manual copies and metadata")
        directory.mkdir(parents=False, exist_ok=False)
        (directory / "video").mkdir()
        started = time.monotonic()
        result = {"schema": 1, **case.manifest(), "actor_label": self.actor_label,
                  "actor_identity_declared": self.actor_identity, "demonstration": bool(demonstration),
                  "benchmark_claim": False, "success": False, "stop_reason": "running",
                  "policy_calls": 0, "policy_pending": False, "actions_dispatched": 0,
                  "action_errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
                  "token_usage_complete": True, "abort_session": False,
                  "events": [], "frames": [], "context_close_returned": False,
                  "clock_start_monotonic": started, "started_unix": time.time(),
                  "browser_version": browser.version,
                  "network_requests_blocked": 0,
                  "scope": "Tiny synthetic screenshot-only browser fixtures; desktop-like/mobile-like layouts, no actual OS/mobile benchmark"}
        context = page = None
        fatal = None
        history = []

        def save():
            write_json(directory / "result.json", result)
            if publish is not None:
                publish(json.loads(canonical(result)))

        def remaining():
            return self.limits.case_timeout_s - (time.monotonic() - started)

        async def timed(awaitable):
            left = remaining()
            if left <= 0:
                if hasattr(awaitable, "close"):
                    awaitable.close()
                raise TimeoutError("original case deadline")
            return await asyncio.wait_for(awaitable, left)

        async def screenshot(label):
            # Playwright screenshot bytes are sent unchanged to the policy.
            before = time.monotonic()
            png = await timed(page.screenshot(type="png", full_page=False, animations="disabled"))
            if len(png) > MAX_PNG:
                raise ValueError("screenshot exceeds1MiB")
            path = directory / f"frame-{len(result['frames']):02d}.png"
            with path.open("xb") as stream:
                stream.write(png)
            result["frames"].append({"path": path.name, "sha256": hashlib.sha256(png).hexdigest(),
                                     "bytes": len(png), "phase": label, "capture_started_monotonic": before,
                                     "capture_returned_monotonic": time.monotonic(),
                                     "viewport": case.viewport, "device_scale_factor": 1})
            return png

        try:
            context = await timed(browser.new_context(viewport=case.viewport, device_scale_factor=1,
                record_video_dir=str(directory / "video"), record_video_size=case.viewport,
                accept_downloads=False, service_workers="block"))
            async def block(route):
                result["network_requests_blocked"] += 1
                await route.abort("blockedbyclient")
            await timed(context.route("**/*", block))
            page = await timed(context.new_page())
            page.set_default_timeout(2000)
            await timed(page.set_content(case.html, wait_until="load", timeout=3000))
            if await timed(page.evaluate(RECEIPT_JS)) is not None:
                raise ValueError("fixture unexpectedly has an initial receipt")
            save()
            while result["actions_dispatched"] < self.limits.max_actions:
                png = await screenshot("before_policy")
                self.sequence += 1
                count = self.limits.max_actions - result["actions_dispatched"]
                messages = policy_messages(case.goal, png, case.viewport, self.sequence, count, history)
                event = {"sequence": self.sequence, "message_sha256": hashlib.sha256(canonical(messages)).hexdigest(),
                         "screenshot_sha256": hashlib.sha256(png).hexdigest(),
                         "inference_started_monotonic": time.monotonic(), "actions": []}
                result["events"].append(event)
                save()
                reply = await self._infer(messages, result, remaining(), save)
                event["inference_returned_monotonic"] = time.monotonic()
                event["response"] = reply.text[:4096]
                actions = parse_actions(reply.text, **case.viewport, remaining=count)
                history.append({"role": "assistant", "content": reply.text})
                for action in actions:
                    action_event = {"action": action, "started_monotonic": time.monotonic()}
                    event["actions"].append(action_event)
                    result["actions_dispatched"] += 1
                    if "done" in action:
                        result["stop_reason"] = "policy_done_without_oracle_success"
                        break
                    await timed(execute_action(page, action))
                    action_event["returned_monotonic"] = time.monotonic()
                    receipt = await timed(page.evaluate(RECEIPT_JS))
                    result["success"] = check_receipt(case, receipt)
                    if result["success"]:
                        result["stop_reason"] = "independent_final_state_match"
                        result["final_receipt"] = receipt
                        break
                    history.append({"role": "user", "content": "UI action returned: " + json.dumps(action)})
                save()
                if result["success"] or result["stop_reason"] != "running":
                    break
            if result["stop_reason"] == "running":
                result["stop_reason"] = "action_limit"
            await screenshot("final")
        except SessionAborted as exc:
            fatal = exc
        except (TimeoutError, asyncio.CancelledError) as exc:
            self.aborted = True
            result.update(stop_reason="case_native_disposition_unknown", abort_session=True, error_type=type(exc).__name__)
            fatal = SessionAborted(result)
        except Exception as exc:
            result["action_errors"] += 1
            result.update(stop_reason="invalid_response_or_ui_error", error_type=type(exc).__name__, error=str(exc)[:512])
        finally:
            result["task_elapsed_s"] = time.monotonic() - started
            save()
            if context is not None:
                try:
                    await asyncio.wait_for(context.close(), self.limits.close_timeout_s)
                    result["context_close_returned"] = True
                except BaseException as exc:
                    self.aborted = True
                    result.update(abort_session=True, close_error_type=type(exc).__name__)
                    fatal = fatal or SessionAborted(result)
            if result["context_close_returned"] and page is not None and page.video is not None:
                try:
                    video = Path(await asyncio.wait_for(page.video.path(), self.limits.close_timeout_s))
                    if video.is_symlink() or video.parent.resolve() != (directory / "video").resolve():
                        raise ValueError("video outside owned artifact directory")
                    size = video.stat().st_size
                    if not 0 < size <= MAX_VIDEO:
                        raise ValueError("video exceeds64MiB")
                    digest = hashlib.sha256()
                    with video.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    result["video"] = {"path": str(video.relative_to(directory)), "bytes": size,
                                       "sha256": digest.hexdigest(), "record_video_size": case.viewport,
                                       "actual_browser_capture": True,
                                       "timing_limit": "Video encoder timestamps and monotonic action times are separate observations; no claimed frame-exact action alignment"}
                except Exception as exc:
                    result["artifact_error"] = f"{type(exc).__name__}: {exc}"[:512]
            result["total_elapsed_s"] = time.monotonic() - started
            save()
        if fatal is not None:
            raise fatal
        return result
