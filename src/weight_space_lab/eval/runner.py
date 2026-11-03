"""Bounded single-action browser evaluator with an injected async policy.

Only call run_task from a separately approved, budgeted experiment.  This module
does not start browsers or providers. Timeouts stop this local coroutine; they
do not establish that a remote inference worker stopped or ceased billing.
"""

import asyncio
import base64
from dataclasses import asdict, dataclass, field
import json
import time
from typing import Awaitable, Callable

from .fixtures import BrowserTask


SYSTEM_PROMPT = """You control a browser on an isolated task page. Complete the user's goal.
Each observation includes a screenshot, visible page text, and visible actionable
elements with stable IDs. Return exactly one JSON object, without markdown:
{"action":"click","id":"e1"}
{"action":"type","id":"e1","text":"replacement text"}
{"action":"select","id":"e1","value":"option value"}
{"action":"done"}
Use only IDs in the current observation. Type replaces the input's contents.
Do not claim completion until the page reflects the requested result. You cannot
navigate to external pages, run JavaScript, or perform actions outside this list.
"""


@dataclass(frozen=True)
class Limits:
    max_steps: int = 8
    max_tokens: int = 192
    call_timeout_s: float = 45.0
    task_timeout_s: float = 180.0
    action_timeout_ms: int = 3000

    def __post_init__(self):
        if not 1 <= self.max_steps <= 8 or not 1 <= self.max_tokens <= 192:
            raise ValueError("maximum eight policy calls and 192 output tokens per call")
        if not 0 < self.call_timeout_s <= 45 or not 0 < self.task_timeout_s <= 180:
            raise ValueError("timeout ceilings are 45 seconds per call / 180 per task")
        if not 1 <= self.action_timeout_ms <= 3000:
            raise ValueError("action timeout must be 1..3000ms")


@dataclass(frozen=True)
class PolicyReply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class TaskResult:
    task_id: str
    split: str
    family: str
    fixture_sha256: str
    success: bool = False
    policy_calls: int = 0
    executed_actions: int = 0
    action_errors: int = 0  # Schema/execution errors, not a comprehensive semantic-error count.
    elapsed_s: float = 0.0
    policy_elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_usage_complete: bool = True
    stop_reason: str = "not_started"
    forced_stop: bool = False
    abort_experiment: bool = False
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


Policy = Callable[[list[dict], int], Awaitable[PolicyReply]]


class PolicyDispositionUnknown(RuntimeError):
    """Fatal: an entered inference call has no normal returned receipt.

    Catch only outside the entire experiment, save result, and stop its inference
    worker. Never proceed to another task, candidate, or call. Cancellation of
    asyncio.to_thread does not cancel the worker thread or its native generation.
    """

    def __init__(self, result: TaskResult):
        self.result = result
        super().__init__(result.stop_reason)

    def __reduce__(self):
        # BaseException's default reduction reconstructs from .args, which is
        # the display string rather than the retained result this constructor
        # requires. Preserve only our constructor payload across trusted worker
        # transport; this is not an interface for unpickling external input.
        return type(self), (self.result,)


def parse_action(text: str, allowed_ids: tuple[str, ...]) -> dict:
    """Strict JSON only: no arbitrary code, extra keys, duplicate keys, or IDs."""
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError("action must be a string of at most 4096 characters")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj

    try:
        action = json.loads(text, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("expected one JSON action object") from exc
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ValueError("expected action object")
    kind = action["action"]
    keys = {"done": {"action"}, "click": {"action", "id"},
            "type": {"action", "id", "text"}, "select": {"action", "id", "value"}}
    if kind not in keys or set(action) != keys[kind]:
        raise ValueError("unknown action or invalid fields")
    if kind == "done":
        return action
    if not isinstance(action["id"], str) or action["id"] not in allowed_ids:
        raise ValueError("unknown element ID")
    for key in ("text", "value"):
        if key in action and (not isinstance(action[key], str) or len(action[key]) > 256):
            raise ValueError(f"{key} must be a string of at most 256 characters")
    return action


OBSERVE_JS = r"""() => {
 const visible = e => !e.closest('[hidden]') && e.getClientRects().length > 0
   && getComputedStyle(e).visibility !== 'hidden' && getComputedStyle(e).display !== 'none';
 const elements = [...document.querySelectorAll('[data-id]')].filter(visible).map(e => ({
   id:e.dataset.id, tag:e.tagName.toLowerCase(), type:e.type || null,
   label:e.getAttribute('aria-label') || e.closest('label')?.innerText.trim() || e.innerText.trim(),
   value:('value' in e ? e.value : null), checked:('checked' in e ? e.checked : null),
   disabled:Boolean(e.disabled),
   options:e.tagName==='SELECT' ? [...e.options].map(o=>({value:o.value,text:o.text,disabled:o.disabled})) : undefined
 }));
 return {text:document.querySelector('main').innerText.slice(0,12000), elements};
}"""

ORACLE_JS = "() => document.querySelector('#result')?.dataset.success === 'true'"


async def _execute(page, action: dict, observation: dict, timeout_ms: int):
    observed = {item["id"]: item for item in observation["elements"]}
    item = observed.get(action["id"])
    if not item or item["disabled"]:
        raise ValueError("element is not currently visible and enabled")
    # IDs originate from the fixture allowlist, not arbitrary selector strings.
    locator = page.locator(f'[data-id="{action["id"]}"]')
    if not await locator.is_visible() or not await locator.is_enabled():
        raise ValueError("element is no longer visible and enabled")
    kind = action["action"]
    if kind == "click":
        if item["tag"] != "button" and not (item["tag"] == "input" and item["type"] in {"checkbox", "radio"}):
            raise ValueError("click requires a button, checkbox, or radio")
        await locator.click(timeout=timeout_ms)
    elif kind == "type":
        if item["tag"] != "input" or item["type"] not in {"text", "number", "search", "email", "tel"}:
            raise ValueError("type requires a text input")
        await locator.fill(action["text"], timeout=timeout_ms)
    elif kind == "select":
        if item["tag"] != "select" or not any(o["value"] == action["value"] and not o["disabled"] for o in item["options"]):
            raise ValueError("select requires an enabled declared option")
        await locator.select_option(action["value"], timeout=timeout_ms)


def _record_usage(reply: PolicyReply, result: TaskResult):
    for name in ("prompt_tokens", "completion_tokens"):
        value = getattr(reply, name)
        if type(value) is not int or value < 0:
            result.token_usage_complete = False
        else:
            setattr(result, name, getattr(result, name) + value)


async def run_task(page, task: BrowserTask, policy: Policy, limits: Limits = Limits()) -> TaskResult:
    """Use a fresh disposable page. Policy signature: await policy(messages, cap).

    All outbound page network requests are aborted. Fixture script alone owns
    success; neither a model 'done' action nor its explanation can make it pass.
    The caller must close the page/context and account for actual provider usage.
    PolicyDispositionUnknown is fatal across the whole experiment, not a normal
    failed-task result. It deliberately propagates past a naive task loop.
    """
    result = TaskResult(task.task_id, task.split, task.family, task.fixture_sha256)
    started = time.monotonic()
    pending_policy = False

    async def abort_network(route):
        await route.abort()

    async def body():
        nonlocal pending_policy
        await page.route("**/*", abort_network)
        await page.set_viewport_size({"width": 1100, "height": 850})
        await page.set_content(task.html, wait_until="load", timeout=limits.action_timeout_ms)
        history: list[dict] = []
        for step in range(limits.max_steps):
            if await page.evaluate(ORACLE_JS):
                result.success, result.stop_reason = True, "oracle_success"
                return
            observation = await page.evaluate(OBSERVE_JS)
            screenshot = await page.screenshot(type="png", full_page=False)
            image = base64.b64encode(screenshot).decode("ascii")
            content = [
                {"type": "text", "text": "Goal: " + task.goal + "\nCurrent browser observation:\n" + json.dumps(observation, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
            ]
            # Only the latest screenshot is sent; textual actions/errors retain history.
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                        {"role": "user", "content": content}]
            event = {"step": step + 1}
            policy_started = time.monotonic()
            result.policy_calls += 1
            pending_policy = True
            try:
                reply = await asyncio.wait_for(policy(messages, limits.max_tokens), limits.call_timeout_s)
                pending_policy = False
            except asyncio.TimeoutError:
                result.token_usage_complete = False
                result.stop_reason, result.forced_stop = "policy_timeout", True
                result.abort_experiment = True
                event["error"] = "policy_timeout_remote_disposition_unknown"
                result.events.append(event)
                raise PolicyDispositionUnknown(result)
            except asyncio.CancelledError:
                # The enclosing task timeout/cancellation handles this while the
                # pending marker remains true; no next policy call is admitted.
                raise
            except Exception as exc:
                result.token_usage_complete = False
                result.stop_reason, result.forced_stop = "policy_error_disposition_unknown", True
                result.abort_experiment = True
                event["error"] = f"{type(exc).__name__}: {exc}"[:512]
                result.events.append(event)
                raise PolicyDispositionUnknown(result) from exc
            finally:
                result.policy_elapsed_s += time.monotonic() - policy_started
            if not isinstance(reply, PolicyReply):
                raise TypeError("policy must return PolicyReply")
            _record_usage(reply, result)
            event["response"] = reply.text[:4096]
            history.append({"role": "assistant", "content": reply.text[:4096]})
            try:
                action = parse_action(reply.text, task.allowed_ids)
                event["action"] = action
                if action["action"] == "done":
                    result.success = bool(await page.evaluate(ORACLE_JS))
                    result.stop_reason = "oracle_success" if result.success else "premature_done"
                    result.events.append(event)
                    return
                await _execute(page, action, observation, limits.action_timeout_ms)
                result.executed_actions += 1
            except Exception as exc:
                result.action_errors += 1
                event["error"] = f"{type(exc).__name__}: {exc}"[:512]
                history.append({"role": "user", "content": "Action rejected: " + event["error"]})
            result.events.append(event)
        result.success = bool(await page.evaluate(ORACLE_JS))
        result.stop_reason = "oracle_success" if result.success else "step_limit"
        result.forced_stop = not result.success

    try:
        await asyncio.wait_for(body(), limits.task_timeout_s)
    except PolicyDispositionUnknown:
        raise
    except asyncio.TimeoutError:
        result.stop_reason, result.forced_stop = "task_timeout", True
        result.token_usage_complete = False
        if pending_policy:
            result.stop_reason = "task_timeout_during_policy_disposition_unknown"
            result.abort_experiment = True
            raise PolicyDispositionUnknown(result)
    except asyncio.CancelledError as exc:
        if pending_policy:
            result.stop_reason = "cancelled_during_policy_disposition_unknown"
            result.forced_stop = result.abort_experiment = True
            result.token_usage_complete = False
            raise PolicyDispositionUnknown(result) from exc
        raise
    except Exception as exc:
        result.stop_reason, result.forced_stop = "harness_or_policy_error", True
        result.token_usage_complete = False
        result.events.append({"error": f"{type(exc).__name__}: {exc}"[:512]})
    finally:
        result.elapsed_s = time.monotonic() - started
    return result
