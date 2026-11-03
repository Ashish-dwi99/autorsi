"""Unlaunched MiniWoB++ native-BU pilot proposal, separate from frozen src.

Uses official MiniWoB observation/action/reward semantics. Environment refs are
not fixture eN identifiers. Native input(clear=True) is supported only on an
observably empty field; it must never silently become append on a nonempty field.
Each task's original native episode clock keeps running during inference. No oracle,
task fields, target flags, HTML, or environment info is sent to the model.
"""
from __future__ import annotations

import asyncio
import base64
import copy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import time

from weight_space_lab.benchmarks.miniwob import (
    Case, FAMILIES, TEXT_TAGS, REQUIRED_PACKAGES,
    visible_observation, initial_fingerprint, require_matching_initial, binary_success,
)
from weight_space_lab.eval.runner import PolicyReply
from verify_contract import cases, VERIFICATION_SEED


# The entire installed package-file metadata is frozen from the consumed v1 CPU
# receipt. This is an observed-content pin, not a claim that a version label binds
# exact bytes. Entry supplies a freshly read runtime_manifest before any reset.
V1_MINIWOB_FILES_SHA256 = '416569c85fbc57b430e0943ff2dd8e3dc49d3015201a705f62de2cc8e2b82a65'
V1_MINIWOB_FILE_COUNT = 380
NATIVE_SOURCE_PINS = {'miniwob/environment.py': {'bytes': 10043, 'sha256': '1fe34ddbb0cabf848ed2c9c22183fee6f0d90ac68b0054ef2b57623abfa81013'}, 'miniwob/html/common/ui_utils.js': {'bytes': 78175, 'sha256': '65301ee11f483df7e9114f3fac2e2938e8b56933fecbed36bce310e2d8d90368'}, 'miniwob/html/core/core.js': {'bytes': 22274, 'sha256': 'db8f489ab947a3194b99e718bce5e59ea9aaf0a3c35410d8c9c79e56e6123d14'}, 'miniwob/html/miniwob/click-button-sequence.html': {'bytes': 1471, 'sha256': '6e47da84af1abd96b52866b3547725f9256d89ae5a349230858a59fb70cb2d06'}, 'miniwob/html/miniwob/click-button.html': {'bytes': 2103, 'sha256': '54568526a8cfb37c2fc90fc09e41d4186065f955d4ee68ec58bbe39b005ac5b3'}, 'miniwob/html/miniwob/click-checkboxes.html': {'bytes': 1973, 'sha256': 'b799bd1749dd603029e7479c56efceb38a425af47c04267fe46f6d7a64ac1895'}, 'miniwob/html/miniwob/click-color.html': {'bytes': 3024, 'sha256': '1a272583b674874291320183f9a62137d26caae489cfcb88edd9d7286ab0f066'}, 'miniwob/html/miniwob/click-link.html': {'bytes': 2036, 'sha256': 'dc73e507c54a4be03b0c3b64fc6c02ea58ec845b17b000532f2f4782e7c7d729'}, 'miniwob/html/miniwob/click-option.html': {'bytes': 1656, 'sha256': 'b918237b51a1b364d990714deffbec9942dc8c1fac0920dce06adb2c07c5da6e'}, 'miniwob/html/miniwob/click-tab.html': {'bytes': 2260, 'sha256': 'd47a0e3ebb7804d4a3f3712424493dd3b61782f2e5e244e0569cc2b00409e5b3'}, 'miniwob/html/miniwob/enter-password.html': {'bytes': 1434, 'sha256': 'fb9dc861a5bacbeaf60da75736b83ba98f4681f06d1bb6fc09b05f4bd7092f6e'}, 'miniwob/html/miniwob/enter-text.html': {'bytes': 1396, 'sha256': 'cea0bd69c5070d069552879007c935fe6d715a05642127fa465b87f7e4bcaf05'}, 'miniwob/html/miniwob/focus-text.html': {'bytes': 1059, 'sha256': '90aa824a79798f11297821997045254d72bcb661b814b139e730e0776efd517e'}, 'miniwob/html/miniwob/login-user.html': {'bytes': 1495, 'sha256': '378df4b6bf6f9100319eff052fdcd25d718916f4812b28a4d5581370ff08b692'}, 'miniwob/html/miniwob/simple-arithmetic.html': {'bytes': 2046, 'sha256': 'aaec2577df9838ea8301820dfbecf257e87f7a9cdf88cfc9e5787200d8294561'}, 'miniwob/selenium_instance.py': {'bytes': 17316, 'sha256': '66136a027ad72db5efc180509208ecf769c588bfd52b875b8d89217cc28d17dc'}}
NATIVE_TIMER_MS_BY_FAMILY = {family: (15000 if family == "enter-password" else 10000)
                             for family in FAMILIES}


def expected_timer_ms(case):
    if case not in cases():
        raise ValueError("case is outside the frozen native timer inventory")
    return NATIVE_TIMER_MS_BY_FAMILY[case.family]


def require_native_source_pins(observed):
    if not isinstance(observed, dict) or len(observed) != V1_MINIWOB_FILE_COUNT:
        raise ValueError("complete frozen installed MiniWoB source inventory required")
    digest = hashlib.sha256(json.dumps(observed, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if digest != V1_MINIWOB_FILES_SHA256:
        raise ValueError("installed MiniWoB package bytes differ from consumed v1 source pins")
    if any(observed.get(name) != expected for name, expected in NATIVE_SOURCE_PINS.items()):
        raise ValueError("native task/core source pins differ")
    return {"package_files_sha256": digest, "package_file_count": len(observed),
            "selected_source_files": copy.deepcopy(NATIVE_SOURCE_PINS)}


SYSTEM_PROMPT = """Complete the MiniWoB task using its screenshot and visible DOM.
Return ONLY one JSON object with exactly the key action. No explanation,
reasoning, memory, markdown or extra keys. Example: {"action":[{"click":{"index":1}}]}.
Positive indexes are current visible element refs; negative refs are not targets.
Tools: click {"index":1}; input {"index":1,"text":"ASCII text","clear":true};
done {"text":"done","success":true}. Use1..8 actions, all targets visible now.
Input text is at most256 printable ASCII characters. clear=true requires an
observably empty field; clear=false appends. done must be last and is only a
claim: native environment reward determines success. No files, code, navigation,
coordinates or other tools. Reply with the shortest complete action JSON.
The original task timer keeps running during inference and actions.
"""


class BenchmarkInferenceUnknown(RuntimeError):
    def __init__(self, partial):
        self.partial = partial
        super().__init__(partial["stop_reason"])


class BenchmarkBudgetStop(RuntimeError):
    """Admission refused before entering a new model request."""


@dataclass(frozen=True)
class Action:
    native: dict
    kind: str
    ref: int | None = None
    text: str | None = None
    clear: bool = False
    identity: tuple | None = None
    claimed_success: bool | None = None


def _string(value, limit, name):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"invalid bounded {name}")
    return value


def _action(action, visible):
    if not isinstance(action, dict) or len(action) != 1:
        raise ValueError("one named native tool per item required")
    name, params = next(iter(action.items()))
    if not isinstance(params, dict):
        raise ValueError("native parameters must be an object")
    if name == "done":
        if "text" not in params or set(params) - {"text", "success", "files_to_display"}:
            raise ValueError("invalid done parameters")
        _string(params["text"], 256, "done text")
        claim = params.get("success", True)
        if type(claim) is not bool or params.get("files_to_display") not in (None, []):
            raise ValueError("done requires boolean claim and no files")
        return Action(copy.deepcopy(action), "done", claimed_success=claim)
    schemas = {"click": ({"index"}, {"index"}),
               "input": ({"index", "text"}, {"index", "text", "clear"})}
    if name not in schemas:
        raise ValueError("unsupported native tool")
    required, allowed = schemas[name]
    if not required <= params.keys() or set(params) - allowed:
        raise ValueError("invalid native tool fields")
    ref = params["index"]
    current = {element["ref"]: element for element in visible["elements"]
               if element["ref"] > 0 and element["tag"] != "t"}
    if type(ref) is not int or ref not in current:
        raise ValueError("index must identify a current positive visible real element")
    item = current[ref]
    # MiniWoB's documented observation has no disabled flag: do not invent one.
    # The original native environment executes the action and owns its outcome.
    identity = (ref, item["tag"], item["text"])
    if name == "click":
        return Action(copy.deepcopy(action), "click", ref=ref, identity=identity)
    clear = params.get("clear", True)
    text = _string(params["text"], 256, "input text")
    if type(clear) is not bool or any(ord(char) < 32 or ord(char) > 126 for char in text):
        raise ValueError("input requires boolean clear and printable ASCII")
    if item["tag"] not in TEXT_TAGS:
        raise ValueError("input requires an observed text input")
    if clear and item["value"] != "":
        raise ValueError("clear=true on a nonempty input is not an append operation")
    return Action(copy.deepcopy(action), "input", ref=ref, text=text, clear=clear, identity=identity)


def parse_batch(text, visible, remaining=8):
    _string(text, 4096, "native response")
    if type(remaining) is not int or not 1 <= remaining <= 8:
        raise ValueError("invalid remaining action budget")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate native JSON key")
            result[key] = value
        return result
    def nonfinite(_value):
        raise ValueError("nonfinite native JSON value")
    try:
        output = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
    except json.JSONDecodeError as exc:
        raise ValueError("one complete native JSON object required") from exc
    if not isinstance(output, dict) or set(output) != {"action"}:
        raise ValueError("concise protocol requires exactly the action key")
    actions = output["action"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= remaining:
        raise ValueError("batch does not fit the remaining eight-action budget")
    parsed = tuple(_action(action, visible) for action in actions)
    if any(action.kind == "done" for action in parsed[:-1]):
        raise ValueError("done must be last")
    return parsed


def revalidate(action, visible):
    current = _action(action.native, visible)
    if current.identity != action.identity:
        raise ValueError("queued element identity changed")
    return current


def native_action(env, action):
    from miniwob.action import ActionTypes
    if action.kind == "done":
        return env.unwrapped.create_action(ActionTypes.NONE)
    if action.kind == "click":
        return env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=action.ref)
    return env.unwrapped.create_action(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=action.ref, text=action.text)


def make_environment(case, expected_versions):
    if case not in cases() or set(expected_versions) != REQUIRED_PACKAGES:
        raise ValueError("frozen case and complete dependency pins required")
    for name, expected in expected_versions.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("runtime dependency version differs: " + name)
    for name in ("MINIWOB_CHROME_BINARY", "MINIWOB_CHROMEDRIVER"):
        if not os.environ.get(name) or not Path(os.environ[name]).is_file():
            raise ValueError("explicit preinstalled browser/driver required")
    import gymnasium
    import miniwob
    from miniwob.action import ActionSpaceConfig, ActionTypes
    from miniwob.reward import get_binary_reward
    gymnasium.register_envs(miniwob)
    config = ActionSpaceConfig(action_types=[ActionTypes.NONE, ActionTypes.CLICK_ELEMENT,
                                             ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT], text_max_len=256)
    return gymnasium.make(case.env_id, render_mode=None, action_space_config=config,
                          reward_processor=get_binary_reward, data_mode="test", refresh_freq=1)


def _reset(env, case, publish):
    """Observe the unchanged task clock; publish returned data before checking it."""
    record = {"case_id": case.case_id, "env_id": case.env_id, "seed": case.seed,
              "expected_native_timer_ms": expected_timer_ms(case), "status": "reset_entered"}
    publish(copy.deepcopy(record))
    try:
        observation, _info = env.reset(seed=case.seed, options={"data_mode": "test", "record_screenshots": True})
        record["status"] = "reset_returned"
        publish(copy.deepcopy(record))
        record["status"] = "timer_query_entered"
        publish(copy.deepcopy(record))
        timer = env.unwrapped.instance.driver.execute_script("return core.EPISODE_MAX_TIME;")
        record.update({"status": "timer_query_returned", "observed_timer_type": type(timer).__name__})
        if timer is None or type(timer) in (str, bool, int) or (type(timer) is float and math.isfinite(timer)):
            record["observed_timer_value"] = timer
            record["observed_timer_encoding"] = "json_scalar"
        else:
            record["observed_timer_value"] = repr(timer)[:512]
            record["observed_timer_encoding"] = "bounded_repr"
        publish(copy.deepcopy(record))
        if type(timer) is not int or timer != record["expected_native_timer_ms"]:
            raise ValueError("frozen task-specific native timer differs; no adjustment permitted")
        record["status"] = "accepted"
        publish(copy.deepcopy(record))
        return observation
    except BaseException as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:512]})
        publish(copy.deepcopy(record))
        raise


def preflight_suite(expected_versions, *, observed_source_files, save_partial):
    """24 paired reserved verification resets and24 official NONE steps after candidate freeze; no model calls."""
    report = {"schema": 3, "phase": "verification", "verification_seed": VERIFICATION_SEED, "status": "running", "cases": [],
              "native_timer_ms_by_family": dict(NATIVE_TIMER_MS_BY_FAMILY),
              "model_calls": 0, "environment_steps": 0, "fresh_reset_attempts": 0, "fresh_resets": 0}
    def save():
        save_partial(copy.deepcopy(report))
    save()
    try:
        report["source_pins"] = require_native_source_pins(observed_source_files)
        save()
        for case in cases():
            pair = []
            case_record = {"case_id": case.case_id, "env_id": case.env_id, "seed": case.seed,
                           "status": "running", "native_timer_ms": expected_timer_ms(case), "resets": []}
            report["cases"].append(case_record)
            save()
            for member in range(2):
                reset_record = {"pair_member": member, "status": "environment_construction_entered",
                                "close_returned": False}
                case_record["resets"].append(reset_record)
                save()
                env = make_environment(case, expected_versions)
                try:
                    report["fresh_reset_attempts"] += 1
                    def publish_timer(value):
                        reset_record["timer"] = value
                        reset_record["status"] = value["status"]
                        if value["status"] == "reset_returned":
                            report["fresh_resets"] += 1
                        save()
                    observation = _reset(env, case, publish_timer)
                    initial = initial_fingerprint(observation)
                    pair.append(initial)
                    reset_record["initial"] = initial
                    reset_record["status"] = "none_step_entered"
                    save()
                    _obs, reward, terminated, truncated, _info = env.step(native_action(env, Action({}, "done")))
                    report["environment_steps"] += 1
                    reset_record.update({"status": "none_step_returned", "none_step": {
                        "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)}})
                    save()
                    if terminated or truncated or reward != 0:
                        raise ValueError("fresh CPU reset already terminal or nonzero; no retry")
                    reset_record["status"] = "completed"
                    save()
                finally:
                    env.close()
                    reset_record["close_returned"] = True
                    save()
            if pair[0] != pair[1]:
                raise ValueError("paired CPU initial observations differ; no retry or case removal")
            case_record.update({"initial": pair[0], "status": "completed"})
            save()
        report["status"] = "completed"
        save()
        return report
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:512]})
        save()
        raise


def messages(observation, history):
    from PIL import Image
    visible = visible_observation(observation)
    output = io.BytesIO()
    Image.fromarray(observation["screenshot"]).save(output, format="PNG")
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(visible, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}}]}]


async def run_episode(env, case, policy, expected_initial, save_partial, *, action_factory=native_action, capture_observation=None):
    """A fixed eight-dispatch episode. Native timers, not model claims, score it.

    Synchronous native Selenium operations require the outer process watchdog.
    A pending inference timeout is fatal to the whole experiment, never a reason
    to issue another request while the previous native worker may still run.
    """
    started = time.monotonic()
    row = {"case_id": case.case_id, "env_id": case.env_id, "seed": case.seed, "initial": None,
           "success": False, "reward": 0.0, "terminated": False, "truncated": False,
           "native_timer_ms": expected_timer_ms(case), "model_calls": 0, "policy_dispatches": 0,
           "environment_steps": 0, "invalid_actions": 0, "prompt_tokens": 0,
           "completion_tokens": 0, "token_usage_complete": True, "events": [], "stop_reason": "running"}
    queue, history = [], []
    try:
        def publish_timer(value):
            row["reset"] = value
            save_partial(row)
        observation = _reset(env, case, publish_timer)
        row["initial"] = require_matching_initial(observation, expected_initial)
        for dispatch in range(1, 9):
            visible = visible_observation(observation)
            event = {"dispatch": dispatch, "kind": "queued_dispatch" if queue else "inference"}
            row["events"].append(event)
            row["policy_dispatches"] += 1
            if capture_observation is not None:
                event["observation_artifact"] = capture_observation(observation, dispatch)
                save_partial(row)
            if not queue:
                request_messages = messages(observation, history)
                try:
                    row["model_calls"] += 1
                    reply = await asyncio.wait_for(policy(request_messages, 96), 40)
                except BenchmarkBudgetStop:
                    row["model_calls"] -= 1
                    row["stop_reason"] = "global_admission_limit"
                    save_partial(row)
                    raise
                except BaseException as exc:
                    row.update({"stop_reason": "native_inference_disposition_unknown", "token_usage_complete": False})
                    event["error"] = f"{type(exc).__name__}: {exc}"[:512]
                    save_partial(row)
                    raise BenchmarkInferenceUnknown(row) from exc
                if not isinstance(reply, PolicyReply) or not isinstance(reply.text, str):
                    raise TypeError("policy must return PolicyReply with text")
                event["native_response"] = reply.text[:4096]
                event["native_response_truncated"] = len(reply.text) > 4096
                for key in ("prompt_tokens", "completion_tokens"):
                    value = getattr(reply, key)
                    if type(value) is int and value >= 0:
                        row[key] += value
                    else:
                        row["token_usage_complete"] = False
                history.append({"role": "assistant", "content": reply.text[:4096]})
                try:
                    queue = list(parse_batch(reply.text, visible, 9 - dispatch))
                except ValueError as exc:
                    row["invalid_actions"] += 1
                    row["stop_reason"] = "invalid_native_batch"
                    event["error"] = str(exc)[:512]
                    return row
            try:
                action = revalidate(queue[0], visible)
                queue.pop(0)
            except ValueError as exc:
                row["invalid_actions"] += 1
                row["stop_reason"] = "stale_queued_action"
                event["error"] = str(exc)[:512]
                return row
            event["native_action"] = action.native
            event["claimed_success"] = action.claimed_success
            observation, reward, terminated, truncated, _info = env.step(action_factory(env, action))
            row["environment_steps"] += 1
            row.update({"reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
                        "success": binary_success(reward, bool(terminated), bool(truncated))})
            if terminated or truncated:
                row["stop_reason"] = "native_success" if row["success"] else "native_failure_or_timeout"
                return row
            if action.kind == "done":
                row["stop_reason"] = "premature_done"
                return row
            history.append({"role": "user", "content": "Native environment step returned nonterminal after "
                            + json.dumps(action.native) + ". Inspect the fresh observation; success is not inferred."})
        row["stop_reason"] = "action_limit"
        return row
    except BaseException as exc:
        if row["stop_reason"] == "running":
            row["stop_reason"] = "episode_setup_or_native_error"
            row["error"] = f"{type(exc).__name__}: {exc}"[:512]
        raise
    finally:
        row["discarded_queued_actions"] = len(queue)
        queue.clear()
        row["elapsed_s"] = time.monotonic() - started
        save_partial(row)
