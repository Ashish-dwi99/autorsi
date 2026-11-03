"""Unlaunched MiniWoB++ native-BU pilot proposal, separate from frozen src.

Uses official MiniWoB observation/action/reward semantics. Environment refs are
not fixture eN identifiers. Native input(clear=True) is supported only on an
observably empty field; it must never silently become append on a nonempty field.
The native 10000ms episode clock keeps running during inference. No oracle,
task fields, target flags, HTML, or environment info is sent to the model.
"""
from __future__ import annotations

import asyncio
import base64
import copy
from dataclasses import dataclass
import importlib.metadata
import io
import json
import os
from pathlib import Path
import time

from weight_space_lab.benchmarks.miniwob import (
    Case, FAMILIES, SEEDS, TEXT_TAGS, REQUIRED_PACKAGES, cases,
    visible_observation, initial_fingerprint, require_matching_initial, binary_success,
)
from weight_space_lab.eval.runner import PolicyReply


SYSTEM_PROMPT = """You are a browser-use agent. Complete the current MiniWoB task
using only its screenshot and observed DOM. Positive indexes refer to the real
elements in this observation; negative refs are text, not action targets.
Return one complete Browser Use JSON object:
{"memory":"Brief status","action":[{"click":{"index":1}}]}
The action array contains1..8 actions in execution order; every target must be
present now. Available tools:
click: {"index":1}, including a link, button, checkbox, option or text-input focus.
input: {"index":1,"text":"text","clear":true}. clear=true (the default) is
supported only when the observed input value is empty. clear=false appends.
done: {"text":"Observed outcome","success":true}. done must be last and is a
claim only; native environment reward determines success.
Use printable ASCII text of at most256 characters. No navigation, files, code,
coordinates, select_dropdown or other tools. A changing page needs a fresh
observation. A rejected action ends this episode. Keep replies brief.
The environment's original ten-second clock runs during inference and actions.
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
    metadata = {"memory", "thinking", "evaluation_previous_goal", "next_goal"}
    if not isinstance(output, dict) or "action" not in output or set(output) - (metadata | {"action"}):
        raise ValueError("unknown native envelope fields")
    for key in metadata & output.keys():
        if output[key] is not None:
            _string(output[key], 1024, key)
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


def _reset(env, case):
    observation, _info = env.reset(seed=case.seed, options={"data_mode": "test", "record_screenshots": True})
    timer = env.unwrapped.instance.driver.execute_script("return core.EPISODE_MAX_TIME;")
    if type(timer) is not int or timer != 10000:
        raise ValueError("frozen native ten-second timer differs; no adjustment permitted")
    return observation


def preflight_suite(expected_versions):
    """48 fresh resets and48 official no-op steps; no model or user-action calls."""
    records = []
    for case in cases():
        pair = []
        for _ in range(2):
            env = make_environment(case, expected_versions)
            try:
                observation = _reset(env, case)
                pair.append(initial_fingerprint(observation))
                _obs, reward, terminated, truncated, _info = env.step(native_action(env, Action({}, "done")))
                if terminated or truncated or reward != 0:
                    raise ValueError("fresh CPU reset already terminal or nonzero; no retry")
            finally:
                env.close()
        if pair[0] != pair[1]:
            raise ValueError("paired CPU initial observations differ; no retry or case removal")
        records.append({"case_id": case.case_id, "env_id": case.env_id, "seed": case.seed,
                        "initial": pair[0], "native_timer_ms": 10000})
    return {"schema": 1, "cases": records, "native_timer_ms": 10000,
            "model_calls": 0, "environment_steps": 48, "fresh_resets": 48}


def messages(observation, history):
    from PIL import Image
    visible = visible_observation(observation)
    output = io.BytesIO()
    Image.fromarray(observation["screenshot"]).save(output, format="PNG")
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(visible, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}}]}]


async def run_episode(env, case, policy, expected_initial, save_partial, *, action_factory=native_action):
    """A fixed eight-dispatch episode. Native timers, not model claims, score it.

    Synchronous native Selenium operations require the outer process watchdog.
    A pending inference timeout is fatal to the whole experiment, never a reason
    to issue another request while the previous native worker may still run.
    """
    started = time.monotonic()
    observation = _reset(env, case)
    initial = require_matching_initial(observation, expected_initial)
    row = {"case_id": case.case_id, "env_id": case.env_id, "seed": case.seed, "initial": initial,
           "success": False, "reward": 0.0, "terminated": False, "truncated": False,
           "native_timer_ms": 10000, "model_calls": 0, "policy_dispatches": 0,
           "environment_steps": 0, "invalid_actions": 0, "prompt_tokens": 0,
           "completion_tokens": 0, "token_usage_complete": True, "events": [], "stop_reason": "running"}
    queue, history = [], []
    try:
        for dispatch in range(1, 9):
            visible = visible_observation(observation)
            event = {"dispatch": dispatch, "kind": "queued_dispatch" if queue else "inference"}
            row["events"].append(event)
            row["policy_dispatches"] += 1
            if not queue:
                request_messages = messages(observation, history)
                try:
                    row["model_calls"] += 1
                    reply = await asyncio.wait_for(policy(request_messages, 192), 40)
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
    finally:
        row["discarded_queued_actions"] = len(queue)
        queue.clear()
        row["elapsed_s"] = time.monotonic() - started
        save_partial(row)
