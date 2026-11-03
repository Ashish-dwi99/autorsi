"""A frozen 12-family x 2-seed MiniWoB++ subset adapter, not full MiniWoB.

Primary API references inspected 2026-11-03:
https://miniwob.farama.org/content/action_space/
https://miniwob.farama.org/content/observation_space/
https://miniwob.farama.org/content/reward/
https://github.com/Farama-Foundation/MiniWoB-plusplus/blob/main/miniwob/environment.py
https://github.com/Farama-Foundation/MiniWoB-plusplus/blob/main/miniwob/selenium_actions.py

Preparation only: no MiniWoB/Selenium/browser/model runtime has been exercised.
Parent must pin and verify package/HTML/Chromium/ChromeDriver builds, complete a
CPU paired-reset preflight, freeze the candidate, and admit a separate budget.
Use a dedicated network-isolated process/container; this adapter does not supply
OS network isolation or a physical process timeout. Native MiniWoB real-time
episode timers stay unchanged (core default 10s). Binary reward does not disable
timeouts. Each model's inference time therefore counts against that same timer.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import io
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import time
from typing import Any

from weight_space_lab.eval.runner import Limits, PolicyReply, PolicyDispositionUnknown


FAMILIES = (
    "click-button", "click-button-sequence", "click-checkboxes", "click-color",
    "click-link", "click-option", "click-tab", "enter-text", "enter-password",
    "login-user", "simple-arithmetic", "focus-text",
)
SEEDS = (2026110317, 2026110353)
REQUIRED_PACKAGES = frozenset({"miniwob", "gymnasium", "selenium", "numpy", "Pillow"})
TEXT_TAGS = frozenset({"input_text", "input_password", "input_email", "input_search",
                       "input_tel", "input_number", "textarea"})

SYSTEM_PROMPT = """Complete the browser task using its screenshot and visible DOM.
Return one JSON object only, without Markdown. Available actions:
{"action":"click","ref":4}
{"action":"focus_type","ref":7,"text":"text to type"}
{"action":"type_text","text":"more text in the focused input"}
{"action":"done"}
Use only positive element refs from the current observation. focus_type clicks
the input then types; it APPENDS to existing text and does not clear it. type_text
requires a currently focused text input. Text must be printable ASCII, <=256
characters. The environment decides success; a done claim does not make it pass.
"""


@dataclass(frozen=True)
class Case:
    family: str
    seed: int

    @property
    def env_id(self):
        return f"miniwob/{self.family}-v1"

    @property
    def case_id(self):
        return f"{self.family}:seed={self.seed}"


def cases() -> tuple[Case, ...]:
    return tuple(Case(family, seed) for family in FAMILIES for seed in SEEDS)


def _check_case(case: Case):
    if case not in cases():
        raise ValueError("case is outside the frozen 12-family/two-seed subset")


def _number(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError("expected a scalar or one-element array")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError("expected a finite numeric observation")
    return float(value)


def _vector(value: Any, length: int) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("unexpected observation vector")
    return [_number(x) for x in value]


def visible_observation(observation: Mapping) -> dict:
    """Expose documented visible inputs only; omit info, oracle, fields, target flags.

    DOM IDs/classes and demonstration-target/tampered flags are intentionally
    absent. The screenshot remains unmodified. This is a declared DOM-assisted
    action/observation subset, not a pixel-only or coordinate-control protocol.
    """
    utterance = observation.get("utterance")
    if not isinstance(utterance, str) or len(utterance) > 4096:
        raise ValueError("invalid instruction")
    screenshot = observation["screenshot"]
    shape = tuple(screenshot.shape)
    if len(shape) != 3 or shape[2] != 3 or not all(type(x) is int and 0 < x <= 4096 for x in shape):
        raise ValueError("expected bounded HxWx3 screenshot")
    if str(screenshot.dtype) != "uint8":
        raise ValueError("expected uint8 screenshot")
    height, width, _ = shape
    elements, refs = [], set()
    dom = observation.get("dom_elements")
    if not isinstance(dom, (tuple, list)) or len(dom) > 512:
        raise ValueError("unexpected DOM collection")
    for item in dom:
        if not isinstance(item, Mapping):
            raise ValueError("DOM element must be a mapping")
        ref = item.get("ref")
        if isinstance(ref, bool) or not isinstance(ref, Integral) or int(ref) == 0:
            raise ValueError("invalid DOM ref")
        ref = int(ref)
        if ref in refs:
            raise ValueError("duplicate DOM ref")
        refs.add(ref)
        box = [_number(item[k]) for k in ("left", "top", "width", "height")]
        left, top, ew, eh = box
        if ew <= 0 or eh <= 0 or left + ew <= 0 or top + eh <= 0 or left >= width or top >= height:
            continue
        strings = {}
        for key in ("tag", "text", "value"):
            value = item.get(key, "")
            if not isinstance(value, str) or len(value) > 4096:
                raise ValueError(f"invalid DOM {key}")
            strings[key] = value
        flags = _vector(item.get("flags", [0, 0, 0, 0]), 4)
        elements.append({"ref": ref, **strings, "box": box,
                         "focused": flags[0] == 1,
                         "bg_color": _vector(item.get("bg_color", [0, 0, 0, 0]), 4),
                         "fg_color": _vector(item.get("fg_color", [0, 0, 0, 0]), 4)})
    return {"utterance": utterance, "screen_shape": list(shape), "elements": elements}


def initial_fingerprint(observation: Mapping) -> dict:
    visible = visible_observation(observation)
    pixels = observation["screenshot"].tobytes(order="C")
    expected_size = math.prod(visible["screen_shape"])
    if len(pixels) != expected_size:
        raise ValueError("screenshot byte count disagrees with shape")
    canonical = json.dumps(visible, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"screenshot_sha256": hashlib.sha256(pixels).hexdigest(),
            "visible_dom_sha256": hashlib.sha256(canonical).hexdigest(),
            "screen_shape": visible["screen_shape"]}


def require_matching_initial(observation: Mapping, expected: Mapping) -> dict:
    actual = initial_fingerprint(observation)
    if actual != dict(expected):
        raise ValueError("paired initial screenshot/visible DOM mismatch; no retry or selective exclusion")
    return actual


def parse_action(text: str, visible: Mapping) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate action key")
            result[key] = value
        return result
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError("action must be a bounded JSON string")
    try:
        action = json.loads(text, object_pairs_hook=unique)
    except json.JSONDecodeError as exc:
        raise ValueError("one JSON object required") from exc
    schemas = {"done": {"action"}, "click": {"action", "ref"},
               "focus_type": {"action", "ref", "text"}, "type_text": {"action", "text"}}
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ValueError("invalid action object")
    kind = action["action"]
    if kind not in schemas or set(action) != schemas[kind]:
        raise ValueError("unknown action or fields")
    elements = {e["ref"]: e for e in visible["elements"] if e["ref"] > 0 and e["tag"] != "t"}
    if "ref" in action:
        if type(action["ref"]) is not int or action["ref"] not in elements:
            raise ValueError("ref must identify a current visible real element")
    if kind == "focus_type" and elements[action["ref"]]["tag"] not in TEXT_TAGS:
        raise ValueError("focus_type requires a text input")
    if kind == "type_text" and not any(e["focused"] and e["tag"] in TEXT_TAGS for e in elements.values()):
        raise ValueError("type_text requires a focused text input")
    if "text" in action and (not isinstance(action["text"], str) or len(action["text"]) > 256
                             or any(ord(c) < 32 or ord(c) > 126 for c in action["text"])):
        raise ValueError("text must be printable ASCII of at most 256 characters")
    return action


def make_native_action(env, action: dict, action_types=None):
    """Translate validated actions using the official helper, never raw JavaScript."""
    if action_types is None:
        from miniwob.action import ActionTypes as action_types
    kind = action["action"]
    if kind == "click":
        return env.unwrapped.create_action(action_types.CLICK_ELEMENT, ref=action["ref"])
    if kind == "focus_type":
        return env.unwrapped.create_action(action_types.FOCUS_ELEMENT_AND_TYPE_TEXT,
                                            ref=action["ref"], text=action["text"])
    if kind == "type_text":
        return env.unwrapped.create_action(action_types.TYPE_TEXT, text=action["text"])
    raise ValueError("done has no native action")


def binary_success(reward: Any, terminated: bool, truncated: bool) -> bool:
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("invalid termination flags")
    number = _number(reward)
    if not terminated and number != 0:
        raise ValueError("nonterminal binary reward must be zero")
    if terminated and number not in (-1, 1):
        raise ValueError("terminal binary reward must be -1 or 1, no partial credit")
    return terminated and not truncated and number == 1


def make_environment(case: Case, expected_versions: Mapping[str, str]):
    """Optional runtime factory; call only in an admitted isolated CPU/GPU worker.

    All five package versions must match the caller's frozen build manifest.
    Explicit Chrome/driver binaries prevent Selenium Manager auto-downloading.
    Parent must separately pin/hash those binaries and the installed task assets.
    """
    _check_case(case)
    if set(expected_versions) != REQUIRED_PACKAGES:
        raise ValueError("a complete frozen MiniWoB dependency manifest is required")
    for package, expected in expected_versions.items():
        if not isinstance(expected, str) or importlib.metadata.version(package) != expected:
            raise ValueError(f"unexpected installed version of {package}")
    for key in ("MINIWOB_CHROME_BINARY", "MINIWOB_CHROMEDRIVER"):
        value = os.environ.get(key)
        if not value or not Path(value).is_file():
            raise ValueError(f"preinstalled explicit {key} is required")
    import gymnasium
    import miniwob
    from miniwob.action import ActionSpaceConfig, ActionTypes
    from miniwob.reward import get_binary_reward
    gymnasium.register_envs(miniwob)
    config = ActionSpaceConfig(action_types=[ActionTypes.CLICK_ELEMENT, ActionTypes.TYPE_TEXT,
                                             ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT], text_max_len=256)
    return gymnasium.make(case.env_id, render_mode=None, action_space_config=config,
                          reward_processor=get_binary_reward, data_mode="test", refresh_freq=1)


def preflight_pair(case: Case, env_factory) -> dict:
    """Two fresh independent resets, no policy/model calls and no environment steps.

    No mismatch is discarded or retried. CPU caller supplies a process deadline.
    Use exactly the same pinned runtime later and compare every runtime reset.
    """
    _check_case(case)
    fingerprints = []
    for _ in range(2):
        env = env_factory(case)
        try:
            observation, _info = env.reset(seed=case.seed, options={"data_mode": "test", "record_screenshots": True})
            fingerprints.append(initial_fingerprint(observation))
        finally:
            env.close()
    if fingerprints[0] != fingerprints[1]:
        raise ValueError("CPU paired reset mismatch; freeze and investigate before model comparisons")
    return {"case_id": case.case_id, "env_id": case.env_id, "seed": case.seed,
            "initial": fingerprints[0], "model_calls": 0, "environment_steps": 0}


def _messages(observation: Mapping, history: list[dict], image_encoder=None) -> list[dict]:
    visible = visible_observation(observation)
    if image_encoder is None:
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(observation["screenshot"]).save(buffer, format="PNG")
        png = buffer.getvalue()
    else:
        png = image_encoder(observation["screenshot"])
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": [
        {"type": "text", "text": json.dumps(visible, ensure_ascii=False)},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}},
    ]}]


@dataclass
class EpisodeResult:
    case_id: str
    env_id: str
    seed: int
    initial: dict
    success: bool = False
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    policy_calls: int = 0
    environment_steps: int = 0
    invalid_actions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_usage_complete: bool = True
    elapsed_s: float = 0.0
    policy_elapsed_s: float = 0.0
    stop_reason: str = "not_started"
    forced_stop: bool = False
    abort_experiment: bool = False
    events: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


async def run_episode(env, case: Case, policy, expected_initial: Mapping,
                      limits: Limits = Limits(), *, image_encoder=None) -> EpisodeResult:
    """Run one already-created env; caller closes it and owns process containment.

    All native timers are left intact. Eight is a maximum action/call count, not
    a promise to complete eight actions before a native timeout. Synchronous
    Selenium reset/step calls require an external process watchdog. A policy
    timeout/exception is fatal to the entire model comparison, as in run_task.
    """
    _check_case(case)
    started = time.monotonic()
    observation, _info = env.reset(seed=case.seed, options={"data_mode": "test", "record_screenshots": True})
    initial = require_matching_initial(observation, expected_initial)
    result = EpisodeResult(case.case_id, case.env_id, case.seed, initial)
    history = []
    try:
        for step in range(limits.max_steps):
            remaining = limits.task_timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                result.stop_reason, result.forced_stop = "host_episode_limit", True
                return result
            visible = visible_observation(observation)
            messages = _messages(observation, history, image_encoder)
            result.policy_calls += 1
            policy_started = time.monotonic()
            try:
                reply = await asyncio.wait_for(policy(messages, limits.max_tokens), min(remaining, limits.call_timeout_s))
            except BaseException as exc:
                result.stop_reason, result.forced_stop = "policy_disposition_unknown", True
                result.abort_experiment, result.token_usage_complete = True, False
                result.events.append({"step": step+1, "error": f"{type(exc).__name__}: {exc}"[:512]})
                raise PolicyDispositionUnknown(result) from exc
            finally:
                result.policy_elapsed_s += time.monotonic() - policy_started
            if not isinstance(reply, PolicyReply):
                raise TypeError("policy must return PolicyReply")
            for key in ("prompt_tokens", "completion_tokens"):
                value = getattr(reply, key)
                if type(value) is int and value >= 0:
                    setattr(result, key, getattr(result, key) + value)
                else:
                    result.token_usage_complete = False
            event = {"step": step+1, "response": reply.text[:4096]}
            try:
                action = parse_action(reply.text, visible)
            except ValueError as exc:
                result.invalid_actions += 1
                result.stop_reason = "invalid_action"
                event["error"] = str(exc)
                result.events.append(event)
                return result
            event["action"] = action
            result.events.append(event)
            if action["action"] == "done":
                result.stop_reason = "premature_done"
                return result
            native = make_native_action(env, action)
            observation, reward, terminated, truncated, _info = env.step(native)
            result.environment_steps += 1
            result.reward = float(reward)
            result.terminated, result.truncated = bool(terminated), bool(truncated)
            result.success = binary_success(reward, result.terminated, result.truncated)
            if result.terminated or result.truncated:
                result.stop_reason = "native_success" if result.success else "native_failure"
                return result
            history.append({"role": "assistant", "content": reply.text[:4096]})
        result.stop_reason, result.forced_stop = "action_limit", True
        return result
    finally:
        result.elapsed_s = time.monotonic() - started


def preparation_plan() -> dict:
    """Explicit outstanding admission steps; this file alone is not runnable proof."""
    return {
        "status": "SOURCE_PREPARED_RUNTIME_UNVERIFIED",
        "benchmark": "MiniWoB++ fixed 12-family x 2-seed subset",
        "cases": [{"case_id": c.case_id, "env_id": c.env_id, "seed": c.seed} for c in cases()],
        "max_actions_per_episode": 8, "reward": "official get_binary_reward",
        "timer": "native unchanged; core default is 10000ms",
        "observations": "unmodified screenshot plus filtered visible DOM; no oracle/info/target flags",
        "pairing": "two independent CPU initial resets must match pixels and visible DOM; each model reset must match the frozen receipt",
        "runtime_prerequisites": [
            "Pin MiniWoB release/commit and task-asset hashes; reviewed main is not a pin.",
            "Pin all five Python dependencies and matching Chromium/ChromeDriver binaries; prohibit Selenium Manager downloads.",
            "Dedicated isolated worker with blocked egress and external wall/process deadline.",
            "Run 24 CPU paired-reset checks without model calls; stop on any mismatch, no retries or selective removal.",
            "Freeze candidate before the 24-case comparison; no patch selection using these outcomes.",
            "Allocate separately bounded remaining Modal credits before runtime execution.",
        ],
        "limitations": ["Not full MiniWoB, pixel-only, broad desktop ability, or Astra comparison.",
                        "Two seeds per family give a small, noisy sample.",
                        "Native wall-time deadlines mix decision quality and inference speed; binary rewards do not remove timeout."]
    }
