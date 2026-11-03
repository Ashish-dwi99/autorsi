"""Pure protocol for a live, unpaired entertainment-game demonstration.

No browser/provider work occurs here. The two actors share the same policy
surface. Labels do not identify this interactive assistant as a verified API
model or turn three unpaired game levels into a benchmark of model superiority.
"""
from __future__ import annotations
import base64
from dataclasses import dataclass
import hashlib
import json
import math
import re

LEVELS = (1, 3, 5)
ACTORS = ("adam", "assistant")
WIDTH, HEIGHT = 465, 600
VIEWPORT = {"width": 970, "height": 650}
MAX_ACTIONS, LEVEL_SECONDS = 32, 90
MAX_REQUESTS = MAX_ACTIONS * len(LEVELS)
EMBED_ORIGIN = "https://embed.neal.fun"
MAX_PNG, MAX_DOM, MAX_MESSAGE = 512 * 1024, 64 * 1024, 896 * 1024
IDENTITY_KEYS = ("session_id", "actor", "level", "seq", "observation_sha256", "frame_id", "task_phase")
TEXT_TAGS = frozenset({"input", "textarea"})
KEYS = frozenset({"Enter", "Tab", "Escape", "Space", "Backspace", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"})

SYSTEM_PROMPT = """Complete the displayed official Neal entertainment game using
only your own screenshot and visible DOM. This is an unpaired qualitative demo.
Return one complete Browser Use JSON object with optional brief memory and an
action array of1..8 actions, fitting the remaining32-action limit.
Tools: click {index:N} OR {coordinate_x:X,coordinate_y:Y}; input
{index:N,text:TEXT,clear:true} (false appends); scroll {down:true,pages:0.5};
send_keys {keys:KEY}; done {text:OBSERVED_OUTCOME,success:true}.
Coordinates are relative to your465x600 iframe screenshot. Keys allowed: Enter,
Tab,Escape,Space,Backspace,ArrowUp,ArrowDown,ArrowLeft,ArrowRight. A scroll has
pages>0 and<=1. No navigation, code, arbitrary selectors, files or downloads.
Every indexed target must be currently visible and enabled. Do not infer hidden
game state. The trusted game completion message, not your done claim, scores
completion. Native game timing and the90-second harness wall continue during
thinking, transport and actions. Never claim that this demo proves superiority.
"""


class DemoAbort(RuntimeError):
    """Whole-demo refusal; no retry or next-level continuation is implied."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(text):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out
    def nonfinite(_):
        raise ValueError("nonfinite JSON value")
    if not isinstance(text, str) or len(text.encode()) > 8192:
        raise ValueError("bounded JSON text required")
    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)


def validate_session(session):
    if not isinstance(session, str) or re.fullmatch(r"[A-Za-z0-9_-]{8,64}", session) is None:
        raise ValueError("session identifier must contain8..64 safe characters")
    return session


def make_observation(*, session_id, actor, level, seq, frame_id, png, visible_dom,
                     remaining_actions, remaining_wall_s, prior_action_result):
    validate_session(session_id)
    if actor not in ACTORS or level not in LEVELS or type(seq) is not int or not 1 <= seq <= MAX_REQUESTS:
        raise ValueError("invalid actor, frozen level or global peractor sequence")
    if not isinstance(frame_id, str) or not frame_id.startswith(session_id + ":" + str(level) + ":" + actor + ":"):
        raise ValueError("fresh session/level/actor frame identity required")
    if not isinstance(png, bytes) or not 0 < len(png) <= MAX_PNG:
        raise ValueError("PNG outside512KiB bound")
    if (not isinstance(visible_dom, dict) or set(visible_dom) != {"text", "elements"}
            or len(_canonical(visible_dom)) > MAX_DOM):
        raise ValueError("bounded visible-only DOM schema required")
    if type(remaining_actions) is not int or not 1 <= remaining_actions <= MAX_ACTIONS:
        raise ValueError("invalid action allowance")
    if type(remaining_wall_s) not in (int, float) or not math.isfinite(remaining_wall_s) or not 0 < remaining_wall_s <= LEVEL_SECONDS:
        raise ValueError("invalid remaining wall time")
    record = {"session_id": session_id, "actor": actor, "level": level, "seq": seq,
              "frame_id": frame_id, "task_phase": "active", "png_base64": base64.b64encode(png).decode(),
              "visible_dom": visible_dom, "remaining_actions": remaining_actions,
              "remaining_wall_s": remaining_wall_s, "prior_action_result": prior_action_result,
              "viewport": {"width": WIDTH, "height": HEIGHT}}
    record["observation_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    if len(_canonical(record)) >= MAX_MESSAGE:
        raise ValueError("observation outside transport bound")
    return record


def validate_command(command, observation):
    if not isinstance(command, dict) or set(command) != {*IDENTITY_KEYS, "native_action_json"}:
        raise ValueError("exact command envelope required")
    if any(command[key] != observation[key] or type(command[key]) is not type(observation[key]) for key in IDENTITY_KEYS):
        raise ValueError("stale or cross-session/frame/actor/phase command")
    if not isinstance(command["native_action_json"], str):
        raise ValueError("native action JSON must be text")
    return command["native_action_json"]


@dataclass(frozen=True)
class Action:
    native: dict
    kind: str
    index: int | None = None
    identity: tuple | None = None


def _action(value, visible):
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("one native tool per action")
    kind, params = next(iter(value.items()))
    if not isinstance(params, dict):
        raise ValueError("tool parameters must be an object")
    elements = {item["index"]: item for item in visible["elements"]}
    index, identity = None, None
    if kind == "click":
        if set(params) == {"index"}:
            index = params["index"]
        elif set(params) == {"coordinate_x", "coordinate_y"}:
            if (type(params["coordinate_x"]) is not int or type(params["coordinate_y"]) is not int
                    or not 0 <= params["coordinate_x"] < WIDTH or not 0 <= params["coordinate_y"] < HEIGHT):
                raise ValueError("coordinates must be inside the actor's iframe")
        else:
            raise ValueError("click uses either index or coordinates, never both")
    elif kind == "input":
        if not {"index", "text"} <= params.keys() or set(params) - {"index", "text", "clear"}:
            raise ValueError("invalid input fields")
        index = params["index"]
        if (not isinstance(params["text"], str) or len(params["text"]) > 256
                or any(ord(c) < 32 for c in params["text"]) or type(params.get("clear", True)) is not bool):
            raise ValueError("bounded text and boolean clear required")
    elif kind == "scroll":
        if (set(params) != {"down", "pages"} or type(params["down"]) is not bool
                or type(params["pages"]) not in (int, float) or not math.isfinite(params["pages"])
                or not 0 < params["pages"] <= 1):
            raise ValueError("bounded directional scroll required")
    elif kind == "send_keys":
        if set(params) != {"keys"} or not isinstance(params["keys"], str) or params["keys"] not in KEYS:
            raise ValueError("key is outside the frozen safe set")
    elif kind == "done":
        if (not {"text"} <= params.keys() or set(params) - {"text", "success"}
                or not isinstance(params["text"], str) or len(params["text"]) > 256
                or type(params.get("success", True)) is not bool):
            raise ValueError("bounded done claim required")
    else:
        raise ValueError("unsupported native action")
    if index is not None:
        if type(index) is not int or index not in elements:
            raise ValueError("index must be currently visible")
        item = elements[index]
        if item["enabled"] is not True:
            raise ValueError("target is disabled")
        if kind == "input" and item["tag"] not in TEXT_TAGS:
            raise ValueError("input requires a visible text field")
        identity = (index, item["tag"], item["text"])
    elif kind == "input" or kind == "click" and "index" in params:
        raise ValueError("missing real index")
    return Action(value, kind, index, identity)


def parse_batch(text, visible, remaining):
    output = strict_json(text)
    metadata = {"memory", "thinking", "evaluation_previous_goal", "next_goal"}
    if not isinstance(output, dict) or "action" not in output or set(output) - (metadata | {"action"}):
        raise ValueError("unknown native output field")
    for name in metadata & output.keys():
        if output[name] is not None and (not isinstance(output[name], str) or len(output[name]) > 2048):
            raise ValueError("bounded native metadata required")
    if type(remaining) is not int or not 1 <= remaining <= MAX_ACTIONS:
        raise ValueError("invalid remaining action count")
    raw = output["action"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= min(8, remaining):
        raise ValueError("batch exceeds remaining action allowance")
    result = tuple(_action(action, visible) for action in raw)
    if any(action.kind == "done" for action in result[:-1]):
        raise ValueError("done must be last")
    return result


def revalidate(action, visible):
    current = _action(action.native, visible)
    if current.identity != action.identity:
        raise ValueError("queued target identity changed")
    return current


def policy_messages(observation, history):
    # No other actor's pane, scorer data, network bodies or hidden game state.
    text = {key: observation[key] for key in ("level", "visible_dom", "remaining_actions", "remaining_wall_s", "prior_action_result", "viewport")}
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": [{"type": "text", "text": json.dumps(text, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + observation["png_base64"]}}]}]


def accept_completion(*, origin, source, expected_window, message, active, expected_level, current_level):
    """Pure specification mirrored by the owned wrapper listener; never emit an event."""
    return (active is True and expected_level == current_level and origin == EMBED_ORIGIN
            and source is expected_window and isinstance(message, dict) and message.get("type") == "captcha-completed")
