"""Screenshot-only browser-fixture actions, independent of a model provider."""
from dataclasses import dataclass
import json

SYSTEM_PROMPT = """Complete the user's task using only the current screenshot.
Coordinates are integer CSS pixels relative to the screenshot's top-left corner.
Return JSON only: {"actions":[{"click":{"x":120,"y":80}}]}.
Use 1..4 actions per reply and at most the remaining total action budget.
Available actions:
click: {"x":120,"y":80} left-clicks a visible position.
type: {"text":"hello"} types into the currently focused editable field; it appends.
key: {"name":"SELECT_ALL"} presses one of TAB, ENTER, ESCAPE, BACKSPACE, DELETE,
ARROWUP, ARROWDOWN, ARROWLEFT, ARROWRIGHT, HOME, END, SELECT_ALL, SPACE.
SELECT_ALL uses the platform's select-all shortcut in a focused text field.
scroll: {"x":200,"y":400,"dy":500} scrolls vertically at that visible position.
done: {} ends your attempt; your claim never determines success.
Use SELECT_ALL then type to replace text after focusing a field. Do not send code,
selectors, element IDs, URLs, prose, or predicted success. A batch uses the
current screenshot: stop the batch when you need to inspect changed content.
"""

KEYS = {"TAB": "Tab", "ENTER": "Enter", "ESCAPE": "Escape",
        "BACKSPACE": "Backspace", "DELETE": "Delete", "ARROWUP": "ArrowUp",
        "ARROWDOWN": "ArrowDown", "ARROWLEFT": "ArrowLeft", "ARROWRIGHT": "ArrowRight",
        "HOME": "Home", "END": "End", "SELECT_ALL": "ControlOrMeta+A", "SPACE": "Space"}


def strict_json(text, max_bytes=4096):
    if not isinstance(text, str) or len(text.encode()) > max_bytes:
        raise ValueError("response exceeds the JSON byte bound")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def reject(_):
        raise ValueError("nonfinite JSON number")
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject)


def parse_actions(text, width, height, remaining=12):
    """Validate the entire batch before dispatch; coordinates are not semantics."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("invalid viewport")
    if type(remaining) is not int or not 1 <= remaining <= 12:
        raise ValueError("invalid action budget")
    value = strict_json(text)
    if not isinstance(value, dict) or set(value) != {"actions"}:
        raise ValueError("only an actions envelope is allowed")
    actions = value["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= min(4, remaining):
        raise ValueError("batch exceeds its action budget")
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or len(action) != 1:
            raise ValueError("one tool per action")
        kind, args = next(iter(action.items()))
        schemas = {"click": {"x", "y"}, "type": {"text"}, "key": {"name"},
                   "scroll": {"x", "y", "dy"}, "done": set()}
        if kind not in schemas or not isinstance(args, dict) or set(args) != schemas[kind]:
            raise ValueError("unknown action or fields")
        if kind in ("click", "scroll"):
            if (type(args["x"]) is not int or type(args["y"]) is not int
                    or not 0 <= args["x"] < width or not 0 <= args["y"] < height):
                raise ValueError("coordinate outside current screenshot")
        if kind == "scroll" and (type(args["dy"]) is not int or not 0 < abs(args["dy"]) <= 700):
            raise ValueError("scroll must be a nonzero integer within700 pixels")
        if kind == "type" and (not isinstance(args["text"], str) or not 1 <= len(args["text"]) <= 128
                               or any(not 32 <= ord(c) <= 126 for c in args["text"])):
            raise ValueError("type requires1..128 printable ASCII characters")
        if kind == "key" and (not isinstance(args["name"], str) or args["name"] not in KEYS):
            raise ValueError("key is not allowed")
        if kind == "done" and index != len(actions) - 1:
            raise ValueError("done must be the final action")
    return actions


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class Limits:
    max_actions: int = 12
    max_output_tokens: int = 128
    call_timeout_s: float = 30.0
    case_timeout_s: float = 90.0
    close_timeout_s: float = 10.0

    def __post_init__(self):
        if type(self.max_actions) is not int or not 1 <= self.max_actions <= 12:
            raise ValueError("maximum12 actions")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 128:
            raise ValueError("maximum128 output tokens")
        for name, ceiling in (("call_timeout_s", 60), ("case_timeout_s", 90), ("close_timeout_s", 10)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value <= ceiling:
                raise ValueError("invalid bounded timeout")


class SessionAborted(RuntimeError):
    """Fatal across cases: unknown pending inference/native disposition, no retry."""
    def __init__(self, result):
        self.result = result
        super().__init__(result.get("stop_reason", "session aborted"))
