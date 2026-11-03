"""Bridge the unchanged smoke runner to the constrained native BU protocol.

This module does not run a browser or model. The injected native policy receives
BU-style messages and returns PolicyReply. Its native response and claims are
kept separately from the single translated action returned to run_task. Invalid
native output becomes an explicitly invalid action response, preserving actual
usage and allowing the runner to count the action error. It never falls back to
accepting the old schema, repairs partial JSON, or invents success.

Cancellation/transport failure is propagated; the adapter remains consumed so
it cannot admit a replacement call. The outer experiment must stop and contain
the actual inference worker. No coroutine cancellation proves native shutdown.
"""

import asyncio
import base64
import binascii
from collections.abc import Mapping
import json

from .bu_protocol import build_messages, parse_agent_output
from .fixtures import BrowserTask
from .runner import Policy, PolicyReply, SYSTEM_PROMPT


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate observation JSON field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite observation value")


def _current_observation(messages, task: BrowserTask):
    """Accept only the runner's known first-system/latest-user message shape."""
    if not isinstance(messages, list) or not 2 <= len(messages) <= 18:
        raise ValueError("unexpected runner message count")
    if messages[0] != {"role": "system", "content": SYSTEM_PROMPT}:
        raise ValueError("runner system prompt changed")
    latest = messages[-1]
    if not isinstance(latest, Mapping) or set(latest) != {"role", "content"} or latest["role"] != "user":
        raise ValueError("missing current runner observation")
    content = latest["content"]
    if not isinstance(content, list) or len(content) != 2:
        raise ValueError("one current text observation and screenshot required")
    text_part, image_part = content
    if not isinstance(text_part, Mapping) or set(text_part) != {"type", "text"} or text_part["type"] != "text":
        raise ValueError("invalid observation text part")
    text = text_part["text"]
    prefix = "Goal: " + task.goal + "\nCurrent browser observation:\n"
    if not isinstance(text, str) or len(text) > 65536 or not text.startswith(prefix):
        raise ValueError("observation is not for this exact task goal")
    try:
        observation = json.loads(text[len(prefix):], object_pairs_hook=_unique, parse_constant=_nonfinite)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed runner observation") from exc
    if not isinstance(observation, dict) or set(observation) != {"text", "elements"}:
        raise ValueError("only visible text and elements are admitted")
    if (not isinstance(image_part, Mapping) or set(image_part) != {"type", "image_url"}
            or image_part["type"] != "image_url"):
        raise ValueError("invalid screenshot part")
    image_url = image_part["image_url"]
    if not isinstance(image_url, Mapping) or set(image_url) != {"url"}:
        raise ValueError("unexpected screenshot fields")
    url = image_url["url"]
    image_prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(image_prefix) or len(url) > 6 * 1024 * 1024:
        raise ValueError("only bounded inline PNG screenshots are admitted")
    try:
        png = base64.b64decode(url[len(image_prefix):], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("malformed screenshot encoding") from exc
    return observation, png


def _native_history(messages, receipts):
    """Replace translated assistant actions with the actual native responses."""
    history = []
    assistant_count = 0
    awaiting_feedback = False
    for item in messages[1:-1]:
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ValueError("unexpected history fields")
        role, content = item["role"], item["content"]
        if not isinstance(content, str) or len(content) > 4096:
            raise ValueError("history must be bounded text")
        if role == "assistant":
            if assistant_count >= len(receipts) or content != receipts[assistant_count]["translated_response"]:
                raise ValueError("history does not match this adapter's returned action")
            history.append({"role": "assistant", "content": receipts[assistant_count]["native_response"]})
            assistant_count += 1
            awaiting_feedback = True
        elif role == "user" and awaiting_feedback and content.startswith("Action rejected: "):
            history.append({"role": "user", "content": content})
            awaiting_feedback = False
        else:
            raise ValueError("unrecognized action feedback")
    if assistant_count != len(receipts):
        raise ValueError("runner history omitted a returned native call")
    return history


def make_native_policy(task: BrowserTask, native_policy: Policy, *, events: list[dict] | None = None) -> Policy:
    """Return an async policy for one fresh run_task invocation.

    events, if supplied, must be an empty list; at most eight bounded records are
    appended. Native token counts pass through without coercion, including None
    for unavailable usage. Exception/cancellation leaves the adapter sealed.
    """
    if events is None:
        events = []
    if not isinstance(events, list) or events:
        raise ValueError("a fresh empty native event list is required")
    receipts = []
    active = False
    calls = 0

    async def policy(messages, cap):
        nonlocal active, calls
        if active:
            raise RuntimeError("prior native call has not completed; adapter cannot be reused")
        if calls >= 8 or len(events) != calls:
            raise RuntimeError("native policy call ceiling or event ownership violated")
        if type(cap) is not int or not 1 <= cap <= 192:
            raise ValueError("native token cap must match the runner ceiling")
        observation, png = _current_observation(messages, task)
        history = _native_history(messages, receipts)
        native_messages = build_messages(task.goal, observation, png, task.allowed_ids, history=history)
        calls += 1
        active = True
        event = {"call": calls, "max_tokens": cap, "disposition": "entered"}
        events.append(event)
        try:
            reply = await native_policy(native_messages, cap)
        except BaseException as exc:
            event["disposition"] = "cancelled_unknown" if isinstance(exc, asyncio.CancelledError) else "error_unknown"
            event["error"] = f"{type(exc).__name__}: {exc}"[:512]
            raise
        if not isinstance(reply, PolicyReply) or not isinstance(reply.text, str):
            event["disposition"] = "returned_invalid_reply_type"
            raise TypeError("native policy must return PolicyReply with string text")
        event.update({"disposition": "returned", "native_response": reply.text[:4096],
                      "native_response_truncated": len(reply.text) > 4096})
        try:
            parsed = parse_agent_output(reply.text, observation, task.allowed_ids)
            translated = json.dumps(parsed.translated_action, ensure_ascii=False, separators=(",", ":"))
            event.update({"valid_native_schema": True, "native_action": parsed.native_action,
                          "claimed_success": parsed.claimed_success, "done_text": parsed.done_text})
        except ValueError as exc:
            # This object is guaranteed to fail runner.parse_action. Passing the
            # raw response through could accidentally accept the old schema.
            error = str(exc)[:512]
            translated = json.dumps({"native_protocol_error": error}, separators=(",", ":"))
            event.update({"valid_native_schema": False, "protocol_error": error})
        event["translated_response"] = translated
        receipts.append({"native_response": event["native_response"], "translated_response": translated})
        active = False
        return PolicyReply(translated, reply.prompt_tokens, reply.completion_tokens)

    return policy
