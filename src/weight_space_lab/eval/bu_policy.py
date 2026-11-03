"""Bridge the unchanged smoke runner to the constrained native BU protocol.

This module does not run a browser or model. The injected native policy receives
BU-style messages and returns PolicyReply. A complete native action batch is
validated before its first dispatch. Later actions use fresh observations and
zero additional model calls/tokens. Its native response and claims are
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
import copy
import json

from .bu_protocol import build_messages, parse_agent_output, revalidate_action
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
    """Join each prior dispatch, retaining each native batch exactly once.

    No runner rejection means only that the executor returned without a reported
    error. It is not a successful-task assertion; the fresh DOM remains decisive.
    """
    history = []
    items = messages[1:-1]
    cursor = 0
    last_rejection = None
    for number, receipt in enumerate(receipts, 1):
        if cursor >= len(items):
            raise ValueError("runner history omitted a returned dispatch")
        item = items[cursor]
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ValueError("unexpected history fields")
        role, content = item["role"], item["content"]
        if role != "assistant" or content != receipt["translated_response"]:
            raise ValueError("history does not match this adapter's returned action")
        cursor += 1
        rejection = None
        if cursor < len(items) and isinstance(items[cursor], Mapping) and items[cursor].get("role") == "user":
            feedback = items[cursor]
            rejection = feedback.get("content")
            if (set(feedback) != {"role", "content"} or not isinstance(rejection, str)
                    or len(rejection) > 4096 or not rejection.startswith("Action rejected: ")):
                raise ValueError("unrecognized action feedback")
            cursor += 1
        if "observed_rejection" in receipt and receipt["observed_rejection"] != rejection:
            raise ValueError("previous runner feedback changed")
        receipt["observed_rejection"] = rejection
        if receipt["native_response"] is not None:
            history.append({"role": "assistant", "content": receipt["native_response"]})
        feedback = rejection or (f"Dispatch {number}: {content}. Executor returned without a reported action rejection; "
                                 "task success is not inferred. Inspect the current browser state.")
        history.append({"role": "user", "content": feedback})
        last_rejection = rejection
    if cursor != len(items):
        raise ValueError("history contains an unowned dispatch or feedback")
    return history, last_rejection


def make_native_policy(task: BrowserTask, native_policy: Policy, *, events: list[dict] | None = None) -> Policy:
    """Return an async policy for one fresh run_task invocation.

    events, if supplied, must be an empty list; at most eight dispatch records are
    appended. kind distinguishes actual inference from queued dispatch. Native
    counts pass through without coercion; queued replies consume exactly 0/0
    tokens. The unchanged runner's policy_calls counts dispatches, not model
    requests. Call policy.finish() in the task's finally block to seal and discard
    unused queue entries when the oracle succeeds or the task otherwise ends.
    Exception/cancellation leaves the adapter sealed.
    """
    if events is None:
        events = []
    if not isinstance(events, list) or events:
        raise ValueError("a fresh empty native event list is required")
    receipts = []
    active = False
    calls = 0
    dispatches = 0
    queue = []
    batch_size = 0
    batch_position = 0
    ended = False

    async def policy(messages, cap):
        nonlocal active, calls, dispatches, queue, batch_size, batch_position, ended
        if active or ended:
            raise RuntimeError("prior native call has not completed; adapter cannot be reused")
        if dispatches >= 8 or len(events) != dispatches:
            raise RuntimeError("native dispatch ceiling or event ownership violated")
        if type(cap) is not int or not 1 <= cap <= 192:
            raise ValueError("native token cap must match the runner ceiling")
        observation, png = _current_observation(messages, task)
        history, rejected = _native_history(messages, receipts)
        discarded = 0
        if rejected and queue:
            discarded = len(queue)
            queue = []
            history[-1]["content"] += f" Remaining {discarded} queued actions were discarded without dispatch."
        kind = "queued_dispatch" if queue else "inference"
        native_messages = None
        if kind == "inference":
            native_messages = build_messages(task.goal, observation, png, task.allowed_ids, history=history)
        dispatches += 1
        active = True
        if kind == "inference":
            calls += 1
        event = {"dispatch": dispatches, "kind": kind, "call": calls,
                 "actual_model_calls": calls, "max_tokens": cap if kind == "inference" else 0,
                 "disposition": "entered"}
        if discarded:
            event["discarded_batch_actions"] = discarded
            event["queue_discard_reason"] = "prior_action_rejected"
        events.append(event)
        native_response = None
        if kind == "inference":
            try:
                reply = await native_policy(native_messages, cap)
            except BaseException as exc:
                event["disposition"] = "cancelled_unknown" if isinstance(exc, asyncio.CancelledError) else "error_unknown"
                event["error"] = f"{type(exc).__name__}: {exc}"[:512]
                raise
            if not isinstance(reply, PolicyReply) or not isinstance(reply.text, str):
                event["disposition"] = "returned_invalid_reply_type"
                raise TypeError("native policy must return PolicyReply with string text")
            native_response = reply.text[:4096]
            event.update({"native_response": native_response,
                          "native_response_truncated": len(reply.text) > 4096})
        else:
            reply = PolicyReply("", 0, 0)
        event["disposition"] = "returned"
        try:
            if kind == "inference":
                batch = parse_agent_output(reply.text, observation, task.allowed_ids)
                if len(batch.actions) > 9 - dispatches:
                    raise ValueError("native batch exceeds remaining dispatch budget")
                batch_size = len(batch.actions)
                batch_position = 1
                parsed = batch.actions[0]
                queue = list(batch.actions[1:])
                event["native_actions"] = copy.deepcopy([action.native_action for action in batch.actions])
            else:
                batch_position += 1
                parsed = revalidate_action(queue[0], observation, task.allowed_ids)
                queue.pop(0)
            translated = json.dumps(parsed.translated_action, ensure_ascii=False, separators=(",", ":"))
            event.update({"valid_native_schema": True, "native_action": copy.deepcopy(parsed.native_action),
                          "claimed_success": parsed.claimed_success, "done_text": parsed.done_text,
                          "batch_size": batch_size, "batch_position": batch_position,
                          "queued_actions_remaining": len(queue)})
            if parsed.translated_action["action"] == "done":
                ended = True
                queue = []
        except ValueError as exc:
            # This object is guaranteed to fail runner.parse_action. Passing the
            # raw response through could accidentally accept the old schema.
            error = str(exc)[:512]
            translated = json.dumps({"native_protocol_error": error}, separators=(",", ":"))
            event.update({"valid_native_schema": False, "protocol_error": error})
            if queue:
                event["discarded_batch_actions"] = len(queue)
                event["queue_discard_reason"] = "queued_action_stale"
            queue = []
            event["queued_actions_remaining"] = 0
        event["translated_response"] = translated
        receipts.append({"native_response": native_response, "translated_response": translated})
        active = False
        return PolicyReply(translated, reply.prompt_tokens, reply.completion_tokens)

    def finish():
        """Seal after task completion; this never cancels or stops native work."""
        nonlocal queue, ended
        if not ended and events:
            events[-1]["task_end_discarded_actions"] = len(queue)
            events[-1]["adapter_finished"] = True
            if active:
                events[-1]["native_disposition_still_unknown"] = True
        queue = []
        ended = True
        return {"actual_model_calls": calls, "policy_dispatches": dispatches,
                "native_disposition_unknown": active}

    policy.finish = finish
    return policy
