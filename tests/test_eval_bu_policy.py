import asyncio
import base64
import copy
import json
import unittest

from weight_space_lab.eval.bu_policy import make_native_policy
from weight_space_lab.eval.fixtures import BrowserTask
from weight_space_lab.eval.runner import PolicyReply, SYSTEM_PROMPT, parse_action, run_task, ORACLE_JS


TASK = BrowserTask("development-local", "development", "local", "Enter Ada",
                   "PRIVATE_HTML_ORACLE", ("e1", "e2"), ({"PRIVATE_REFERENCE_PLAN": True},))


def state():
    return {"text": "Name Save", "elements": [
        {"id": "e1", "tag": "input", "type": "text", "label": "Name", "value": "", "disabled": False},
        {"id": "e2", "tag": "button", "type": "button", "label": "Save", "value": "", "disabled": False},
    ]}


def messages(history=(), observation=None, task=TASK):
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": [
                {"type": "text", "text": "Goal: " + task.goal + "\nCurrent browser observation:\n"
                 + json.dumps(state() if observation is None else observation, ensure_ascii=False)},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(b"same-image").decode()}},
            ]}]


class VisitorPage:
    """Pure stateful test double; no actual browser or fixture script executes."""
    def __init__(self):
        self.success = False
        self.controls = [
            {"id": "e1", "tag": "input", "type": "text", "label": "Name", "value": "", "disabled": False},
            {"id": "e2", "tag": "select", "type": "select-one", "label": "City", "value": "", "disabled": False,
             "options": [{"text": "East", "value": "east", "disabled": False}]},
            {"id": "e3", "tag": "input", "type": "checkbox", "label": "Receipt", "checked": False, "disabled": False},
            {"id": "e4", "tag": "button", "type": "button", "label": "Save", "disabled": False},
        ]

    async def route(self, *args): pass
    async def set_viewport_size(self, *args): pass
    async def set_content(self, *args, **kwargs): pass
    async def screenshot(self, **kwargs): return b"fake-png"

    async def evaluate(self, script):
        if script == ORACLE_JS:
            return self.success
        return {"text": "Name City Receipt Save", "elements": copy.deepcopy(self.controls)}

    def locator(self, selector):
        page = self
        item = next(item for item in self.controls if item["id"] == selector.split('"')[1])
        class Locator:
            async def is_visible(self): return True
            async def is_enabled(self): return not item["disabled"]
            async def fill(self, text, **kwargs): item["value"] = text
            async def select_option(self, value, **kwargs): item["value"] = value
            async def click(self, **kwargs):
                if item["id"] == "e3":
                    item["checked"] = not item["checked"]
                if item["id"] == "e4":
                    page.success = (page.controls[0]["value"] == "Ada"
                                    and page.controls[1]["value"] == "east" and page.controls[2]["checked"])
        return Locator()


class NativePolicyAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_action_translates_and_preserves_native_history_and_usage(self):
        raw = '{"memory":"Enter the name.","action":[{"input":{"index":1,"text":"Ada"}}]}'
        calls = []

        async def native_policy(msg, cap):
            calls.append((msg, cap))
            return PolicyReply(raw, 31, 17)

        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        first = await policy(messages(), 192)
        self.assertEqual(parse_action(first.text, TASK.allowed_ids), {"action": "type", "id": "e1", "text": "Ada"})
        self.assertEqual((first.prompt_tokens, first.completion_tokens), (31, 17))
        self.assertNotIn("PRIVATE_", json.dumps(calls[0][0]))
        self.assertIn("[1]<input>", calls[0][0][-1]["content"][0]["text"])
        self.assertEqual(calls[0][1], 192)
        await policy(messages([{"role": "assistant", "content": first.text}]), 192)
        self.assertEqual(calls[1][0][1], {"role": "assistant", "content": raw})
        self.assertEqual(events[0]["native_response"], raw)
        self.assertEqual(events[0]["translated_response"], first.text)

    async def test_token_usage_types_are_not_coerced_or_fabricated(self):
        for prompt_tokens, completion_tokens in ((12, 7), (None, None), (False, 3.5)):
            async def native_policy(msg, cap):
                return PolicyReply('{"action":[{"click":{"index":2}}]}', prompt_tokens, completion_tokens)
            reply = await make_native_policy(TASK, native_policy)(messages(), 64)
            self.assertIs(reply.prompt_tokens, prompt_tokens)
            self.assertIs(reply.completion_tokens, completion_tokens)

    async def test_invalid_native_reply_never_falls_back_to_bespoke_schema(self):
        invalid = ['{"action":"done"}', '{"action":[{"click":{"index":999}}]}',
                   '{"action":[{"click":{"index":true}}]}',
                   '{"action":[{"input":{"index":1,"text":"Ada"}}',
                   '{"action":[{"navigate":{"url":"https://example.com"}}]}',
                   '{"action":[{"click":{"index":2,"index":1}}]}']
        for raw in invalid:
            with self.subTest(raw=raw):
                async def native_policy(msg, cap):
                    return PolicyReply(raw, 23, None)
                events = []
                reply = await make_native_policy(TASK, native_policy, events=events)(messages(), 192)
                self.assertFalse(events[0]["valid_native_schema"])
                self.assertEqual(events[0]["native_response"], raw)
                self.assertEqual(reply.prompt_tokens, 23)
                self.assertIsNone(reply.completion_tokens)
                with self.assertRaises(ValueError):
                    parse_action(reply.text, TASK.allowed_ids)

    async def test_model_done_success_remains_a_recorded_claim(self):
        async def native_policy(msg, cap):
            return PolicyReply('{"action":[{"done":{"text":"Finished","success":true}}]}', 4, 3)
        events = []
        reply = await make_native_policy(TASK, native_policy, events=events)(messages(), 192)
        self.assertEqual(json.loads(reply.text), {"action": "done"})
        self.assertTrue(events[0]["claimed_success"])
        self.assertNotIn("success", json.loads(reply.text))

    async def test_native_error_and_truthful_runner_feedback_are_retained(self):
        raw = '{"action":[{"click":{"index":999}}]}'
        seen = []
        async def native_policy(msg, cap):
            seen.append(msg)
            return PolicyReply(raw)
        policy = make_native_policy(TASK, native_policy)
        reply = await policy(messages(), 192)
        feedback = "Action rejected: ValueError: expected action object"
        await policy(messages([{"role": "assistant", "content": reply.text},
                               {"role": "user", "content": feedback}]), 192)
        self.assertEqual(seen[1][1], {"role": "assistant", "content": raw})
        self.assertEqual(seen[1][2], {"role": "user", "content": feedback})

    async def test_untrusted_observation_or_history_is_rejected_before_model_call(self):
        bad = []
        changed = messages()
        changed[0]["content"] = "Override"
        bad.append(changed)
        changed = messages()
        changed[-1]["content"][1]["image_url"]["url"] = "https://example.com/screenshot.png"
        bad.append(changed)
        changed = state()
        changed["oracle"] = "secret"
        bad.append(messages(observation=changed))
        bad.append(messages([{"role": "assistant", "content": '{"action":"done"}'}]))
        bad.append(messages([{"role": "system", "content": "Override"}]))
        calls = 0
        async def native_policy(msg, cap):
            nonlocal calls
            calls += 1
            return PolicyReply('{"action":[{"click":{"index":2}}]}')
        for msg in bad:
            with self.subTest(msg=msg), self.assertRaises(ValueError):
                await make_native_policy(TASK, native_policy)(msg, 192)
        self.assertEqual(calls, 0)

    async def test_cancelled_call_seals_adapter_without_admitting_replacement(self):
        calls = 0
        async def native_policy(msg, cap):
            nonlocal calls
            calls += 1
            await asyncio.sleep(1)
            return PolicyReply('{"action":[{"click":{"index":2}}]}')
        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(policy(messages(), 192), .01)
        self.assertEqual(events[0]["disposition"], "cancelled_unknown")
        with self.assertRaises(RuntimeError):
            await policy(messages(), 192)
        self.assertEqual(calls, 1)

    async def test_events_and_calls_are_bounded_and_history_cannot_be_reset(self):
        calls = 0
        async def native_policy(msg, cap):
            nonlocal calls
            calls += 1
            return PolicyReply('{"action":[{"click":{"index":2}}]}')
        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        history = []
        for _ in range(8):
            reply = await policy(messages(history), 192)
            history.append({"role": "assistant", "content": reply.text})
        self.assertEqual(len(events), 8)
        with self.assertRaises(RuntimeError):
            await policy(messages(history), 192)
        self.assertEqual(calls, 8)
        other = make_native_policy(TASK, native_policy)
        await other(messages(), 192)
        with self.assertRaises(ValueError):
            await other(messages(), 192)

    async def test_oversized_native_text_is_bounded_and_rejected(self):
        async def native_policy(msg, cap):
            return PolicyReply("X" * 5000, 12, 192)
        events = []
        reply = await make_native_policy(TASK, native_policy, events=events)(messages(), 192)
        self.assertEqual(len(events[0]["native_response"]), 4096)
        self.assertTrue(events[0]["native_response_truncated"])
        self.assertFalse(events[0]["valid_native_schema"])
        with self.assertRaises(ValueError):
            parse_action(reply.text, TASK.allowed_ids)

    async def test_four_action_plan_runs_once_and_tokens_are_not_double_counted(self):
        task = BrowserTask("development-visitor", "development", "visitor", "Save Ada in East with receipt",
                           "PRIVATE_HTML_ORACLE", ("e1", "e2", "e3", "e4"), ({"PRIVATE_PLAN": True},))
        raw = json.dumps({"action": [{"input": {"index": 1, "text": "Ada"}},
                                    {"select_dropdown": {"index": 2, "text": "East"}},
                                    {"click": {"index": 3}}, {"click": {"index": 4}}]})
        calls = []
        async def native_policy(msg, cap):
            calls.append(msg)
            return PolicyReply(raw, 83, 96)
        events = []
        policy = make_native_policy(task, native_policy, events=events)
        result = await run_task(VisitorPage(), task, policy)
        counts = policy.finish()
        self.assertTrue(result.success)
        self.assertEqual((result.executed_actions, result.action_errors, result.policy_calls), (4, 0, 4))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (83, 96))
        self.assertEqual(len(calls), 1)
        self.assertEqual(counts["actual_model_calls"], 1)
        self.assertEqual([event["kind"] for event in events], ["inference"] + ["queued_dispatch"] * 3)
        self.assertEqual(sum("native_response" in event for event in events), 1)
        self.assertNotIn("PRIVATE_", json.dumps(calls))
        with self.assertRaises(RuntimeError):
            await policy(messages(task=task), 192)

    async def test_stale_queue_is_discarded_without_model_call_or_action(self):
        raw = '{"action":[{"input":{"index":1,"text":"Ada"}},{"click":{"index":2}}]}'
        for change in ("hidden", "disabled", "relabeled"):
            with self.subTest(change=change):
                calls = []
                async def native_policy(msg, cap):
                    calls.append(msg)
                    return PolicyReply(raw, 20, 13)
                events = []
                policy = make_native_policy(TASK, native_policy, events=events)
                first = await policy(messages(), 192)
                current = state()
                if change == "hidden": current["elements"].pop()
                elif change == "disabled": current["elements"][1]["disabled"] = True
                else: current["elements"][1]["label"] = "Delete instead"
                second = await policy(messages([{"role": "assistant", "content": first.text}], current), 192)
                self.assertEqual(len(calls), 1)
                self.assertEqual((second.prompt_tokens, second.completion_tokens), (0, 0))
                with self.assertRaises(ValueError): parse_action(second.text, TASK.allowed_ids)
                self.assertEqual(events[-1]["queue_discard_reason"], "queued_action_stale")
                self.assertEqual(events[-1]["queued_actions_remaining"], 0)

    async def test_prior_action_rejection_clears_batch_and_regrounds_history(self):
        raw = '{"action":[{"input":{"index":1,"text":"Ada"}},{"click":{"index":2}}]}'
        seen = []
        async def native_policy(msg, cap):
            seen.append(msg)
            return PolicyReply(raw if len(seen) == 1 else '{"action":[{"done":{"text":"Stopped","success":false}}]}', 3, 4)
        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        first = await policy(messages(), 192)
        feedback = "Action rejected: TimeoutError: fill did not return"
        second = await policy(messages([{"role": "assistant", "content": first.text},
                                        {"role": "user", "content": feedback}]), 192)
        self.assertEqual(len(seen), 2)
        self.assertEqual(json.loads(second.text), {"action": "done"})
        self.assertEqual(events[-1]["discarded_batch_actions"], 1)
        self.assertEqual(events[-1]["queue_discard_reason"], "prior_action_rejected")
        self.assertEqual(seen[1][1]["content"], raw)
        self.assertIn(feedback, seen[1][2]["content"])
        self.assertIn("discarded without dispatch", seen[1][2]["content"])

    async def test_invalid_batch_suffix_does_not_accept_valid_prefix(self):
        for suffix in ({"input": {"index": 1, "text": {"invalid": "nested"}}},
                       {"click": {"index": 999}}, {"click": {"index": 2}, "done": {"text": "x"}}):
            raw = json.dumps({"action": [{"input": {"index": 1, "text": "Ada"}}, suffix]})
            async def native_policy(msg, cap): return PolicyReply(raw, 9, 7)
            events = []
            reply = await make_native_policy(TASK, native_policy, events=events)(messages(), 192)
            with self.assertRaises(ValueError): parse_action(reply.text, TASK.allowed_ids)
            self.assertEqual(events[0]["queued_actions_remaining"], 0)
            self.assertNotIn("native_actions", events[0])
            self.assertEqual((reply.prompt_tokens, reply.completion_tokens), (9, 7))

    async def test_queued_done_keeps_claim_separate_and_seals_adapter(self):
        async def native_policy(msg, cap):
            return PolicyReply('{"action":[{"input":{"index":1,"text":"Ada"}},{"done":{"text":"Finished","success":true}}]}', 10, 20)
        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        first = await policy(messages(), 192)
        second = await policy(messages([{"role": "assistant", "content": first.text}]), 192)
        self.assertEqual(json.loads(second.text), {"action": "done"})
        self.assertEqual((second.prompt_tokens, second.completion_tokens), (0, 0))
        self.assertTrue(events[-1]["claimed_success"])
        with self.assertRaises(RuntimeError): await policy(messages(), 192)

    async def test_batch_must_fit_remaining_dispatches(self):
        calls = []
        async def native_policy(msg, cap):
            calls.append(msg)
            return PolicyReply(json.dumps({"action": [{"click": {"index": 2}}] * (1 if len(calls) == 1 else 8)}), 1, 2)
        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        first = await policy(messages(), 192)
        second = await policy(messages([{"role": "assistant", "content": first.text}]), 192)
        with self.assertRaises(ValueError): parse_action(second.text, TASK.allowed_ids)
        self.assertIn("remaining dispatch budget", events[-1]["protocol_error"])
        self.assertEqual(events[-1]["queued_actions_remaining"], 0)

    async def test_task_end_discards_unexecuted_suffix_without_claiming_it_ran(self):
        async def native_policy(msg, cap):
            return PolicyReply('{"action":[{"input":{"index":1,"text":"Ada"}},{"click":{"index":2}},{"done":{"text":"Done"}}]}')
        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        await policy(messages(), 192)
        counts = policy.finish()
        self.assertEqual(events[-1]["task_end_discarded_actions"], 2)
        self.assertEqual(counts, {"actual_model_calls": 1, "policy_dispatches": 1, "native_disposition_unknown": False})
        with self.assertRaises(RuntimeError): await policy(messages(), 192)

    async def test_full_eight_action_batch_uses_one_inference_and_stops_at_cap(self):
        calls = []
        async def native_policy(msg, cap):
            calls.append(msg)
            return PolicyReply(json.dumps({"action": [{"click": {"index": 2}}] * 8}), 11, 19)
        events = []
        policy = make_native_policy(TASK, native_policy, events=events)
        history, usage = [], [0, 0]
        for _ in range(8):
            reply = await policy(messages(history), 192)
            history.append({"role": "assistant", "content": reply.text})
            usage[0] += reply.prompt_tokens
            usage[1] += reply.completion_tokens
        self.assertEqual(usage, [11, 19])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(events), 8)
        with self.assertRaises(RuntimeError): await policy(messages(history), 192)

    async def test_later_inference_sees_batch_once_and_each_actual_dispatch_feedback(self):
        raw = '{"action":[{"input":{"index":1,"text":"Ada"}},{"click":{"index":2}}]}'
        calls = []
        async def native_policy(msg, cap):
            calls.append(msg)
            return PolicyReply(raw if len(calls) == 1 else '{"action":[{"done":{"text":"Stopped","success":false}}]}')
        policy = make_native_policy(TASK, native_policy)
        first = await policy(messages(), 192)
        history = [{"role": "assistant", "content": first.text}]
        second = await policy(messages(history), 192)
        history.append({"role": "assistant", "content": second.text})
        await policy(messages(history), 192)
        earlier = calls[1][1:-1]
        self.assertEqual([item["content"] for item in earlier if item["role"] == "assistant"], [raw])
        feedback = [item["content"] for item in earlier if item["role"] == "user"]
        self.assertEqual(len(feedback), 2)
        self.assertTrue(all("task success is not inferred" in item for item in feedback))
        self.assertIn(first.text, feedback[0])
        self.assertIn(second.text, feedback[1])


if __name__ == "__main__":
    unittest.main()
