import asyncio
import base64
import copy
import json
import unittest

from weight_space_lab.eval.bu_policy import make_native_policy
from weight_space_lab.eval.fixtures import BrowserTask
from weight_space_lab.eval.runner import PolicyReply, SYSTEM_PROMPT, parse_action


TASK = BrowserTask("development-local", "development", "local", "Enter Ada",
                   "PRIVATE_HTML_ORACLE", ("e1", "e2"), ({"PRIVATE_REFERENCE_PLAN": True},))


def state():
    return {"text": "Name Save", "elements": [
        {"id": "e1", "tag": "input", "type": "text", "label": "Name", "value": "", "disabled": False},
        {"id": "e2", "tag": "button", "type": "button", "label": "Save", "value": "", "disabled": False},
    ]}


def messages(history=(), observation=None):
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history,
            {"role": "user", "content": [
                {"type": "text", "text": "Goal: " + TASK.goal + "\nCurrent browser observation:\n"
                 + json.dumps(state() if observation is None else observation, ensure_ascii=False)},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(b"same-image").decode()}},
            ]}]


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


if __name__ == "__main__":
    unittest.main()
