import asyncio
import json
import pickle
import threading
import time
import unittest

from weight_space_lab.eval.fixtures import build_suite
from weight_space_lab.eval.runner import Limits, PolicyReply, PolicyDispositionUnknown, TaskResult, parse_action, run_task, ORACLE_JS


class FakeLocator:
    def __init__(self, page, element_id):
        self.page, self.element_id = page, element_id

    async def is_visible(self):
        return True

    async def is_enabled(self):
        return True

    async def fill(self, text, **kwargs):
        self.page.value = text

    async def click(self, **kwargs):
        self.page.success = self.page.value == "42"


class FakePage:
    """A test double, not a browser or model execution."""
    def __init__(self):
        self.success, self.value = False, ""
        self.network_blocked = False

    async def route(self, pattern, callback):
        self.network_blocked = pattern == "**/*"

    async def set_viewport_size(self, size):
        pass

    async def set_content(self, html, **kwargs):
        pass

    async def screenshot(self, **kwargs):
        return b"fake-image"

    async def evaluate(self, script):
        if script == ORACLE_JS:
            return self.success
        return {"text": "Example", "elements": [
            {"id": "e1", "tag": "input", "type": "text", "disabled": False},
            {"id": "e2", "tag": "button", "type": "button", "disabled": False},
        ]}

    def locator(self, selector):
        return FakeLocator(self, selector.split('"')[1])


class ActionTests(unittest.TestCase):
    def test_fatal_policy_receipt_survives_pickle_transport(self):
        result = TaskResult("development-example", "development", "example", "f" * 64,
                            policy_calls=5, action_errors=4, elapsed_s=180.001,
                            policy_elapsed_s=179.61, prompt_tokens=5829,
                            completion_tokens=50, token_usage_complete=False,
                            stop_reason="task_timeout_during_policy_disposition_unknown",
                            forced_stop=True, abort_experiment=True,
                            events=[{"step": 1, "response": '{"',
                                     "error": "ValueError: expected one JSON action object"}])
        original = PolicyDispositionUnknown(result)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(original, protocol=protocol))
                self.assertIs(type(restored), PolicyDispositionUnknown)
                self.assertIs(type(restored.result), TaskResult)
                self.assertIsNot(restored.result, result)
                self.assertEqual(restored.result.to_dict(), result.to_dict())
                self.assertEqual(restored.args, (result.stop_reason,))
                self.assertEqual(str(restored), result.stop_reason)
                self.assertTrue(restored.result.abort_experiment)
                self.assertTrue(restored.result.forced_stop)
                self.assertFalse(restored.result.token_usage_complete)

    def test_rejects_code_unknown_ids_and_schema_ambiguity(self):
        bad = ['{"action":"evaluate","code":"alert(1)"}',
               '{"action":"click","id":"e999"}',
               '{"action":"click","id":"e1","extra":true}',
               '{"action":"done","action":"click","id":"e1"}',
               '```json\n{"action":"done"}\n```', '[]',
               '{"action":"type","id":"e1","text":123}']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action(raw, ("e1",))

    def test_ceiling_cannot_be_silently_widened(self):
        for kwargs in ({"max_steps": 9}, {"max_tokens": 193}, {"call_timeout_s": 46}, {"task_timeout_s": 181}):
            with self.assertRaises(ValueError):
                Limits(**kwargs)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_claim_cannot_make_success(self):
        async def policy(messages, cap):
            return PolicyReply('{"action":"done"}', 5, 2)
        page = FakePage()
        result = await run_task(page, build_suite("development")[1], policy)
        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "premature_done")
        self.assertTrue(page.network_blocked)

    async def test_dom_success_and_usage_accounting(self):
        actions = iter([{"action": "type", "id": "e1", "text": "42"}, {"action": "click", "id": "e2"}])
        async def policy(messages, cap):
            self.assertEqual(cap, 192)
            self.assertEqual(messages[-1]["content"][1]["type"], "image_url")
            return PolicyReply(json.dumps(next(actions)), 10, 4)
        result = await run_task(FakePage(), build_suite("development")[1], policy)
        self.assertTrue(result.success)
        self.assertEqual(result.policy_calls, 2)
        self.assertEqual(result.executed_actions, 2)
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(result.completion_tokens, 8)

    async def test_invalid_action_consumes_step_and_does_not_execute(self):
        async def policy(messages, cap):
            return PolicyReply('{"action":"click","id":"e999"}')
        result = await run_task(FakePage(), build_suite("development")[1], policy, Limits(max_steps=2))
        self.assertEqual(result.stop_reason, "step_limit")
        self.assertEqual(result.action_errors, 2)
        self.assertEqual(result.executed_actions, 0)
        self.assertFalse(result.token_usage_complete)

    async def test_timeout_reports_unknown_remote_disposition(self):
        async def policy(messages, cap):
            await asyncio.sleep(1)
            return PolicyReply('{"action":"done"}')
        with self.assertRaises(PolicyDispositionUnknown) as caught:
            await run_task(FakePage(), build_suite("development")[1], policy, Limits(call_timeout_s=.01))
        result = caught.exception.result
        self.assertEqual(result.stop_reason, "policy_timeout")
        self.assertTrue(result.forced_stop)
        self.assertFalse(result.token_usage_complete)
        self.assertIn("remote_disposition_unknown", result.events[-1]["error"])

    async def test_thread_timeout_propagates_past_task_loop(self):
        native_finished = threading.Event()
        calls = []

        def simulated_native_call():
            time.sleep(.08)
            native_finished.set()
            return PolicyReply('{"action":"done"}')

        async def policy(messages, cap):
            calls.append(1)
            return await asyncio.to_thread(simulated_native_call)

        with self.assertRaises(PolicyDispositionUnknown) as caught:
            for task in build_suite("development")[:2]:
                await run_task(FakePage(), task, policy, Limits(call_timeout_s=.01))
        self.assertEqual(len(calls), 1)
        self.assertFalse(native_finished.is_set())
        self.assertTrue(caught.exception.result.abort_experiment)
        # Test cleanup only: the fake thread is finite. This wait is not a
        # production prescription to resume after an unknown native disposition.
        await asyncio.to_thread(native_finished.wait, .5)
        self.assertTrue(native_finished.is_set())

    async def test_total_task_timeout_during_policy_is_fatal(self):
        async def policy(messages, cap):
            await asyncio.sleep(1)
            return PolicyReply('{"action":"done"}')
        with self.assertRaises(PolicyDispositionUnknown) as caught:
            await run_task(FakePage(), build_suite("development")[1], policy,
                           Limits(task_timeout_s=.01, call_timeout_s=.1))
        self.assertEqual(caught.exception.result.stop_reason, "task_timeout_during_policy_disposition_unknown")

    async def test_task_deadline_preserves_four_returned_errors_and_pending_fifth_call(self):
        calls = 0

        async def policy(messages, cap):
            nonlocal calls
            calls += 1
            if calls <= 4:
                return PolicyReply('{"', 7, 1)
            await asyncio.sleep(1)
            return PolicyReply('{"action":"done"}')

        with self.assertRaises(PolicyDispositionUnknown) as caught:
            for task in build_suite("development")[:2]:
                await run_task(FakePage(), task, policy,
                               Limits(task_timeout_s=.05, call_timeout_s=.5))
        result = caught.exception.result
        self.assertEqual(calls, 5)
        self.assertEqual(result.policy_calls, 5)
        self.assertEqual(result.action_errors, 4)
        self.assertEqual(result.executed_actions, 0)
        self.assertEqual(len(result.events), 4)
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (28, 4))
        self.assertEqual(result.stop_reason, "task_timeout_during_policy_disposition_unknown")
        self.assertTrue(result.abort_experiment)
        self.assertFalse(result.token_usage_complete)

    async def test_policy_transport_error_is_fatal(self):
        async def policy(messages, cap):
            raise ConnectionError("response was lost")
        with self.assertRaises(PolicyDispositionUnknown) as caught:
            await run_task(FakePage(), build_suite("development")[1], policy)
        self.assertEqual(caught.exception.result.policy_calls, 1)
        self.assertTrue(caught.exception.result.abort_experiment)


if __name__ == "__main__":
    unittest.main()
