"""Pure proposal tests: no MiniWoB, Selenium, model, provider or browser calls."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

import dev_native as native
from dev_benchmark import validate_preflight, summarize
from dev_contract import SCALES, cases, variant_order, select_development, experiment_spec, validate_spec
from weight_space_lab.eval.runner import PolicyReply


class Pixels:
    shape = (10, 10, 3)
    dtype = "uint8"
    def tobytes(self, order):
        return b"\0" * 300


def observation():
    def item(ref, tag, text, value=""):
        return {"ref": ref, "tag": tag, "text": text, "value": value,
                "left": [0], "top": [0], "width": [5], "height": [5],
                "flags": [0, 1, 1, 1], "id": "secret_target", "classes": "oracle"}
    return {"utterance": "Fill the two fields then submit.", "screenshot": Pixels(),
            "fields": [("oracle", "hidden")], "dom_elements": [
                item(1, "input_text", ""), item(2, "input_password", ""),
                item(3, "button", "Submit"), item(-1, "t", "Text")]}


def batch(actions):
    return json.dumps({"action": actions})


class NativeValidationTests(unittest.TestCase):
    def test_atomic_batch_and_invalid_nested_actions(self):
        visible = native.visible_observation(observation())
        invalid = [{"click": {"index": 3}}, {"input": {"index": 1, "text": "x", "js": "evil"}}]
        with self.assertRaises(ValueError):
            native.parse_batch(batch(invalid), visible)
        for index in (True, -1, 999, "3"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                native.parse_batch(batch([{"click": {"index": index}}]), visible)
        with self.assertRaises(ValueError):
            native.parse_batch('{"action":[{"click":{"index":3,"index":1}}]}', visible)
        with self.assertRaises(ValueError):
            native.parse_batch(batch([{"done": {"text": "yes"}}, {"click": {"index": 3}}]), visible)

    def test_clear_true_requires_empty_initial_and_fresh_values(self):
        obs = observation()
        action = native.parse_batch(batch([{"input": {"index": 1, "text": "x"}}]), native.visible_observation(obs))[0]
        obs["dom_elements"][0]["value"] = "existing"
        with self.assertRaisesRegex(ValueError, "nonempty"):
            native.revalidate(action, native.visible_observation(obs))
        append = native.parse_batch(batch([{"input": {"index": 1, "text": "x", "clear": False}}]), native.visible_observation(obs))
        self.assertFalse(append[0].clear)

    def test_stale_reference_identity_is_rejected(self):
        obs = observation()
        action = native.parse_batch(batch([{"click": {"index": 3}}]), native.visible_observation(obs))[0]
        obs["dom_elements"][2]["text"] = "Delete everything"
        with self.assertRaisesRegex(ValueError, "identity"):
            native.revalidate(action, native.visible_observation(obs))

    def test_visible_view_excludes_oracle(self):
        encoded = json.dumps(native.visible_observation(observation()))
        for token in ("oracle", "secret_target", '"fields":', "targeted", "tampered"):
            self.assertNotIn(token, encoded)

    def test_native_mapping_preserves_focus_append_semantics(self):
        # No import of MiniWoB; injected official enum names are sufficient.
        from types import SimpleNamespace
        seen = []
        env = SimpleNamespace(unwrapped=SimpleNamespace(create_action=lambda kind, **kw: seen.append((kind, kw))))
        types = SimpleNamespace(NONE="none", CLICK_ELEMENT="click", FOCUS_ELEMENT_AND_TYPE_TEXT="focus_type")
        with patch.dict("sys.modules", {"miniwob.action": SimpleNamespace(ActionTypes=types)}):
            native.native_action(env, native.Action({}, "input", ref=1, text="hello", clear=True))
            native.native_action(env, native.Action({}, "done"))
        self.assertEqual(seen, [("focus_type", {"ref": 1, "text": "hello"}), ("none", {})])


class FakeEnv:
    def __init__(self, *, success_at=3, stale=False):
        self.obs, self.steps, self.success_at, self.stale = observation(), [], success_at, stale
    def step(self, action):
        self.steps.append(action)
        if action.kind == "input":
            self.obs["dom_elements"][action.ref - 1]["value"] += action.text
        if self.stale:
            self.obs["dom_elements"][1]["tag"] = "div"
        terminal = len(self.steps) == self.success_at
        return self.obs, 1.0 if terminal else 0.0, terminal, False, {"hidden_answer": "never expose"}


class NativeEpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake(self, env, policy):
        saved = []
        def encode(obs, history):
            return [{"role": "user", "content": json.dumps(native.visible_observation(obs))}, *history]
        with patch.object(native, "_reset", side_effect=lambda env, case, publish: env.obs), patch.object(native, "messages", side_effect=encode):
            row = await native.run_episode(env, native.cases()[0], policy,
                native.initial_fingerprint(observation()), lambda row: saved.append(copy.deepcopy(row)),
                action_factory=lambda env, action: action)
        return row, saved

    async def test_full_native_batch_uses_one_call_and_counts_tokens_once(self):
        calls = []
        async def policy(messages, cap):
            calls.append((messages, cap))
            self.assertNotIn("hidden_answer", json.dumps(messages))
            return PolicyReply(batch([{"input": {"index": 1, "text": "a"}},
                {"input": {"index": 2, "text": "b"}}, {"click": {"index": 3}},
                {"done": {"text": "submitted", "success": True}}]), 100, 96)
        row, saved = await self.run_fake(FakeEnv(), policy)
        self.assertTrue(row["success"])
        self.assertEqual(calls[0][1],96)
        self.assertEqual((len(calls), row["model_calls"], row["environment_steps"]), (1, 1, 3))
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"]), (100, 96))
        self.assertEqual(row["discarded_queued_actions"], 1)
        self.assertEqual([event["kind"] for event in row["events"]], ["inference", "queued_dispatch", "queued_dispatch"])
        self.assertEqual(saved[-1], row)

    async def test_invalid_full_batch_never_executes_valid_prefix(self):
        env = FakeEnv()
        async def policy(messages, cap):
            return PolicyReply(batch([{"click": {"index": 3}}, {"click": {"index": 999}}]), 4, 5)
        row, _ = await self.run_fake(env, policy)
        self.assertEqual(env.steps, [])
        self.assertEqual(row["stop_reason"], "invalid_native_batch")

    async def test_stale_queue_ends_without_new_inference(self):
        calls = []
        async def policy(messages, cap):
            calls.append(True)
            return PolicyReply(batch([{"input": {"index": 1, "text": "a"}}, {"input": {"index": 2, "text": "b"}}]), 1, 2)
        row, _ = await self.run_fake(FakeEnv(stale=True), policy)
        self.assertEqual(row["stop_reason"], "stale_queued_action")
        self.assertEqual(row["environment_steps"], 1)
        self.assertEqual(len(calls), 1)

    async def test_done_claim_does_not_override_native_reward(self):
        async def policy(messages, cap):
            return PolicyReply(batch([{"done": {"text": "won", "success": True}}]), 1, 1)
        row, _ = await self.run_fake(FakeEnv(success_at=10), policy)
        self.assertFalse(row["success"])
        self.assertEqual(row["stop_reason"], "premature_done")

    async def test_native_timer_failure_is_recorded_without_model_retry(self):
        env = FakeEnv()
        env.step = lambda action: (env.obs, -1.0, True, False, {"reason": "timed out"})
        async def policy(messages, cap):
            return PolicyReply(batch([{"click": {"index": 3}}, {"click": {"index": 3}}]), 1, 2)
        row, _ = await self.run_fake(env, policy)
        self.assertEqual(row["stop_reason"], "native_failure_or_timeout")
        self.assertFalse(row["success"])
        self.assertEqual(row["model_calls"], 1)

    async def test_timeout_is_fatal_and_no_followup_native_action(self):
        env, calls = FakeEnv(), []
        async def policy(messages, cap):
            calls.append(True)
            raise TimeoutError("pending native request")
        with self.assertRaises(native.BenchmarkInferenceUnknown) as caught:
            await self.run_fake(env, policy)
        self.assertEqual(env.steps, [])
        self.assertEqual(len(calls), 1)
        self.assertFalse(caught.exception.partial["token_usage_complete"])
        self.assertEqual(caught.exception.partial["stop_reason"], "native_inference_disposition_unknown")

    async def test_budget_refusal_is_not_counted_as_an_actual_call(self):
        async def policy(messages, cap):
            raise native.BenchmarkBudgetStop("full")
        with self.assertRaises(native.BenchmarkBudgetStop):
            await self.run_fake(FakeEnv(), policy)


class AdmissionTests(unittest.TestCase):
    def test_incomplete_denominator_stays12(self):
        row = {"success": True, "invalid_actions": 0, "model_calls": 1, "environment_steps": 1, "elapsed_s": 2}
        result = summarize([row])
        self.assertEqual(result["success_fraction_of_planned"], 1 / 12)
        self.assertEqual(result["missing_episodes"], 11)
        self.assertFalse(result["coverage_complete"])

    def test_preflight_requires_every_case_in_fixed_order(self):
        receipt = complete_preflight_fixture()
        self.assertEqual(len(validate_preflight(receipt)), 12)
        receipt["cases"].reverse()
        with self.assertRaises(ValueError):
            validate_preflight(receipt)


def source_pin_fixture():
    return {"package_files_sha256": native.V1_MINIWOB_FILES_SHA256,
            "package_file_count": native.V1_MINIWOB_FILE_COUNT,
            "selected_source_files": copy.deepcopy(native.NATIVE_SOURCE_PINS)}


def complete_preflight_fixture():
    initial = native.initial_fingerprint(observation())
    return {"schema": 3, "phase":"development", "development_seed": native.DEVELOPMENT_SEED, "status": "completed", "source_pins": source_pin_fixture(),
            "native_timer_ms_by_family": dict(native.NATIVE_TIMER_MS_BY_FAMILY), "model_calls": 0,
            "environment_steps": 24, "fresh_resets": 24, "fresh_reset_attempts": 24, "cases": [
                {"case_id": c.case_id, "env_id": c.env_id, "seed": c.seed,
                 "native_timer_ms": native.expected_timer_ms(c), "status": "completed", "initial": initial,
                 "resets": [{"pair_member": member, "status": "completed", "close_returned": True,
                             "initial": initial, "none_step": {"reward": 0.0, "terminated": False, "truncated": False},
                             "timer": {"case_id": c.case_id, "env_id": c.env_id, "seed": c.seed,
                                       "status": "accepted", "expected_native_timer_ms": native.expected_timer_ms(c),
                                       "observed_timer_type": "int", "observed_timer_encoding": "json_scalar",
                                       "observed_timer_value": native.expected_timer_ms(c)}} for member in range(2)]}
                for c in native.cases()]}


class FakeResetEnv:
    def __init__(self, timer, error=None):
        from types import SimpleNamespace
        self.timer, self.error, self.closed = timer, error, False
        self.queries, self.reset_calls, self.step_calls = [], [], []
        self.unwrapped = SimpleNamespace(instance=SimpleNamespace(driver=self))
    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        if self.error:
            raise self.error
        return observation(), {"hidden_answer": "never expose"}
    def execute_script(self, script):
        self.queries.append(script)
        return self.timer
    def step(self, action):
        self.step_calls.append(action)
        return observation(), 0.0, False, False, {}
    def close(self):
        self.closed = True


class NativeTimerCorrectionTests(unittest.TestCase):
    def test_original_password_override_and_other_defaults(self):
        self.assertEqual(set(native.NATIVE_TIMER_MS_BY_FAMILY), set(native.FAMILIES))
        self.assertEqual([key for key, value in native.NATIVE_TIMER_MS_BY_FAMILY.items() if value != 10000],
                         ["enter-password"])
        for case in native.cases():
            env = FakeResetEnv(native.expected_timer_ms(case))
            events = []
            self.assertEqual(native._reset(env, case, events.append)["utterance"], observation()["utterance"])
            self.assertEqual(events[-1]["status"], "accepted")
            self.assertEqual(events[-1]["observed_timer_value"], env.timer)
            self.assertEqual(env.queries, ["return core.EPISODE_MAX_TIME;"])
            self.assertEqual(len(env.reset_calls), 1)

    def test_timer_value_and_type_survive_refusal(self):
        password = next(c for c in native.cases() if c.family == "enter-password")
        for value in (10000, 15000.0, None, True, "15000", float("nan")):
            events, env = [], FakeResetEnv(value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "task-specific"):
                native._reset(env, password, events.append)
            self.assertEqual(events[-1]["status"], "failed")
            self.assertEqual(events[-1]["case_id"], password.case_id)
            self.assertEqual(events[-1]["observed_timer_type"], type(value).__name__)
            self.assertEqual(len(env.queries), 1)
            json.dumps(events, allow_nan=False)

    def test_reset_exception_has_no_invented_timer_observation(self):
        events, env = [], FakeResetEnv(10000, RuntimeError("native reset failed"))
        with self.assertRaises(RuntimeError):
            native._reset(env, native.cases()[0], events.append)
        self.assertEqual(env.queries, [])
        self.assertEqual(events[-1]["status"], "failed")
        self.assertNotIn("observed_timer_value", events[-1])

    def test_source_mismatch_refuses_before_any_environment(self):
        saved = []
        with patch.object(native, "make_environment") as make, self.assertRaises(ValueError):
            native.preflight_suite({}, observed_source_files={}, save_partial=saved.append)
        make.assert_not_called()
        self.assertEqual(saved[-1]["status"], "failed")
        self.assertEqual(saved[-1]["fresh_reset_attempts"], 0)

    def test_preflight_publishes_partial_reset_failure_with_close(self):
        saved, env = [], FakeResetEnv(12345)
        with patch.object(native, "require_native_source_pins", return_value=source_pin_fixture()), \
             patch.object(native, "make_environment", return_value=env), self.assertRaises(ValueError):
            native.preflight_suite({}, observed_source_files={}, save_partial=saved.append)
        last = saved[-1]
        self.assertEqual(last["status"], "failed")
        self.assertEqual((last["fresh_reset_attempts"], last["fresh_resets"], last["environment_steps"]), (1, 1, 0))
        reset = last["cases"][0]["resets"][0]
        self.assertEqual(reset["timer"]["observed_timer_value"], 12345)
        self.assertEqual(reset["timer"]["status"], "failed")
        self.assertTrue(reset["close_returned"])
        self.assertTrue(env.closed)

    def test_complete_incremental_preflight_preserves_all_pairs(self):
        saved = []
        with patch.object(native, "require_native_source_pins", return_value=source_pin_fixture()), \
             patch.object(native, "make_environment", side_effect=lambda case, versions: FakeResetEnv(native.expected_timer_ms(case))), \
             patch.object(native, "native_action", side_effect=lambda env, action: action):
            result = native.preflight_suite({}, observed_source_files={}, save_partial=saved.append)
        self.assertEqual(result, saved[-1])
        self.assertEqual(len(validate_preflight(result)), 12)
        self.assertTrue(any(len(item["cases"]) == 1 and item["fresh_resets"] == 1 for item in saved))
        self.assertEqual(result["environment_steps"], 24)

    def test_preflight_rejects_stale_uniform_timer_and_incomplete_evidence(self):
        for change in ("timer", "source", "reset_count", "close", "pair_initial", "status", "float_timer"):
            receipt = complete_preflight_fixture()
            password = receipt["cases"][8]
            if change == "timer":
                password["native_timer_ms"] = 10000
            elif change == "source":
                receipt["source_pins"]["package_files_sha256"] = "0" * 64
            elif change == "reset_count":
                receipt["fresh_reset_attempts"] = 49
            elif change == "close":
                password["resets"][1]["close_returned"] = False
            elif change == "pair_initial":
                password["resets"][1]["initial"] = {}
            elif change == "status":
                receipt["status"] = "failed"
            elif change == "float_timer":
                password["resets"][1]["timer"]["observed_timer_value"] = 15000.0
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_preflight(receipt)


if __name__ == "__main__":
    unittest.main()


