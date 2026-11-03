"""Pure proposal tests: no MiniWoB, Selenium, model, provider or browser calls."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

import miniwob_native as native
from miniwob_benchmark import validate_candidate, validate_preflight, summarize
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
    return json.dumps({"memory": "Observed task", "action": actions})


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
        with patch.object(native, "_reset", side_effect=lambda env, case: env.obs), patch.object(native, "messages", side_effect=encode):
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
    def admission(self, scores):
        evidence = {"candidate_scores": scores, "frozen_selection_rule": "successes descending; action errors ascending; .75 tie",
                    "v4_report_sha256": "a" * 64, "v4_status": "no_development_improvement"}
        selection = {"v4_report_sha256": "a" * 64, "v4_status": evidence["v4_status"],
                     "eligible_for_broader_benchmark": False,
                     "development": {str(s): {"tasks": 8, "successes": n, "action_errors": e} for n, e, s in scores}}
        return {"candidate_evidence": evidence, "candidate_selection": selection, "candidate_scale": .75}

    def test_prespecified_ties_and_errors_and_failed_promotion(self):
        for scores in ([[3, 1, .5], [3, 1, .75]], [[4, 2, .5], [4, 1, .75]]):
            admission = self.admission(scores)
            self.assertEqual(validate_candidate(admission), .75)
        wrong = self.admission([[8, 0, .5], [3, 0, .75]])
        with self.assertRaises(ValueError):
            validate_candidate(wrong)
        wrong["candidate_scale"] = .5
        self.assertEqual(validate_candidate(wrong), .5)

    def test_incomplete_denominator_stays24(self):
        row = {"success": True, "invalid_actions": 0, "model_calls": 1, "environment_steps": 1, "elapsed_s": 2}
        result = summarize([row])
        self.assertEqual(result["success_fraction_of_planned"], 1 / 24)
        self.assertEqual(result["missing_episodes"], 23)
        self.assertFalse(result["coverage_complete"])

    def test_preflight_requires_every_case_in_fixed_order(self):
        receipt = {"schema": 1, "native_timer_ms": 10000, "model_calls": 0,
                   "environment_steps": 48, "fresh_resets": 48, "cases": [
                       {"case_id": c.case_id, "env_id": c.env_id, "seed": c.seed,
                        "native_timer_ms": 10000, "initial": native.initial_fingerprint(observation())} for c in native.cases()]}
        self.assertEqual(len(validate_preflight(receipt)), 24)
        receipt["cases"].reverse()
        with self.assertRaises(ValueError):
            validate_preflight(receipt)


if __name__ == "__main__":
    unittest.main()
