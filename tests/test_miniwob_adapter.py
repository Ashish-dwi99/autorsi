import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from weight_space_lab.benchmarks.miniwob import (
    cases, visible_observation, initial_fingerprint, require_matching_initial,
    parse_action, make_native_action, binary_success, preflight_pair, preparation_plan,
    run_episode,
)
from weight_space_lab.eval.runner import PolicyReply, PolicyDispositionUnknown


class Pixels:
    shape = (4, 5, 3)
    dtype = "uint8"

    def __init__(self, fill=0):
        self.fill = fill

    def tobytes(self, order):
        return bytes([self.fill]) * 60


def observation(fill=0):
    return {"utterance": "Type hello and click submit.", "fields": (("secret", "oracle"),),
            "screenshot": Pixels(fill), "raw_reward": 1, "dom_elements": [
                {"ref": 1, "parent": 0, "tag": "input_text", "text": "", "value": "", "id": "answer",
                 "classes": "target", "flags": [1, 1, 1, 1], "left": [0], "top": [0], "width": [3], "height": [2]},
                {"ref": 2, "tag": "button", "text": "Submit", "value": "", "flags": [0, 0, 0, 1],
                 "left": [0], "top": [2], "width": [3], "height": [2]},
                {"ref": -1, "tag": "t", "text": "Label", "value": "", "flags": [0, 0, 0, 1],
                 "left": [3], "top": [2], "width": [1], "height": [1]},
            ]}


class MiniWoBAdapterTests(unittest.TestCase):
    def test_fixed_manifest_has_12_families_two_seeds(self):
        result = cases()
        self.assertEqual(len(result), 24)
        self.assertEqual(len({c.family for c in result}), 12)
        self.assertEqual(len({c.seed for c in result}), 2)
        self.assertEqual(preparation_plan()["status"], "SOURCE_PREPARED_RUNTIME_UNVERIFIED")

    def test_policy_observation_excludes_oracle_and_target_flags(self):
        result = visible_observation(observation())
        encoded = json.dumps(result)
        for secret in ("oracle", "fields", "raw_reward", "target", "classes", "answer", "tampered"):
            self.assertNotIn(secret, encoded)
        self.assertTrue(result["elements"][0]["focused"])

    def test_pairing_checks_pixels_and_visible_dom(self):
        original = observation()
        expected = initial_fingerprint(original)
        self.assertEqual(require_matching_initial(observation(), expected), expected)
        with self.assertRaises(ValueError):
            require_matching_initial(observation(1), expected)
        changed = observation()
        changed["utterance"] = "Different goal"
        with self.assertRaises(ValueError):
            require_matching_initial(changed, expected)

    def test_actions_cannot_use_pseudo_elements_or_hidden_fields(self):
        visible = visible_observation(observation())
        valid = {"action": "focus_type", "ref": 1, "text": "hello"}
        self.assertEqual(parse_action(json.dumps(valid), visible), valid)
        bad = [
            '{"action":"click","ref":-1}', '{"action":"click","ref":999}',
            '{"action":"focus_type","ref":2,"text":"hello"}',
            '{"action":"click","ref":2,"script":"hack"}',
            '{"action":"click","action":"done"}',
            json.dumps({"action":"type_text", "text":"\ue003"}),
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_action(text, visible)

    def test_native_mapping_uses_official_action_helper(self):
        records = []
        env = SimpleNamespace(unwrapped=SimpleNamespace(create_action=lambda kind, **kw: records.append((kind, kw))))
        types = SimpleNamespace(CLICK_ELEMENT="click", TYPE_TEXT="type", FOCUS_ELEMENT_AND_TYPE_TEXT="focus")
        make_native_action(env, {"action":"focus_type", "ref":1, "text":"hello"}, types)
        self.assertEqual(records, [("focus", {"ref":1, "text":"hello"})])

    def test_binary_reward_cannot_count_partial_or_self_reported_success(self):
        self.assertTrue(binary_success(1, True, False))
        self.assertFalse(binary_success(-1, True, False))
        self.assertFalse(binary_success(0, False, False))
        with self.assertRaises(ValueError):
            binary_success(.5, True, False)
        with self.assertRaises(ValueError):
            binary_success(1, False, False)

    def test_pair_preflight_closes_both_envs_without_steps_or_policy(self):
        closed = []
        class FakeEnv:
            def reset(self, seed, options):
                return observation(), {"do_not_expose": "oracle"}
            def close(self):
                closed.append(True)
            def step(self, action):
                raise AssertionError("No steps during paired reset validation")
        record = preflight_pair(cases()[0], lambda case: FakeEnv())
        self.assertEqual(len(closed), 2)
        self.assertEqual(record["model_calls"], 0)
        self.assertEqual(record["environment_steps"], 0)


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_reward_controls_success_and_oracle_not_in_messages(self):
        class FakeEnv:
            def reset(self, seed, options):
                return observation(), {"secret_oracle": "not visible"}
            def step(self, native):
                return observation(), 1.0, True, False, {"secret_oracle": "not visible"}
        async def policy(messages, cap):
            self.assertNotIn("secret_oracle", json.dumps(messages))
            return PolicyReply('{"action":"click","ref":2}', 12, 3)
        with patch("weight_space_lab.benchmarks.miniwob.make_native_action", return_value={"native": "stub"}):
            result = await run_episode(FakeEnv(), cases()[0], policy, initial_fingerprint(observation()),
                                       image_encoder=lambda pixels: b"fake_png")
        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "native_success")
        self.assertEqual(result.policy_calls, 1)
        self.assertEqual(result.environment_steps, 1)

    async def test_pair_mismatch_prevents_any_policy_call(self):
        class FakeEnv:
            def reset(self, seed, options):
                return observation(1), {}
        async def policy(messages, cap):
            raise AssertionError("Must never be called on mismatched initial state")
        with self.assertRaisesRegex(ValueError, "paired initial"):
            await run_episode(FakeEnv(), cases()[0], policy, initial_fingerprint(observation()),
                              image_encoder=lambda pixels: b"fake_png")

    async def test_policy_failure_is_fatal_not_an_ordinary_task_loss(self):
        class FakeEnv:
            def reset(self, seed, options):
                return observation(), {}
        async def policy(messages, cap):
            raise ConnectionError("Lost generation receipt")
        with self.assertRaises(PolicyDispositionUnknown) as caught:
            await run_episode(FakeEnv(), cases()[0], policy, initial_fingerprint(observation()),
                              image_encoder=lambda pixels: b"fake_png")
        self.assertTrue(caught.exception.result.abort_experiment)
        self.assertEqual(caught.exception.result.environment_steps, 0)


if __name__ == "__main__":
    unittest.main()
