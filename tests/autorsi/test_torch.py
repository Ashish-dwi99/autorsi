from pathlib import Path
import tempfile
import unittest

from autorsi import AutoRSI, EditRecipe, Goal, Guard, RunLimits, SequenceProposer, TensorEdit, TorchBackend

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class TorchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = torch.nn.ModuleDict({"target": torch.nn.Linear(1, 1, bias=False),
                                          "other": torch.nn.Linear(1, 1, bias=False)})
        with torch.no_grad():
            self.model["target"].weight.fill_(1.0)
            self.model["other"].weight.fill_(3.0)
        self.x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])

    def evaluator(self, model):
        return {"mse": ((model["target"](self.x) - 2 * self.x) ** 2).mean().item(),
                "other_mse": ((model["other"](self.x) - 3 * self.x) ** 2).mean().item()}

    def test_real_weight_edit_regression_rejection_and_checkpoint_reload(self):
        backend = TorchBackend(self.model, editable={"target.weight", "other.weight"}, max_abs_delta=1.0)
        bad = EditRecipe((TensorEdit("target.weight", "add", 1.0), TensorEdit("other.weight", "add", -1.0)))
        good = EditRecipe((TensorEdit("target.weight", "add", 1.0),))
        controller = AutoRSI(backend=backend, evaluator=self.evaluator, goal=Goal("mse", "minimize", target=0.0),
                            guards=[Guard("other_mse", "minimize")], proposer=SequenceProposer([bad, good]),
                            output_dir=self.temp.name)
        original = backend.capture().fingerprint
        rejected = controller.step()
        self.assertEqual(rejected.accepted, 0)
        self.assertEqual(backend.capture().fingerprint, original)
        result = controller.step()
        self.assertEqual(result.status, "goal_reached")
        self.assertEqual(result.metrics, {"mse": 0.0, "other_mse": 0.0})
        # The model supplied by the caller is unchanged.
        self.assertEqual(self.model["target"].weight.item(), 1.0)
        state = torch.load(result.checkpoint, weights_only=True, map_location="cpu")
        self.assertEqual(state["target.weight"].item(), 2.0)
        self.assertTrue(torch.equal(state["other.weight"], self.model["other"].weight))
        self.model.load_state_dict(state)
        self.assertEqual(self.evaluator(self.model)["mse"], 0.0)
        with backend.inference() as served:
            self.assertFalse(served.training)
            self.assertTrue(torch.equal(served["target"](self.x), 2 * self.x))
        controller.rollback()
        self.assertEqual(backend.capture().fingerprint, original)

    def test_tied_parameters_and_nonfinite_inputs_are_rejected(self):
        self.model["other"].weight = self.model["target"].weight
        with self.assertRaisesRegex(ValueError, "tied/aliased"):
            TorchBackend(self.model, editable={"target.weight"})
        ordinary = torch.nn.Linear(1, 1)
        with torch.no_grad():
            ordinary.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            TorchBackend(ordinary, editable={"weight"})

    def test_rounding_is_checked_against_realized_delta(self):
        model = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.float16)
        with torch.no_grad():
            model.weight.fill_(1.0)
        backend = TorchBackend(model, editable={"weight"}, max_abs_delta=0.0006)
        with self.assertRaisesRegex(ValueError, "max_abs_delta"):
            backend.candidate(backend.capture(), EditRecipe((TensorEdit("weight", "add", 0.0006),)))

    def test_buffers_mutated_by_evaluator_are_rejected(self):
        model = torch.nn.BatchNorm1d(2).eval()
        backend = TorchBackend(model, editable={"weight"})
        def evaluator(candidate):
            candidate.running_mean.add_(1)
            return {"metric": 1.0}
        with self.assertRaisesRegex(ValueError, "mutated"):
            backend.evaluate(backend.capture(), evaluator)

    def test_targeted_cumulative_bound(self):
        backend = TorchBackend(self.model, editable={"target.weight"}, max_abs_delta=0.5, max_total_delta=0.6)
        first = backend.candidate(backend.capture(), EditRecipe((TensorEdit("target.weight", "add", 0.5),)))
        backend.promote(first, expected_revision=0)
        with self.assertRaisesRegex(ValueError, "max_total_delta"):
            backend.candidate(backend.capture(), EditRecipe((TensorEdit("target.weight", "add", 0.5),)))


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class HuggingFaceIntegrationTests(unittest.TestCase):
    def test_tiny_random_causal_lm_edit_and_reload_without_downloads(self):
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            self.skipTest("optional transformers package is not installed")
        torch.manual_seed(7)
        config = GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16, n_positions=8,
                            tie_word_embeddings=False, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        model = GPT2LMHeadModel(config).eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            original = model(tokens, output_hidden_states=True)
            desired_logits = original.logits * 0.5
            protected_hidden = original.hidden_states[-1].clone()
        def evaluate(candidate):
            output = candidate(tokens, output_hidden_states=True)
            return {"logit_mse": (output.logits - desired_logits).square().mean().item(),
                    "hidden_mse": (output.hidden_states[-1] - protected_hidden).square().mean().item()}
        backend = TorchBackend(model, editable={"lm_head.weight"}, max_abs_delta=0.2)
        recipe = EditRecipe((TensorEdit("lm_head.weight", "scale", 0.5),))
        with tempfile.TemporaryDirectory() as directory:
            controller = AutoRSI(backend=backend, evaluator=evaluate,
                                goal=Goal("logit_mse", "minimize", target=1e-12),
                                guards=[Guard("hidden_mse", "minimize")],
                                proposer=SequenceProposer([recipe]), output_dir=directory)
            result = controller.step()
            self.assertEqual(result.status, "goal_reached", result.to_dict())
            self.assertEqual(result.accepted, 1)
            self.assertGreater(result.baseline["logit_mse"], result.metrics["logit_mse"])
            self.assertEqual(result.metrics["hidden_mse"], 0.0)
            restored = GPT2LMHeadModel(config).eval()
            restored.load_state_dict(torch.load(result.checkpoint, weights_only=True, map_location="cpu"))
            with torch.inference_mode():
                self.assertTrue(torch.equal(restored(tokens).logits, desired_logits))
            for key, value in model.state_dict().items():
                if key != "lm_head.weight":
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
