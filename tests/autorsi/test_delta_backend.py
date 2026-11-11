import tempfile
import unittest

from autorsi import (AutoRSI, DeltaEdit, DeltaTorchBackend, EditRecipe, GradientProposer, Goal, Guard,
                     RunLimits, SequenceProposer, TensorEdit)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class DeltaBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = torch.nn.ModuleDict({"target": torch.nn.Linear(2, 2, bias=False),
                                          "other": torch.nn.Linear(2, 2, bias=False)})
        with torch.no_grad():
            self.model["target"].weight.copy_(torch.eye(2))
            self.model["other"].weight.copy_(3 * torch.eye(2))
        self.x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def evaluator(self, model):
        return {"mse": ((model["target"](self.x) - 2 * self.x) ** 2).mean().item(),
                "other": ((model["other"](self.x) - 3 * self.x) ** 2).mean().item()}

    def test_no_model_copies_and_delta_edit_promotes(self):
        backend = DeltaTorchBackend(self.model, editable={"target.weight"}, max_abs_delta=1.0, copy=False)
        before = self.model["other"].weight.data_ptr()
        recipe = EditRecipe((DeltaEdit("target.weight", torch.eye(2)),))
        c = AutoRSI(backend=backend, evaluator=self.evaluator, goal=Goal("mse", "minimize", target=0.0),
                    guards=[Guard("other", "minimize")], proposer=SequenceProposer([recipe]),
                    output_dir=self.temp.name)
        result = c.step()
        self.assertEqual(result.status, "goal_reached")
        self.assertEqual(self.model["other"].weight.data_ptr(), before)  # frozen weights never copied
        with backend.inference() as m:
            self.assertTrue(torch.equal(m["target"].weight, 2 * torch.eye(2)))
        self.assertIn("sha256", result.attempts[0]["recipe"]["edits"][0])
        state = torch.load(result.checkpoint, weights_only=True)
        self.assertEqual(set(state), {"target.weight"})
        c.rollback()
        with backend.inference() as m:
            self.assertTrue(torch.equal(m["target"].weight, torch.eye(2)))

    def test_rejection_leaves_serving_weights_and_bounds_hold(self):
        backend = DeltaTorchBackend(self.model, editable={"target.weight", "other.weight"},
                                    max_abs_delta=0.5, max_total_delta=0.6, copy=True)
        bad = EditRecipe((TensorEdit("other.weight", "add", 0.4),))
        c = AutoRSI(backend=backend, evaluator=self.evaluator, goal=Goal("mse", "minimize"),
                    guards=[Guard("other", "minimize")], proposer=SequenceProposer(
                        [bad, EditRecipe((TensorEdit("target.weight", "add", 2.0),))]),
                    output_dir=self.temp.name)
        r = c.run(limits=RunLimits(max_steps=2))
        self.assertEqual(r.accepted, 0)
        self.assertEqual(r.attempts[0]["status"], "rejected")
        self.assertEqual(r.attempts[1]["status"], "failed")  # exceeds max_abs_delta
        self.assertEqual(backend.revision, 0)

    def test_only_editable_parameters_can_receive_gradients(self):
        DeltaTorchBackend(self.model, editable={"target.weight"}, copy=False)
        self.assertFalse(self.model["other"].weight.requires_grad)   # HF/torch default is True: a hidden RAM cost
        self.assertFalse(self.model["target"].weight.requires_grad)

    def test_evaluator_mutation_detected(self):
        backend = DeltaTorchBackend(self.model, editable={"target.weight"}, max_abs_delta=1.0)
        def evil(m):
            m["target"].weight.data.add_(1.0)
            return {"mse": 0.0}
        with self.assertRaisesRegex(ValueError, "mutated"):
            backend.evaluate(backend.capture(), evil)

    def test_gradient_proposer_reduces_loss_through_gates(self):
        backend = DeltaTorchBackend(self.model, editable={"target.weight"}, max_abs_delta=0.5, copy=True)
        loss_fn = lambda m: ((m["target"](self.x) - 2 * self.x) ** 2).mean()
        prop = GradientProposer(backend, loss_fn, scale=0.1, grow=1.0)
        c = AutoRSI(backend=backend, evaluator=self.evaluator, goal=Goal("mse", "minimize"),
                    guards=[Guard("other", "minimize")], proposer=prop, output_dir=self.temp.name)
        start = c.run(limits=RunLimits(max_steps=1)).baseline["mse"]
        r = c.run(limits=RunLimits(max_steps=8, max_accepted=8, max_evaluations=20))
        self.assertGreater(r.accepted, 0)
        self.assertLess(r.metrics["mse"], start)


if __name__ == "__main__":
    unittest.main()
