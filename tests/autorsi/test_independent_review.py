"""Independent transaction regressions using finite scalar fixtures only."""
from pathlib import Path
import tempfile
import threading
import time
import unittest

from autorsi import AutoRSI, EditRecipe, Goal, Guard, RunLimits, ScalarBackend, SequenceProposer, TensorEdit
from autorsi.backends import PromotionDeadlineError


def addition(value):
    return EditRecipe((TensorEdit("w", "add", value),), "local test hypothesis")


class IndependentTransactionTests(unittest.TestCase):
    def test_foreign_backend_snapshot_cannot_change_unallowlisted_weights(self):
        first = ScalarBackend({"w": 0.0, "protected": 1.0}, editable=["w"])
        other = ScalarBackend({"w": 0.0, "protected": 99.0}, editable=["w"])
        original = first.capture()
        foreign = other.capture()
        with tempfile.TemporaryDirectory() as directory:
            operations = (
                lambda: first.candidate(foreign, addition(0.05)),
                lambda: first.evaluate(foreign, lambda model: {"score": model["w"]}),
                lambda: first.promote(foreign, expected_revision=0),
                lambda: first.export(foreign, Path(directory)),
            )
            for operation in operations:
                with self.assertRaises(ValueError):
                    operation()
                self.assertEqual(first.capture().fingerprint, original.fingerprint)
                self.assertEqual(first.revision, 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_serialized_recipe_does_not_mutate_frozen_edits(self):
        recipe = addition(0.05)
        serialized = recipe.to_dict()
        serialized["edits"][0]["value"] = 99.0
        serialized["edits"][0]["parameter"] = "not-allowed"
        serialized["edits"].append({"parameter": "w", "operation": "add", "value": 1.0})
        self.assertEqual(recipe.edits, (TensorEdit("w", "add", 0.05),))
        self.assertEqual(recipe.to_dict()["edits"], [{"parameter": "w", "operation": "add", "value": 0.05}])

    def test_external_promotion_does_not_relabel_stale_metrics(self):
        backend = ScalarBackend({"w": 0.0}, editable=["w"], max_abs_delta=2.0, max_total_delta=2.0)

        class ConcurrentWriter:
            def propose(self, _):
                previous = backend.capture()
                backend.promote(backend.candidate(previous, addition(0.1)), expected_revision=previous.revision)
                return None

        with tempfile.TemporaryDirectory() as directory:
            controller = AutoRSI(backend=backend, evaluator=lambda m: {"score": m["w"]},
                                 goal=Goal("score"), proposer=ConcurrentWriter(), output_dir=directory)
            result = controller.run()
        self.assertEqual(result.status, "stale_revision")
        self.assertEqual(result.revision, 0)
        self.assertEqual(result.serving_revision, 1)
        self.assertEqual(result.metrics["score"], 0.0)
        self.assertEqual(result.accepted, 0)
        self.assertEqual(backend.capture().payload, {"w": 0.1})

    def test_target_reached_never_overrides_original_guard(self):
        for direction, protected in (("maximize", lambda w: 1.0-w), ("minimize", lambda w: w)):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as directory:
                backend = ScalarBackend({"w": 0.0}, editable=["w"], max_abs_delta=2.0, max_total_delta=2.0)
                controller = AutoRSI(backend=backend, evaluator=lambda m: {"score": m["w"], "protected": protected(m["w"])},
                                     goal=Goal("score", target=1.0), guards=[Guard("protected", direction=direction)],
                                     proposer=SequenceProposer([]), output_dir=directory)
                initial = backend.capture()
                backend.promote(backend.candidate(initial, addition(1.0)), expected_revision=initial.revision)
                result = controller.run()
                self.assertEqual(result.status, "current_regression")
                self.assertEqual(result.accepted, 0)
                self.assertEqual(result.revision, result.serving_revision)
                self.assertEqual(result.baseline["protected"], protected(0.0))
                self.assertEqual(result.metrics["protected"], protected(1.0))

    def test_safe_point_wait_deadline_does_not_publish_after_inference(self):
        backend = ScalarBackend({"w": 0.0}, editable=["w"])
        initial = backend.capture()
        candidate = backend.candidate(initial, addition(0.05))
        attempted = threading.Event()
        returned = threading.Event()
        errors = []

        def promote():
            attempted.set()
            try:
                backend.promote(candidate, expected_revision=0, deadline=time.monotonic()+0.04)
            except Exception as exc:
                errors.append(exc)
            finally:
                returned.set()

        with backend.inference() as serving:
            worker = threading.Thread(target=promote)
            worker.start()
            self.assertTrue(attempted.wait(1))
            self.assertTrue(returned.wait(1), "deadline must refuse while the inference safe point is still held")
            self.assertEqual(serving, {"w": 0.0})
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PromotionDeadlineError)
        self.assertEqual(backend.revision, 0)
        self.assertEqual(backend.capture().fingerprint, initial.fingerprint)

    def test_engine_passes_deadline_to_safe_point_promotion(self):
        backend = ScalarBackend({"w": 0.0}, editable=["w"])
        original_promote = backend.promote
        seen = []

        def observe(state, *, expected_revision, deadline=None):
            seen.append(deadline)
            self.assertIsNotNone(deadline)
            self.assertGreater(deadline, time.monotonic())
            return original_promote(state, expected_revision=expected_revision, deadline=deadline)

        backend.promote = observe
        with tempfile.TemporaryDirectory() as directory:
            controller = AutoRSI(backend=backend, evaluator=lambda m: {"score": m["w"]}, goal=Goal("score"),
                                 proposer=SequenceProposer([addition(0.05)]), output_dir=directory)
            result = controller.run(limits=RunLimits(max_seconds=10.0))
            self.assertTrue(Path(result.checkpoint).is_file())
        self.assertEqual(result.accepted, 1)
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
