import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from autorsi import (AutoRSI, EditRecipe, Goal, Guard, RunLimits, ScalarBackend,
                     SequenceProposer, StaleRevisionError, TensorEdit, RollbackJournalError)


def edit(value=1.0, protected=None):
    edits = [TensorEdit("target", "add", value)]
    if protected is not None:
        edits.append(TensorEdit("other", "add", protected))
    return EditRecipe(tuple(edits))


def evaluate(model):
    return {"score": model["target"], "other": model["other"]}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = ScalarBackend({"target": 0.0, "other": 1.0}, editable={"target", "other"},
                                     max_abs_delta=2.0, max_total_delta=5.0)

    def controller(self, recipes=(), **kwargs):
        return AutoRSI(backend=self.backend, evaluator=kwargs.pop("evaluator", evaluate),
                       proposer=kwargs.pop("proposer", SequenceProposer(recipes)),
                       goal=kwargs.pop("goal", Goal("score")),
                       guards=kwargs.pop("guards", [Guard("other")]), output_dir=self.temp.name, **kwargs)

    def test_reject_then_recursive_accept_export_and_rollback(self):
        source = self.backend.capture()
        controller = self.controller([edit(protected=-1), edit(), edit()])
        result = controller.run(limits=RunLimits(max_steps=3, max_accepted=2))
        self.assertEqual([a["status"] for a in result.attempts], ["rejected", "accepted", "accepted"])
        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.metrics, {"score": 2.0, "other": 1.0})
        self.assertEqual(json.loads(Path(result.checkpoint).read_text()), {"target": 2.0, "other": 1.0})
        events = [json.loads(line) for line in Path(result.journal).read_text().splitlines()]
        self.assertEqual(sum(e["event"] == "candidate_accepted" for e in events), 2)
        self.assertEqual(controller.rollback(), 3)
        self.assertEqual(controller.rollback(), 4)
        self.assertEqual(self.backend.capture().fingerprint, source.fingerprint)

    def test_rejected_candidate_preserves_exact_original(self):
        before = self.backend.capture().fingerprint
        result = self.controller([edit(protected=-1)]).step()
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.attempts[0]["status"], "rejected")
        self.assertEqual(self.backend.capture().fingerprint, before)
        self.assertIsNone(result.checkpoint)

    def test_original_guard_reference_across_separate_steps(self):
        controller = self.controller([edit(protected=-0.25), edit(protected=-0.25)],
                                     guards=[Guard("other", max_regression=0.3)])
        self.assertEqual(controller.step().accepted, 1)
        result = controller.step()
        self.assertEqual(result.accepted, 0)
        self.assertEqual(self.backend.capture().payload["other"], 0.75)

    def test_cumulative_bound_cannot_be_bypassed_by_recursion(self):
        self.backend = ScalarBackend({"target": 0.0, "other": 1.0}, editable={"target"},
                                     max_abs_delta=1.0, max_total_delta=1.5)
        result = self.controller([edit(), edit()]).run(limits=RunLimits(max_steps=2, max_accepted=2))
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.status, "error")
        self.assertEqual(self.backend.capture().payload["target"], 1.0)

    def test_exception_and_nan_leave_active_state_unchanged(self):
        before = self.backend.capture().fingerprint
        for bad in (lambda model: {"score": float("nan"), "other": 1.0},
                    lambda model: (_ for _ in ()).throw(RuntimeError("private credential"))):
            with self.subTest(callback=bad):
                result = self.controller([edit()], evaluator=bad).step()
                self.assertEqual(result.status, "error")
                self.assertEqual(self.backend.capture().fingerprint, before)
                self.assertNotIn("private credential", result.error)

    def test_candidate_evaluator_failure_preserves_live_weights(self):
        def bad_candidate(model):
            if model["target"]:
                raise RuntimeError("failed evaluation")
            return evaluate(model)
        result = self.controller([edit()], evaluator=bad_candidate).step()
        self.assertEqual(result.status, "error")
        self.assertEqual(self.backend.revision, 0)

    def test_mutating_evaluator_is_rejected(self):
        def mutate(model):
            model["target"] = 20.0
            return evaluate(model)
        result = self.controller([edit()], evaluator=mutate).step()
        self.assertEqual(result.status, "error")
        self.assertEqual(self.backend.capture().payload["target"], 0.0)

    def test_missing_metric_refused(self):
        result = self.controller([edit()], evaluator=lambda _: {"score": 0.0}).step()
        self.assertEqual(result.status, "error")
        self.assertEqual(self.backend.revision, 0)

    def test_proposer_exception_sanitized(self):
        class Broken:
            def propose(self, context):
                raise ValueError("secret api key")
        result = self.controller(proposer=Broken()).step()
        self.assertEqual(result.status, "error")
        self.assertNotIn("secret api key", result.error)
        self.assertEqual(self.backend.revision, 0)

    def test_stale_candidate_cannot_overwrite_new_revision(self):
        original = self.backend.capture()
        candidate = self.backend.candidate(original, edit())
        self.backend.promote(candidate, expected_revision=0)
        with self.assertRaises(StaleRevisionError):
            self.backend.promote(original, expected_revision=0)
        self.assertEqual(self.backend.capture().payload["target"], 1.0)

    def test_promotion_waits_for_inference_exit(self):
        candidate = self.backend.candidate(self.backend.capture(), edit())
        attempted, done = threading.Event(), threading.Event()
        errors = []
        def promote():
            attempted.set()
            try:
                self.backend.promote(candidate, expected_revision=0)
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()
        with self.backend.inference() as current:
            thread = threading.Thread(target=promote)
            thread.start()
            self.assertTrue(attempted.wait(1))
            self.assertFalse(done.wait(0.05))
            self.assertEqual(current["target"], 0.0)
        self.assertTrue(done.wait(2))
        thread.join()
        self.assertFalse(errors)
        self.assertEqual(self.backend.capture().payload["target"], 1.0)

    def test_promotion_inside_own_inference_is_refused(self):
        candidate = self.backend.candidate(self.backend.capture(), edit())
        with self.backend.inference():
            with self.assertRaisesRegex(RuntimeError, "inside inference"):
                self.backend.promote(candidate, expected_revision=0)

    def test_validation_gate_and_private_feedback(self):
        contexts = []
        class Propose:
            def propose(self, context):
                contexts.append(context)
                return edit()
        def validation(model):
            return {"score": -model["target"], "other": model["other"]}
        result = self.controller(proposer=Propose(), validation_evaluator=validation).run(
            limits=RunLimits(max_steps=2))
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.evaluations, 6)
        self.assertNotIn("validation_metrics", contexts[1].history[0])
        self.assertEqual(contexts[1].history[0]["reasons"], ["validation gate rejected candidate"])

    def test_limit_counts_baseline_and_rejects_late_evaluation(self):
        result = self.controller([edit()]).run(limits=RunLimits(max_evaluations=1))
        self.assertEqual(result.status, "limit_reached")
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(self.backend.revision, 0)
        def slow(model):
            if model["target"]:
                time.sleep(0.03)
            return evaluate(model)
        result = self.controller([edit()], evaluator=slow).run(limits=RunLimits(max_seconds=0.02))
        self.assertEqual(result.status, "limit_reached")
        self.assertEqual(result.accepted, 0)

    def test_export_failure_leaves_active_weights_unchanged(self):
        with patch.object(self.backend, "export", side_effect=OSError("disk full")):
            result = self.controller([edit()]).step()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.accepted, 0)
        self.assertEqual(self.backend.revision, 0)

    def test_pre_promotion_journal_failure_leaves_active_unchanged(self):
        with patch("autorsi.engine.os.fsync", side_effect=OSError("disk full")):
            result = self.controller([edit()]).step()
        self.assertEqual(result.status, "journal_error")
        self.assertEqual(result.accepted, 0)
        self.assertEqual(self.backend.revision, 0)

    def test_post_promotion_journal_failure_reports_actual_acceptance(self):
        import os
        real_fsync = os.fsync
        def fail_after_accept(fd):
            if self.backend.revision:
                raise OSError("disk full after promotion")
            return real_fsync(fd)
        with patch("autorsi.engine.os.fsync", side_effect=fail_after_accept):
            result = self.controller([edit()]).step()
        self.assertEqual(result.status, "journal_error")
        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.revision, 1)
        self.assertTrue(Path(result.checkpoint).is_file())
        self.assertEqual(self.backend.capture().payload["target"], 1.0)

    def test_goal_already_reached_does_not_request_proposal(self):
        result = self.controller([edit()], goal=Goal("score", target=0.0)).step()
        self.assertEqual(result.status, "goal_reached")
        self.assertEqual(result.accepted, 0)
        self.assertEqual(result.attempts, ())

    def test_rollback_prepared_journal_failure_does_not_restore_or_pop(self):
        controller = self.controller([edit()])
        controller.step()
        with patch("autorsi.engine.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "journal write failed"):
                controller.rollback()
        self.assertEqual(self.backend.revision, 1)
        self.assertEqual(self.backend.capture().payload["target"], 1.0)
        self.assertEqual(controller.rollback(), 2)
        self.assertEqual(self.backend.capture().payload["target"], 0.0)

    def test_rollback_final_journal_failure_reports_restored_revision(self):
        controller = self.controller([edit()])
        controller.step()
        import os
        real_fsync = os.fsync
        def fail_after_restore(fd):
            if self.backend.revision == 2:
                raise OSError("disk full")
            return real_fsync(fd)
        with patch("autorsi.engine.os.fsync", side_effect=fail_after_restore):
            with self.assertRaises(RollbackJournalError) as caught:
                controller.rollback()
        self.assertEqual(caught.exception.revision, 2)
        self.assertEqual(self.backend.capture().payload["target"], 0.0)
        with self.assertRaisesRegex(ValueError, "no accepted change"):
            controller.rollback()


if __name__ == "__main__":
    unittest.main()
