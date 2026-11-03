"""Fake adapter integration tests; no model, provider, browser or native calls."""
import unittest
from dataclasses import replace

from framework.adapters import (AdapterRegistry, AdapterOperationError, Operation,
                                Proposal, ResourceQuote, Usage)
from framework.contracts import (ExperimentConfig, GoalSpec, MetricSpec, MetricThreshold,
    RegressionGuard, EditorSpec, TargetSpec, RecipeSpec, EvaluatorSpec, BudgetSpec, EvaluationResult)
from framework.engine import run_experiment, BudgetExhausted, _Ledger


QUOTE = ResourceQuote(.1, 1, 1)
USAGE = Usage(.01, .1, 1)
ARCHITECTURES = frozenset({"dense", "moe", "vision_language"})


def config(domain="computer_use", architecture="vision_language", metrics=None, **budget):
    names = {"computer_use": "task_success", "math": "exact_accuracy", "coding": "tests_passed"}
    goal_metrics = metrics or (MetricSpec(names[domain], "maximize", "fraction", MetricThreshold(.9, ">=")),)
    bounds = dict(max_candidates=2, max_evaluations=3, max_editor_cost_usd=2,
                  max_target_cost_usd=10, max_wall_seconds=100, max_model_calls=100)
    bounds.update(budget)
    return ExperimentConfig(1, GoalSpec("goal", domain, "improve", goal_metrics),
        EditorSpec("fake_editor", "mock_editor"), TargetSpec("fake_target", "target", architecture, "revision", {"weights": "a" * 64}),
        RecipeSpec("bounded_strength", {"strength": 1.0}), EvaluatorSpec("measured", "development", "v1"), BudgetSpec(**bounds))


class Editor:
    provider = "fake_editor"
    supported_models = frozenset({"mock_editor", "independent_editor"})
    def __init__(self, proposals):
        self.proposals, self.calls, self.unknown_usage = list(proposals), 0, False
    def quote(self, config):
        return QUOTE
    def propose(self, config, baseline, history):
        self.calls += 1
        proposal = self.proposals.pop(0) if self.proposals else None
        return Operation(proposal, Usage(None, None, None) if self.unknown_usage else USAGE)


class Target:
    adapter_id = "fake_target"
    supported_architectures = ARCHITECTURES
    def __init__(self):
        self.state, self.restores, self.accepts = 0.0, 0, 0
        self.fail_restore = False
    def quote(self, operation, config):
        if operation not in {"snapshot", "restore", "accept"}:
            raise ValueError("unsupported operation")
        return QUOTE
    def snapshot(self, config):
        return Operation(self.state, USAGE)
    def restore(self, config, snapshot):
        self.restores += 1
        if self.fail_restore:
            raise AdapterOperationError("restore failed", disposition_known=True, usage=USAGE)
        self.state = snapshot
        return Operation(None, USAGE)
    def accept(self, config, snapshot, proposal):
        self.accepts += 1
        return Operation("owned-artifact-receipt", USAGE)


class Recipe:
    recipe_id = "bounded_strength"
    supported_architectures = ARCHITECTURES
    def __init__(self):
        self.failure = None
    def validate_parameters(self, parameters):
        if set(parameters) != {"strength"} or type(parameters["strength"]) not in (int, float) or not 0 <= parameters["strength"] <= 1:
            raise ValueError("only bounded strength is supported")
    def quote(self, config, proposal):
        return QUOTE
    def apply(self, config, target, snapshot, proposal):
        target.state = proposal.parameters["strength"]
        if self.failure:
            raise self.failure
        return Operation(None, USAGE)


class Evaluator:
    evaluator_id = "measured"
    supported_architectures = ARCHITECTURES
    supported_domains = frozenset({"math", "coding", "computer_use"})
    feedback_allowed = True
    def __init__(self):
        self.calls, self.tamper, self.regress, self.incomplete = 0, None, False, False
    def quote(self, config):
        return QUOTE
    def evaluate(self, config, target):
        self.calls += 1
        metrics = {spec.name: (.5 if target.state == 0 else target.state) for spec in config.goal.metrics}
        if "retained_quality" in metrics:
            metrics["retained_quality"] = .9 if target.state == 0 else (.8 if self.regress else .9)
        suite = config.evaluator.suite_revision
        if self.calls > 1 and self.tamper == "revision":
            suite = "easier-suite"
        if self.calls > 1 and self.tamper == "metric":
            metrics = {"claimed_success": 1.0}
        return Operation(EvaluationResult("measured", "development", suite, metrics,
            complete=not (self.calls > 1 and self.incomplete), artifact_id="measured-receipt"), USAGE)


def registry(proposals):
    editor, target, recipe, evaluator = Editor(proposals), Target(), Recipe(), Evaluator()
    result = AdapterRegistry()
    result.register_editor(editor)
    result.register_target(target)
    result.register_recipe(recipe)
    result.register_evaluator(evaluator)
    return result, editor, target, recipe, evaluator


class EngineTests(unittest.TestCase):
    def test_proposal_deep_freeze_survives_nested_source_mutation(self):
        parameters = {"nested": {"scales": [.5, .75]}}
        proposal = Proposal("bounded_strength", parameters)
        parameters["nested"]["scales"][0] = 999
        self.assertEqual(proposal.parameters["nested"]["scales"], (.5, .75))
        with self.assertRaises(TypeError):
            proposal.parameters["nested"]["new"] = True
        with self.assertRaises(TypeError):
            proposal.parameters["nested"]["scales"][0] = 1

    def test_independent_editor_and_target_support_math_coding_computer_use(self):
        for domain, architecture in (("math", "dense"), ("coding", "moe"), ("computer_use", "vision_language")):
            with self.subTest(domain=domain):
                reg, editor, target, _, evaluator = registry([Proposal("bounded_strength", {"strength": 1.0})])
                cfg = config(domain, architecture)
                cfg = replace(cfg, editor=EditorSpec("fake_editor", "independent_editor"))
                result = run_experiment(cfg, reg, clock=lambda: 0)
                self.assertEqual(result.status, "accepted")
                self.assertTrue(result.succeeded)
                self.assertEqual((target.accepts, target.restores, editor.calls, evaluator.calls), (1, 0, 1, 2))
                self.assertEqual(result.to_dict()["artifact_id"], "owned-artifact-receipt")

    def test_incompatible_architecture_rejected_before_any_callback(self):
        reg, editor, target, recipe, evaluator = registry([])
        recipe.supported_architectures = frozenset({"dense"})
        with self.assertRaisesRegex(ValueError, "architecture"):
            run_experiment(config(), reg)
        self.assertEqual((editor.calls, evaluator.calls, target.accepts), (0, 0, 0))

    def test_unavailable_editor_and_sealed_feedback_are_refused(self):
        reg, _, _, _, evaluator = registry([])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            run_experiment(replace(config(), editor=EditorSpec("fake_editor", "unavailable_Astra")), reg)
        evaluator.feedback_allowed = False
        with self.assertRaisesRegex(ValueError, "sealed"):
            run_experiment(config(), reg)

    def test_unmet_targets_restore_even_when_editor_claims_perfect_success(self):
        proposal = Proposal("bounded_strength", {"strength": .6}, "Perfect; all goals passed")
        reg, editor, target, _, _ = registry([proposal, proposal])
        result = run_experiment(config(), reg, clock=lambda: 0)
        self.assertEqual(result.status, "target_unmet")
        self.assertFalse(result.succeeded)
        self.assertEqual((target.state, target.restores, target.accepts), (0.0, 2, 0))

    def test_goal_pass_with_regression_restores(self):
        metrics = (MetricSpec("task_success", "maximize", "fraction", MetricThreshold(.9, ">=")),
                   MetricSpec("retained_quality", "maximize", "fraction", regression=RegressionGuard()))
        reg, _, target, _, evaluator = registry([Proposal("bounded_strength", {"strength": 1.0})])
        evaluator.regress = True
        result = run_experiment(config(metrics=metrics, max_candidates=1), reg, clock=lambda: 0)
        self.assertFalse(result.succeeded)
        self.assertEqual((target.restores, target.accepts), (1, 0))

    def test_metric_gaming_and_suite_change_restore_and_stop(self):
        for tamper in ("metric", "revision"):
            with self.subTest(tamper=tamper):
                reg, _, target, _, evaluator = registry([Proposal("bounded_strength", {"strength": 1.0})])
                evaluator.tamper = tamper
                result = run_experiment(config(), reg, clock=lambda: 0)
                self.assertEqual(result.status, "candidate_failed")
                self.assertEqual((target.restores, target.accepts), (1, 0))

    def test_incomplete_evaluation_is_not_success(self):
        reg, _, target, _, evaluator = registry([Proposal("bounded_strength", {"strength": 1.0})])
        evaluator.incomplete = True
        result = run_experiment(config(max_candidates=1), reg, clock=lambda: 0)
        self.assertEqual(result.status, "target_unmet")
        self.assertEqual(target.restores, 1)

    def test_reserved_rollback_prevents_mutation_when_budget_too_small(self):
        reg, _, target, _, evaluator = registry([Proposal("bounded_strength", {"strength": 1.0})])
        result = run_experiment(config(max_target_cost_usd=.3), reg, clock=lambda: 0)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual((target.state, target.restores, evaluator.calls), (0.0, 0, 1))

    def test_unknown_usage_retains_editor_reservation(self):
        reg, editor, _, _, _ = registry([None])
        editor.unknown_usage = True
        result = run_experiment(config(), reg, clock=lambda: 0)
        self.assertEqual(result.resources["editor_cost_charged_or_reserved_usd"], .1)
        self.assertTrue(result.resources["events"][-1]["unknown_usage_retains_reservation"])

    def test_apply_unknown_exception_does_not_issue_overlapping_restore(self):
        reg, _, target, recipe, evaluator = registry([Proposal("bounded_strength", {"strength": 1.0})])
        recipe.failure = ValueError("ordinary exception does not establish native completion")
        result = run_experiment(config(), reg, clock=lambda: 0)
        self.assertEqual(result.status, "blocked_target_state")
        self.assertEqual(result.target_state, "unknown_operation_disposition")
        self.assertEqual((target.restores, evaluator.calls, target.accepts), (0, 1, 0))

    def test_known_stopped_apply_failure_restores_once(self):
        reg, _, target, recipe, _ = registry([Proposal("bounded_strength", {"strength": 1.0})])
        recipe.failure = AdapterOperationError("stopped operation", disposition_known=True, usage=USAGE)
        result = run_experiment(config(), reg, clock=lambda: 0)
        self.assertEqual(result.target_state, "restored")
        self.assertEqual(target.restores, 1)

    def test_failed_restore_is_never_retried(self):
        reg, _, target, _, _ = registry([Proposal("bounded_strength", {"strength": .6})])
        target.fail_restore = True
        result = run_experiment(config(), reg, clock=lambda: 0)
        self.assertEqual(result.status, "blocked_target_state")
        self.assertEqual(target.restores, 1)

    def test_baseline_consumes_evaluation_slot(self):
        reg, editor, target, _, evaluator = registry([])
        result = run_experiment(config(max_evaluations=1), reg, clock=lambda: 0)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual((editor.calls, evaluator.calls, target.state), (0, 1, 0))

    def test_elapsed_time_reduces_future_admission(self):
        current = [0]
        ledger = _Ledger(config().budget, lambda: current[0])
        current[0] = 95
        with self.assertRaises(BudgetExhausted):
            ledger.admit([("target", ResourceQuote(0, 6, 0))])


if __name__ == "__main__":
    unittest.main()
