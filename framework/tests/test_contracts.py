import json
from dataclasses import replace
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

from framework.contracts import (
    BudgetSpec, ConfigurationError, EditorSpec, EvaluationResult, EvaluatorSpec,
    ExperimentConfig, GoalSpec, MetricSpec, MetricThreshold, RecipeSpec,
    RegressionGuard, TargetSpec, assess_metrics, config_from_dict, config_to_dict,
    dumps_config, loads_config, to_dict, validate_config,
)


def config(domain="math"):
    # Synthetic artifact IDs belong only to these pure unit fixtures.
    return ExperimentConfig(
        schema_version=1,
        goal=GoalSpec("test-goal", domain, "test-capability", (
            MetricSpec("accuracy", "maximize", "fraction", MetricThreshold(0.9, ">="), RegressionGuard()),
            MetricSpec("latency", "minimize", "seconds", MetricThreshold(0.1, "<=", "baseline")),
        )),
        editor=EditorSpec("test-editor", "editor-large", "https://editor.example/v1", "TEST_EDITOR_KEY"),
        target=TargetSpec("fake-target", "target-small", "fixture-architecture", "fixture-r1", {"checkpoint": "a" * 64}),
        recipe=RecipeSpec("fixture-scale", {"scale": 0.5, "layers": [1, 2]}),
        evaluator=EvaluatorSpec("trusted-fixture", "fixture-suite", "fixture-v1"),
        budget=BudgetSpec(2, 4, 0.25, 2.0, 60.0, 20),
    )


def result(metrics, **kwargs):
    return EvaluationResult("trusted-fixture", "fixture-suite", "fixture-v1", metrics, **kwargs)


class ConfigurationTests(unittest.TestCase):
    def test_editor_and_target_are_independent_identities(self):
        original = config()
        changed = replace(original, editor=EditorSpec("another-provider", "another-editor"))
        self.assertEqual(original.target, changed.target)
        self.assertNotEqual(original.editor, changed.editor)
        self.assertEqual(validate_config(changed), changed)

    def test_named_domains_are_configuration_not_hardcoded_logic(self):
        for domain in ("math", "coding", "computer_use", "molecule_prediction"):
            self.assertEqual(config_from_dict(config_to_dict(config(domain))).goal.domain, domain)

    def test_json_roundtrip_preserves_all_contracts(self):
        original = config()
        self.assertEqual(loads_config(dumps_config(original)), original)
        self.assertEqual(json.loads(dumps_config(original)), config_to_dict(original))

    def test_mutating_input_does_not_mutate_frozen_config(self):
        raw = config_to_dict(config())
        parsed = config_from_dict(raw)
        raw["target"]["artifact_pins"]["checkpoint"] = "b" * 64
        raw["recipe"]["parameters"]["layers"].append(3)
        self.assertEqual(parsed.target.artifact_pins["checkpoint"], "a" * 64)
        self.assertEqual(parsed.recipe.parameters["layers"], (1, 2))
        with self.assertRaises(TypeError):
            parsed.recipe.parameters["scale"] = 1.0

    def test_split_budgets_do_not_assume_editor_is_target(self):
        budget = BudgetSpec(1, 2, 0, 30, 100, 10)
        self.assertEqual(budget.max_cost_usd, 30)
        self.assertEqual(budget.max_editor_cost_usd, 0)
        self.assertEqual(budget.max_target_cost_usd, 30)

    def test_unknown_fields_and_embedded_secrets_rejected(self):
        raw = config_to_dict(config())
        raw["editor"]["api_key"] = "not-an-accepted-field"
        with self.assertRaises(ConfigurationError):
            config_from_dict(raw)
        for endpoint in ("https://user:password@example.com", "https://example.com?api_key=secret",
                         "file:///checkpoint", "https://exa mple.com", "https://example.com:invalid"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ConfigurationError):
                EditorSpec("provider", "editor", endpoint=endpoint)
        with self.assertRaises(ConfigurationError):
            EditorSpec("provider", "editor", credential_env="sk-not-an-env-reference")

    def test_duplicate_json_fields_do_not_silently_override(self):
        with self.assertRaises(ConfigurationError):
            loads_config('{"schema_version":1,"schema_version":2}')

    def test_nonfinite_and_boolean_numeric_fields_rejected(self):
        for value in (True, float("nan"), float("inf"), -(float("inf")), 10 ** 1000):
            with self.subTest(value=type(value).__name__), self.assertRaises(ConfigurationError):
                MetricThreshold(value, ">=")
        with self.assertRaises(ConfigurationError):
            BudgetSpec(True, 2, 0, 2, 10, 1)
        with self.assertRaises(ConfigurationError):
            loads_config('{"schema_version":NaN}')

    def test_negative_or_zero_invalid_budget_limits_rejected(self):
        for overrides in ({"max_candidates": 0}, {"max_evaluations": 0},
                          {"max_editor_cost_usd": -1}, {"max_target_cost_usd": -1},
                          {"max_wall_seconds": 0}, {"max_model_calls": -1}):
            with self.subTest(overrides=overrides), self.assertRaises(ConfigurationError):
                replace(config().budget, **overrides)

    def test_pin_and_schema_validation(self):
        for pins in ({}, {"checkpoint": "short"}, {"checkpoint": "A" * 64}):
            with self.subTest(pins=pins), self.assertRaises(ConfigurationError):
                replace(config().target, artifact_pins=pins)
        with self.assertRaises(ConfigurationError):
            replace(config(), schema_version=2)

    def test_metric_names_and_direction_are_unambiguous(self):
        metric = config().goal.metrics[0]
        with self.assertRaises(ConfigurationError):
            replace(config().goal, metrics=(metric, metric))
        with self.assertRaises(ConfigurationError):
            MetricSpec("accuracy", "maximize", "fraction", MetricThreshold(0.9, "<="))
        with self.assertRaises(ConfigurationError):
            MetricSpec("unconstrained", "maximize", "fraction")
        with self.assertRaises(ConfigurationError):
            MetricThreshold(1, "==")

    def test_invalid_unhashable_enum_values_report_configuration_error(self):
        for value in ([], {}, True):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                MetricSpec("score", value, "points", MetricThreshold(1, ">="))
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                RegressionGuard(mode=value)
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                MetricThreshold(1, value)

    def test_non_json_recipe_values_rejected(self):
        with self.assertRaises(ConfigurationError):
            RecipeSpec("recipe", {"callback": lambda: None})
        with self.assertRaises(ConfigurationError):
            RecipeSpec("recipe", {"value": float("nan")})


class MetricDecisionTests(unittest.TestCase):
    def test_absolute_relative_and_retention_must_all_pass(self):
        baseline = result({"accuracy": 0.92, "latency": 100})
        candidate = result({"accuracy": 0.94, "latency": 10})
        assessment = assess_metrics(config().goal, candidate, {"baseline": baseline})
        self.assertTrue(assessment.passed)
        self.assertEqual(len(assessment.checks), 3)
        self.assertEqual(to_dict(assessment)["failures"], [])

    def test_target_success_does_not_override_a_regression(self):
        baseline = result({"accuracy": 0.95, "latency": 100})
        candidate = result({"accuracy": 0.91, "latency": 5})
        assessment = assess_metrics(config().goal, candidate, {"baseline": baseline})
        self.assertFalse(assessment.passed)
        self.assertIn("accuracy: regression guard failed", assessment.failures)

    def test_missing_reference_does_not_invent_comparison(self):
        assessment = assess_metrics(config().goal, result({"accuracy": 1, "latency": 1}))
        self.assertFalse(assessment.passed)
        self.assertTrue(any("missing or incomplete reference" in item for item in assessment.failures))

    def test_named_comparator_requires_actual_matching_measurement(self):
        goal = GoalSpec("compare", "coding", "latency", (
            MetricSpec("latency", "minimize", "seconds", MetricThreshold(0.1, "<=", "reference_model")),))
        candidate = result({"latency": 1})
        self.assertFalse(assess_metrics(goal, candidate, {}).passed)
        self.assertTrue(assess_metrics(goal, candidate, {"reference_model": result({"latency": 10})}).passed)

    def test_zero_or_negative_ratio_reference_refuses(self):
        goal = GoalSpec("ratio", "math", "speed", (
            MetricSpec("latency", "minimize", "seconds", MetricThreshold(0.1, "<=", "baseline")),))
        for denominator in (0, -10):
            with self.subTest(denominator=denominator):
                self.assertFalse(assess_metrics(goal, result({"latency": 0}),
                    {"baseline": result({"latency": denominator})}).passed)

    def test_incomplete_and_different_suite_references_refuse(self):
        baseline = result({"accuracy": 0.9, "latency": 10})
        candidate = result({"accuracy": 1, "latency": 1})
        for reference in (replace(baseline, complete=False), replace(baseline, suite_revision="other"),
                          replace(baseline, evaluator_id="unmatched")):
            with self.subTest(reference=reference.suite_revision):
                self.assertFalse(assess_metrics(config().goal, candidate, {"baseline": reference}).passed)

    def test_incomplete_and_missing_or_extra_candidate_metrics_refuse(self):
        candidates = (result({"accuracy": 1, "latency": 1}, complete=False), result({"accuracy": 1}),
                      result({"accuracy": 1, "latency": 1, "invented_success": 1}))
        for candidate in candidates:
            self.assertFalse(assess_metrics(config().goal, candidate, {}).passed)

    def test_nonfinite_result_cannot_be_constructed(self):
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                result({"metric": value})

    def test_strict_threshold_does_not_add_hidden_epsilon(self):
        goal = GoalSpec("strict", "any", "score", (
            MetricSpec("score", "maximize", "points", MetricThreshold(5, ">")),))
        self.assertFalse(assess_metrics(goal, result({"score": 5})).passed)
        self.assertTrue(assess_metrics(goal, result({"score": 6})).passed)

    def test_relative_regression_is_directional_and_zero_reference_is_defined(self):
        goal = GoalSpec("retention", "any", "latency", (
            MetricSpec("latency", "minimize", "seconds", regression=RegressionGuard(tolerance=0.05, mode="relative")),))
        self.assertTrue(assess_metrics(goal, result({"latency": 105}), {"baseline": result({"latency": 100})}).passed)
        self.assertFalse(assess_metrics(goal, result({"latency": 106}), {"baseline": result({"latency": 100})}).passed)
        self.assertTrue(assess_metrics(goal, result({"latency": 0}), {"baseline": result({"latency": 0})}).passed)
        self.assertFalse(assess_metrics(goal, result({"latency": 1}), {"baseline": result({"latency": 0})}).passed)

    def test_computed_overflow_is_a_failure_not_success(self):
        goal = GoalSpec("huge", "any", "score", (
            MetricSpec("score", "maximize", "points", MetricThreshold(1, ">=", "baseline")),))
        self.assertFalse(assess_metrics(goal, result({"score": 1e308}), {"baseline": result({"score": 1e-308})}).passed)


@unittest.skipUnless(Draft202012Validator is not None, "optional JSON Schema validator is not installed")
class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "config.schema.json"
        cls.schema = json.loads(path.read_text())
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_runtime_serialization_matches_schema(self):
        self.validator.validate(config_to_dict(config()))
        raw = config_to_dict(config())
        raw["goal"]["metrics"][0].pop("target")
        raw["goal"]["metrics"][0]["regression"] = {}
        raw["editor"].pop("endpoint")
        raw["editor"].pop("credential_env")
        self.validator.validate(raw)
        config_from_dict(raw)

    def test_schema_rejects_ambiguous_or_unsafe_contracts(self):
        mutations = (
            lambda raw: raw["editor"].update(api_key="secret"),
            lambda raw: raw["editor"].update(credential_env="TOKEN\n"),
            lambda raw: raw["editor"].update(endpoint="https://user:pass@example.com"),
            lambda raw: raw["target"].update(architecture="arch\n"),
            lambda raw: raw["target"]["artifact_pins"].update(checkpoint="a" * 64 + "\n"),
            lambda raw: raw["budget"].update(max_editor_cost_usd=-1),
            lambda raw: raw["goal"]["metrics"][0]["target"].update(comparator="<="),
            lambda raw: raw["goal"]["metrics"][0].update(target=None, regression=None),
            lambda raw: raw["goal"]["metrics"][1]["target"].update(value=-1),
        )
        for mutation in mutations:
            raw = config_to_dict(config())
            mutation(raw)
            self.assertFalse(self.validator.is_valid(raw), raw)
            with self.assertRaises(ConfigurationError):
                config_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
