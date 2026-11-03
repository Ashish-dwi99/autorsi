"""Executable numerical test doubles, not model training or benchmark results.

Run: python3 -m framework.offline_example
The fixed editor proposes .5 then1. The first fabricated numerical result misses
the declared ratio, so it is restored. The second meets the numerical contract.
There is no network, model inference, weight checkpoint or artifact publication.
"""
from pathlib import Path
import json

from .adapters import AdapterRegistry, Operation, Proposal, ResourceQuote, Usage
from .contracts import EvaluationResult, loads_config
from .engine import run_experiment


ARCH = frozenset({"scalar_test_double"})
QUOTE = ResourceQuote(0, 1, 0)
USAGE = Usage(0, 0, 0)


class ScriptedEditor:
    provider = "offline_scripted"
    supported_models = frozenset({"fixed_two_proposals"})
    def quote(self, config):
        return QUOTE
    def propose(self, config, baseline, history):
        strength = (.5, 1.0)[len(history)]
        return Operation(Proposal("bounded_scalar", {"strength": strength},
                                  "Scripted numerical fixture; no model insight or capability transfer"), USAGE)


class NumericTarget:
    adapter_id = "numeric_target"
    supported_architectures = ARCH
    def __init__(self):
        self.strength, self.restore_count = 0.0, 0
    def quote(self, operation, config):
        if operation not in {"snapshot", "restore", "accept"}:
            raise ValueError("unknown numeric operation")
        return QUOTE
    def snapshot(self, config):
        return Operation(self.strength, USAGE)
    def restore(self, config, snapshot):
        self.strength = snapshot
        self.restore_count += 1
        return Operation(None, USAGE)
    def accept(self, config, snapshot, proposal):
        return Operation("numeric_fixture_only:no_model_artifact", USAGE)


class ScalarRecipe:
    recipe_id = "bounded_scalar"
    supported_architectures = ARCH
    def validate_parameters(self, parameters):
        if (set(parameters) != {"strength"} or type(parameters["strength"]) not in (int, float)
                or not 0 <= parameters["strength"] <= 1):
            raise ValueError("only strength within0..1 is supported")
    def quote(self, config, proposal):
        return QUOTE
    def apply(self, config, target, snapshot, proposal):
        target.strength = proposal.parameters["strength"]
        return Operation(None, USAGE)


class NumericEvaluator:
    evaluator_id = "numeric_evaluator"
    supported_architectures = ARCH
    supported_domains = frozenset({"computer_use"})
    feedback_allowed = True
    def quote(self, config):
        return QUOTE
    def evaluate(self, config, target):
        # Deliberate test formulas, never actual model quality or latency.
        metrics = {"latency": 100 * (1 - .95 * target.strength), "accuracy": .9}
        return Operation(EvaluationResult("numeric_evaluator", "offline_formula", "v1", metrics,
                                         artifact_id="numerical_formula_only"), USAGE)


def main():
    config = loads_config(Path(__file__).with_name("example_config.json").read_text())
    registry, target = AdapterRegistry(), NumericTarget()
    registry.register_editor(ScriptedEditor())
    registry.register_target(target)
    registry.register_recipe(ScalarRecipe())
    registry.register_evaluator(NumericEvaluator())
    result = run_experiment(config, registry)
    print(json.dumps({"example_type": "OFFLINE_NUMERICAL_TEST_DOUBLES_NOT_MODEL_RESULTS",
                      "actual_model_calls": 0, "provider_requests": 0, "restore_count": target.restore_count,
                      "result": result.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
