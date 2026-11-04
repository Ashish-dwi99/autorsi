"""Run real arithmetic on a tiny two-output linear model, without dependencies."""
from __future__ import annotations

import argparse
import json

from . import AutoRSI, EditRecipe, Goal, Guard, RunLimits, ScalarBackend, SequenceProposer, TensorEdit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="autorsi-runs", help="directory for local checkpoints and journals")
    args = parser.parse_args()
    backend = ScalarBackend({"target": 1.0, "protected": 3.0},
                            editable={"target", "protected"}, max_abs_delta=1.0, max_total_delta=2.0)
    inputs = (-2.0, -1.0, 1.0, 2.0)

    def evaluate(weights):
        # The target learns y=2*x; the other task must retain y=3*x.
        return {"target_mse": sum((weights["target"] * x - 2 * x) ** 2 for x in inputs) / len(inputs),
                "protected_mse": sum((weights["protected"] * x - 3 * x) ** 2 for x in inputs) / len(inputs)}

    proposer = SequenceProposer([
        EditRecipe((TensorEdit("target", "add", 1.0), TensorEdit("protected", "add", -1.0)),
                   "This target improvement should be rejected because it damages the protected task."),
        EditRecipe((TensorEdit("target", "add", 0.5),), "First bounded improvement."),
        EditRecipe((TensorEdit("target", "add", 0.5),), "Improve the accepted model again."),
    ])
    controller = AutoRSI(backend=backend, evaluator=evaluate,
                        goal=Goal("target_mse", "minimize", target=0.0),
                        guards=[Guard("protected_mse", "minimize")], proposer=proposer,
                        output_dir=args.output)
    result = controller.run(limits=RunLimits(max_steps=3, max_accepted=2))
    print(json.dumps(result.to_dict(), indent=2))
    if result.status != "goal_reached" or result.accepted != 2:
        raise SystemExit("demo did not reach its expected measured result")


if __name__ == "__main__":
    main()
