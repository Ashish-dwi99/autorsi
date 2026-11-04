"""Opt-in improvement between completed agent tasks; dependency-free example."""
from autorsi import AutoRSI, EditRecipe, Goal, Guard, RunLimits, ScalarBackend, SequenceProposer, TensorEdit


def main():
    backend = ScalarBackend({"target": 1.0, "other": 3.0}, editable={"target"},
                            max_abs_delta=0.5, max_total_delta=1.0)
    def evaluate(weights):
        return {"error": abs(weights["target"] - 2.0), "other": weights["other"]}
    controller = AutoRSI(backend=backend, evaluator=evaluate, goal=Goal("error", "minimize", target=0.0),
                        guards=[Guard("other")],
                        proposer=SequenceProposer([EditRecipe((TensorEdit("target", "add", 0.5),)) for _ in range(2)]),
                        output_dir="autorsi-runs/agent")
    # The host app owns trigger frequency and a lifetime limit across step calls.
    remaining_improvement_steps = 2
    for task_input in (2.0, 4.0, 6.0):
        with backend.inference() as weights:
            print("task result:", task_input * weights["target"])
        if remaining_improvement_steps:
            result = controller.step(limits=RunLimits(max_evaluations=3, max_seconds=5))
            remaining_improvement_steps -= 1
            print("improvement:", result.status, result.metrics)


if __name__ == "__main__":
    main()
