"""Real, deterministic CPU weight editing; no provider keys or model downloads.

Run after installing the checkout: python examples/autorsi/torch_quickstart.py
"""
import json
import torch

from autorsi import AutoRSI, EditRecipe, Goal, Guard, SequenceProposer, TensorEdit, TorchBackend


def main():
    model = torch.nn.ModuleDict({"target": torch.nn.Linear(1, 1, bias=False),
                                 "protected": torch.nn.Linear(1, 1, bias=False)})
    with torch.no_grad():
        model["target"].weight.fill_(1)
        model["protected"].weight.fill_(3)
    x = torch.tensor([[-2.], [-1.], [1.], [2.]])

    def evaluate(candidate):
        return {"target_mse": (candidate["target"](x) - 2 * x).square().mean().item(),
                "protected_mse": (candidate["protected"](x) - 3 * x).square().mean().item()}

    backend = TorchBackend(model, editable={"target.weight"}, max_abs_delta=1.0)
    controller = AutoRSI(
        backend=backend, evaluator=evaluate,
        goal=Goal("target_mse", "minimize", target=0.0),
        guards=[Guard("protected_mse", "minimize")],
        proposer=SequenceProposer([EditRecipe((TensorEdit("target.weight", "scale", 2.0),))]),
        output_dir="autorsi-runs/torch",
    )
    result = controller.step()
    print(json.dumps(result.to_dict(), indent=2))
    if result.accepted != 1:
        raise RuntimeError("expected a measured target improvement")

    # All application inference uses this context for its whole duration.
    with backend.inference() as serving_model:
        assert torch.equal(serving_model["target"](x), 2 * x)

    # A checkpoint contains only state_dict tensors. Supply its architecture.
    model.load_state_dict(torch.load(result.checkpoint, weights_only=True, map_location="cpu"))
    assert evaluate(model) == {"target_mse": 0.0, "protected_mse": 0.0}
    controller.rollback()


if __name__ == "__main__":
    main()
