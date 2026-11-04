"""Tiny random Hugging Face causal LM: no download, training or provider calls.

This verifies real logit edits and serialization, not language capability gains.
Install the optional demo dependencies: python -m pip install -e '.[hf-demo]'
"""
import json
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from autorsi import AutoRSI, EditRecipe, Goal, Guard, SequenceProposer, TensorEdit, TorchBackend


def main():
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
    controller = AutoRSI(backend=backend, evaluator=evaluate,
                        goal=Goal("logit_mse", "minimize", target=1e-12),
                        guards=[Guard("hidden_mse", "minimize")],
                        proposer=SequenceProposer([EditRecipe((TensorEdit("lm_head.weight", "scale", 0.5),))]),
                        output_dir="autorsi-runs/hf")
    result = controller.step()
    print(json.dumps(result.to_dict(), indent=2))
    if result.accepted != 1:
        raise RuntimeError("expected a measured logit change")
    restored = GPT2LMHeadModel(config).eval()
    restored.load_state_dict(torch.load(result.checkpoint, weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        assert torch.equal(restored(tokens).logits, desired_logits)


if __name__ == "__main__":
    main()
