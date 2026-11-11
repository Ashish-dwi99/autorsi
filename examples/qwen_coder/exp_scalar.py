"""E1: stock-shaped AutoRSI search (scalar scale/add edits), Qwen2.5-Coder-0.5B-Instruct, validation-gated."""
import json, time, torch
from common import *
from autorsi import *
torch.set_num_threads(2); torch.manual_seed(0)
tok, model = load(); D = split_data(); T = text_set(tok)
names = ["model.norm.weight", "model.layers.23.input_layernorm.weight", "model.layers.23.mlp.down_proj.weight",
         "model.layers.12.mlp.down_proj.weight"]
backend = DeltaTorchBackend(model, editable=names, max_abs_delta=0.5, max_total_delta=1.0, copy=False,
                            compute_dtype=torch.bfloat16)
def ev(split):
    def f(m): return {"code_nll": batch_nll(m, tok, D[split]).item(), "text_nll": text_nll(m, T)}
    return f
recipes = []
for n in names:
    for s in (0.98, 1.02): recipes.append(EditRecipe((TensorEdit(n, "scale", s),), f"scale {n} by {s}"))
    recipes.append(EditRecipe((TensorEdit(n, "add", 0.003),), f"add 0.003 to {n}"))
c = AutoRSI(backend=backend, evaluator=ev("gate"), validation_evaluator=ev("val"),
            goal=Goal("code_nll", "minimize", min_improvement=1e-4), guards=[Guard("text_nll", "minimize", 0.02)],
            proposer=SequenceProposer(recipes), output_dir=f"{OUT}/runs/scalar")
r = c.run(limits=RunLimits(max_steps=len(recipes), max_accepted=len(recipes), max_evaluations=400, max_seconds=7200))
print(r.status, "accepted", r.accepted, "evals", r.evaluations)
print("baseline", dict(r.baseline), "final", dict(r.metrics))
for a in r.attempts: print(a["status"], a["recipe"]["rationale"], a.get("metrics"))
