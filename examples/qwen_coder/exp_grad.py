"""E2: AutoRSI + GradientProposer (full / lowrank) on mlp.down_proj of the last layers.
usage: exp_grad.py <full|lowrank|sign> <scale> <steps> <tag> [layers e.g. 20,21,22,23]"""
import json, sys, time, torch, random, os
from common import *
from autorsi import *
mode, scale, steps, tag = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
layers = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "20,21,22,23").split(",")]
torch.set_num_threads(2); torch.manual_seed(0)
tok, model = load(); D = split_data(); T = text_set(tok)
names = [f"model.layers.{i}.mlp.down_proj.weight" for i in layers]
backend = DeltaTorchBackend(model, editable=names, max_abs_delta=0.01, max_total_delta=0.05, copy=False,
                            compute_dtype=torch.bfloat16)
def ev(split):
    def f(m): return {"code_nll": batch_nll(m, tok, D[split]).item(), "text_nll": text_nll(m, T)}
    return f
rnd = random.Random(1)
def loss_fn(m): return batch_nll(m, tok, rnd.sample(D["grad"], 4), grad=True)
prop = GradientProposer(backend, loss_fn, scale=scale, mode=mode, rank=16)
c = AutoRSI(backend=backend, evaluator=ev("gate"), validation_evaluator=ev("val"),
            goal=Goal("code_nll", "minimize", min_improvement=1e-3), guards=[Guard("text_nll", "minimize", 0.05)],
            proposer=prop, output_dir=f"{OUT}/runs/{tag}")
t = time.time(); rows = []; CH = 4
for k in range(0, steps, CH):
    r = c.run(limits=RunLimits(max_steps=CH, max_accepted=CH, max_evaluations=40, max_seconds=1800))
    for a in r.attempts:
        rows.append({"k": len(rows), "status": a["status"], "step": a["recipe"]["rationale"], "metrics": a.get("metrics"), "reasons": a.get("reasons")})
        print(rows[-1]["k"], a["status"], a["recipe"]["rationale"], a.get("metrics"), f"{time.time()-t:.0f}s", flush=True)
    json.dump({"baseline": dict(r.baseline), "current": dict(r.metrics), "rows": rows}, open(f"{OUT}/{tag}_result.json", "w"), indent=1)
    if r.status == "proposer_stopped": break
    final = c.backend.capture(); os.makedirs(f"{OUT}/{tag}_ckpt", exist_ok=True)
    torch.save(dict(final.payload), f"{OUT}/{tag}_ckpt/edited.pt")   # checkpoint every chunk
print("baseline", dict(r.baseline), "final", dict(r.metrics), "accepted revision", r.revision)
