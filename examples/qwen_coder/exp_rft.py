"""E3: execution-gated expert iteration. Proposer = gradient step on the model's OWN passing samples;
AutoRSI gate = greedy PASS RATE of executed tests (objective), validated on separate problems; guard = general-text NLL.
usage: exp_rft.py <scale> <steps> <tag>"""
import json, sys, time, torch, random, os
from common import *
from autorsi import *
scale, steps, tag = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
layers = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "16,17,18,19,20,21,22,23").split(",")]
torch.set_num_threads(2); torch.manual_seed(0)
tok, model = load(); D = split_data(); T = text_set(tok)
rows = [json.loads(l) for l in open(f"{OUT}/rft_data.jsonl")]
print("self-samples", len(rows), "problems", len({r['id'] for r in rows}), flush=True)
names = [f"model.layers.{i}.mlp.down_proj.weight" for i in layers]
backend = DeltaTorchBackend(model, editable=names, max_abs_delta=0.01, max_total_delta=0.05, copy=False, compute_dtype=torch.bfloat16)
def passrate(m, probs):
    ok = 0
    for p in probs:
        resp = generate(tok, m, mbpp_user(p), max_new=256)
        ok += run_program(mbpp_program(p, resp))[0]
    return ok / len(probs)
def ev(split):
    def f(m): return {"pass": passrate(m, D[split]), "text_nll": text_nll(m, T)}
    return f
def self_nll(m, batch):
    tot, n = 0.0, 0
    with torch.enable_grad():
        for r in batch:
            prompt = tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False), return_tensors="pt").input_ids[0]
            resp = tok(r["resp"] + "<|im_end|>", return_tensors="pt").input_ids[0]
            ids = torch.cat([prompt, resp])[:512]; lab = ids.clone(); lab[: len(prompt)] = -100
            tgt = lab[1:]; mk = tgt != -100
            h = m.model(ids[None]).last_hidden_state[0, :-1][mk]
            tot = tot + torch.nn.functional.cross_entropy(m.lm_head(h).float(), tgt[mk], reduction="sum"); n += int(mk.sum())
    return tot / n
rnd = random.Random(1)
prop = GradientProposer(backend, lambda m: self_nll(m, rnd.sample(rows, 4)), scale=scale, mode="full")
c = AutoRSI(backend=backend, evaluator=ev("gate"), validation_evaluator=ev("val"),
            goal=Goal("pass", "maximize", min_improvement=0.01), guards=[Guard("text_nll", "minimize", 0.05)],
            proposer=prop, output_dir=f"{OUT}/runs/{tag}")
t = time.time(); log = []
for k in range(steps):
    r = c.run(limits=RunLimits(max_steps=1, max_accepted=1, max_evaluations=10, max_seconds=3000))
    a = r.attempts[-1] if r.attempts else {}
    log.append({"k": k, "status": a.get("status", r.status), "step": a.get("recipe", {}).get("rationale"), "metrics": a.get("metrics"), "reasons": a.get("reasons")})
    print(k, log[-1]["status"], log[-1]["step"], log[-1]["metrics"], f"{time.time()-t:.0f}s", flush=True)
    json.dump({"baseline": dict(r.baseline), "current": dict(r.metrics), "rows": log}, open(f"{OUT}/{tag}_result.json", "w"), indent=1)
    os.makedirs(f"{OUT}/{tag}_ckpt", exist_ok=True)
    torch.save(dict(c.backend.capture().payload), f"{OUT}/{tag}_ckpt/edited.pt")
