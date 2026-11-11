"""bench.py <base|tag> <he|mbpp>: one chunk of 40 problems then exit (avoids RSS growth). Prints DONE/MORE."""
import sys, torch
from common import *
torch.set_num_threads(2)
tag, which = sys.argv[1], sys.argv[2]
tok, m = load(); D = split_data()
ctx = torch.autocast("cpu", enabled=False)
if tag != "base":
    ed = torch.load(f"{OUT}/{tag}_ckpt/edited.pt", weights_only=True)
    params = dict(m.named_parameters())
    for k, v in ed.items(): params[k].data = v.float()
    ctx = torch.autocast("cpu", dtype=torch.bfloat16)
probs, uf, pf = (he_problems()[:80], he_user, he_program) if which == "he" else (D["test"][:60], mbpp_user, mbpp_program)
with ctx:
    r = run_bench(tok, m, probs, uf, pf, f"{OUT}/{tag}_{which}.jsonl", chunk=40)
print("DONE" if r is not None else "MORE", r, flush=True)
