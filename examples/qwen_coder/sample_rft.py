"""Sample the model's own solutions (T=0.8, 4 per problem) on MBPP-train problems; keep those that PASS their tests.
Resumable. Output: $AUTORSI_OUT/rft_data.jsonl  {id, prompt, resp, pass_rate}"""
import json, os, sys, torch
from common import *
torch.set_num_threads(2); torch.manual_seed(0)
N_PROB, K = int(sys.argv[1]), 4
tok, m = load(); D = split_data()
path = f"{OUT}/rft_data.jsonl"
done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
f = open(path, "a"); t0 = __import__("time").time()
for p in D["grad"][:N_PROB]:
    if str(p["task_id"]) in done: continue
    enc = chat_ids(tok, mbpp_user(p)); n = enc.input_ids.shape[1]
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.95, num_return_sequences=K,
                         pad_token_id=tok.eos_token_id)
    res = []
    for o in out:
        text = tok.decode(o[n:], skip_special_tokens=True)
        i = text.find("```", text.find("```") + 3) if "```" in text else -1
        if i < 0: res.append((text, False)); continue
        resp = text[: i + 3]
        ok, _ = run_program(mbpp_program(p, resp))
        res.append((resp, ok))
    rate = sum(ok for _, ok in res) / K
    for resp, ok in res:
        if ok: f.write(json.dumps({"id": str(p["task_id"]), "prompt": mbpp_user(p), "resp": resp, "pass_rate": rate}) + "\n")
    f.flush(); f.write("") 
    json.dump({"id": str(p["task_id"]), "rate": rate}, open(f"{OUT}/rft_progress.json", "w"))
    print(p["task_id"], rate, f"{__import__('time').time()-t0:.0f}s", flush=True)
