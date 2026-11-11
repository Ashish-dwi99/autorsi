"""Eval harness: HumanEval / MBPP pass@1 (greedy, chat prompt). CPU, local models only, no API calls."""
import json, os, re, subprocess, sys, tempfile, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
OUT = os.environ.get("AUTORSI_OUT", "/tmp/w")

def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(NAME)
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=dtype)
    m.eval(); return tok, m

class CloseFence(StoppingCriteria):
    def __init__(self, tok, start): self.tok, self.start = tok, start
    def __call__(self, ids, scores, **kw):
        if ids.shape[1] - self.start < 8: return False
        return self.tok.decode(ids[0, self.start:][-200:]).count("```") >= 2

def chat_ids(tok, user):
    s = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
    return tok(s, return_tensors="pt")

@torch.no_grad()
def generate(tok, m, user, max_new=320):
    enc = chat_ids(tok, user); n = enc.input_ids.shape[1]
    out = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                     stopping_criteria=StoppingCriteriaList([CloseFence(tok, n)]), pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True)

def extract_code(resp):
    mt = re.search(r"```(?:python|py)?\n(.*?)(?:```|$)", resp, re.S)
    return mt.group(1) if mt else resp

def run_program(src, timeout=10):
    os.makedirs(f"{OUT}/run", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=f"{OUT}/run") as f:
        f.write(src); p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, timeout=timeout, text=True,
                           stdin=subprocess.DEVNULL, cwd=f"{OUT}/run")
        return r.returncode == 0, r.stderr[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(p)

def he_user(p): return f"Complete the following Python function. Return the full function in one ```python code block.\n\n```python\n{p['prompt']}```"
def he_program(p, resp):
    code = extract_code(resp)
    head = p["prompt"] + "\n    pass\n\n" if f"def {p['entry_point']}" in code else p["prompt"]
    return head + code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"

def mbpp_user(p): return f"{p['prompt']}\nYour code should pass this test:\n{p['test_list'][0]}\nReturn the function in one ```python code block."
def mbpp_program(p, resp):
    return "\n".join(p.get("test_imports", [])) + "\n" + extract_code(resp) + "\n\n" + "\n".join(p["test_list"]) + "\n"

def run_bench(tok, m, probs, user_fn, prog_fn, out_path, chunk=None, log_every=10):
    """Resumable. With chunk=N, handles at most N new problems then returns None (caller relaunches)."""
    done = {}
    if os.path.exists(out_path):
        for l in open(out_path): r = json.loads(l); done[r["id"]] = r
    f = open(out_path, "a"); fresh = 0
    for p in probs:
        pid = str(p["task_id"])
        if pid in done: continue
        if chunk and fresh >= chunk: return None
        fresh += 1
        resp = generate(tok, m, user_fn(p))
        ok, err = run_program(prog_fn(p, resp))
        done[pid] = {"id": pid, "pass": ok, "err": err, "resp": resp}; f.write(json.dumps(done[pid]) + "\n"); f.flush()
    ids = [str(p["task_id"]) for p in probs]
    return sum(done[i]["pass"] for i in ids) / len(ids)

def he_problems():
    from datasets import load_dataset
    return list(load_dataset("openai_humaneval")["test"])
