import random, torch
from evalkit import *
def mbpp_full(split):
    from datasets import load_dataset
    return [{"task_id": r["task_id"], "prompt": r["text"], "code": r["code"].replace("\r\n", "\n").strip("\n"),
             "test_list": r["test_list"], "test_imports": [r["test_setup_code"]] if r["test_setup_code"] else []}
            for r in load_dataset("mbpp", "full")[split]]
def split_data():
    tr, va, te = mbpp_full("train"), mbpp_full("validation"), mbpp_full("test")
    random.Random(0).shuffle(tr)
    # grad: proposer's training data. gate/val: AutoRSI objective + validation gate. test: never touched by gates.
    return {"grad": tr[:300], "gate": tr[300:324], "val": va[:24], "test": te}
def example_ids(tok, p):
    prompt = chat_ids(tok, mbpp_user(p)).input_ids[0]
    resp = tok("```python\n" + p["code"] + "\n```<|im_end|>", return_tensors="pt").input_ids[0]
    ids = torch.cat([prompt, resp]); labels = ids.clone(); labels[: len(prompt)] = -100
    return ids[:512], labels[:512]
def batch_nll(model, tok, probs, grad=False):
    """Mean NLL of reference solutions (response tokens only). Logits only for response tokens: saves ~1GB RAM."""
    tot, n = 0.0, 0
    with (torch.enable_grad() if grad else torch.no_grad()):
        for p in probs:
            ids, lab = example_ids(tok, p)
            tgt = lab[1:]; m = tgt != -100
            h = model.model(ids[None]).last_hidden_state[0, :-1][m]
            l = torch.nn.functional.cross_entropy(model.lm_head(h).float(), tgt[m], reduction="sum")
            tot = tot + l; n += int(m.sum())
    return tot / n
def text_set(tok, k=12, seqlen=160):
    from datasets import load_dataset
    txt = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1")["validation"]["text"] if len(t) > 400][:k]
    return [tok(t, return_tensors="pt").input_ids[0, :seqlen] for t in txt]
@torch.no_grad()
def text_nll(model, ids_list):
    tot, n = 0.0, 0
    for ids in ids_list:
        logits = model(ids[None]).logits[0, :-1].float()
        tot += torch.nn.functional.cross_entropy(logits, ids[1:], reduction="sum").item(); n += len(ids) - 1
    return tot / n

