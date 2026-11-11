# Experiment state (update every slice)
Box: 2 CPU / 2GB RAM; background jobs only advance while a command is active. Outputs in /tmp/w (lost if the VM resets) - copy small results into results/ and push.
Model: Qwen/Qwen2.5-Coder-0.5B-Instruct (bf16; editable tensors fp32 via DeltaTorchBackend). Seeds: torch 0, data split Random(0), proposer sampling Random(1).
Eval sets (never used by gates): HumanEval first 80, MBPP-full test first 60. Gate sets: MBPP-train[300:324] (objective), MBPP-validation[:24] (validation), wikitext-2 val 12 passages (text guard).

## Done
- Baseline (bf16, greedy): HE80 57/80 = 0.7125 (earlier run), MBPP60 21/60=0.35 (22/60 earlier). Gate NLL 0.809, text NLL 3.514.
- E1 scalar scale/add (exp_scalar.py): NLL 0.8089 -> 0.8013, 2 accepted / 12. 
- E2 gradient full (exp_grad.py full 2e-4 16 gfull, stopped after 6 attempts for time): gate NLL 0.8077 -> 0.7366, text 3.514->3.520, 2 accepted/6. MBPP60 22/60, HE80 52/80.
## Running / next
- pipeline.sh finishing base HE80 (paired baseline) -> /tmp/w/base_he.jsonl
- pipeline2.sh (waits for pipeline.done): sample_rft.py 80 (own samples, keep passing) -> exp_rft.py 5e-5 6 rft1 (gate = executed pass rate) -> bench rft1 mbpp/he.
- Then: write docs/EXPERIMENT_qwen_coder.md, speech protocol (docs/speech_protocol.md + examples/speech_eval.py), push, open DRAFT PR (head Ashish-dwi99:gated-gradient-delta-backend -> base main on Ashish-dwi99/autorsi via tools github create-pull-request; never merge).
## Push recipe (no remote edits allowed)
git -c credential.helper='!f(){ echo username=Ashish-dwi99; echo "password=$(cat /home/sandbox/.vault/github-cli-ashish-dwi99-5d49e5/password)"; };f' push -q https://github.com/Ashish-dwi99/autorsi.git gated-gradient-delta-backend
(vault file recreated with `tools vault write-file --key password --name "GitHub CLI Ashish-dwi99" --return-channel whatsapp` in tools box if missing, then tools workspace put --path <file>)

## Update 19:44
- Paired baseline (this VM): HE80 base 56/80 (0.70) vs gfull 52/80 (0.65): base-only 8, edit-only 4 (sign test p~0.39, not significant, wrong direction). MBPP60 base 21 vs gfull 22 (3 vs 4). Per-problem results in results/passfail_base_vs_gfull.json.
- sample_rft.py running: ~8s/problem, ~30% of problems yield any passing sample; 80 problems ~ 32 min.
