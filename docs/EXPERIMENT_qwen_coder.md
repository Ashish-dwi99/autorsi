# Experiment: evaluation-gated weight editing on Qwen2.5-Coder-0.5B-Instruct

Status: measured results below are real but small. Benchmarks are tiny (HumanEval first 80, MBPP first 60), so differences under about 5 points are noise.

## Setup
- Model: Qwen2.5-Coder-0.5B-Instruct, bf16, greedy decoding, CPU only (2 CPUs, 2GB RAM).
- Gates (never overlap eval): objective MBPP-train[300:324], validation MBPP-validation[:24], guard 12 wikitext-2 passages.
- Eval: HumanEval first 80, MBPP test first 60. Seeds: torch 0, split Random(0), proposer Random(1).

## Fork changes
- `DeltaTorchBackend`: one model copy, fp32 editable tensors only, all else frozen (stock backend keeps 5+ copies and cannot run in 2GB).
- `DeltaEdit` contract and `GradientProposer` (full / lowrank / sign).
- Fast fingerprint: stock hashing of `untyped_storage()` is ~1.5 s/MB (about 25 min for a 1GB model); numpy view is ~10 ms.
- 40 tests pass.

## Results (measured)
| Experiment | Gate-set effect | HumanEval-80 | MBPP-60 |
|---|---|---|---|
| Baseline | NLL 0.809 | 56/80 | 21/60 |
| E1 scalar scale/add edits | NLL -0.9%, 2/12 accepted | not benched | not benched |
| E2 gated gradient edit | NLL -8.8%, text guard held, 2/6 accepted | 52/80 (8 base-only vs 4 edit-only, n.s.) | 22/60 (3 vs 4, noise) |

Conclusion: the gates work (they block damaging edits) but NLL on reference solutions mostly measures dataset style. Lower NLL did not give higher pass@1. Likelihood is not capability.

## E3: execution-gated expert iteration
Model samples 4 solutions per problem at T=0.8, keeps those passing tests (117 passing samples over 40 of 80 problems), fine-tunes down_proj of layers 16-23 on them, and the gate is greedy executed pass rate on 24 problems (baseline 11/24) with validation gate and text guard. See STATE.md for the result once finished; it is not claimed here unless listed above.

## Speculation (not measured)
- Merge-gated learning on a team's own repos: merged PRs as the positive signal, gated by that repo's tests and protected benchmarks.
- A 0.5B model has hard ceilings; expect small gains at best, and only with execution-based objectives.

Reproduce: `examples/qwen_coder/` (STATE.md has commands).
