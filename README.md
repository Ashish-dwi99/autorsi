# Weight Space Lab

Reproducible experiments on the owner's fine-tuned BU-30B-A3B browser model. The current experiment tests explicit weight edits against an unchanged checkpoint, with a total Modal testing budget of $30.

**No improved model or Astra comparison has been demonstrated yet.** The first GPU pilot loaded the real checkpoint but produced four incomplete or malformed replies, then hit its first task deadline on call five. No browser action executed and no edited candidate ran. A subsequent timing diagnostic failed remote startup because a helper was outside its import path; it was cancelled. Both failure records are preserved.

The next experiment uses a standalone vLLM package, an explicit compatibility bridge for the checkpoint's packed expert tensors, and a Browser Use action-protocol subset. An actual CPU import/spawn check gates GPU use. These changes apply equally to baseline and edited variants; improvements over the earlier serving path must not be attributed to weight editing.

## Model provenance

The recovered Modal artifacts are a merged `Qwen3VLMoeForConditionalGeneration` checkpoint and an ordinary rank-16 attention LoRA whose configuration names `browser-use/bu-30b-a3b-preview`. BU itself derives from Qwen3-VL-30B-A3B-Instruct. This is separate from the owner's Qwen2.5-7B experiment.

Original volumes are read-only. Private checkpoints, training data, screenshots and raw run receipts are excluded from Git. Source and artifact fingerprints live in `evidence/`; detailed local receipts remain in ignored directories.

## Initial intervention

For each original target matrix `W*`, test `BF16(FP32(W*) + (lambda - 1) * 2 * (FP32(B) @ FP32(A)))` using the recovered LoRA matrices. Every variant starts from retained originals. Lambda 1 restores the original bytes; lambda 0 is an adapter-subtracted control and is **not** verified to reconstruct raw BU. Candidate strengths are 0.5 and 0.75.

This tests fine-tuning strength. It does not copy another model's hidden knowledge, shrink BU's architecture or establish general capability transfer.

## Validation

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The local tests use small numerical examples and evaluator doubles; they do not run a model. Sixteen synthetic reference browser paths have separately passed CPU Chromium validation. The MiniWoB adapter is source-only and still requires runtime validation.

See `evidence/benchmark_protocol_v1.json` for the planned success, latency, cost, regression and demonstration measurements. Neal's game is a demonstration; established benchmarks provide broader coverage. A comparison with this interactive assistant must identify its tools and access differences, and cannot be labelled a verified Astra model benchmark.

## Running remote work

Modal entrypoints launch billable resources. Historical `modal_preflight.py` and `modal_pilot.py` invocations are consumed and must not be rerun under their old IDs. Check the current source admission, remaining budget and once-only run markers before any new invocation. Source timeouts and spend reservations do not constitute provider-enforced billing caps.

Eden models, datasets, evaluation and previous run approvals are outside this project.
