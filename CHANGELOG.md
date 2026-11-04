# Changelog

## 0.1.0a1 — prepared, not yet published

First alpha of AutoRSI's developer framework, under the MIT license. The API is experimental and may change before a stable release.

- One-shot and bounded iterative weight editing with separate model, evaluator and proposer contracts.
- Detached scalar and dense PyTorch candidates, strict objective improvement and protected metric gates against the original baseline.
- Per-candidate and cumulative edit bounds, revision-checked inference safe points and explicit rollback.
- Local weight-only checkpoints and journals with tested failure handling.
- Provider-neutral structured JSON proposals; no generated code execution or built-in provider requests.
- Offline examples, including a tiny randomly initialized Hugging Face causal LM, plus 35 core and model integration tests.
- Manual release workflow with isolated PyPI trusted publishing, exact source/tag validation and distribution checks.

This alpha does not establish broad capability gains, preservation of unmeasured abilities, or compatibility with quantized, sharded, compiled, vLLM or inaccessible hosted weights. Full model copies and fingerprints favor small local experiments over large production deployments.
