# Contributing

AutoRSI's core coordinates edits and evaluations. Model-specific behavior belongs in a backend; application-specific metrics belong in an evaluator. Keep provider SDKs optional.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests/autorsi -v
python -m autorsi
```

Install `.[torch]` to run the real PyTorch backend tests. Optional Hugging Face tests construct a tiny random model locally; they must not download a checkpoint or require credentials.

Changes to editing, promotion, rollback or persistence should include a failing-path test. In particular, rejected candidates must leave the active model unchanged, and iterative edits must respect the original cumulative edit bounds and protected-metric reference.

Document the exact model types a backend supports. Do not infer compatibility with quantized, sharded or hosted models from a dense PyTorch test. Do not add benchmark claims without a reproducible evaluation and its limitations.

The manual GitHub Actions workflow provides package and optional model-backend checks. Local tests do not establish a passing remote workflow.

AutoRSI is licensed under MIT; see [LICENSE](LICENSE). Release preparation and the separate manual publication step are documented in [docs/releasing.md](docs/releasing.md). Regenerate the package-index README with `python scripts/render_package_readme.py` after changing `README.md`.
