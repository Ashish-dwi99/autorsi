# Experiment orchestration

This framework separates the **editor**, which proposes changes, from the **target**, whose weights may be changed. An independent evaluator measures the declared goal. It supports explicitly registered adapters; it does not make arbitrary architectures interchangeable or guarantee capability transfer.

The repository's wheel includes both `framework` and `weight_space_lab`, plus the example configuration and JSON schema. Test modules are excluded. Application-specific model, GPU and browser dependencies remain optional; installing this package does not install or launch them.

Run the offline example from the repository root:

```sh
python3 -m framework.offline_example
```

The example uses scalar numerical test doubles. No model, browser, provider, checkpoint or GPU runs. Its fabricated latency formula lets the first proposal fail and roll back, then lets the second pass. Those numbers demonstrate control flow and are **not model results**.

`example_config.json` declares latency at most0.1 times the measured baseline, with zero allowed accuracy regression. Its editor and target are independently specified. Real claims such as10x lower latency require an actual evaluator using matched workloads, timing boundaries and quality checks; writing the ratio in a goal does not establish it.

## Connecting an application

Implement the protocols in `adapters.py` and register concrete instances with `AdapterRegistry`. Imports are explicit; configuration cannot import a plugin, execute shell commands, run model-generated code or expand an adapter's declared compatibility.

- `EditorAdapter` supplies a structured `Proposal` or stops. `OpenAIResponsesEditor` and `GeminiInteractionsEditor` accept explicit available model identifiers, a reviewed recipe schema and validator, an environment-variable name for credentials, and a per-request resource quote. No request occurs on import. Naming Astra or another model does not make it available through an API.
- `TargetAdapter` verifies its application's actual artifact provenance and owns snapshot, restore and local acceptance. Acceptance retains a candidate; it is not authorization to publish a model or deploy a service.
- `RecipeAdapter` declares supported architectures, validates all editable parameters and implements the actual edit. A new architecture needs an appropriate implementation and evidence; a configuration label is insufficient.
- `EvaluatorAdapter` declares supported domains and architectures and produces a typed result for the frozen suite/revision. Metrics must exactly match the goal. It must derive results from actual trusted evaluation, never the editor's assertions.

The engine measures a baseline, asks the editor for a proposal, reserves the complete candidate path plus rollback, snapshots the target, applies the edit, evaluates it, and checks every target and regression guard. Unmet targets restore the snapshot before another proposal. Only an evaluation that passes all configured checks can reach target acceptance.

The existing BU runtime and its source-bound experimental results are separate integrations. This module does not yet turn the generic interface into working math, coding or arbitrary-model runtime adapters. Unit tests use simulated adapters for those domains.

## Budgets and failures

Every potentially resource-consuming operation returns `Operation(value, Usage(...))` and supplies a prospective `ResourceQuote`. Editor and target spending ceilings are separate. Missing usage retains its quoted reservation instead of becoming zero. The baseline counts against the evaluation cap, and the engine also checks model-call and wall-time budgets before dispatch.

Quotes are integration contracts, not provider billing receipts or physical cancellation guarantees. Adapters must enforce native request bounds and report honest usage. Unexpected resource overruns stop the loop; they cannot undo resources already consumed. An exhausted budget can leave restoration unconfirmed, which is reported explicitly.

Ordinary adapter exceptions imply unknown operation disposition. If an operation could still be running, the engine stops without a potentially overlapping restore or another model call. `AdapterOperationError(disposition_known=True)` is reserved for a trusted adapter that knows the operation stopped. A failed restore is never silently retried. The application owns process supervision and any required human recovery.

## Evaluation boundaries

The editing loop accepts only evaluators with `feedback_allowed=True`. Sealed/heldout evaluation must use a separate non-search workflow; do not register a heldout test as development feedback. Results must retain suite identity, completed coverage, actual resource records and target artifact provenance. A passed narrow goal is not a claim of universal intelligence or superiority over another model.

Pure tests:

```sh
python3 -m unittest discover -s framework/tests
```

These commands use no live provider APIs. Provider transport tests use injected fake responses and do not prove remote API availability or model compatibility.
