# Integration contract

## The three application boundaries

`Backend` owns model state, isolation, safe-point promotion and checkpoint serialization. `Evaluator` is a synchronous callable from an isolated model to a nonempty mapping of finite numeric metrics. `Proposer.propose(ProposalContext)` returns `EditRecipe` or `None`; its context contains metrics and metadata, never model objects or weights.

These contracts are exported from `autorsi`. Importing the package does not import Torch, contact a provider or read a checkpoint. Torch loads only when constructing `TorchBackend`.

### Backend protocol

| Member | Required behavior |
| --- | --- |
| `capture()` | Return an owned `ModelState` snapshot with revision, fingerprint and backend owner identity. Refuse snapshots from another backend in every consuming method. |
| `candidate(state, recipe)` | Create an isolated candidate. Validate all edits, finite realized values, per-step bounds and cumulative bounds. Never touch serving weights. |
| `evaluate(state, evaluator)` | Run the evaluator on an independent copy; reject persistent weight/buffer mutation. Return finite metric values. |
| `promote(state, expected_revision=..., deadline=...)` | Atomically publish at a safe point only if the current revision matches and the absolute monotonic deadline has not expired. On failure, leave serving state unchanged. Return the new revision. |
| `export(state, directory)` | Write a reloadable checkpoint into a new directory and return its `Path`. An exported candidate has not necessarily been accepted. |
| `inference()` | Context manager protecting the entire read-only inference operation. |
| `revision` / `recipe_space` | Current monotonically increasing version and bounded JSON-compatible edit metadata. |

`InMemoryBackend` implements cloning, isolation, fingerprint checks and lock/version handling. Subclasses implement `_validate_model`, `_fingerprint`, `_apply`, `_validate_total_delta` and `export`; optionally override `_clone` and `_inference_context`. `ScalarBackend` is a small complete reference adapter. Use a dedicated implementation of the public protocol for remote or partitioned systems; the in-memory lock does not coordinate other processes or machines.

The model passed to `TorchBackend` remains untouched. The backend serves its own copy, so the application must route inference through that backend to use accepted weights. Ordinary `state_dict` semantics and trustworthy `deepcopy` implementations are required. Arbitrary Python hooks, model methods, custom backend code, evaluators and transports are trusted application code and are not sandboxed. Model behavior held outside the state dictionary is outside the fingerprint guarantee.

### Evaluator design

Use fixed task inputs and representative protected tasks. Return Python `float`/`int` values (use `.item()` for Torch scalars), with the exact required goal and guard names. Missing metrics, booleans, infinities and NaN are errors. Additional finite metrics are retained in feedback but are not implicit gates.

`Goal("latency", "minimize", min_improvement=0.01)` requires strict improvement of at least 0.01 units over the current accepted model. `Guard("accuracy", "maximize", max_regression=0.002)` permits at most 0.002 absolute loss against the controller's construction-time baseline. The original reference persists across `step()` and `run()` calls. Recreating a controller creates a new metric reference; recreating a backend creates a new cumulative weight reference.

An objective target controls when search stops; it is not a requirement that each intermediate accepted candidate already meet the final target. Every intermediate candidate must still improve the objective and pass all guards. There is one objective per controller. Several measurements can be combined in a clearly defined evaluator objective, while additional independent requirements belong in guards.

Evaluation callbacks must not train the model or change persistent buffers. They should use deterministic inference settings and controlled environments. TorchBackend supplies `eval()` copies and `torch.inference_mode()`. State dictionaries are fingerprinted before and after evaluation. External environment changes, random sampling and custom Python attributes need application-level control. Reference metrics are cached; do not change evaluation data or semantics while reusing a controller.

A validation evaluator uses the same objective and guard names, comparing validation objective against the current validation score and guards against the original validation baseline. Numeric validation scores appear only in the local journal, not proposer context. Its binary selection feedback means it is not a permanently sealed holdout.

### Proposer and editor LLM

`SequenceProposer` consumes an iterator once. `JSONProposer(complete)` calls your chosen editor transport once per proposal, without retries. The core owns no provider credentials. The complete callable must return JSON text; configure request cancellation, timeouts, output limits and provider spending caps in that callable. The supplied prompt contains developer-selected metric names, values, recipes and rationales; decide whether those metadata can be sent to your editor provider.

Custom proposers can examine immutable goal/current/reference metrics and a detached recent history. They cannot authorize acceptance or alter the evaluator. No executable model-generated code is supported. Only scalar addition and scaling of explicitly allowed parameter names are implemented by the shipped backends. More sophisticated LoRA, low-rank or optimizer-based edits require a reviewed backend/recipe extension; they are not inferred from a natural-language goal.

## State and failure semantics

`RunResult.revision` identifies the state described by `metrics`. `serving_revision` is the backend revision observed at the end of the run. If another writer changed serving weights, `stale_revision` is returned; those metrics must not be attributed to that new serving revision. This is an observation at completion, not a perpetual guarantee against later writers.

`accepted` counts promotions made by that call. `checkpoint` is the latest accepted artifact from that call, or `None`. A subsequent rejection, limit or error does not undo an earlier accepted improvement. Rejected candidates never become live. `rollback()` deliberately restores one accepted predecessor and increments the version; it refuses concurrent or stale restoration. Repeated rollback is available for accepted states retained by the same controller. The history and rollback snapshots are in-memory; rebuilding a controller does not automatically resume them from a journal.

Before promotion, the engine writes a candidate checkpoint and a prepared journal event. It then atomically swaps the backend model and writes the accepted event. If that final journal write fails, the result says `journal_error` but still reports the actual accepted count and checkpoint. Do not infer acceptance from the existence of a staged file or a prepared record. A process crash between the swap and the accepted record cannot be resolved by the journal alone; inspect the serving fingerprint. This is not a durable distributed commit protocol.

Rollback similarly journals preparation before changing weights. Failure of that first write prevents rollback. If the final write fails after successful restoration, `RollbackJournalError` reports the committed revision. Other callback failures are returned with sanitized exception types so provider error strings do not accidentally expose credentials. Framework validation errors retain specific actionable messages.

The total edit bound is relative to backend construction weights, not each successive accepted model. Repeated small edits therefore cannot evade it. The bound does not determine which capabilities will be affected; that remains an empirical evaluation problem.

## Bounded execution

`RunLimits` caps proposals, evaluator invocations, accepted changes and elapsed time per call. Both search and validation evaluations count; a failed admitted evaluator invocation still consumes its slot. Time is checked between synchronous operations and again under the promotion lock. A model callback, exporter, evaluator or transport can overrun the deadline before returning. A returned late candidate is not promoted. No thread/process cancellation, provider billing cap or GPU preemption is supplied.

All application inference must hold `backend.inference()` until it is completely finished. Nested read contexts are allowed; promoting from inside one's own inference context is refused. Different inference calls are serialized by this reference backend. Candidate creation/evaluation proceeds on isolated copies while the serving model remains available; promotion waits for a safe point. For asynchronous generation, join/await completion before releasing the context.
