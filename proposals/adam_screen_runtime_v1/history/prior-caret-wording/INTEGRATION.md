This is a small new screenshot-only browser-fixture harness. Four development
cases use a 1024×768 desktop-like viewport and four use a 390×844 mobile-like
viewport. Eight verification variants have independent fixed seeds, disjoint
word labels, different goals, and the opposite viewport for each family. They
are browser pages, not native desktop/mobile applications. No result here proves
general computer use, unseen real-site ability, perfect performance, or Astra
superiority.

`screen_cases.cases("development")` and `cases("verification")` return the frozen
case objects. `demo_cases()` returns two development fixtures (contact and
dialog), explicitly for qualitative demonstrations. Keep verification out of
candidate selection. Local preflight rendered verification pages only; it did
not run a policy or score their task outcomes. The case manifest pins the complete
scenario (HTML, goal, oracle, viewport and seed); generic apps can share HTML.

`ScreenSession(policy, actor_label=..., actor_identity=..., limits=...)` owns one
policy lifetime. Call `await session.run_case(browser, case, new_artifact_dir,
demonstration=False, publish=None)`. The injected async policy receives
`(messages, max_output_tokens)` and returns `screen_protocol.Reply`. Messages
contain only the task goal, unchanged PNG, viewport, remaining action count,
sequence, and previously issued action feedback. There are no DOM records,
element IDs, target boxes, task oracle, expected answer, or page source. Trusted
Python callbacks must not use their filesystem/browser access to supply hidden
information to a model. Policy code is an application trust boundary, not a
Python security sandbox.

The JSON envelope is `{"actions":[...]}` with one to four actions. Tools are
`click:{x,y}`, `type:{text}`, `key:{name}`, `scroll:{x,y,dy}`, and `done:{}`. Every
coordinate uses viewport CSS pixels at device scale1. `SELECT_ALL` is the logical
platform select-all shortcut (`ControlOrMeta+A`) in a focused text field. Type
appends. No selectors, JavaScript, external navigation, filesystem tools or model
success claims are admitted. The whole batch is syntax-checked before dispatch.
Coordinate bounds do not prove semantic element identity. Use a one-action batch
when changed UI needs inspection; every subsequent batch gets a fresh screenshot.
No self-reported confidence changes the budget or the independent oracle.

Default limits:12 actions/case,4 actions/batch,128 output tokens/call,30s policy
timeout and90s case clock starting before context creation. Interactive manual
policy allows at most60s per reply within the same90s case clock. Invalid action
ends the case. Unknown inference/native timeout raises `SessionAborted` and
permanently latches the session; no further case or policy call is allowed.
Cancellation does not kill a background generation thread or remote service.
An external process owner must terminate/reap that worker before replacement.
Context close and video.path share one total10s original finalization deadline,
separate from scored work.

Success comes from a trusted Python comparison of the actual normal UI receipt
with the withheld expected final state. Fixture JavaScript records the user's
form/selection action; it does not contain a goal-specific success predicate.
No `done` claim grants success. Reports distinguish returned policy calls from
UI actions, retain action errors and timing, and mark unknown token usage.

Each case records actual Playwright WebM, before-policy/final PNG hashes,
viewport, actor declaration, and monotonic action/inference/capture times. Video
timestamps and monotonic events are separate observations; frame-exact alignment
is not asserted. A finalized video must be decoded/validated before publishing.
Validated artifact limits are1MiB/PNG and64MiB/video; these are checked after
returns, not physical encoder allocation guarantees. A fresh220MiB disk floor
includes128MiB reserve and conservative per-case video/PNG/manual-copy bounds.
Partial artifacts stay owned when close/receipt disposition is uncertain.

Manual local demonstration commands use the installed Playwright interpreter:

```sh
PYTHONPATH=proposals/adam_screen_v1 /Users/samuelekpe/Convert/.venv/bin/python proposals/adam_screen_v1/screen_cli.py run --run-dir /absolute/new-run --session-id unique-session --actor-label 'Actual observed actor' --browser-executable '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
PYTHONPATH=proposals/adam_screen_v1 /Users/samuelekpe/Convert/.venv/bin/python proposals/adam_screen_v1/screen_cli.py next --run-dir /absolute/new-run
PYTHONPATH=proposals/adam_screen_v1 /Users/samuelekpe/Convert/.venv/bin/python proposals/adam_screen_v1/screen_cli.py submit --run-dir /absolute/new-run --response '{"actions":[{"click":{"x":100,"y":200}}]}'
```

The run process remains alive while an actual assistant views each returned PNG
and submits its typed action. Observation and command identities join session,
case, sequence and content hash. Commands are atomically published once, never
overwritten; expired/mismatched commands abort. `next` is read-only. An actor
label is declared metadata, not verified API identity. Retain the actual requested
and observed model/tool provenance externally. Manual elapsed time includes tool
and coordination delay and cannot be presented as API-only inference latency.
Compositing should join actual actors' source/result/video receipts and disclose
different tools/timing; do not fabricate another actor or a winning score.

Local validation is opt-in through `ADAM_SCREEN_LOCAL_BROWSER_TESTS=1`. It uses
an explicitly selected existing Chrome executable and fresh Playwright temporary
profiles, never the user's profile. The eight development reference UI checks
and the done-only bridge/video test are labelled test doubles, not model results.
No old MiniWoB/Eden/provider experiment is rerun. Remote deployment, model edits,
budget admission and actor comparisons are outside this module.
