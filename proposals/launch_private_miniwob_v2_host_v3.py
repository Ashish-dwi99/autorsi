"""Corrected HOST-ONLY attempt v3 for the unchanged deployed MiniWoB v2.

No App, image, build, deploy, CPU replay, or remote runtime modification.
The earlier local v2 attempt failed resolving a nonexistent Function.from_id
before spawn. This new once-only host path uses public name lookup + hydration,
requires the exact observed function ID, and retains remote run_id v2.
Importing this module performs no provider call. Actual launch needs separate
source review and the existing authorized/shared resource scope.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ATTEMPT_ID = "private-miniwob-benchmark-host-v3"
REMOTE_RUN_ID = "private-miniwob-benchmark-v2"
EXPECTED_APP_ID = "ap-ah4kX7k6SeDM0GDOqHb8fM"
EXPECTED_FUNCTION_ID = "fu-QNmCQAocBpghFWF9KgMhqN"
HOST_DEADLINE_SECONDS = 2000
CANCELLATION_GRACE_SECONDS = 20
APP_NAME = "adam-1-cu-private-miniwob-v2"
PROPOSAL_FILES = ("modal_private_miniwob_v2.py", "miniwob_process_entry_v2.py",
                  "build_miniwob_runtime.py", "miniwob_native_v2.py", "miniwob_benchmark_v2.py")
RUN_ID = REMOTE_RUN_ID
REVISION = "28f795cd9ec73aedf2970a8914bbbf0d18accd82"
CONFIG_SHA = "9743d159f38b383b746c54c9d286da79da71768e7997c73b37024da307ffcaa0"
INDEX_SHA = "1bb83a632abd9efef296e6394e35c705195f5b8d7cec394742753196b952aec0"
ADAPTER_SHA = "49311fe8c20663926e6474cc3e2a8c9e86eff2821259e22e85a533eb5ba08933"
ADAPTER_CONFIG_SHA = "52e05adc5f8595942879747f1cb4df8f45c06f547005bf7dc3094e6233723e92"
CANDIDATE_RULE_SHA = "e0edf8b2c5d0778ab241c4a1bc8e128cb3d8471e1631f651e5fb3b17102a99d9"
CHILD_WALL_SECONDS = 1500
GENERATION_HARD_SECONDS = 45
RESERVATION_USD = 2.80
MAX_PROGRESS_BYTES = 8 * 1024 * 1024
NOT_AFTER_UNIX = 1793793600

SDK_PINS = {'_functions.py': 'aa18b6dcfa00f09807d1719fc592f96fa69d51c492eecf059aecec865702a2bf', '_object.py': 'b49b4c10fdf18e32dc09effa598e65f523f37c0424f48844297534faf72511c9', 'functions.py': '11945c1c57e7a186ef6b05099c01c5e6722fa28f465f60e3cbb99498e0979977'}
SAVED_RECEIPT_PINS = {
    'runs/private-miniwob-benchmark-v2.failure.json': '46c1ac59d6fbaed645322c9dd664fc16c2d70f3147e6f48ab83fa87389c92183',
    'runs/private-miniwob-benchmark-v2.log': 'fead854bd61ee55ce37c6b4dcb0d2d76503703b84bc29924e05057b538b510d3',
    'runs/private-miniwob-benchmark-v2.started.json': 'c654fcc58a0ed5aefaa47a3383da346127ecbe89d65e7296a9fd9e007879e4b4',
    'runs/private-miniwob-imports-v2.json': '51d72e183c8cdb8611117d5180fe7a8f9835210844774c7d4f6b35405293afdf',
    'runs/private-miniwob-deployment-v2.json': 'f75f81c1c976743fae527cd88723f656087c5533c8fb651488c3b963e006473b',
    'runs/vllm-pilot-v4.json': '2298d155c13e117ed8837fee509a485c7e8ac4118d5fc57cba0b052b379d90a6',
    'evidence/private_miniwob_v2_pre_spawn_failure_reconciliation.json': '5de5bb1c2babea56b231e4bc8cb004dc9b19f414b78a8302dbf1f9471e7d34e6',
    'evidence/private_miniwob_v2_host_failure_ui.json': 'ecafd674005326a9080d42a3e34f5c9f603cd1a3f165d56dffd3287634fdb020',
    'runs/pilot-v1-remote-report.json': '1525f269113797acb31f52ede41ad1792208d243c88a9b92a807a36afba6a9f5',
    'runs/fixtures-v1.json': '3f979fc23eb0456d2574fedbf12608c4ab7e5198ac36b880726af360e737173d',
}

def source_manifest(package_root: Path, entrypoint: Path) -> dict[str, str]:
    """Pin the whole local package source closure, including package initializers."""
    paths = {"proposals/" + name: entrypoint.parent / name for name in PROPOSAL_FILES}
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("source symlinks are not admitted")
        if path.is_file():
            if path.suffix not in {".py", ".json", ".txt", ".jinja", ".md"}:
                raise ValueError("unreviewed non-source package file: " + str(relative))
            paths["src/weight_space_lab/" + relative.as_posix()] = path
    if len(paths) > 80:
        raise ValueError("bounded complete child source inventory required")
    total = 0
    manifest = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("ordinary readable source file required")
        data = path.read_bytes()
        total += len(data)
        if total > 2 * 1024 * 1024:
            raise ValueError("source inventory exceeds 2 MiB")
        manifest[name] = hashlib.sha256(data).hexdigest()
    return manifest

def build_admission(root: Path = ROOT):
    """Saved-only reconstruction of the exact v2 benchmark admission."""
    ROOT = Path(root)
    for name, expected in SAVED_RECEIPT_PINS.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("ordinary bounded saved receipt required: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("saved evidence changed: " + name)
    saved_deployment = json.loads((ROOT / "runs/private-miniwob-deployment-v2.json").read_text())
    initial_manifest = source_manifest(ROOT / "src/weight_space_lab", ROOT / "proposals/modal_private_miniwob_v2.py")
    if len(initial_manifest) != 23 or initial_manifest != saved_deployment["source_manifest"]:
        raise ValueError("frozen runtime source differs before importing its pure reconciliation helper")
    if (ROOT / "runs/private-miniwob-benchmark-v2.call.json").exists():
        raise ValueError("prior attempt unexpectedly has a call handle; correction admission refuses")
    prior_failure = json.loads((ROOT / "runs/private-miniwob-benchmark-v2.failure.json").read_text())
    if prior_failure.get("error") != {"type": "AttributeError", "message": "type object 'Function' has no attribute 'from_id'"}:
        raise ValueError("exact pre-spawn attribute lookup failure required")
    ui = json.loads((ROOT / "evidence/private_miniwob_v2_host_failure_ui.json").read_text())
    target = ui["deployed_target"]
    if (target["app_id"] != EXPECTED_APP_ID or target["function_id"] != EXPECTED_FUNCTION_ID
            or target["live_containers"] != 0 or target["running_calls"] != 0
            or target["benchmark_activity"] != "No activity"):
        raise ValueError("saved actual target inactivity reconciliation required")
    runs = ROOT / "runs"
    run_id = REMOTE_RUN_ID
    phase = "benchmark"
    prior_bytes = (runs / "pilot-v1-remote-report.json").read_bytes()
    prior = json.loads(prior_bytes)
    prior_source = prior.get("source", {})
    shards = prior_source.get("full_merged_shard_hashes", {})
    if prior.get("status") != "aborted" or len(shards) != 13:
        raise ValueError("complete saved pilot source receipt required; no original shard rehash")
    if prior_source.get("merged_config_sha256") != CONFIG_SHA or prior_source.get("merged_index_sha256") != INDEX_SHA:
        raise ValueError("saved merged source identity differs")
    if prior_source.get("adapter_sha256") != ADAPTER_SHA or prior_source.get("cached_base_revision") != REVISION:
        raise ValueError("saved adapter/base source identity differs")
    fixtures = json.loads((runs / "fixtures-v1.json").read_text())
    if len(fixtures.get("records", [])) != 16 or not all(item["success"] for item in fixtures["records"]):
        raise ValueError("saved CPU fixture admission incomplete")
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "proposals"))
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    reconciliation = reconcile_fixture_admission(fixtures["fixture_manifest"])
    manifest = source_manifest(ROOT / "src/weight_space_lab", ROOT / "proposals/modal_private_miniwob_v2.py")
    pilot_bytes = (runs / "vllm-pilot-v4.json").read_bytes()
    pilot = json.loads(pilot_bytes)
    pilot_child = pilot.get("child", {})
    rule_bytes = (ROOT / "evidence/miniwob_candidate_rule_v1.json").read_bytes()
    if hashlib.sha256(rule_bytes).hexdigest() != CANDIDATE_RULE_SHA:
        raise ValueError("prospective candidate-selection rule changed")
    if pilot.get("status") != "completed":
        raise ValueError("completed actual v4 result required")
    candidates = []
    task_orders = []
    for scale in (0.5, 0.75):
        measured = pilot_child.get("variants", {}).get(str(scale), [])
        if len(measured) != 8 or any(type(row.get("success")) is not bool
                or type(row.get("action_errors")) is not int or row["action_errors"] < 0 for row in measured):
            raise ValueError("each candidate requires eight actual completed v4 development rows")
        task_orders.append([row["task_id"] for row in measured])
        candidates.append((sum(row["success"] for row in measured), sum(row["action_errors"] for row in measured), scale))
    if task_orders[0] != task_orders[1] or len(set(task_orders[0])) != 8:
        raise ValueError("paired v4 development cases do not match")
    # Frozen rule: most successes, then fewest action errors, then .75 on a tie.
    candidates.sort(key=lambda item: (-item[0], item[1], -item[2]))
    candidate_scale = candidates[0][2]
    candidate_selection = {
        "rule_sha256": CANDIDATE_RULE_SHA, "rule": json.loads(rule_bytes),
        "development": {str(scale): {"tasks": 8, "successes": successes, "action_errors": errors}
                        for successes, errors, scale in candidates},
        "v4_status": pilot_child.get("status"),
        "eligible_for_broader_benchmark": pilot_child.get("eligible_for_broader_benchmark", False),
        "v4_report_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
    }
    admission = {"run_id": run_id, "revision": REVISION, "fixtures": fixtures,
                 "candidate_scale": candidate_scale, "experimental_only": True,
                 "candidate_selection": candidate_selection,
                 "candidate_evidence": {"v4_report_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
                     "v4_status": pilot_child.get("status"), "promotion": "experimental measured scale only",
                     "frozen_selection_rule": "successes descending; action errors ascending; .75 tie",
                     "candidate_scores": candidates},
                 "runner_reconciliation": reconciliation, "source_manifest": manifest,
                 "prior_source": prior_source, "prior_report_sha256": hashlib.sha256(prior_bytes).hexdigest(),
                 "merged_metadata_pins": {"config.json": CONFIG_SHA, "model.safetensors.index.json": INDEX_SHA},
                 "adapter_pins": {"adapter_model.safetensors": ADAPTER_SHA, "adapter_config.json": ADAPTER_CONFIG_SHA},
                 "limits": {"child_wall_seconds": CHILD_WALL_SECONDS, "generation_hard_seconds": GENERATION_HARD_SECONDS}}
    if phase == "benchmark":
        imports_bytes = (runs / "private-miniwob-imports-v2.json").read_bytes()
        cpu = json.loads(imports_bytes)
        if (cpu.get("status") != "completed" or cpu.get("child_status") != "imports_passed"
                or cpu.get("source_manifest") != manifest):
            raise ValueError("GPU benchmark requires successful CPU import receipt for exactly this source")
        if cpu["child"]["prior_full_shard_receipt_sha256"] != hashlib.sha256(prior_bytes).hexdigest():
            raise ValueError("saved prior source receipt differs from actual CPU admission")
        admission["imports_receipt"] = cpu
        admission["imports_receipt_sha256"] = hashlib.sha256(imports_bytes).hexdigest()
        admission["miniwob_preflight"] = cpu["child"]["miniwob_preflight"]
        admission["miniwob_versions"] = cpu["child"]["miniwob_versions"]
        admission["runtime_manifest"] = cpu["child"]["runtime_manifest"]
        deployment_bytes = (runs / "private-miniwob-deployment-v2.json").read_bytes()
        deployment = json.loads(deployment_bytes)
        if (deployment.get("app_name") != APP_NAME or deployment.get("source_manifest") != manifest
                or not isinstance(deployment.get("app_id"), str) or not deployment["app_id"].startswith("ap-")
                or not isinstance(deployment.get("function_id"), str) or not deployment["function_id"].startswith("fu-")):
            raise ValueError("observed exact private deployment receipt required")
        admission["deployment_receipt_sha256"] = hashlib.sha256(deployment_bytes).hexdigest()
        admission["deployment_identity"] = {key: deployment[key] for key in ("app_name", "app_id", "function_id")}
    if deployment["app_id"] != EXPECTED_APP_ID or deployment["function_id"] != EXPECTED_FUNCTION_ID:
        raise ValueError("deployment identity differs from the fixed correction target")
    if len(manifest) != 23:
        raise ValueError("exact original 23-file runtime closure required")
    return admission, deployment


def verify_sdk(modal_api):
    """Inspect the installed SDK only; no credential lookup or hydration here."""
    if importlib.metadata.version("modal") != "1.4.3":
        raise ValueError("reviewed host SDK 1.4.3 required")
    directory = Path(modal_api.__file__).resolve().parent
    for name, expected in SDK_PINS.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("installed SDK source changed: " + name)
    for name in ("from_name", "hydrate", "spawn"):
        if not callable(getattr(modal_api.Function, name, None)):
            raise ValueError("required public SDK interface unavailable: " + name)
    return {"version": "1.4.3", "sources": SDK_PINS,
            "lookup": "Function.from_name(app, benchmark, environment_name=main); hydrate; exact object_id join"}


def write_once(path: Path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False)
    if len(encoded.encode()) > 16 * 1024 * 1024:
        raise ValueError("host receipt exceeds 16 MiB")
    with path.open("x") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def dispatch_once(admission, deployment, runs, function_factory, *,
                  sdk_receipt, source_sha256, save=write_once,
                  clock=time.monotonic, now=time.time, before_cancel=lambda: None):
    """One lookup/hydration, then at most one spawn; injected fakes support tests.

    A lookup failure is pre-submit. Entered spawn without a returned handle is
    uncertain and never retried. A returned handle remains in cancellation
    coverage even if writing its local receipt fails.
    """
    runs = Path(runs)
    if not now() < NOT_AFTER_UNIX:
        raise RuntimeError("fixed remote experiment expired")
    if (admission.get("run_id") != REMOTE_RUN_ID or deployment.get("app_name") != APP_NAME
            or deployment.get("app_id") != EXPECTED_APP_ID or deployment.get("function_id") != EXPECTED_FUNCTION_ID
            or admission.get("source_manifest") != deployment.get("source_manifest")):
        raise ValueError("exact remote v2 identity and source admission required")
    started = clock()
    base = {"schema": 1, "local_attempt_id": LOCAL_ATTEMPT_ID, "remote_run_id": REMOTE_RUN_ID,
            "target_app_name": APP_NAME, "target_app_id": EXPECTED_APP_ID,
            "target_function_id": EXPECTED_FUNCTION_ID, "environment": "main",
            "host_source_sha256": source_sha256, "source_manifest": admission["source_manifest"],
            "sdk": sdk_receipt, "saved_receipt_pins": SAVED_RECEIPT_PINS,
            "new_app_created": False, "new_deployment": False, "cpu_repeated": False,
            "new_reservation_usd": 0, "retained_shared_hosting_benchmark_reservation_usd": 2.80,
            "total_retained_reservations_usd": 29.60,
            "reservation_note": "Existing reviewed shared allocation retained; no settled-cost or reservation-release inference",
            "host_deadline_seconds": HOST_DEADLINE_SECONDS,
            "cancellation_grace_seconds": CANCELLATION_GRACE_SECONDS}
    # An existing local v3 marker refuses before any provider lookup. The old
    # local v2 marker and failure are preserved and are never deleted/reset.
    save(runs / (LOCAL_ATTEMPT_ID + ".started.json"), {**base, "unix": now()})
    call = None
    stage = "before_lookup"
    spawn_entered = False
    try:
        stage = "lookup"
        function = function_factory(APP_NAME, "benchmark", environment_name="main")
        stage = "hydrate"
        function.hydrate()
        actual_id = function.object_id
        if actual_id != EXPECTED_FUNCTION_ID:
            raise ValueError("hydrated function ID differs from the reviewed deployed target")
        stage = "identity_verified"
        save(runs / (LOCAL_ATTEMPT_ID + ".lookup.json"), {
            **base, "actual_function_id": actual_id, "unix": now(), "spawn_entered": False,
            "app_identity_scope": "Named lookup and exact function ID; app ID comes from the separately observed deployment receipt"})
        if clock() - started >= HOST_DEADLINE_SECONDS - CANCELLATION_GRACE_SECONDS:
            raise TimeoutError("host admission deadline reached before dispatch")
        # Do not replace this handle after identity verification. SDK1.4.3
        # hydrates the fixed object and spawn uses that hydrated identity.
        stage = "spawn_entered"
        spawn_entered = True
        call = function.spawn(admission)
        stage = "handle_returned"
        save(runs / (LOCAL_ATTEMPT_ID + ".call.json"), {
            **base, "function_call_id": call.object_id, "actual_function_id": function.object_id,
            "unix": now(), "spawn_entered": True})
        remaining = HOST_DEADLINE_SECONDS - CANCELLATION_GRACE_SECONDS - (clock() - started)
        if remaining <= 0:
            raise TimeoutError("host result deadline reached after submission")
        stage = "waiting_for_result"
        result = call.get(timeout=remaining)
        stage = "result_returned"
        save(runs / (LOCAL_ATTEMPT_ID + ".remote-report.json"), result)
        if (not isinstance(result, dict) or result.get("function_call_id") != call.object_id
                or result.get("run") != REMOTE_RUN_ID
                or result.get("source_manifest") != admission["source_manifest"]):
            raise RuntimeError("remote result does not join the exact call/run/source")
        if result.get("container_termination_required", True):
            raise RuntimeError("remote supervision requires container termination")
        summary = {**base, "status": "remote_result_received", "remote_status": result.get("status"),
                   "child_status": result.get("child_status"), "function_call_id": call.object_id,
                   "elapsed_s": clock() - started, "unix": now(),
                   "physical_stop_and_billing": "Require actual provider terminal and billing evidence"}
        save(runs / (LOCAL_ATTEMPT_ID + ".json"), summary)
        return summary
    except BaseException as original_error:
        failure = {**base, "stage": stage, "unix": now(), "spawn_entered": spawn_entered,
                   "error": {"type": type(original_error).__name__, "message": str(original_error)[:2000]},
                   "submission_disposition": ("pre_submit_no_spawn_entered" if not spawn_entered else
                       "entered_unknown_no_returned_handle" if call is None else "returned_handle"),
                   "retry_authorized": False}
        if call is None:
            failure["cancellation"] = {"attempted": False, "reason": "No returned call handle"}
        else:
            cancellation_entered = False
            try:
                before_cancel()
                cancellation_entered = True
                call.cancel(terminate_containers=True)
                cancellation = {"attempted": True, "request_returned": True}
            except BaseException as cancel_error:
                cancellation = {"attempted": cancellation_entered, "request_returned": False,
                                "error": {"type": type(cancel_error).__name__, "message": str(cancel_error)[:1000]}}
            cancellation["physical_stop_or_billing_stop"] = "Not established by cancellation acknowledgement"
            failure["cancellation"] = cancellation
            try:
                failure["function_call_id"] = call.object_id
            except BaseException:
                failure["function_call_id_unavailable"] = True
        try:
            save(runs / (LOCAL_ATTEMPT_ID + ".failure.json"), failure)
        except BaseException as save_error:
            failure["failure_save_error"] = type(save_error).__name__
            try:
                print(json.dumps(failure), file=sys.stderr)
            except BaseException:
                pass
        raise


def alarm_handlers(arm):
    """Create testable phase-aware handlers; cancellation never rearms itself."""
    cancellation_phase = False

    def deadline_handler(signum, frame):
        nonlocal cancellation_phase
        if not cancellation_phase:
            cancellation_phase = True
            arm(CANCELLATION_GRACE_SECONDS)
        raise TimeoutError("bounded host lifecycle deadline")

    def bound_cancel():
        nonlocal cancellation_phase
        cancellation_phase = True
        arm(CANCELLATION_GRACE_SECONDS)

    return deadline_handler, bound_cancel


def main():
    admission, deployment = build_admission()
    import modal
    sdk = verify_sdk(modal)
    previous_handler = signal.getsignal(signal.SIGALRM)
    deadline_handler, bound_cancel = alarm_handlers(
        lambda seconds: signal.setitimer(signal.ITIMER_REAL, seconds))

    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, HOST_DEADLINE_SECONDS)
    try:
        summary = dispatch_once(admission, deployment, ROOT / "runs", modal.Function.from_name,
            sdk_receipt=sdk, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            before_cancel=bound_cancel)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
