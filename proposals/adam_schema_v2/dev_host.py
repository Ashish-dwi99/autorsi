"""Private once-only host driver. Import/prepare is local only; main dispatches once.

New source-specific CPU admission is mandatory before the development GPU phase.
Saved older records supply artifact identity only; they are not fresh CPU passes.
No verification phase, public endpoint, deployment or resource purchase is here.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dev_contract import (APP_NAME, IMPORT_RUN, GPU_RUN, NOT_AFTER_UNIX, source_manifest,
    experiment_spec, validate_spec)
HOST_DEADLINE_SECONDS = 1800
CANCELLATION_GRACE_SECONDS = 20
SDK_PINS = {'_functions.py': 'aa18b6dcfa00f09807d1719fc592f96fa69d51c492eecf059aecec865702a2bf', '_object.py': 'b49b4c10fdf18e32dc09effa598e65f523f37c0424f48844297534faf72511c9', 'functions.py': '11945c1c57e7a186ef6b05099c01c5e6722fa28f465f60e3cbb99498e0979977'}
SAVED_RECEIPT_PINS = {
 "runs/pilot-v1-remote-report.json":"1525f269113797acb31f52ede41ad1792208d243c88a9b92a807a36afba6a9f5",
 "runs/fixtures-v1.json":"3f979fc23eb0456d2574fedbf12608c4ab7e5198ac36b880726af360e737173d"}
REVISION="28f795cd9ec73aedf2970a8914bbbf0d18accd82"
CONFIG_SHA="9743d159f38b383b746c54c9d286da79da71768e7997c73b37024da307ffcaa0"
INDEX_SHA="1bb83a632abd9efef296e6394e35c705195f5b8d7cec394742753196b952aec0"
ADAPTER_SHA="49311fe8c20663926e6474cc3e2a8c9e86eff2821259e22e85a533eb5ba08933"
ADAPTER_CONFIG_SHA="52e05adc5f8595942879747f1cb4df8f45c06f547005bf7dc3094e6233723e92"


def read_json(path, limit=16*1024**2):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError("ordinary bounded receipt required")
    raw=path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def build_admission(phase, allocation_path, deployment_path, terminal_path, root=ROOT, now=time.time):
    if phase not in ("imports", "development"):
        raise ValueError("no verification phase is executable here")
    root=Path(root); proposal=root/"proposals/adam_schema_v2"
    manifest=source_manifest(root/"src/weight_space_lab",proposal)
    allocation, allocation_hash=read_json(allocation_path)
    if (allocation.get("source_manifest") != manifest or allocation.get("app_name") != APP_NAME
            or type(allocation.get("allocation_usd")) not in (int,float)
            or not 0<allocation["allocation_usd"]<=2.5
            or type(allocation.get("total_reserved_usd")) not in (int,float)
            or not allocation["allocation_usd"]<=allocation["total_reserved_usd"]<=30
            or not now()<allocation.get("not_after_unix",0)<=NOT_AFTER_UNIX):
        raise ValueError("actual root-reviewed source-bound budget allocation required")
    terminal, terminal_hash=read_json(terminal_path)
    apps=terminal.get("applications",[])
    required=allocation.get("required_inactive_app_ids",[])
    if (not required or len(set(required))!=len(required) or
            not 0<=now()-terminal.get("observed_unix",0)<=300 or
            {row.get("app_id") for row in apps}!=set(required) or len(apps)!=len(required) or
            any(row.get("live_containers")!=0 or row.get("running_calls")!=0 for row in apps)):
        raise ValueError("fresh exact no-overlap provider observation required")
    deployment, deployment_hash=read_json(deployment_path)
    function_name="verify_runtime_imports" if phase=="imports" else "development"
    function_id=deployment.get("functions",{}).get(function_name)
    if (deployment.get("app_name")!=APP_NAME or deployment.get("source_manifest")!=manifest
            or not str(deployment.get("app_id","")).startswith("ap-")
            or not isinstance(function_id,str) or not function_id.startswith("fu-")):
        raise ValueError("observed actual deployment/function IDs bound to this source required")
    evidence={}
    for name, expected in SAVED_RECEIPT_PINS.items():
        receipt,digest=read_json(root/name)
        if digest!=expected: raise ValueError("saved artifact history changed")
        evidence[name]=receipt
    prior=evidence["runs/pilot-v1-remote-report.json"]; source=prior["source"]
    if (len(source.get("full_merged_shard_hashes",{}))!=13
            or source.get("merged_config_sha256")!=CONFIG_SHA or source.get("merged_index_sha256")!=INDEX_SHA
            or source.get("adapter_sha256")!=ADAPTER_SHA or source.get("cached_base_revision")!=REVISION):
        raise ValueError("exact saved original artifact identities required")
    fixtures=evidence["runs/fixtures-v1.json"]
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    admission={"run_id":IMPORT_RUN if phase=="imports" else GPU_RUN,
        "experiment_spec":experiment_spec(),"source_manifest":manifest,"revision":REVISION,
        "fixtures":fixtures,"runner_reconciliation":reconcile_fixture_admission(fixtures["fixture_manifest"]),
        "prior_source":source,"prior_report_sha256":SAVED_RECEIPT_PINS["runs/pilot-v1-remote-report.json"],
        "merged_metadata_pins":{"config.json":CONFIG_SHA,"model.safetensors.index.json":INDEX_SHA},
        "adapter_pins":{"adapter_model.safetensors":ADAPTER_SHA,"adapter_config.json":ADAPTER_CONFIG_SHA},
        "root_allocation":allocation,"root_allocation_sha256":allocation_hash,
        "no_overlap_observation":terminal,"no_overlap_receipt_sha256":terminal_hash,
        "deployment_receipt_sha256":deployment_hash}
    if phase=="development":
        cpu,cpu_hash=read_json(root/"runs"/(IMPORT_RUN+".remote-report.json"))
        child=cpu.get("child",{})
        from dev_benchmark import validate_preflight
        if (cpu.get("run")!=IMPORT_RUN or cpu.get("status")!="completed"
                or cpu.get("source_manifest")!=manifest or not cpu.get("child_reaped") or cpu.get("child_exitcode")!=0
                or child.get("status")!="imports_passed" or child.get("source_manifest")!=manifest
                or child.get("cuda_initialized") is not False or child.get("model_calls")!=0
                or child.get("tensor_payloads_loaded")!=0 or child.get("experiment_spec")!=experiment_spec()
                or child.get("prior_full_shard_receipt_sha256")!=admission["prior_report_sha256"]):
            raise ValueError("fresh exact new CPU admission required; older passes cannot substitute")
        from action_schema import validate_cpu_grammar
        validate_cpu_grammar(child["grammar_preflight"])
        validate_preflight(child["miniwob_preflight"])
        admission.update(imports_receipt=cpu,imports_receipt_sha256=cpu_hash,
            miniwob_preflight=child["miniwob_preflight"],miniwob_versions=child["miniwob_versions"],
            runtime_manifest=child["runtime_manifest"],grammar_preflight=child["grammar_preflight"])
    validate_spec(admission)
    return admission,{**deployment,"function_id":function_id}


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
            "lookup": "Function.from_name(app, exact_phase_function, environment_name=main); hydrate; exact object_id join"}


def write_once(path: Path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False)
    if len(encoded.encode()) > 16 * 1024 * 1024:
        raise ValueError("host receipt exceeds 16 MiB")
    with path.open("x") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_disk(path, required_bytes):
    if shutil.disk_usage(path).free < required_bytes:
        raise RuntimeError("fresh local preservation/download floor unavailable")


def dispatch_once(admission, deployment, runs, function_factory, *,
                  sdk_receipt, source_sha256, save=write_once,
                  clock=time.monotonic, now=time.time, before_cancel=lambda: None,
                  disk_guard=require_disk):
    """One lookup/hydration, then at most one spawn; injected fakes support tests.

    A lookup failure is pre-submit. Entered spawn without a returned handle is
    uncertain and never retried. A returned handle remains in cancellation
    coverage even if writing its local receipt fails.
    """
    runs = Path(runs)
    remote_run = admission["run_id"]
    if remote_run not in (IMPORT_RUN, GPU_RUN):
        raise ValueError("development-only fixed run identity required")
    local_attempt = remote_run
    expected_app, expected_function = deployment["app_id"], deployment["function_id"]
    function_name = "verify_runtime_imports" if remote_run == IMPORT_RUN else "development"
    if not now() < NOT_AFTER_UNIX:
        raise RuntimeError("fixed remote experiment expired")
    if (admission.get("run_id") != remote_run or deployment.get("app_name") != APP_NAME
            or deployment.get("app_id") != expected_app or deployment.get("function_id") != expected_function
            or admission.get("source_manifest") != deployment.get("source_manifest")):
        raise ValueError("exact new development identity and source admission required")
    started = clock()
    base = {"schema": 1, "local_attempt_id": local_attempt, "remote_run_id": remote_run,
            "target_app_name": APP_NAME, "target_app_id": expected_app,
            "target_function_id": expected_function, "environment": "main",
            "host_source_sha256": source_sha256, "source_manifest": admission["source_manifest"],
            "sdk": sdk_receipt, "saved_artifact_receipt_pins": SAVED_RECEIPT_PINS,
            "new_app_created": False, "new_deployment": False, "cpu_repeated": False,
            "root_allocation": admission["root_allocation"],
            "reservation_note": "Existing reviewed shared allocation retained; no settled-cost or reservation-release inference",
            "host_deadline_seconds": HOST_DEADLINE_SECONDS,
            "cancellation_grace_seconds": CANCELLATION_GRACE_SECONDS}
    # An existing exact phase marker refuses before any provider lookup.
    # Prior experiments and failures stay unchanged; no reset/retry path.
    disk_guard(runs, 400 * 1024**2)
    save(runs / (local_attempt + ".started.json"), {**base, "unix": now()})
    call = None
    stage = "before_lookup"
    spawn_entered = False
    try:
        stage = "lookup"
        function = function_factory(APP_NAME, function_name, environment_name="main")
        stage = "hydrate"
        function.hydrate()
        actual_id = function.object_id
        if actual_id != expected_function:
            raise ValueError("hydrated function ID differs from the reviewed deployed target")
        stage = "identity_verified"
        save(runs / (local_attempt + ".lookup.json"), {
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
        save(runs / (local_attempt + ".call.json"), {
            **base, "function_call_id": call.object_id, "actual_function_id": function.object_id,
            "unix": now(), "spawn_entered": True})
        remaining = HOST_DEADLINE_SECONDS - CANCELLATION_GRACE_SECONDS - (clock() - started)
        if remaining <= 0:
            raise TimeoutError("host result deadline reached after submission")
        stage = "waiting_for_result"
        disk_guard(runs, (200 + 16) * 1024**2)
        result = call.get(timeout=remaining)
        stage = "result_returned"
        save(runs / (local_attempt + ".remote-report.json"), result)
        if (not isinstance(result, dict) or result.get("function_call_id") != call.object_id
                or result.get("run") != remote_run
                or result.get("source_manifest") != admission["source_manifest"]):
            raise RuntimeError("remote result does not join the exact call/run/source")
        if result.get("container_termination_required", True):
            raise RuntimeError("remote supervision requires container termination")
        summary = {**base, "status": "remote_result_received", "remote_status": result.get("status"),
                   "child_status": result.get("child_status"), "function_call_id": call.object_id,
                   "elapsed_s": clock() - started, "unix": now(),
                   "physical_stop_and_billing": "Require actual provider terminal and billing evidence"}
        save(runs / (local_attempt + ".json"), summary)
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
            save(runs / (local_attempt + ".failure.json"), failure)
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
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("imports","development"),required=True)
    parser.add_argument("--allocation",required=True)
    parser.add_argument("--deployment",required=True)
    parser.add_argument("--terminal",required=True)
    args=parser.parse_args()
    admission,deployment=build_admission(args.phase,args.allocation,args.deployment,args.terminal)
    import modal
    sdk=verify_sdk(modal)
    old=signal.getsignal(signal.SIGALRM)
    handler,cancel=alarm_handlers(lambda seconds:signal.setitimer(signal.ITIMER_REAL,seconds))
    signal.signal(signal.SIGALRM,handler);signal.setitimer(signal.ITIMER_REAL,HOST_DEADLINE_SECONDS)
    try:
        result=dispatch_once(admission,deployment,ROOT/"runs",modal.Function.from_name,
            sdk_receipt=sdk,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),before_cancel=cancel)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    print(json.dumps(result))

if __name__=="__main__":
    main()
