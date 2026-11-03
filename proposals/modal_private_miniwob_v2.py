"""SOURCE-ONLY private MiniWoB hosting proposal; importing this file starts no remote function.

The image is independent of modal_pilot.py. A fresh spawned child owns CUDA and
all model calls; the supervisor enforces bounded source deadlines and retains
JSON receipts. No existing endpoint is modified and no model is published.
Provider replacement/retry and billing termination require actual observations;
retries=0 and a cancellation acknowledgement do not establish those facts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "adam-1-cu-private-miniwob-v2"
PROPOSAL_FILES = ("modal_private_miniwob_v2.py", "miniwob_process_entry_v2.py",
                  "build_miniwob_runtime.py", "miniwob_native_v2.py", "miniwob_benchmark_v2.py")
RUN_ID = "private-miniwob-benchmark-v2"
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
NOT_AFTER_UNIX = 1793793600  # 2026-11-04 12:00:00 UTC

app = modal.App(APP_NAME)
image = (modal.Image.debian_slim(python_version="3.11")
    .run_commands("timeout --signal=TERM --kill-after=10s 900s python -m pip install --no-cache-dir "
                  "vllm==0.18.0 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 "
                  "transformers==4.57.6 safetensors==0.7.0 pillow==12.2.0 playwright==1.60.0")
    .run_commands("timeout --signal=TERM --kill-after=10s 600s python -m playwright install --with-deps chromium")
    .run_commands("timeout --signal=TERM --kill-after=10s 300s python -m pip install --no-cache-dir "
                  "gymnasium==1.2.3 selenium==4.40.0 numpy==2.2.6 "
                  "'miniwob @ https://files.pythonhosted.org/packages/fc/a1/a34e0cc33dcba9de00001695979d2b9e3cc2de03bacdf010f045fb084afe/miniwob-1.1.0-py3-none-any.whl#sha256=297006e84219d56ad037e843527ed773a4a82509dc43b6031ec4a3ad3e3e61d9'")
    .env({"PYTHONPATH": "/opt/weight-space/src:/opt/weight-space/proposals", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          "WEIGHT_SPACE_PRIVATE_PACKAGED_RUNTIME": "1",
          "MINIWOB_CHROME_BINARY": "/opt/cft/chrome-linux64/chrome",
          "MINIWOB_CHROMEDRIVER": "/opt/cft/chromedriver-linux64/chromedriver",
          "SE_OFFLINE": "true",
          "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}))
# The exact packaged entry module is also imported during CPU admission. Remote
# imports must not look for the host checkout or construct local source mounts.
# modal.is_local() is True in spawned children, so use an explicit image marker.
if os.environ.get("WEIGHT_SPACE_PRIVATE_PACKAGED_RUNTIME") != "1":
    image = (image.add_local_file(ROOT / "proposals/build_miniwob_runtime.py", "/opt/build_miniwob_runtime.py", copy=True)
                  .run_commands("timeout --signal=TERM --kill-after=10s 660s python /opt/build_miniwob_runtime.py")
                  .add_local_dir(ROOT / "src", "/opt/weight-space/src", copy=True))
    for name in PROPOSAL_FILES:
        image = image.add_local_file(ROOT / "proposals" / name, "/opt/weight-space/proposals/" + name, copy=True)
outputs = modal.Volume.from_name("bu-weight-space-results-v3", create_if_missing=True, version=2)
once = modal.Dict.from_name("adam-1-cu-private-miniwob-v2-once", create_if_missing=True)
volumes = {
    "/base": modal.Volume.from_name("hf-cache").with_mount_options(read_only=True),
    "/adapter": modal.Volume.from_name("bu-lora").with_mount_options(read_only=True),
    "/merged": modal.Volume.from_name("bu-merged").with_mount_options(read_only=True),
    "/results": outputs,
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


def _supervise_worker(admission, kind):
    import multiprocessing
    import os
    import signal
    import tempfile
    import weight_space_lab.diagnostic as diagnostic_source
    from weight_space_lab.diagnostic import atomic_json
    from miniwob_process_entry_v2 import worker_entry

    if kind not in {"imports", "benchmark"}:
        raise ValueError("unknown supervisor kind")
    run_id = "private-miniwob-imports-v2" if kind == "imports" else RUN_ID
    wall_limit = 450 if kind == "imports" else CHILD_WALL_SECONDS

    # No torch/vLLM import or CUDA initialization in this process.
    actual_manifest = source_manifest(Path(diagnostic_source.__file__).parent,
                                     Path("/opt/weight-space/proposals/modal_private_miniwob_v2.py"))
    if actual_manifest != admission["source_manifest"] or admission["run_id"] != run_id:
        return {"schema": 1, "run": run_id, "status": "source_admission_failed", "model_calls": 0,
                "container_termination_required": False, "actual_source_manifest": actual_manifest}
    if kind == "benchmark":
        cpu = admission.get("imports_receipt", {})
        child_receipt = cpu.get("child", {})
        if (cpu.get("status") != "completed" or not cpu.get("child_reaped") or cpu.get("child_exitcode") != 0
                or child_receipt.get("status") != "imports_passed" or child_receipt.get("cuda_initialized") is not False
                or child_receipt.get("model_calls") != 0 or child_receipt.get("tensor_payloads_loaded") != 0
                or cpu.get("source_manifest") != actual_manifest or child_receipt.get("source_manifest") != actual_manifest
                or len(child_receipt.get("miniwob_preflight", {}).get("cases", [])) != 24
                or child_receipt.get("candidate_scale") != admission.get("candidate_scale")):
            return {"schema": 1, "run": run_id, "status": "cpu_import_admission_failed", "model_calls": 0,
                    "container_termination_required": False}
    # Fixed experiment expiry prevents a Dict inactivity expiry from rearming
    # this exact source. Provider startup retries can still incur charges.
    if not time.time() < NOT_AFTER_UNIX:
        return {"status": "fixed_experiment_expired", "container_termination_required": False}
    if not once.put(kind + "-entered", {"unix": time.time(), "source_manifest": actual_manifest}, skip_if_exists=True):
        return {"status": "refused_consumed_entry", "container_termination_required": False}
    run_dir = Path("/results") / run_id
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "started.json").open("x") as stream:
        json.dump({"unix": time.time(), "source_manifest": actual_manifest}, stream)
    outputs.commit()
    spool = tempfile.TemporaryDirectory(prefix="bu-private-miniwob-benchmark-v2-")
    progress_path = Path(spool.name) / "child-progress.json"
    child = multiprocessing.get_context("spawn").Process(target=worker_entry,
        args=(kind, admission, str(progress_path)), daemon=False)
    started = time.monotonic()
    report = {"schema": 1, "run": run_id, "status": "supervising", "child": None,
              "function_call_id": modal.current_function_call_id(), "declared_app_name": APP_NAME,
              "source_manifest": actual_manifest, "lifecycle": [],
              "limits": {"child_wall_seconds": wall_limit,
                         "generation_hard_seconds": GENERATION_HARD_SECONDS,
                         "terminal_exit_grace_seconds": 20},
              "original_volumes_read_only": True, "endpoint_created": False,
              "weight_publication": False, "provider_retry_limit_is_source_only": True,
              "physical_gpu_release_and_billing_stop": "Requires provider terminal/billing evidence"}
    last_mtime = None

    def read_progress():
        nonlocal last_mtime
        if not progress_path.exists():
            return
        stat = progress_path.stat()
        if stat.st_size > MAX_PROGRESS_BYTES:
            raise ValueError("child progress exceeds 8 MiB")
        if stat.st_mtime_ns != last_mtime:
            partial = json.loads(progress_path.read_text())
            if not isinstance(partial, dict):
                raise ValueError("child progress must be a JSON object")
            report["child"] = partial
            last_mtime = stat.st_mtime_ns

    def send_owned_signal(sig):
        try:
            if os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, sig)
            elif sig == signal.SIGTERM:
                child.terminate()
            else:
                child.kill()
        except ProcessLookupError:
            report["lifecycle"].append({"event": "exited_during_signal_request", "signal": int(sig), "unix": time.time()})

    def stop_and_reap():
        if child.pid is None:
            report["child_started"] = False
            report["child_reaped"] = True
            return
        report["child_started"] = True
        if child.is_alive():
            report["lifecycle"].append({"event": "terminate_requested", "unix": time.time()})
            send_owned_signal(signal.SIGTERM)
            child.join(timeout=5)
        if child.is_alive():
            report["lifecycle"].append({"event": "kill_requested", "unix": time.time()})
            send_owned_signal(signal.SIGKILL)
            child.join(timeout=5)
        else:
            child.join(timeout=1)
        report["child_exitcode"] = child.exitcode
        report["child_reaped"] = child.exitcode is not None and not child.is_alive()

    try:
        child.start()
        report["lifecycle"].append({"event": "spawned", "pid": child.pid, "unix": time.time()})
        while child.is_alive():
            now = time.monotonic()
            read_progress()
            partial = report.get("child") or {}
            stage = partial.get("stage")
            stage_start = partial.get("stage_started_monotonic", started)
            if isinstance(stage_start, bool) or not isinstance(stage_start, (int, float)) or not math.isfinite(stage_start):
                raise ValueError("invalid stage timestamp")
            reason = None
            if now - started >= wall_limit:
                reason = "child_wall_deadline"
            elif stage in {"generation", "warmup"} and now - stage_start >= GENERATION_HARD_SECONDS:
                reason = "native_generation_hard_deadline"
            elif stage in {"done", "completed", "failed"} and now - stage_start >= 20:
                reason = "terminal_exit_grace_deadline"
            if reason:
                report.update({"status": "aborted", "abort_reason": reason})
                break
            # No provider RPC/commit in the live-child watchdog loop.
            child.join(timeout=0.2)
    except BaseException as exc:
        report.update({"status": "aborted", "supervisor_error": {"type": type(exc).__name__, "message": str(exc)[:4000]}})
    finally:
        try:
            stop_and_reap()
        except BaseException as exc:
            report["child_reaped"] = False
            report["reap_error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        try:
            read_progress()
        except BaseException as exc:
            report["progress_error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        if report["status"] == "supervising":
            returned_stage = (report.get("child") or {}).get("stage")
            browser_closed = kind == "imports" or (report.get("child") or {}).get("browser_close_returned") is True
            report["status"] = "completed" if (report.get("child_reaped") and report.get("child_exitcode") == 0
                and returned_stage in {"done", "completed"} and browser_closed) else "failed"
        if not report.get("child_reaped", False):
            report["status"] = "child_not_reaped"
        report["child_status"] = (report.get("child") or {}).get("status")
        report["direct_cuda_owner_exit_confirmed"] = bool(kind == "benchmark" and report.get("child_started") and report.get("child_reaped"))
        report["descendant_disposition"] = "No independent descendant inventory; direct child reap does not prove descendant stop"
        report["container_termination_required"] = report["status"] != "completed"
        report["elapsed_s"] = time.monotonic() - started
        if report.get("child") is not None:
            atomic_json(run_dir / "child-progress.json", report["child"])
        atomic_json(run_dir / "report.json", report)
        outputs.commit()
        spool.cleanup()
    return report


@app.function(image=image, cpu=(4, 4), memory=(4096, 16384), volumes=volumes,
              timeout=600, startup_timeout=240, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, single_use_containers=True, block_network=True)
def verify_runtime_imports(admission):
    return _supervise_worker(admission, "imports")


@app.function(image=image, gpu="A100-80GB", cpu=(4, 4), memory=(8192, 98304), volumes=volumes,
              timeout=1560, startup_timeout=240, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, scaledown_window=2, single_use_containers=True, block_network=True)
def benchmark(admission):
    """Private SDK function; no HTTP route, secret, or new access permission."""
    return _supervise_worker(admission, "benchmark")


@app.local_entrypoint()
def main(phase: str = "imports"):
    """Once-only launch boundary; exact source requires separate review/admission."""
    runs = ROOT / "runs"
    if not time.time() < NOT_AFTER_UNIX:
        raise RuntimeError("fixed private experiment expired; no renewed call authority")
    if phase not in {"imports", "benchmark"}:
        raise ValueError("phase must be imports or benchmark")
    run_id = "private-miniwob-imports-v2" if phase == "imports" else RUN_ID
    marker = runs / (run_id + ".started.json")
    if marker.exists():
        raise RuntimeError(run_id + " is consumed; no repeat invocation")
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
    with (runs / (run_id + ".source-manifest.json")).open("x") as stream:
        json.dump(manifest, stream, indent=2)
    with marker.open("x") as stream:
        json.dump({"run": run_id, "unix": time.time(), "app_id": app.app_id,
                   "shared_v4_budget_envelope_usd": 6, "combined_hosting_benchmark_suballocation_usd": RESERVATION_USD,
                   "v4_pilot_suballocation_usd": 3.20, "retained_prior_reservations_usd": 23.60,
                   "new_reservation_usd": 0, "total_retained_reservations_usd": 29.60,
                   "reservation_reference": "vllm-imports-v4.started.json", "source_manifest": manifest}, stream, indent=2)
    call = None
    try:
        # A returned handle remains under cancellation coverage even if saving
        # its receipt fails. A failed spawn without a handle is never retried.
        # The GPU call is looked up from the separately deployed private app.
        # No production app is stopped, overwritten, or given a public endpoint.
        function = verify_runtime_imports if phase == "imports" else modal.Function.from_id(deployment["function_id"])
        call = function.spawn(admission)
        with (runs / (run_id + ".call.json")).open("x") as stream:
            json.dump({"function_call_id": call.object_id, "function_id": function.object_id,
                       "caller_app_id": app.app_id, "target_deployed_app_name": APP_NAME if phase == "benchmark" else None,
                       "target_deployed_app_id": deployment["app_id"] if phase == "benchmark" else None,
                       "unix": time.time()}, stream)
        result = call.get(timeout=1000 if phase == "imports" else 2000)
        with (runs / (run_id + ".json")).open("x") as stream:
            json.dump(result, stream, indent=2)
        if result.get("function_call_id") != call.object_id:
            raise RuntimeError("returned supervisor receipt does not join the actual function call")
        if result.get("container_termination_required", True):
            raise RuntimeError("vLLM child did not complete normally; request container termination")
    except BaseException as original_error:
        failure = {"run": run_id, "unix": time.time(),
                   "error": {"type": type(original_error).__name__, "message": str(original_error)[:4000]},
                   "submission_disposition": "unknown_no_returned_handle" if call is None else "returned_handle",
                   "retry_authorized": False}
        if call is None:
            # The provider may have accepted work before the SDK raised. There
            # is no exact returned handle to cancel or legitimate basis to retry.
            failure["cancellation"] = {"attempted": False, "reason": "No returned function-call handle"}
        else:
            try:
                call.cancel(terminate_containers=True)
                cancellation = {"attempted": True, "request_returned": True}
            except BaseException as cancel_error:
                cancellation = {"attempted": True, "request_returned": False,
                                "error": {"type": type(cancel_error).__name__, "message": str(cancel_error)[:1000]}}
            cancellation["physical_stop_or_billing_stop"] = "Not proved by cancellation ACK"
            failure["cancellation"] = cancellation
            # Receipt attributes must not prevent the cancellation request.
            try:
                failure["function_call_id"] = call.object_id
            except BaseException as receipt_error:
                failure["handle_receipt_error"] = str(receipt_error)[:1000]
        try:
            with (runs / (run_id + ".failure.json")).open("x") as stream:
                json.dump(failure, stream, indent=2)
        except BaseException as save_error:
            failure["failure_receipt_save_error"] = {"type": type(save_error).__name__, "message": str(save_error)[:1000]}
            # Keep the original exception and attempt a console receipt if the
            # filesystem refused the exclusive failure-evidence write.
            try:
                print(json.dumps(failure), file=sys.stderr)
            except BaseException:
                pass
        raise
    print(json.dumps({"saved": str(runs / (run_id + ".json")), "status": result["status"], "child_status": result.get("child_status")}))
