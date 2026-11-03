"""SOURCE-ONLY vLLM v4 proposal; importing this file starts no remote function.

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


ROOT = Path(__file__).resolve().parent
RUN_ID = "vllm-pilot-v4"
REVISION = "28f795cd9ec73aedf2970a8914bbbf0d18accd82"
CONFIG_SHA = "9743d159f38b383b746c54c9d286da79da71768e7997c73b37024da307ffcaa0"
INDEX_SHA = "1bb83a632abd9efef296e6394e35c705195f5b8d7cec394742753196b952aec0"
ADAPTER_SHA = "49311fe8c20663926e6474cc3e2a8c9e86eff2821259e22e85a533eb5ba08933"
ADAPTER_CONFIG_SHA = "52e05adc5f8595942879747f1cb4df8f45c06f547005bf7dc3094e6233723e92"
CHILD_WALL_SECONDS = 1800
GENERATION_HARD_SECONDS = 60
RESERVATION_USD = 6
MAX_PROGRESS_BYTES = 8 * 1024 * 1024

app = modal.App("bu-weight-space-vllm-pilot-v4")
image = (modal.Image.debian_slim(python_version="3.11")
    .run_commands("timeout --signal=TERM --kill-after=10s 900s python -m pip install --no-cache-dir "
                  "vllm==0.18.0 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 "
                  "transformers==4.57.6 safetensors==0.7.0 pillow==12.2.0 playwright==1.60.0")
    .run_commands("timeout --signal=TERM --kill-after=10s 600s python -m playwright install --with-deps chromium")
    .env({"PYTHONPATH": "/opt/weight-space/src:/opt/weight-space/source", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          "WEIGHT_SPACE_PACKAGED_RUNTIME": "1",
          "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}))
# The exact packaged entry module is also imported during CPU admission. Remote
# imports must not look for the host checkout or construct local source mounts.
# modal.is_local() is True in spawned children, so use an explicit image marker.
if os.environ.get("WEIGHT_SPACE_PACKAGED_RUNTIME") != "1":
    image = (image.add_local_dir(ROOT / "src", "/opt/weight-space/src", copy=True)
             .add_local_file(ROOT / "modal_vllm_pilot_v4.py", "/opt/weight-space/source/modal_vllm_pilot_v4.py", copy=True))
outputs = modal.Volume.from_name("bu-weight-space-results-v3", create_if_missing=True, version=2)
volumes = {
    "/base": modal.Volume.from_name("hf-cache").with_mount_options(read_only=True),
    "/adapter": modal.Volume.from_name("bu-lora").with_mount_options(read_only=True),
    "/merged": modal.Volume.from_name("bu-merged").with_mount_options(read_only=True),
    "/results": outputs,
}


def source_manifest(package_root: Path, entrypoint: Path) -> dict[str, str]:
    """Pin the whole local package source closure, including package initializers."""
    paths = {"modal_vllm_pilot_v4.py": entrypoint}
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
    if "src/weight_space_lab/vllm_pilot_v4.py" not in paths or len(paths) > 64:
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
    from weight_space_lab.process_entry_v4 import worker_entry

    if kind not in {"imports", "pilot"}:
        raise ValueError("unknown supervisor kind")
    run_id = "vllm-imports-v4" if kind == "imports" else RUN_ID
    wall_limit = 300 if kind == "imports" else CHILD_WALL_SECONDS

    # No torch/vLLM import or CUDA initialization in this process.
    actual_manifest = source_manifest(Path(diagnostic_source.__file__).parent,
                                     Path("/opt/weight-space/source/modal_vllm_pilot_v4.py"))
    if actual_manifest != admission["source_manifest"] or admission["run_id"] != run_id:
        return {"schema": 1, "run": run_id, "status": "source_admission_failed", "model_calls": 0,
                "container_termination_required": False, "actual_source_manifest": actual_manifest}
    if kind == "pilot":
        cpu = admission.get("imports_receipt", {})
        child_receipt = cpu.get("child", {})
        if (cpu.get("status") != "completed" or not cpu.get("child_reaped") or cpu.get("child_exitcode") != 0
                or child_receipt.get("status") != "imports_passed" or child_receipt.get("cuda_initialized") is not False
                or child_receipt.get("model_calls") != 0 or child_receipt.get("tensor_payloads_loaded") != 0
                or cpu.get("source_manifest") != actual_manifest or child_receipt.get("source_manifest") != actual_manifest):
            return {"schema": 1, "run": run_id, "status": "cpu_import_admission_failed", "model_calls": 0,
                    "container_termination_required": False}
    run_dir = Path("/results") / run_id
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "started.json").open("x") as stream:
        json.dump({"unix": time.time(), "source_manifest": actual_manifest}, stream)
    outputs.commit()
    spool = tempfile.TemporaryDirectory(prefix="bu-vllm-pilot-v4-")
    progress_path = Path(spool.name) / "child-progress.json"
    child = multiprocessing.get_context("spawn").Process(target=worker_entry,
        args=(kind, admission, str(progress_path)), daemon=False)
    started = time.monotonic()
    report = {"schema": 1, "run": run_id, "status": "supervising", "child": None,
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
            elif stage == "generation" and now - stage_start >= GENERATION_HARD_SECONDS:
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
        report["direct_cuda_owner_exit_confirmed"] = bool(kind == "pilot" and report.get("child_started") and report.get("child_reaped"))
        report["descendant_disposition"] = "No independent descendant inventory; direct child reap does not prove descendant stop"
        report["container_termination_required"] = report["status"] != "completed"
        report["elapsed_s"] = time.monotonic() - started
        if report.get("child") is not None:
            atomic_json(run_dir / "child-progress.json", report["child"])
        atomic_json(run_dir / "report.json", report)
        outputs.commit()
        spool.cleanup()
    return report


@app.function(image=image, cpu=4, memory=(4096, 16384), volumes=volumes,
              timeout=600, startup_timeout=600, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, single_use_containers=True, block_network=True)
def verify_runtime_imports(admission):
    return _supervise_worker(admission, "imports")


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=(8192, 98304), volumes=volumes,
              timeout=2100, startup_timeout=600, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, single_use_containers=True, block_network=True)
def supervise(admission):
    return _supervise_worker(admission, "pilot")


@app.local_entrypoint()
def main(phase: str = "imports"):
    """Once-only launch boundary; exact source requires separate review/admission."""
    runs = ROOT / "runs"
    if phase not in {"imports", "pilot"}:
        raise ValueError("phase must be imports or pilot")
    run_id = "vllm-imports-v4" if phase == "imports" else RUN_ID
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
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    reconciliation = reconcile_fixture_admission(fixtures["fixture_manifest"])
    manifest = source_manifest(ROOT / "src/weight_space_lab", ROOT / "modal_vllm_pilot_v4.py")
    admission = {"run_id": run_id, "revision": REVISION, "fixtures": fixtures,
                 "runner_reconciliation": reconciliation, "source_manifest": manifest,
                 "prior_source": prior_source, "prior_report_sha256": hashlib.sha256(prior_bytes).hexdigest(),
                 "merged_metadata_pins": {"config.json": CONFIG_SHA, "model.safetensors.index.json": INDEX_SHA},
                 "adapter_pins": {"adapter_model.safetensors": ADAPTER_SHA, "adapter_config.json": ADAPTER_CONFIG_SHA},
                 "limits": {"child_wall_seconds": CHILD_WALL_SECONDS, "generation_hard_seconds": GENERATION_HARD_SECONDS}}
    if phase == "pilot":
        imports_bytes = (runs / "vllm-imports-v4.json").read_bytes()
        cpu = json.loads(imports_bytes)
        if (cpu.get("status") != "completed" or cpu.get("child_status") != "imports_passed"
                or cpu.get("source_manifest") != manifest):
            raise ValueError("GPU pilot requires successful CPU import receipt for exactly this source")
        admission["imports_receipt"] = cpu
        admission["imports_receipt_sha256"] = hashlib.sha256(imports_bytes).hexdigest()
    with (runs / (run_id + ".source-manifest.json")).open("x") as stream:
        json.dump(manifest, stream, indent=2)
    with marker.open("x") as stream:
        json.dump({"run": run_id, "unix": time.time(), "app_id": app.app_id,
                   "shared_imports_and_pilot_budget_envelope_usd": RESERVATION_USD,
                   "reservation_usd": RESERVATION_USD if phase == "imports" else 0,
                   "reservation_reference": "vllm-imports-v4.started.json", "source_manifest": manifest}, stream, indent=2)
    call = None
    try:
        # A returned handle remains under cancellation coverage even if saving
        # its receipt fails. A failed spawn without a handle is never retried.
        call = (verify_runtime_imports if phase == "imports" else supervise).spawn(admission)
        with (runs / (run_id + ".call.json")).open("x") as stream:
            json.dump({"function_call_id": call.object_id, "app_id": app.app_id, "unix": time.time()}, stream)
        result = call.get(timeout=1500 if phase == "imports" else 3000)
        with (runs / (run_id + ".json")).open("x") as stream:
            json.dump(result, stream, indent=2)
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
