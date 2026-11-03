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


from dev_contract import (APP_NAME, FILES, IMPORT_RUN, GPU_RUN, CHILD_WALL_SECONDS,
    GENERATION_HARD_SECONDS, NOT_AFTER_UNIX, source_manifest, validate_spec, MAX_ARTIFACT_BYTES)
PACKAGED = os.environ.get("ADAM_DEV_PACKAGED_RUNTIME") == "1"
ROOT = Path("/opt/weight-space") if PACKAGED else Path(__file__).resolve().parents[2]
PROPOSAL_ROOT = Path("/opt/weight-space/adam_dev") if PACKAGED else Path(__file__).resolve().parent
MAX_PROGRESS_BYTES = 8 * 1024 * 1024

app = modal.App(APP_NAME)
image = (modal.Image.debian_slim(python_version="3.11")
    .run_commands("timeout --signal=TERM --kill-after=10s 900s python -m pip install --no-cache-dir "
                  "vllm==0.18.0 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 "
                  "transformers==4.57.6 safetensors==0.7.0 pillow==12.2.0 playwright==1.60.0")
    .run_commands("timeout --signal=TERM --kill-after=10s 600s python -m playwright install --with-deps chromium")
    .run_commands("timeout --signal=TERM --kill-after=10s 300s python -m pip install --no-cache-dir "
                  "gymnasium==1.2.3 selenium==4.40.0 numpy==2.2.6 "
                  "'miniwob @ https://files.pythonhosted.org/packages/fc/a1/a34e0cc33dcba9de00001695979d2b9e3cc2de03bacdf010f045fb084afe/miniwob-1.1.0-py3-none-any.whl#sha256=297006e84219d56ad037e843527ed773a4a82509dc43b6031ec4a3ad3e3e61d9'")
    .env({"PYTHONPATH": "/opt/weight-space/src:/opt/weight-space/adam_dev", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          "ADAM_DEV_PACKAGED_RUNTIME": "1",
          "MINIWOB_CHROME_BINARY": "/opt/cft/chrome-linux64/chrome",
          "MINIWOB_CHROMEDRIVER": "/opt/cft/chromedriver-linux64/chromedriver",
          "SE_OFFLINE": "true",
          "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}))
# The exact packaged entry module is also imported during CPU admission. Remote
# imports must not look for the host checkout or construct local source mounts.
# modal.is_local() is True in spawned children, so use an explicit image marker.
if not PACKAGED:
    image = (image.add_local_file(PROPOSAL_ROOT / "build_miniwob_runtime.py", "/opt/build_miniwob_runtime.py", copy=True)
                  .run_commands("timeout --signal=TERM --kill-after=10s 660s python /opt/build_miniwob_runtime.py")
                  .add_local_dir(ROOT / "src", "/opt/weight-space/src", copy=True))
    for name in FILES:
        image = image.add_local_file(PROPOSAL_ROOT / name, "/opt/weight-space/adam_dev/" + name, copy=True)
outputs = modal.Volume.from_name("bu-weight-space-results-v3", create_if_missing=True, version=2)
once = modal.Dict.from_name("adam-1-dev-miniwob-20261103-v1-once", create_if_missing=True)
volumes = {
    "/base": modal.Volume.from_name("hf-cache").with_mount_options(read_only=True),
    "/adapter": modal.Volume.from_name("bu-lora").with_mount_options(read_only=True),
    "/merged": modal.Volume.from_name("bu-merged").with_mount_options(read_only=True),
    "/results": outputs,
}


def _supervise_worker(admission, kind):
    import multiprocessing
    import os
    import signal
    import tempfile
    import weight_space_lab.diagnostic as diagnostic_source
    from weight_space_lab.diagnostic import atomic_json
    from dev_entry import worker_entry
    from dev_contract import executing_source_join

    if kind not in {"imports", "development"}:
        raise ValueError("unknown supervisor kind")
    run_id = IMPORT_RUN if kind == "imports" else GPU_RUN
    wall_limit = 450 if kind == "imports" else CHILD_WALL_SECONDS

    # No torch/vLLM import or CUDA initialization in this process.
    actual_manifest = source_manifest(Path(diagnostic_source.__file__).parent,
                                     PROPOSAL_ROOT)
    executing_join = executing_source_join(actual_manifest, {"modal_dev.py": __file__, "dev_entry.py": __import__("dev_entry").__file__})
    validate_spec(admission)
    if actual_manifest != admission["source_manifest"] or admission["run_id"] != run_id:
        return {"schema": 1, "run": run_id, "status": "source_admission_failed", "model_calls": 0,
                "container_termination_required": False, "actual_source_manifest": actual_manifest}
    if kind == "development":
        cpu = admission.get("imports_receipt", {})
        child_receipt = cpu.get("child", {})
        if (cpu.get("status") != "completed" or not cpu.get("child_reaped") or cpu.get("child_exitcode") != 0
                or child_receipt.get("status") != "imports_passed" or child_receipt.get("cuda_initialized") is not False
                or child_receipt.get("model_calls") != 0 or child_receipt.get("tensor_payloads_loaded") != 0
                or cpu.get("source_manifest") != actual_manifest or child_receipt.get("source_manifest") != actual_manifest
                or len(child_receipt.get("miniwob_preflight", {}).get("cases", [])) != 12
                or child_receipt.get("experiment_spec") != admission.get("experiment_spec")):
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
    spool = Path(tempfile.mkdtemp(prefix="adam-dev-miniwob-v1-"))
    progress_path = spool / "child-progress.json"
    child = multiprocessing.get_context("spawn").Process(target=worker_entry,
        args=(kind, admission, str(progress_path)), daemon=False)
    started = time.monotonic()
    report = {"schema": 1, "run": run_id, "status": "supervising", "child": None,
              "function_call_id": modal.current_function_call_id(), "declared_app_name": APP_NAME,
              "source_manifest": actual_manifest, "executing_source_join": executing_join, "lifecycle": [],
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
        report["direct_cuda_owner_exit_confirmed"] = bool(kind == "development" and report.get("child_started") and report.get("child_reaped"))
        report["descendant_disposition"] = "No independent descendant inventory; direct child reap does not prove descendant stop"
        report["container_termination_required"] = report["status"] != "completed"
        report["elapsed_s"] = time.monotonic() - started
        if report.get("child") is not None:
            atomic_json(run_dir / "child-progress.json", report["child"])
        atomic_json(run_dir / "report.json", report)
        # Preserve exact bounded source-observation artifacts; never weights.
        import shutil
        artifact_source = spool / "artifacts"
        artifact_receipts = (report.get("child") or {}).get("artifacts", [])
        total = 0
        if len(artifact_receipts) > 288:
            raise ValueError("artifact count exceeds frozen bound")
        if artifact_receipts:
            destination = run_dir / "artifacts"
            destination.mkdir(exist_ok=False)
            for item in artifact_receipts:
                name = item["relative_path"]
                if Path(name).name != name or not name.endswith(".png"):
                    raise ValueError("invalid observation artifact name")
                path = artifact_source / name
                if path.is_symlink() or path.stat().st_size != item["bytes"] or item["bytes"] > 128*1024:
                    raise ValueError("bounded ordinary observation artifact required")
                total += item["bytes"]
                if total > MAX_ARTIFACT_BYTES:
                    raise ValueError("aggregate artifact cap")
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    raise ValueError("artifact hash differs")
                with (destination / name).open("xb") as stream:
                    stream.write(raw)
        report["artifact_preservation"] = {"files": len(artifact_receipts), "bytes": total,
            "kind": "actual pre-action screenshots with elapsed timestamps; not continuous video"}
        atomic_json(run_dir / "report.json", report)
        outputs.commit()
        if report["status"] == "completed":
            shutil.rmtree(spool)
        else:
            report["partial_spool_retained"] = True
    return report


@app.function(image=image, cpu=(4, 4), memory=(4096, 16384), volumes=volumes,
              timeout=600, startup_timeout=240, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, single_use_containers=True, block_network=True)
def verify_runtime_imports(admission):
    return _supervise_worker(admission, "imports")


@app.function(image=image, gpu="A100-80GB", cpu=(4, 4), memory=(8192, 98304), volumes=volumes,
              timeout=1260, startup_timeout=240, retries=0, max_containers=1,
              min_containers=0, buffer_containers=0, scaledown_window=2, single_use_containers=True, block_network=True)
def development(admission):
    """Private SDK function; no HTTP route, secret, or new access permission."""
    return _supervise_worker(admission, "development")


# Deployment is separate. dev_host.py owns reviewed once-only authenticated invocation.
