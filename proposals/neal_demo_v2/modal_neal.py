"""UNREVIEWED source-only private Neal lifecycle proposal. No HTTP endpoint.

Remote execution remains refused without a source-bound budget allocation and,
for GPU work, this exact source's successful CPU/browser/video receipt. Local
JSON records do not issue permission. This file neither releases reservations
nor calls deploy. Review and actual account admission precede any provider call.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time

import modal
from neal_limits import *
from neal_lifecycle import source_manifest, manifest_sha, validate_admission, artifact_inventory

PACKAGED_RUNTIME = os.environ.get("NEAL_PACKAGED_RUNTIME") == "1"
# Modal may auto-mount this module at /root/modal_neal.py. Never derive a
# repository ancestor from that shallow runtime path. Explicit image paths
# own all packaged imports; local ancestor lookup is evaluated only locally.
ROOT = Path("/opt/weight-space") if PACKAGED_RUNTIME else Path(__file__).resolve().parents[2]
app = modal.App(APP_NAME)
image = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .run_commands("timeout --signal=TERM --kill-after=10s 900s python -m pip install --no-cache-dir "
        "vllm==0.18.0 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 "
        "transformers==4.57.6 safetensors==0.7.0 pillow==12.2.0 playwright==1.60.0 modal==1.4.3")
    .run_commands("timeout --signal=TERM --kill-after=10s 600s python -m playwright install --with-deps chromium")
    .env({"PYTHONPATH": "/opt/weight-space/src:/opt/weight-space/neal", "NEAL_PACKAGED_RUNTIME": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}))
if not PACKAGED_RUNTIME:
    image = (image.add_local_dir(ROOT / "src", "/opt/weight-space/src", copy=True,
                                ignore=["__pycache__", "*.pyc"])
                  .add_local_dir(Path(__file__).parent, "/opt/weight-space/neal", copy=True,
                                 ignore=["__pycache__", "*.pyc"]))
outputs = modal.Volume.from_name("bu-weight-space-results-v3", create_if_missing=False, version=2)
once = modal.Dict.from_name(ONCE_NAME, create_if_missing=True)
bridge = modal.Dict.from_name(BRIDGE_NAME, create_if_missing=True)
volumes = {
    "/base": modal.Volume.from_name("hf-cache").with_mount_options(read_only=True),
    "/adapter": modal.Volume.from_name("bu-lora").with_mount_options(read_only=True),
    "/merged": modal.Volume.from_name("bu-merged").with_mount_options(read_only=True),
    "/results": outputs}


def executed_source_bindings(actual):
    """Join the SDK entry copy and imported helpers to explicit image bytes."""
    import neal_limits
    import neal_lifecycle
    import neal_worker
    import weight_space_lab.diagnostic as diagnostic
    paths = {
        "proposals/neal_demo_v2/modal_neal.py": Path(__file__),
        "proposals/neal_demo_v2/neal_limits.py": Path(neal_limits.__file__),
        "proposals/neal_demo_v2/neal_lifecycle.py": Path(neal_lifecycle.__file__),
        "proposals/neal_demo_v2/neal_worker.py": Path(neal_worker.__file__),
        "src/weight_space_lab/diagnostic.py": Path(diagnostic.__file__),
    }
    bindings = {}
    for key, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != actual.get(key):
            raise ValueError("executing or imported source differs from image manifest: " + key)
        bindings[key] = {"actual_file_path": str(path), "sha256": digest,
            "scope": "Imported source-file bytes joined; not a compiler/runtime/native attestation"}
    return bindings


def supervise(admission, kind):
    import multiprocessing
    import signal
    import tempfile
    import shutil
    import weight_space_lab.diagnostic as diagnostic
    import neal_worker
    from neal_worker import worker_entry

    actual = source_manifest(Path("/opt/weight-space/src/weight_space_lab"), Path("/opt/weight-space/neal"))
    bindings = executed_source_bindings(actual)
    validate_admission(admission, kind, actual)
    if not once.put(kind + "-entered", {"unix": time.time(), "source_manifest": actual}, skip_if_exists=True):
        return {"status": "refused_consumed_entry", "container_termination_required": False}
    registration = {"session_id": SESSION_ID, "source_manifest_sha256": manifest_sha(actual),
                    "expires_unix": NOT_AFTER_UNIX}
    if not bridge.put(SESSION_ID + ":registration", registration, skip_if_exists=True):
        if bridge.get(SESSION_ID + ":registration") != registration:
            raise ValueError("private session registration was already bound to other source")
    run_id = CPU_RUN_ID if kind == "preflight" else GPU_RUN_ID
    run_dir = Path("/results") / run_id
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "started.json").open("x") as stream:
        json.dump({"unix": time.time(), "source_manifest": actual, "kind": kind}, stream)
    outputs.commit()
    spool = Path(tempfile.mkdtemp(prefix="adam-neal-owned-"))
    artifact_dir = spool / "artifacts"
    artifact_dir.mkdir()
    progress = spool / "child-progress.json"
    child = multiprocessing.get_context("spawn").Process(target=worker_entry,
        args=(kind, admission, str(progress), str(artifact_dir)), daemon=False)
    started = time.monotonic()
    wall = CPU_CHILD_WALL_SECONDS if kind == "preflight" else CHILD_WALL_SECONDS
    report = {"schema": 1, "run": run_id, "kind": kind, "status": "supervising", "child": None,
        "function_call_id": modal.current_function_call_id(), "declared_app_name": APP_NAME,
        "source_manifest": actual, "executed_source_bindings": bindings, "lifecycle": [], "artifact_inventory": [],
        "limits": {"child_wall_seconds": wall, "generation_hard_seconds": GENERATION_HARD_SECONDS,
            "finalize_hard_seconds": FINALIZE_HARD_SECONDS, "artifact_bytes": MAX_ARTIFACT_BYTES},
        "private_sdk_only": True, "public_endpoint": False, "original_volumes_read_only": True,
        "provider_retry_limit_is_source_only": True, "new_model_export": False}
    previous_mtime = None

    def read_progress():
        nonlocal previous_mtime
        if progress.exists():
            stat = progress.stat()
            if stat.st_size > MAX_PROGRESS_BYTES:
                raise ValueError("child progress exceeds8MiB")
            if stat.st_mtime_ns != previous_mtime:
                report["child"] = json.loads(progress.read_text())
                if not isinstance(report["child"], dict):
                    raise ValueError("child progress must be an object")
                previous_mtime = stat.st_mtime_ns

    def owned_signal(sig):
        try:
            if os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, sig)
            elif sig == signal.SIGTERM:
                child.terminate()
            else:
                child.kill()
        except ProcessLookupError:
            report["lifecycle"].append({"event": "exited_during_signal", "signal": int(sig)})

    try:
        child.start()
        report["lifecycle"].append({"event": "spawned", "pid": child.pid, "unix": time.time()})
        while child.is_alive():
            read_progress()
            artifact_inventory(artifact_dir)
            now = time.monotonic()
            partial = report.get("child") or {}
            stage, phase_start = partial.get("stage"), partial.get("stage_started_monotonic", started)
            if type(phase_start) not in (int, float) or not math.isfinite(phase_start):
                raise ValueError("invalid source phase timestamp")
            reason = None
            if now - started >= wall:
                reason = "original_child_wall_deadline"
            elif stage == "generation" and now - phase_start >= GENERATION_HARD_SECONDS:
                reason = "native_generation_disposition_unknown"
            elif stage == "waiting_assistant_ready" and now - phase_start >= ASSISTANT_READY_SECONDS + 5:
                reason = "assistant_not_ready"
            elif stage == "finalizing" and now - phase_start >= FINALIZE_HARD_SECONDS:
                reason = "video_finalization_deadline"
            elif stage in {"completed", "failed"} and now - phase_start >= 20:
                reason = "terminal_child_exit_deadline"
            if reason:
                report.update(status="aborted", abort_reason=reason)
                break
            # No SDK RPC/volume commit runs in the live-child watchdog loop.
            child.join(timeout=0.2)
    except BaseException as exc:
        report.update(status="aborted", supervisor_error={"type": type(exc).__name__, "message": str(exc)[:2000]})
    finally:
        if child.pid is not None:
            try:
                if child.is_alive():
                    report["lifecycle"].append({"event": "terminate_requested", "unix": time.time()})
                    owned_signal(signal.SIGTERM)
                    child.join(timeout=5)
                if child.is_alive():
                    report["lifecycle"].append({"event": "kill_requested", "unix": time.time()})
                    owned_signal(signal.SIGKILL)
                    child.join(timeout=5)
                else:
                    child.join(timeout=1)
            except BaseException as exc:
                report["reap_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        report["child_exitcode"] = child.exitcode
        report["child_reaped"] = child.pid is not None and child.exitcode is not None and not child.is_alive()
        try:
            read_progress()
        except BaseException as exc:
            report["progress_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        partial = report.get("child") or {}
        if report["status"] == "supervising":
            report["status"] = "completed" if (report["child_reaped"] and child.exitcode == 0
                and partial.get("status") == "completed" and partial.get("stage") == "completed"
                and partial.get("browser_close_returned") is True and partial.get("native_pending") is False) else "failed"
        report["container_termination_required"] = report["status"] != "completed"
        report["direct_cuda_owner_exit_confirmed"] = bool(kind == "demo" and report["child_reaped"])
        report["descendant_disposition"] = "Direct child reap is not an independent browser/SDK descendant inventory"
        report["physical_stop_and_billing"] = "Require actual provider terminal and billing observations"
        report["elapsed_s"] = time.monotonic() - started
        try:
            if not report["child_reaped"]:
                raise RuntimeError("live/unknown child; no concurrent artifact copy or cleanup")
            inventory = artifact_inventory(artifact_dir, hash_files=True)
            destination = run_dir / "artifacts"
            destination.mkdir(exist_ok=False)
            for item in inventory:
                source = artifact_dir / item["relative_path"]
                target = destination / item["relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if artifact_inventory(destination, hash_files=True) != inventory:
                raise ValueError("artifact copy readback identity mismatch")
            report["artifact_inventory"] = inventory
            report["artifacts_finalized"] = partial.get("browser_close_returned") is True
        except BaseException as exc:
            report["artifact_preservation_error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            report["artifacts_finalized"] = False
            report["status"] = "artifact_preservation_failed"
            report["container_termination_required"] = True
        if report.get("child") is not None:
            diagnostic.atomic_json(run_dir / "child-progress.json", report["child"])
        diagnostic.atomic_json(run_dir / "report.json", report)
        outputs.commit()
        if (report.get("child_reaped") and partial.get("browser_close_returned") is True
                and "artifact_preservation_error" not in report):
            shutil.rmtree(spool)
    return report


@app.function(image=image, cpu=(4, 4), memory=(4096, 16384), volumes=volumes,
    timeout=240, startup_timeout=240, retries=0, max_containers=1, min_containers=0,
    buffer_containers=0, single_use_containers=True)
def cpu_preflight(admission):
    return supervise(admission, "preflight")


@app.function(image=image, gpu="A100-80GB", cpu=(4, 4), memory=(8192, 98304), volumes=volumes,
    timeout=960, startup_timeout=240, retries=0, max_containers=1, min_containers=0,
    buffer_containers=0, scaledown_window=2, single_use_containers=True)
def live_demo(admission):
    return supervise(admission, "demo")
