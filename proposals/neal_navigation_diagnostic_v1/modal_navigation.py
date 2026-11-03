"""Separate private CPU-only diagnostic; no model mounts, GPU or HTTP route."""
import json
import os
from pathlib import Path
import time
import modal
from nav_common import *

PACKAGED = os.environ.get("NEAL_NAV_PACKAGED") == "1"
app = modal.App(APP)
# These dependency layers exactly reuse the v2 image declaration. Extra model
# packages are present in cached layers but never imported by this diagnostic.
image = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .run_commands("timeout --signal=TERM --kill-after=10s 900s python -m pip install --no-cache-dir "
        "vllm==0.18.0 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 "
        "transformers==4.57.6 safetensors==0.7.0 pillow==12.2.0 playwright==1.60.0 modal==1.4.3")
    .run_commands("timeout --signal=TERM --kill-after=10s 600s python -m playwright install --with-deps chromium")
    .env({"PYTHONPATH": "/opt/weight-space/navigation", "NEAL_NAV_PACKAGED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}))
if not PACKAGED:
    image = image.add_local_dir(Path(__file__).parent, "/opt/weight-space/navigation", copy=True,
        ignore=["__pycache__", "*.pyc"])
outputs = modal.Volume.from_name("bu-weight-space-results-v3", create_if_missing=False, version=2)
once = modal.Dict.from_name(ONCE, create_if_missing=True)


def executing_bindings(actual):
    import nav_common
    import nav_worker
    paths = {SOURCE_PREFIX + "modal_navigation.py": Path(__file__),
        SOURCE_PREFIX + "nav_common.py": Path(nav_common.__file__),
        SOURCE_PREFIX + "nav_worker.py": Path(nav_worker.__file__)}
    result = {}
    for key, path in paths.items():
        digest = sha(path)
        if digest != actual.get(key):
            raise ValueError("actual imported/entry bytes differ: " + key)
        result[key] = {"actual_file_path": str(path), "sha256": digest}
    return result


def supervise(admission):
    import multiprocessing
    import shutil
    import signal
    import tempfile
    from nav_worker import worker_entry
    actual = manifest(Path("/opt/weight-space/navigation"))
    bindings = executing_bindings(actual)
    validate_admission(admission, actual)
    if not once.put(RUN, {"unix": time.time(), "manifest_sha256": canonical_sha(actual)}, skip_if_exists=True):
        return {"status": "refused_consumed", "container_termination_required": False}
    destination = Path("/results") / RUN
    destination.mkdir(exist_ok=False)
    atomic_json(destination / "entered.json", {"source_manifest": actual, "unix": time.time()})
    outputs.commit()
    spool = Path(tempfile.mkdtemp(prefix="neal-nav-owned-"))
    progress, artifacts = spool / "progress.json", spool / "artifacts"
    child = multiprocessing.get_context("spawn").Process(target=worker_entry,
        args=(admission, str(progress), str(artifacts)), daemon=False)
    report = {"schema": 1, "run": RUN, "status": "supervising", "source_manifest": actual,
        "executed_source_bindings": bindings, "function_call_id": modal.current_function_call_id(),
        "gpu_requested": False, "model_mounts": [], "model_calls": 0, "game_actions": 0,
        "lifecycle": [], "artifact_inventory": [], "not_a_preflight_or_gpu_admission": True}
    started = time.monotonic()
    def read_progress():
        if progress.exists():
            if progress.stat().st_size > MAX_REPORT:
                raise ValueError("progress exceeded2MiB")
            report["child"] = json.loads(progress.read_bytes())
    try:
        child.start()
        report["lifecycle"].append({"event": "spawned", "pid": child.pid})
        while child.is_alive():
            read_progress()
            inventory(artifacts, hash_files=False)
            partial = report.get("child", {})
            now = time.monotonic()
            if now - started >= CHILD_SECONDS:
                raise TimeoutError("original120s child wall")
            if (partial.get("stage") == "navigation_observation"
                    and now - partial["stage_started_monotonic"] >= OBSERVATION_SECONDS):
                raise TimeoutError("original30s observation watchdog; partial video may remain unfinalized")
            if (partial.get("stage") == "finalizing"
                    and now - partial["stage_started_monotonic"] >= 45):
                raise TimeoutError("original finalization watchdog")
            if (partial.get("stage") in ("completed", "failed")
                    and now - partial["stage_started_monotonic"] >= 20):
                raise TimeoutError("terminal child did not exit")
            child.join(0.2)
    except BaseException as exc:
        report.update(status="aborted", error={"type": type(exc).__name__, "message": str(exc)[:2000]})
    finally:
        if child.pid is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not child.is_alive():
                    break
                report["lifecycle"].append({"event": "signal_requested", "signal": int(sig)})
                try:
                    if os.getpgid(child.pid) == child.pid:
                        os.killpg(child.pid, sig)
                    elif sig == signal.SIGTERM:
                        child.terminate()
                    else:
                        child.kill()
                except ProcessLookupError:
                    pass
                child.join(5)
            if not child.is_alive():
                child.join(1)
        report.update(child_exitcode=child.exitcode,
            child_reaped=child.pid is not None and child.exitcode is not None and not child.is_alive())
        try:
            read_progress()
        except BaseException as exc:
            report["progress_error"] = {"type": type(exc).__name__, "message": str(exc)[:512]}
        partial = report.get("child", {})
        closed = partial.get("browser_close_returned") is True
        if report["status"] == "supervising":
            report["status"] = "diagnostic_returned" if report["child_reaped"] and child.exitcode == 0 and closed else "failed"
        report["container_termination_required"] = not (report["child_reaped"] and child.exitcode == 0 and closed)
        report["physical_disposition"] = "Direct child reap/close returns do not prove independent descendant inventory or billing termination"
        if report["child_reaped"] and closed:
            try:
                expected = inventory(artifacts)
                shutil.copytree(artifacts, destination / "artifacts")
                if inventory(destination / "artifacts") != expected:
                    raise ValueError("artifact copy/readback differs")
                report["artifact_inventory"] = expected
                report["artifacts_finalized"] = True
            except BaseException as exc:
                report.update(status="artifact_preservation_failed", container_termination_required=True,
                    artifact_error={"type": type(exc).__name__, "message": str(exc)[:512]})
        else:
            report["partial_spool_preserved"] = str(spool)
        report["elapsed_s"] = time.monotonic() - started
        atomic_json(destination / "report.json", report)
        outputs.commit()
        if report.get("artifacts_finalized"):
            shutil.rmtree(spool)
    return report


@app.function(image=image, cpu=(4, 4), memory=(4096, 16384), volumes={"/results": outputs},
    timeout=180, startup_timeout=240, retries=0, min_containers=0, max_containers=1,
    buffer_containers=0, single_use_containers=True)
def diagnose_navigation(admission):
    return supervise(admission)
