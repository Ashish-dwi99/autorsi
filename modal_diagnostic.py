"""SOURCE-ONLY until separately reviewed: one supervised baseline vision call.

Imports reuse the pinned pilot image and read-only original mounts. No function
is invoked on import. This entry point has a distinct once marker and consumes
only diagnostic-v2; it never resumes pilot-v1 or publishes model weights.
"""
import hashlib
import json
from pathlib import Path
import time

import modal
from modal_pilot import ROOT, REVISION, CONFIG_SHA, INDEX_SHA, image, originals, outputs


app = modal.App("bu-weight-space-diagnostic-v2")
SOURCE_FILES = ("modal_diagnostic.py", "modal_pilot.py", "src/weight_space_lab/diagnostic.py",
                "src/weight_space_lab/__init__.py", "src/weight_space_lab/eval/__init__.py",
                "src/weight_space_lab/policy_io.py", "src/weight_space_lab/patch.py",
                "src/weight_space_lab/eval/runner.py", "src/weight_space_lab/eval/fixtures.py")
diagnostic_image = (image
    .add_local_file(ROOT / "modal_diagnostic.py", "/opt/weight-space/source/modal_diagnostic.py", copy=True)
    .add_local_file(ROOT / "modal_pilot.py", "/opt/weight-space/source/modal_pilot.py", copy=True))


def local_manifest():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


@app.function(image=diagnostic_image, gpu="A100-80GB", cpu=(2, 4), memory=(8192, 98304),
              volumes=originals, timeout=900, startup_timeout=600, retries=0,
              max_containers=1, min_containers=0, buffer_containers=0,
              single_use_containers=True, block_network=True)
def diagnose(admission):
    # The supervising process never imports torch or initializes CUDA.
    import multiprocessing
    import os
    import signal
    import sys
    import tempfile
    import weight_space_lab.diagnostic as diagnostic_source
    from weight_space_lab.diagnostic import atomic_json, run_cuda_diagnostic

    if "torch" in sys.modules:
        raise RuntimeError("supervisor unexpectedly imported torch before spawn")
    package_root = Path(diagnostic_source.__file__).parent
    module_sources = {name: Path("/opt/weight-space/source") / name for name in ("modal_diagnostic.py", "modal_pilot.py")}
    for relative, expected in admission["source_manifest"].items():
        if relative not in SOURCE_FILES:
            raise ValueError("unexpected source manifest entry")
        path = package_root / relative.removeprefix("src/weight_space_lab/") if relative.startswith("src/") else module_sources[relative]
        if path.suffix != ".py":
            raise ValueError("readable exact Python module source required for manifest")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("source manifest changed: " + relative)
    if set(admission["source_manifest"]) != set(SOURCE_FILES):
        raise ValueError("incomplete source manifest")
    run_dir = Path("/results/diagnostic-v2")
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "started.json").open("x") as stream:
        json.dump({"unix": time.time(), "source_manifest": admission["source_manifest"]}, stream)
    outputs.commit()
    # Keep watchdog progress on local temporary storage. No provider RPC or
    # volume commit is made while an inference child might need termination.
    temporary_dir = tempfile.TemporaryDirectory(prefix="bu-diagnostic-v2-")
    progress_path = Path(temporary_dir.name) / "child-progress.json"
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=run_cuda_diagnostic, args=(admission, str(progress_path)), daemon=False)
    started = time.monotonic()
    report = {"schema": 1, "run": "diagnostic-v2", "status": "supervising",
              "source_manifest": admission["source_manifest"], "child_wall_limit_s": 600,
              "generation_hard_limit_s": 150, "lifecycle": [], "child": None,
              "provider_retry_limit_is_source_only": True,
              "physical_gpu_release_and_billing_stop": "Requires provider terminal/billing evidence"}
    last_mtime = None
    deadline_reason = None

    def read_progress():
        nonlocal last_mtime
        if progress_path.exists():
            stat = progress_path.stat()
            if stat.st_size > 1024 * 1024:
                raise ValueError("child progress exceeded 1 MiB")
            if stat.st_mtime_ns != last_mtime:
                report["child"] = json.loads(progress_path.read_text())
                last_mtime = stat.st_mtime_ns

    def stop_and_reap():
        if child.pid is None:
            return
        if child.is_alive():
            report["lifecycle"].append({"event": "terminate_requested", "unix": time.time()})
            try:
                if os.getpgid(child.pid) == child.pid:
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
            except ProcessLookupError:
                report["lifecycle"].append({"event": "exited_during_terminate_request", "unix": time.time()})
            child.join(timeout=5)
        if child.is_alive():
            report["lifecycle"].append({"event": "kill_requested", "unix": time.time()})
            try:
                if os.getpgid(child.pid) == child.pid:
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
            except ProcessLookupError:
                report["lifecycle"].append({"event": "exited_during_kill_request", "unix": time.time()})
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
            if now - started >= 600:
                deadline_reason = "child_wall_deadline"
            elif stage == "generation" and now - stage_start >= 150:
                deadline_reason = "native_generation_hard_deadline"
            elif stage in {"completed", "failed"} and now - stage_start >= 20:
                deadline_reason = "child_exit_grace_deadline"
            if deadline_reason:
                report["status"] = "aborted"
                report["abort_reason"] = deadline_reason
                break
            child.join(timeout=0.2)
    except BaseException as exc:
        report["status"] = "aborted"
        report["supervisor_error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
    finally:
        try:
            stop_and_reap()
        except BaseException as exc:
            report["reap_error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
            report["child_reaped"] = False
        try:
            read_progress()
        except BaseException as exc:
            report["progress_error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        if report["status"] == "supervising":
            report["status"] = "completed" if (report.get("child_reaped") and report.get("child_exitcode") == 0
                and (report.get("child") or {}).get("status") == "diagnostic_returned") else "failed"
        if not report.get("child_reaped", False):
            report["status"] = "child_not_reaped"
        report["direct_cuda_owner_exit_confirmed"] = report.get("child_reaped", False)
        report["browser_close_returned"] = bool((report.get("child") or {}).get("browser_close_returned", False))
        report["descendant_disposition"] = "No independent descendant process inventory; child reap does not prove descendant stop"
        report["container_termination_required"] = report["status"] != "completed" or not report["browser_close_returned"]
        report["elapsed_s"] = time.monotonic() - started
        if report.get("child") is not None:
            atomic_json(run_dir / "child-progress.json", report["child"])
        atomic_json(run_dir / "report.json", report)
        outputs.commit()
        # Only regenerable local diagnostic spool data is removed here; the
        # saved progress/report remain in the separate result volume.
        temporary_dir.cleanup()
    return report


@app.local_entrypoint()
def main():
    # This is the only launch boundary. A separate review must admit this exact source.
    runs = ROOT / "runs"
    marker = runs / "diagnostic-v2.started.json"
    if marker.exists():
        raise RuntimeError("diagnostic-v2 is consumed; no repeat invocation")
    previous_path = runs / "pilot-v1-remote-report.json"
    previous_bytes = previous_path.read_bytes()
    previous = json.loads(previous_bytes)
    prior_source = previous.get("source", {})
    shards = prior_source.get("full_merged_shard_hashes", {})
    if previous.get("status") != "aborted" or len(shards) != 13:
        raise ValueError("complete prior source receipt required; no shard audit replay")
    fixtures = json.loads((runs / "fixtures-v1.json").read_text())
    if len(fixtures.get("records", [])) != 16 or not all(r["success"] for r in fixtures["records"]):
        raise ValueError("prior CPU fixture admission is incomplete")
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    reconciliation = reconcile_fixture_admission(fixtures["fixture_manifest"])
    manifest = local_manifest()
    admission = {"revision": REVISION, "fixtures": fixtures, "source_manifest": manifest,
                 "runner_reconciliation": reconciliation,
                 "prior_source": prior_source,
                 "prior_report_sha256": hashlib.sha256(previous_bytes).hexdigest(),
                 "merged_metadata_pins": {"config.json": CONFIG_SHA, "model.safetensors.index.json": INDEX_SHA}}
    with (runs / "diagnostic-v2.source-manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    with marker.open("x") as stream:
        json.dump({"run": "diagnostic-v2", "unix": time.time(), "app_id": app.app_id,
                   "reservation_usd": 3, "source_manifest": manifest}, stream, indent=2)
    call = diagnose.spawn(admission)
    with (runs / "diagnostic-v2.call.json").open("x") as stream:
        json.dump({"function_call_id": call.object_id, "app_id": app.app_id, "unix": time.time()}, stream)
    try:
        result = call.get(timeout=1600)
        with (runs / "diagnostic-v2.json").open("x") as stream:
            json.dump(result, stream, indent=2)
        if result.get("container_termination_required", True):
            raise RuntimeError("diagnostic incomplete or child/descendant disposition uncertain; request container termination")
    except BaseException:
        try:
            call.cancel(terminate_containers=True)
            cancellation = {"request_returned": True}
        except BaseException as exc:
            cancellation = {"request_returned": False, "error": str(exc)[:1000]}
        cancellation.update({"unix": time.time(), "physical_stop_or_billing_stop": "Not proved by cancellation ACK"})
        with (runs / "diagnostic-v2.cancel.json").open("x") as stream:
            json.dump(cancellation, stream, indent=2)
        raise
    print(json.dumps({"saved": str(runs / "diagnostic-v2.json"), "status": result["status"]}))
