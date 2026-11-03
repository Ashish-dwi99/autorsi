"""Existing private SDK identity; separate once-only launch and exact recovery."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from nav_common import *

ROOT = Path("/opt/weight-space") if os.environ.get("NEAL_NAV_PACKAGED") == "1" else Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / RUN


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REPORT:
        raise ValueError("bounded ordinary JSON required")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def invalid(_):
        raise ValueError("nonfinite JSON")
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def save_once(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False).encode()
    if len(payload) > MAX_REPORT:
        raise ValueError("host receipt exceeds2MiB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def preserve_failure(path, value):
    try:
        save_once(path, value)
    except BaseException as exc:
        print(json.dumps({"receipt_error": type(exc).__name__, "original_failure": value}), file=sys.stderr)


def sdk():
    import modal
    if importlib.metadata.version("modal") != "1.4.3":
        raise ValueError("reviewed Modal1.4.3 required")
    for name, digest in SDK_PINS.items():
        if sha(Path(modal.__file__).parent / name) != digest:
            raise ValueError("SDK source differs")
    return modal


def admission_from_saved(allocation_path, terminal_path, deployment_path):
    actual = manifest(Path(__file__).parent)
    allocation, terminal, deployment = read(allocation_path), read(terminal_path), read(deployment_path)
    if allocation.get("ledger_sha256") != sha(ROOT / "evidence/budget_ledger.json"):
        raise ValueError("shared envelope ledger changed")
    if (terminal.get("observation_type") != "actual_provider_terminal_observation"
            or terminal.get("app_id") != PRIOR_APP
            or any(terminal.get(key) != 0 for key in ("active_gpu_calls", "active_gpu_containers", "active_cpu_calls", "active_cpu_containers"))
            or type(terminal.get("observed_unix")) not in (int, float)
            or not 0 <= time.time() - terminal["observed_unix"] <= 300):
        raise ValueError("fresh actual no-overlap terminal evidence required")
    if (deployment.get("app_name") != APP or deployment.get("environment") != "main"
            or deployment.get("source_manifest") != actual
            or not str(deployment.get("app_id", "")).startswith("ap-")
            or set(deployment.get("functions", {})) != {FUNCTION}
            or not str(deployment["functions"][FUNCTION]).startswith("fu-")):
        raise ValueError("exact observed private CPU deployment identity required")
    admission = {"run_id": RUN, "source_manifest": actual, "source_manifest_sha256": canonical_sha(actual),
        "allowed_origins": list(ORIGINS), "budget_allocation": allocation,
        "terminal_receipt_sha256": sha(terminal_path)}
    validate_admission(admission, actual)
    return admission, deployment


def launch(allocation, terminal, deployment_path):
    admission, deployment = admission_from_saved(allocation, terminal, deployment_path)
    modal = sdk()
    save_once(RUNS / "started.json", {"admission": admission, "deployment": deployment, "retry_authorized": False})
    call = None
    entered = False
    stage = "lookup"
    try:
        function = modal.Function.from_name(APP, FUNCTION, environment_name="main")
        stage = "hydrate"
        function.hydrate()
        if function.object_id != deployment["functions"][FUNCTION]:
            raise ValueError("actual hydrated function ID differs")
        save_once(RUNS / "lookup.json", {"function_id": function.object_id, "app_id": deployment["app_id"]})
        entered, stage = True, "spawn_entered"
        call = function.spawn(admission)
        stage = "handle_returned"
        save_once(RUNS / "call.json", {"function_call_id": call.object_id, "function_id": function.object_id,
            "app_id": deployment["app_id"], "source_manifest": admission["source_manifest"]})
        result = call.get(timeout=480)
        save_once(RUNS / "remote-report.json", result)
        if (result.get("function_call_id") != call.object_id or result.get("run") != RUN
                or result.get("source_manifest") != admission["source_manifest"]):
            raise RuntimeError("actual result identity differs")
        if result.get("container_termination_required", True):
            raise RuntimeError("supervisor requires container termination")
        return {"report": str(RUNS / "remote-report.json"), "status": result["status"],
            "not_gpu_admission": True, "provider_terminal_evidence_still_required": True}
    except BaseException as exc:
        failure = {"stage": stage, "error": type(exc).__name__, "message": str(exc)[:1500],
            "submission_disposition": "pre_submit" if not entered else "entered_unknown" if call is None else "returned_handle",
            "retry_authorized": False}
        if call is not None:
            cancel_entered = False
            try:
                signal.setitimer(signal.ITIMER_REAL, 20)
                cancel_entered = True
                call.cancel(terminate_containers=True)
                failure["cancellation"] = {"attempted": True, "request_returned": True, "physical_stop_proved": False}
            except BaseException as cancel_error:
                failure["cancellation"] = {"attempted": cancel_entered, "request_returned": False, "error": type(cancel_error).__name__}
        preserve_failure(RUNS / "failure.json", failure)
        raise


def recover():
    modal = sdk()
    report, call = read(RUNS / "remote-report.json"), read(RUNS / "call.json")
    if (report.get("function_call_id") != call["function_call_id"] or report.get("run") != RUN
            or report.get("source_manifest") != manifest(Path(__file__).parent)):
        raise ValueError("exact saved call/source/run identity required")
    items = report.get("artifact_inventory", [])
    if not isinstance(items, list) or not 1 <= len(items) <= 3:
        raise ValueError("bounded actual artifact inventory required")
    seen, total, videos = set(), 0, 0
    for item in items:
        path = Path(item["relative_path"])
        cap = MAX_VIDEO if path.suffix == ".webm" else MAX_PNG
        videos += int(path.suffix == ".webm")
        if (path.is_absolute() or ".." in path.parts or str(path) in seen or path.suffix not in (".webm", ".png")
                or type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= cap
                or len(item.get("sha256", "")) != 64):
            raise ValueError("invalid artifact path/size/hash")
        seen.add(str(path)); total += item["bytes"]
    if total > MAX_ARTIFACTS or videos > 1:
        raise ValueError("aggregate artifacts exceed limit")
    save_once(RUNS / "recovery-started.json", {"inventory": items, "retry_authorized": False})
    output = RUNS / "artifacts"
    output.mkdir(exist_ok=False)
    volume = modal.Volume.from_name("bu-weight-space-results-v3", create_if_missing=False)
    completed, remaining = [], total
    try:
        for item in items:
            if shutil.disk_usage(output).free < 200 * 1024 * 1024 + remaining:
                raise RuntimeError("fresh200MiB plus remaining aggregate reserve required")
            target = output / item["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with target.open("xb") as stream:
                for block in volume.read_file(RUN + "/artifacts/" + item["relative_path"]):
                    if not isinstance(block, bytes) or written + len(block) > item["bytes"]:
                        raise ValueError("stream type/size mismatch")
                    stream.write(block); written += len(block)
            if written != item["bytes"] or sha(target) != item["sha256"]:
                raise ValueError("artifact bytes/hash mismatch")
            completed.append({**item, "local_path": str(target.resolve())})
            remaining -= written
        result = {"completed": completed, "raw_video_finalized": report.get("artifacts_finalized") is True,
            "transport": "One application read_file per exact file; SDK1.4.3 internally retries/prefetches blocks"}
        save_once(RUNS / "recovery.json", result)
        return result
    except BaseException as exc:
        preserve_failure(RUNS / "recovery-failed.json", {"completed": completed, "error": type(exc).__name__,
            "message": str(exc)[:1000], "partial_files_preserved": True, "retry_authorized": False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "recover"))
    parser.add_argument("--allocation"); parser.add_argument("--terminal"); parser.add_argument("--deployment")
    args = parser.parse_args()
    def deadline(*_):
        raise TimeoutError("bounded host operation expired; no automatic retry")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 500 if args.command == "launch" else 600)
    try:
        result = launch(args.allocation, args.terminal, args.deployment) if args.command == "launch" else recover()
        print(json.dumps(result, allow_nan=False))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
