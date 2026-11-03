"""Source-only private host control: launch, readiness, PNG read, submit, recover.

No operation occurs on import. Commands use existing Modal SDK authentication;
no public endpoint, secret extraction, fallback origin, or automatic retry.
`next` saves the actual actor PNG and prints its absolute path for view_image.
Deployment and a concrete budget record are separate reviewed prerequisites.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from neal_limits import *
from neal_bridge import (IDENTITY_FIELDS, bounded_json, validate_observation,
    validate_action_identity, private_store)

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "neal-demo-v1"
SDK_PINS = {"_functions.py": "aa18b6dcfa00f09807d1719fc592f96fa69d51c492eecf059aecec865702a2bf",
    "_object.py": "b49b4c10fdf18e32dc09effa598e65f523f37c0424f48844297534faf72511c9",
    "functions.py": "11945c1c57e7a186ef6b05099c01c5e6722fa28f465f60e3cbb99498e0979977"}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path, limit=16 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("ordinary bounded JSON file required")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON field")
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError("nonfinite JSON")
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=nonfinite)


def save_once(path, value, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=None if compact else 2,
        separators=(",", ":") if compact else None, allow_nan=False).encode()
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("host receipt exceeds16MiB")
    with path.open("xb") as stream:
        stream.write(payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def preserve_failure(path, value):
    """Receipt-write failure never replaces the original uncertain operation."""
    try:
        save_once(path, value)
    except BaseException as receipt_error:
        print(json.dumps({"failure_receipt_error": type(receipt_error).__name__,
            "original_failure": value}, allow_nan=False), file=sys.stderr)


def check_sdk():
    import modal
    if importlib.metadata.version("modal") != "1.4.3":
        raise ValueError("reviewed SDK1.4.3 required")
    folder = Path(modal.__file__).resolve().parent
    for name, expected in SDK_PINS.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("installed SDK source differs")
    if not all(callable(getattr(modal.Function, name, None)) for name in ("from_name", "hydrate", "spawn")):
        raise ValueError("public SDK interface unavailable")
    return modal


def current_manifest():
    from neal_lifecycle import source_manifest
    return source_manifest(ROOT / "src/weight_space_lab", Path(__file__).parent)


def prepare_admission(phase, allocation_path, prior_terminal_path, deployment_path):
    """Saved-only builder. Its output is not an approval or provider probe."""
    if phase not in {"preflight", "demo"}:
        raise ValueError("unknown phase")
    from neal_worker import MODEL_PINS, REVISION
    source = current_manifest()
    source_sha = canonical_sha(source)
    allocation = read_json(allocation_path)
    if (allocation.get("source_manifest_sha256") != source_sha
            or type(allocation.get("allocated_usd")) not in (int, float)
            or not PROPOSED_RESERVATION_USD <= allocation["allocated_usd"] <= 4
            or type(allocation.get("total_committed_usd")) not in (int, float)
            or not allocation["allocated_usd"] <= allocation["total_committed_usd"] <= 30):
        raise ValueError("concrete allocation within existing30USD ceiling required")
    if allocation.get("ledger_sha256") != hashlib.sha256((ROOT / "evidence/budget_ledger.json").read_bytes()).hexdigest():
        raise ValueError("budget ledger changed after allocation review")
    terminal = read_json(prior_terminal_path)
    if (terminal.get("active_gpu_calls") != 0 or terminal.get("active_gpu_containers") != 0
            or terminal.get("observation_type") != "actual_provider_terminal_observation"
            or type(terminal.get("observed_unix")) not in (int, float)
            or not 0 <= time.time() - terminal["observed_unix"] <= 300):
        raise ValueError("fresh actual prior GPU terminal evidence required before a new launch")
    deployment = read_json(deployment_path)
    if (deployment.get("app_name") != APP_NAME or deployment.get("environment") != "main"
            or deployment.get("source_manifest") != source
            or not isinstance(deployment.get("app_id"), str) or not deployment["app_id"].startswith("ap-")
            or set(deployment.get("functions", {})) != {"cpu_preflight", "live_demo"}
            or any(not isinstance(value, str) or not value.startswith("fu-") for value in deployment["functions"].values())):
        raise ValueError("observed exact deployed function identities required")
    prior_path = ROOT / "runs/pilot-v1-remote-report.json"
    prior = read_json(prior_path)
    prior_sha = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    if prior_sha != "1525f269113797acb31f52ede41ad1792208d243c88a9b92a807a36afba6a9f5":
        raise ValueError("saved original source receipt differs; do not rehash original shards")
    fixtures_path = ROOT / "runs/fixtures-v1.json"
    if hashlib.sha256(fixtures_path.read_bytes()).hexdigest() != "3f979fc23eb0456d2574fedbf12608c4ab7e5198ac36b880726af360e737173d":
        raise ValueError("saved original fixture/processor receipt differs")
    fixtures = read_json(fixtures_path)
    selected_path = ROOT / "runs/vllm-pilot-v4.json"
    selected_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    if selected_sha != "2298d155c13e117ed8837fee509a485c7e8ac4118d5fc57cba0b052b379d90a6":
        raise ValueError("prior selected-smoke receipt differs")
    selected = read_json(selected_path)
    if (selected.get("status") != "completed" or selected.get("child", {}).get("selected_scale") != 0.5
            or selected["child"].get("status") != "accepted_smoke_improvement"):
        raise ValueError("exact0.5 synthetic-smoke selection required; no broad-superiority inference")
    sys.path.insert(0, str(ROOT / "src"))
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    admission = {"run_id": CPU_RUN_ID if phase == "preflight" else GPU_RUN_ID,
        "session_id": SESSION_ID, "levels": list(LEVELS), "source_manifest": source,
        "source_manifest_sha256": source_sha, "budget_allocation": allocation,
        "prior_terminal_receipt_sha256": hashlib.sha256(Path(prior_terminal_path).read_bytes()).hexdigest(),
        "deployment_identity": deployment, "revision": REVISION, "candidate_scale": 0.5,
        "model_pins": MODEL_PINS, "fixtures": fixtures, "selection_receipt_sha256": selected_sha,
        "runner_reconciliation": reconcile_fixture_admission(fixtures["fixture_manifest"]),
        "prior_source": prior["source"], "prior_report_sha256": prior_sha,
        "allowed_asset_origins": ["https://neal.fun", "https://embed.neal.fun", "https://plausible.neal.fun"],
        "actual_model_identity": {"name": "Adam-1-CU", "candidate_scale": 0.5, "model_pins": MODEL_PINS},
        "assistant_model_identity": "Interactive assistant; exact model identity/tokens/cost not supplied by trusted runtime metadata"}
    if phase == "demo":
        cpu = read_json(RUNS / "preflight.remote-report.json")
        child = cpu.get("child") or {}
        result = child.get("demo") or {}
        if (cpu.get("source_manifest") != source or cpu.get("status") != "completed"
                or result.get("status") != "completed" or result.get("blocked_browser_requests")):
            raise ValueError("exact successful CPU/video preflight required; no silent origin widening")
        admission.update(cpu_receipt=cpu, runtime_versions=child["runtime_versions"],
                         browser_runtime_manifest=child["browser_runtime_manifest"])
    return admission, deployment


def load_session(path=RUNS / "session.json"):
    value = read_json(path)
    if (value.get("session_id") != SESSION_ID or value.get("bridge_name") != BRIDGE_NAME
            or value.get("source_manifest") != current_manifest()
            or value.get("source_manifest_sha256") != canonical_sha(value["source_manifest"])
            or not time.time() < NOT_AFTER_UNIX):
        raise ValueError("fixed active session/source receipt required")
    return value


def once_mutation(name, payload, operation):
    """Consume local intent before one mutating RPC; preserve uncertain delivery."""
    save_once(RUNS / (name + ".intent.json"), {"unix": time.time(), "payload": payload,
        "sha256": canonical_sha(payload), "retry_authorized": False})
    try:
        created = operation()
        if created is not True:
            raise RuntimeError("remote immutable key already exists; no overwrite")
        receipt = {"created": True, "transport_only": True, "actual_action_or_game_success": False}
        save_once(RUNS / (name + ".delivery.json"), receipt)
        return receipt
    except BaseException as exc:
        preserve_failure(RUNS / (name + ".failure.json"), {"type": type(exc).__name__,
            "message": str(exc)[:1000], "delivery_disposition": "unknown_or_existing", "retry_authorized": False})
        raise


def ready():
    session = load_session()
    store = private_store()
    registration = store.get(SESSION_ID + ":registration")
    if registration != {"session_id": SESSION_ID, "source_manifest_sha256": session["source_manifest_sha256"],
                        "expires_unix": NOT_AFTER_UNIX}:
        raise ValueError("actual private session registration does not match")
    payload = {"session_id": SESSION_ID, "actor": "assistant", "ready": True,
        "source_manifest_sha256": session["source_manifest_sha256"],
        "expires_unix": min(time.time() + 180, NOT_AFTER_UNIX)}
    return once_mutation("assistant-ready", payload,
        lambda: store.put(SESSION_ID + ":ready", payload, skip_if_exists=True))


def next_observation(seq):
    load_session()
    if type(seq) is not int or not 1 <= seq <= MAX_ASSISTANT_COMMANDS:
        raise ValueError("sequence1..96 required")
    metadata = RUNS / "observations" / f"{seq}.json"
    png_path = metadata.with_suffix(".png")
    if metadata.exists():
        value = read_json(metadata)
        png = validate_observation(value)
        if png_path.is_symlink() or not png_path.is_file() or png_path.read_bytes() != png:
            raise ValueError("cached observed PNG identity differs")
        return {"available": True, "cached": True, "observation": {key: item for key, item in value.items() if key != "png_base64"},
                "png_path": str(png_path.resolve()), "render": "Display this actual PNG with view_image before choosing an action"}
    RUNS.mkdir(parents=True, exist_ok=True)
    remaining_observation_bytes = ((MAX_ASSISTANT_COMMANDS + 1 - seq)
        * (MAX_PNG_BYTES + MAX_MESSAGE_BYTES + 1))
    if shutil.disk_usage(RUNS).free < 200 * 1024 * 1024 + remaining_observation_bytes:
        raise RuntimeError("fresh observation disk reserve is below200MiB plus remaining PNG and compact metadata caps")
    value = private_store().get(SESSION_ID + f":observation:{seq}")
    if value is None:
        return {"available": False, "seq": seq}
    png = validate_observation(value)
    if value["seq"] != seq:
        raise ValueError("response sequence mismatch")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with png_path.open("xb") as stream:
        stream.write(png)
    save_once(metadata, value, compact=True)
    return {"available": True, "cached": False, "observation": {key: item for key, item in value.items() if key != "png_base64"},
            "png_path": str(png_path.resolve()), "render": "Display this actual PNG with view_image before choosing an action"}


def submit(seq, action_file):
    load_session()
    observation = read_json(RUNS / "observations" / f"{seq}.json")
    validate_observation(observation)
    path = Path(action_file)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError("ordinary bounded native action JSON file required")
    raw = path.read_text()
    from neal_protocol import parse_batch
    parse_batch(raw, observation["visible_dom"], observation["remaining_actions"])
    command = {key: observation[key] for key in IDENTITY_FIELDS}
    command["native_action_json"] = raw
    validate_action_identity(command, observation)
    store = private_store()
    return once_mutation(f"action-{seq}", command,
        lambda: store.put(SESSION_ID + f":action:{seq}", command, skip_if_exists=True))


def launch(phase, allocation, terminal, deployment_path):
    admission, deployment = prepare_admission(phase, allocation, terminal, deployment_path)
    modal = check_sdk()
    function_name = "cpu_preflight" if phase == "preflight" else "live_demo"
    expected_id = deployment["functions"][function_name]
    save_once(RUNS / (phase + ".started.json"), {"unix": time.time(), "admission": admission,
        "deployment": deployment, "retry_authorized": False})
    if phase == "demo":
        save_once(RUNS / "session.json", {"session_id": SESSION_ID, "bridge_name": BRIDGE_NAME,
            "source_manifest": admission["source_manifest"], "source_manifest_sha256": admission["source_manifest_sha256"]})
    call = None
    spawn_entered = False
    stage = "before_lookup"
    try:
        function = modal.Function.from_name(APP_NAME, function_name, environment_name="main")
        stage = "hydrate"
        function.hydrate()
        if function.object_id != expected_id:
            raise ValueError("hydrated deployed function ID mismatch")
        save_once(RUNS / (phase + ".lookup.json"), {"function_id": function.object_id,
            "app_id_from_observed_deployment": deployment["app_id"], "function_name": function_name})
        spawn_entered = True
        stage = "spawn_entered"
        call = function.spawn(admission)
        stage = "handle_returned"
        save_once(RUNS / (phase + ".call.json"), {"function_call_id": call.object_id,
            "function_id": function.object_id, "app_id": deployment["app_id"]})
        result = call.get(timeout=600 if phase == "preflight" else 1250)
        save_once(RUNS / (phase + ".remote-report.json"), result)
        if (result.get("function_call_id") != call.object_id or result.get("run") != admission["run_id"]
                or result.get("source_manifest") != admission["source_manifest"]):
            raise RuntimeError("result call/run/source identity mismatch")
        if result.get("container_termination_required", True):
            raise RuntimeError("supervisor requested container termination")
        return {"saved": str(RUNS / (phase + ".remote-report.json")), "status": result["status"],
            "physical_stop_and_billing": "Require actual provider terminal evidence"}
    except BaseException as exc:
        failure = {"stage": stage, "spawn_entered": spawn_entered, "type": type(exc).__name__,
            "message": str(exc)[:2000], "retry_authorized": False,
            "submission_disposition": "pre_submit" if not spawn_entered else "entered_unknown" if call is None else "returned_handle"}
        if call is not None:
            cancellation_entered = False
            try:
                signal.setitimer(signal.ITIMER_REAL, 20)
                cancellation_entered = True
                call.cancel(terminate_containers=True)
                failure["cancellation"] = {"attempted": True, "request_returned": True, "proves_physical_stop": False}
            except BaseException as cancel_error:
                failure["cancellation"] = {"attempted": cancellation_entered, "request_returned": False, "type": type(cancel_error).__name__}
        preserve_failure(RUNS / (phase + ".failure.json"), failure)
        raise


def recover(phase):
    """Once-only exact artifact streaming; failed partial files remain preserved."""
    modal = check_sdk()
    report = read_json(RUNS / (phase + ".remote-report.json"))
    call = read_json(RUNS / (phase + ".call.json"))
    run_id = CPU_RUN_ID if phase == "preflight" else GPU_RUN_ID
    if (report.get("function_call_id") != call["function_call_id"] or report.get("run") != run_id
            or report.get("source_manifest") != current_manifest()):
        raise ValueError("exact call/run/source-bound artifact manifest required")
    inventory = report.get("artifact_inventory", [])
    if not isinstance(inventory, list) or not 1 <= len(inventory) <= MAX_ARTIFACT_FILES:
        raise ValueError("bounded nonempty artifact inventory required")
    seen = set()
    for item in inventory:
        relative = Path(item["relative_path"])
        cap = MAX_VIDEO_BYTES if relative.suffix == ".webm" else MAX_PNG_BYTES if relative.suffix == ".png" else MAX_PROGRESS_BYTES
        if (relative.is_absolute() or ".." in relative.parts or str(relative) in seen
                or relative.suffix not in {".webm", ".png", ".json", ".jsonl"}
                or type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= cap
                or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64):
            raise ValueError("invalid exact artifact path/length/hash")
        seen.add(str(relative))
    remaining = sum(item["bytes"] for item in inventory)
    if remaining > MAX_ARTIFACT_BYTES:
        raise ValueError("aggregate artifact budget exceeded")
    save_once(RUNS / (phase + ".recovery-started.json"), {"inventory": inventory,
        "function_call_id": call["function_call_id"], "retry_authorized": False})
    volume = modal.Volume.from_name("bu-weight-space-results-v3", create_if_missing=False)
    output = RUNS / (phase + "-artifacts")
    output.mkdir(exist_ok=False)
    completed = []
    try:
        for item in inventory:
            if shutil.disk_usage(output).free < 200 * 1024 * 1024 + remaining:
                raise RuntimeError("fresh200MiB plus remaining artifact aggregate disk reserve required")
            target = output / item["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            count, digest = 0, hashlib.sha256()
            with target.open("xb") as stream:
                for chunk in volume.read_file(run_id + "/artifacts/" + item["relative_path"]):
                    if not isinstance(chunk, bytes) or count + len(chunk) > item["bytes"]:
                        raise ValueError("unexpected streamed artifact type/length")
                    stream.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
            if count != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError("artifact stream length/hash mismatch")
            completed.append({**item, "local_path": str(target.resolve())})
            remaining -= count
        result = {"completed": completed, "raw_finalized_video": report.get("artifacts_finalized") is True,
                  "source_call_id": call["function_call_id"],
                  "transport_scope": "One source-level Volume.read_file per exact file; Modal1.4.3 internally retries/prefetches blocks; not a physical-request bound"}
        save_once(RUNS / (phase + ".recovery.json"), result)
        return result
    except BaseException as exc:
        preserve_failure(RUNS / (phase + ".recovery-failed.json"), {"completed": completed,
            "type": type(exc).__name__, "message": str(exc)[:1000], "retry_authorized": False,
            "partial_files_preserved": True})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "ready", "next", "submit", "recover"))
    parser.add_argument("--phase", choices=("preflight", "demo"), default="demo")
    parser.add_argument("--seq", type=int)
    parser.add_argument("--action-file")
    parser.add_argument("--allocation")
    parser.add_argument("--prior-terminal")
    parser.add_argument("--deployment")
    args = parser.parse_args()
    check_sdk()
    def timeout(signum, frame):
        raise TimeoutError("bounded host command timeout; no blind retry")
    signal.signal(signal.SIGALRM, timeout)
    seconds = 1300 if args.command == "launch" else 600 if args.command == "recover" else 8
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        if args.command == "launch":
            result = launch(args.phase, args.allocation, args.prior_terminal, args.deployment)
        elif args.command == "ready":
            result = ready()
        elif args.command == "next":
            result = next_observation(args.seq)
        elif args.command == "submit":
            result = submit(args.seq, args.action_file)
        else:
            result = recover(args.phase)
        print(json.dumps(result, allow_nan=False))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
