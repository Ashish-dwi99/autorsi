"""Pure source, admission, and artifact limits shared by the private wrapper."""
import hashlib
import json
import math
from pathlib import Path
import time
from neal_limits import *

def source_manifest(source_root, demo_root):
    paths = {}
    for prefix, folder in (("src/weight_space_lab", Path(source_root)), ("proposals/neal_demo_v2", Path(demo_root))):
        for path in sorted(folder.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError("source symlinks are not admitted")
            if path.is_file():
                if path.suffix not in {".py", ".json", ".txt", ".md", ".jinja"}:
                    raise ValueError("unreviewed package file")
                paths[prefix + "/" + path.relative_to(folder).as_posix()] = path
    if not 20 <= len(paths) <= 96 or sum(path.stat().st_size for path in paths.values()) > 2 * 1024 * 1024:
        raise ValueError("bounded complete source closure required")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}

def manifest_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def validate_admission(admission, kind, actual):
    if kind not in {"preflight", "demo"}:
        raise ValueError("unknown function phase")
    expected_run = CPU_RUN_ID if kind == "preflight" else GPU_RUN_ID
    if (admission.get("run_id") != expected_run or admission.get("session_id") != SESSION_ID
            or tuple(admission.get("levels", ())) != LEVELS
            or admission.get("source_manifest") != actual
            or admission.get("source_manifest_sha256") != manifest_sha(actual)):
        raise ValueError("exact run/session/three-level/source admission required")
    budget = admission.get("budget_allocation", {})
    if (budget.get("source_manifest_sha256") != manifest_sha(actual)
            or type(budget.get("allocated_usd")) not in (int, float)
            or not math.isfinite(budget["allocated_usd"])
            or budget["allocated_usd"] < PROPOSED_RESERVATION_USD
            or type(budget.get("total_committed_usd")) not in (int, float)
            or not budget["allocated_usd"] <= budget["total_committed_usd"] <= 30
            or not isinstance(budget.get("ledger_sha256"), str) or len(budget["ledger_sha256"]) != 64):
        raise ValueError("no concrete source-bound allocation within the existing30USD ceiling")
    if kind == "demo":
        cpu = admission.get("cpu_receipt", {})
        child = cpu.get("child") or {}
        demo = child.get("demo") or {}
        if (cpu.get("status") != "completed" or cpu.get("child_exitcode") != 0
                or cpu.get("child_reaped") is not True or cpu.get("source_manifest") != actual
                or child.get("status") != "completed" or child.get("model_calls") != 0
                or child.get("cuda_initialized") is not False
                or demo.get("status") != "completed" or demo.get("browser_close_returned") is not True
                or demo.get("blocked_browser_requests") or demo.get("planned_levels") != list(LEVELS)
                or len(demo.get("levels", [])) != 3 or len(demo.get("artifacts", [])) != 1):
            raise ValueError("successful exact-source CPU/browser/video preflight required")
        if admission.get("runtime_versions") != child.get("runtime_versions"):
            raise ValueError("actual CPU runtime identity required")
        if admission.get("browser_runtime_manifest") != child.get("browser_runtime_manifest"):
            raise ValueError("actual CPU browser binary/source identity required")
        media = child.get("media_inspection") or {}
        artifact = demo["artifacts"][0]
        if (media.get("status") != "decoded_frames_verified" or media.get("width") != 970
                or media.get("height") != 650 or type(media.get("frames")) is not int or media["frames"] <= 0
                or media.get("video_sha256") != artifact.get("sha256")
                or media.get("video_bytes") != artifact.get("bytes")
                or media.get("ffprobe_returncode") != 0 or media.get("stderr_bytes") != 0):
            raise ValueError("actual bounded media decode/dimension receipt required")
    if not time.time() < NOT_AFTER_UNIX:
        raise ValueError("fixed demo expired; no automatic renewal")

def artifact_inventory(folder, *, hash_files=False):
    folder = Path(folder)
    files = []
    total = videos = 0
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlink refused")
        if not path.is_file():
            continue
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if suffix == ".webm":
            videos += 1
            cap = MAX_VIDEO_BYTES
        elif suffix == ".png":
            cap = MAX_PNG_BYTES
        elif suffix in {".json", ".jsonl"}:
            cap = MAX_PROGRESS_BYTES
        else:
            raise ValueError("unreviewed artifact type")
        total += size
        if size > cap or total > MAX_ARTIFACT_BYTES or videos > 1 or len(files) >= MAX_ARTIFACT_FILES:
            raise ValueError("artifact byte/count bound exceeded")
        item = {"relative_path": path.relative_to(folder).as_posix(), "bytes": size}
        if hash_files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        files.append(item)
    return files
