"""Pure limits and provenance for a single CPU navigation diagnostic."""
import hashlib
import json
from pathlib import Path
import time

APP = "adam-1-cu-neal-navigation-diagnostic-v1"
RUN = "neal-navigation-diagnostic-v1"
ONCE = APP + "-once"
FUNCTION = "diagnose_navigation"
PRIOR_APP = "ap-1qgHoLFsIavzrIdLnD905u"  # Actual failed Neal v2 deployment.
EXPIRES = 1793793600
SOURCE_PREFIX = "proposals/neal_navigation_diagnostic_v1/"
ORIGINS = ("https://neal.fun", "https://embed.neal.fun", "https://plausible.neal.fun")
INITIAL = "https://embed.neal.fun/not-a-robot/1"
CANONICAL = "https://embed.neal.fun/not-a-robot/embed/1"
DOCUMENTS = frozenset((INITIAL, INITIAL + "/", CANONICAL, CANONICAL + "/"))
OBSERVATION_SECONDS = 30
CHILD_SECONDS = 120
MAX_REPORT = 2 * 1024 * 1024
MAX_VIDEO = 256 * 1024 * 1024
MAX_PNG = 1024 * 1024
MAX_ARTIFACTS = 258 * 1024 * 1024
SDK_PINS = {
    "_functions.py": "aa18b6dcfa00f09807d1719fc592f96fa69d51c492eecf059aecec865702a2bf",
    "_object.py": "b49b4c10fdf18e32dc09effa598e65f523f37c0424f48844297534faf72511c9",
    "functions.py": "11945c1c57e7a186ef6b05099c01c5e6722fa28f465f60e3cbb99498e0979977"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def manifest(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.is_file())
    if not 4 <= len(files) <= 12 or sum(p.stat().st_size for p in files) > 512 * 1024:
        raise ValueError("bounded exact diagnostic source required")
    if any(p.is_symlink() or p.suffix not in (".py", ".json") for p in files):
        raise ValueError("unexpected diagnostic payload")
    return {SOURCE_PREFIX + p.name: sha(p) for p in files}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False).encode()
    if len(payload) > MAX_REPORT:
        raise ValueError("diagnostic JSON exceeds2MiB")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def validate_admission(admission, actual):
    budget = admission.get("budget_allocation", {})
    if (admission.get("run_id") != RUN or admission.get("source_manifest") != actual
            or admission.get("source_manifest_sha256") != canonical_sha(actual)
            or admission.get("allowed_origins") != list(ORIGINS)
            or budget.get("source_manifest_sha256") != canonical_sha(actual)
            or budget.get("allocated_usd") != 3
            or budget.get("shared_existing_neal_envelope") is not True
            or type(budget.get("total_committed_usd")) not in (int, float)
            or not 3 <= budget["total_committed_usd"] <= 30
            or not isinstance(budget.get("ledger_sha256"), str)
            or len(budget["ledger_sha256"]) != 64 or time.time() >= EXPIRES):
        raise ValueError("exact once-only source/origin/shared3USD admission required")


def inventory(folder, *, hash_files=True):
    result, total, videos = [], 0, 0
    for path in sorted(Path(folder).rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlink refused")
        if not path.is_file():
            continue
        if path.suffix not in (".webm", ".png"):
            raise ValueError("only observed video/PNG artifacts are admitted")
        cap = MAX_VIDEO if path.suffix == ".webm" else MAX_PNG
        size = path.stat().st_size
        total += size
        videos += int(path.suffix == ".webm")
        if size > cap or total > MAX_ARTIFACTS or videos > 1 or len(result) >= 3:
            raise ValueError("bounded artifact inventory exceeded")
        item = {"relative_path": str(path.relative_to(folder)), "bytes": size}
        if hash_files:
            item["sha256"] = sha(path)
        result.append(item)
    return result
