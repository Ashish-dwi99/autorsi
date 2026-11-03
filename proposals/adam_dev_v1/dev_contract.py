"""Frozen development design. Verification identities exist but are not runnable here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from weight_space_lab.benchmarks.miniwob import Case, FAMILIES

DEVELOPMENT_SEED = 2026110389
VERIFICATION_SEED = 2026110397
SCALES = (1.0, 0.5, 0.25)
APP_NAME = "adam-1-dev-miniwob-20261103-v1"
IMPORT_RUN = "adam-dev-imports-20261103-v1"
GPU_RUN = "adam-dev-miniwob-20261103-v1"
CHILD_WALL_SECONDS = 1200
NEW_WORK_SECONDS = 1140
GENERATION_HARD_SECONDS = 45
MAX_MODEL_REQUESTS = 288  # Includes three neutral warmups, not288+3.
MAX_IMAGE_BYTES = 128 * 1024
MAX_IMAGES = 288
MAX_ARTIFACT_BYTES = MAX_IMAGES * MAX_IMAGE_BYTES
NOT_AFTER_UNIX = 1793880000  # 2026-11-05 12:00UTC; no automatic renewal.
SELECTION_RULE = "complete36episodes; successes descending; invalid actions ascending; output tokens ascending; scale descending"
FILES = ("dev_contract.py", "dev_patch.py", "dev_native.py", "dev_benchmark.py",
         "dev_entry.py", "modal_dev.py", "dev_host.py", "build_miniwob_runtime.py", "SCOPE.json")


def cases():
    return tuple(Case(family, DEVELOPMENT_SEED) for family in FAMILIES)


def variant_order(index):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError("development case index required")
    # Each recipe occupies each ordinal four times; no adaptive ordering.
    offset = index % 3
    return SCALES[offset:] + SCALES[:offset]


def reset_variant_caches(engine):
    """Same cold image/text cache state for each paired variant under native clocks."""
    if engine.reset_prefix_cache() is not True:
        raise RuntimeError("text prefix cache reset did not confirm success; no inference permitted")
    engine.reset_mm_cache()
    engine.llm_engine.reset_encoder_cache()
    return {"prefix_reset_returned_true": True, "mm_reset_returned": True,
            "encoder_reset_returned": True, "purpose": "Equal fresh image/text cache state per variant"}


def experiment_spec():
    return {"phase": "development", "development_seed": DEVELOPMENT_SEED,
            "verification_seed_reserved_unconsumed": VERIFICATION_SEED,
            "scales": list(SCALES), "case_ids": [case.case_id for case in cases()],
            "selection_rule": SELECTION_RULE, "max_model_requests_including_warmups": MAX_MODEL_REQUESTS,
            "protocol": "action-only-json-v1", "max_completion_tokens": 96,
            "actions_per_episode": 8, "native_timers_unchanged": True}


def validate_spec(admission):
    if admission.get("experiment_spec") != experiment_spec():
        raise ValueError("fresh frozen development spec required; verification is not admitted")


def select_development(variants):
    """No partial denominator, verification input, model claim or stale winner."""
    if set(variants) != {str(scale) for scale in SCALES}:
        raise ValueError("all three frozen variants required")
    expected = [case.case_id for case in cases()]
    scores = []
    for scale in SCALES:
        rows = variants[str(scale)]
        if len(rows) != 12 or [row.get("case_id") for row in rows] != expected:
            raise ValueError("all twelve fresh paired cases in order required")
        if any(type(row.get("success")) is not bool or type(row.get("invalid_actions")) is not int
               or row["invalid_actions"] < 0 or type(row.get("completion_tokens")) is not int
               or row["completion_tokens"] < 0 or row.get("token_usage_complete") is not True for row in rows):
            raise ValueError("complete native outcomes and measured token counts required")
        scores.append({"scale": scale, "successes": sum(row["success"] for row in rows),
                       "invalid_actions": sum(row["invalid_actions"] for row in rows),
                       "completion_tokens": sum(row["completion_tokens"] for row in rows)})
    chosen = sorted(scores, key=lambda s: (-s["successes"], s["invalid_actions"],
                                         s["completion_tokens"], -s["scale"]))[0]
    baseline = variants["1.0"]
    edited = variants[str(chosen["scale"])]
    gains = [a["case_id"] for a, b in zip(baseline, edited) if not a["success"] and b["success"]]
    regressions = [a["case_id"] for a, b in zip(baseline, edited) if a["success"] and not b["success"]]
    return {"rule": SELECTION_RULE, "scores": scores, "selected_scale": chosen["scale"],
            "paired_gains": gains, "paired_regressions": regressions,
            "verification_required": True, "verification_performed": False,
            "promotion": "NONE; development selection only", "official_prior_score_unchanged": "21/24"}


def source_manifest(package_root, proposal_root):
    package_root, proposal_root = Path(package_root), Path(proposal_root)
    paths = {"proposals/adam_dev_v1/" + name: proposal_root / name for name in FILES}
    for path in sorted(package_root.rglob("*")):
        rel = path.relative_to(package_root)
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("source symlink refused")
        if path.is_file():
            if path.suffix not in {".py", ".json", ".txt", ".jinja", ".md"}:
                raise ValueError("unexpected package source type")
            paths["src/weight_space_lab/" + str(rel)] = path
    if len(paths) > 80:
        raise ValueError("source inventory too large")
    output, total = {}, 0
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing ordinary source: " + name)
        raw = path.read_bytes(); total += len(raw)
        if total > 2 * 1024**2:
            raise ValueError("source bytes exceed2MiB")
        output[name] = hashlib.sha256(raw).hexdigest()
    return output


def executing_source_join(manifest, modules):
    """Join SDK auto-mounted modules to explicit packaged copies, by actual bytes."""
    modules = {"dev_contract.py": __file__, **modules}
    output = {}
    for name, filename in modules.items():
        expected = manifest["proposals/adam_dev_v1/" + name]
        path = Path(filename)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("executing source differs from packaged manifest: " + name)
        output[name] = {"actual_path": str(path), "sha256": actual}
    return output


def save_observation(observation, destination, *, case_id, scale, dispatch, elapsed_s):
    """Exact returned pre-action screenshot, not a reconstruction or continuous video."""
    import io
    from PIL import Image
    if type(dispatch) is not int or not 1 <= dispatch <= 8 or scale not in SCALES:
        raise ValueError("unexpected observation identity")
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("invalid original elapsed time")
    image = Image.fromarray(observation["screenshot"])
    if image.size != (160, 210):
        raise ValueError("unchanged MiniWoB screenshot geometry required")
    data = io.BytesIO(); image.save(data, format="PNG"); raw = data.getvalue()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("observation image exceeds128KiB")
    path = Path(destination)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"relative_path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "case_id": case_id, "scale": scale, "dispatch": dispatch, "elapsed_s": elapsed_s,
            "width":160,"height":210,"kind":"actual_pre_action_observation"}
