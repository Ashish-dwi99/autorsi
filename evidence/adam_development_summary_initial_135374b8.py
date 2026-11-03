"""Saved-only summary for the once Adam development run, not new evaluation.

This is a root reconciliation, not a substitute for distinct result review.
Development selection cannot establish a verification/benchmark/Astra win.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "proposals/adam_dev_v1")]
from dev_contract import cases, SCALES, select_development, experiment_spec


def summarize(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 16 * 1024**2:
        raise ValueError("oversized saved receipt")
    outer = json.loads(raw)
    child = outer["child"]
    source = json.loads((ROOT / "proposals/adam_dev_v1/SOURCE_RECORD_V2.json").read_bytes())
    if (outer["status"] != "completed" or outer.get("child_reaped") is not True
            or outer.get("child_exitcode") != 0 or child["status"] != "completed_development_only"
            or child["browser_close_returned"] is not True
            or child["experiment_spec"] != experiment_spec()
            or outer["source_manifest"] != source["source_manifest"]
            or child["source_manifest"] != source["source_manifest"]):
        raise ValueError("complete exact saved development result required")
    chosen = select_development(child["variants"])
    if chosen != child["development_selection"]:
        raise ValueError("selection does not reconcile")
    generations = child["generation_records"]
    warmups = child["warmups"]
    if [x["scale"] for x in warmups] != list(SCALES):
        raise ValueError("missing fixed warmups")
    if not len(generations) == child["model_calls"] <= 288:
        raise ValueError("generation count differs")
    if [x["call"] for x in generations] != list(range(1, len(generations) + 1)):
        raise ValueError("generation numbering differs")
    expected = [case.case_id for case in cases()]
    rows = {}
    for scale in SCALES:
        episodes = child["variants"][str(scale)]
        if [x["case_id"] for x in episodes] != expected:
            raise ValueError("fresh paired task coverage differs")
        requests = [x for x in generations if x["scale"] == scale and x["scope"] == "scored_miniwob_episode"]
        if (len(requests) != sum(x["model_calls"] for x in episodes)
                or sum(x["output_tokens"] for x in requests) != sum(x["completion_tokens"] for x in episodes)
                or sum(x["prompt_tokens"] for x in requests) != sum(x["prompt_tokens"] for x in episodes)):
            raise ValueError("per-variant usage does not reconcile")
        for item in requests:
            if not math.isfinite(item["elapsed_s"]) or item["elapsed_s"] < 0:
                raise ValueError("invalid latency")
        rows[str(scale)] = {"successes": sum(x["success"] for x in episodes), "planned": 12,
            "invalid_actions": sum(x["invalid_actions"] for x in episodes),
            "requests": len(requests), "output_tokens": sum(x["output_tokens"] for x in requests),
            "prompt_tokens": sum(x["prompt_tokens"] for x in requests),
            "sum_episode_seconds": sum(x["elapsed_s"] for x in episodes),
            "median_full_request_seconds": statistics.median(x["elapsed_s"] for x in requests) if requests else None,
            "failed_case_ids": [x["case_id"] for x in episodes if not x["success"]]}
    if sum(x["requests"] for x in rows.values()) + 3 != len(generations):
        raise ValueError("unaccounted request scope")
    inspection = child["weight_value_inspection"]
    if len(inspection["entries"]) != 192 or inspection["parameter_mutations"] != 0:
        raise ValueError("actual value inspection incomplete")
    for scale in SCALES:
        audit = child["patch_audits"][str(scale)]
        if audit["pair_count"] != 192 or audit["scale"] != scale or audit["totals"]["elements"] != 905969664:
            raise ValueError("actual edit scope mismatch")
        if scale == 1.0 and audit["totals"]["changed_elements"] != 0:
            raise ValueError("baseline was changed")
        if scale != 1.0 and audit["totals"]["changed_elements"] <= 0:
            raise ValueError("edit did not change actual weights")
    return {"schema": 1, "development_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "selected_attention_scale": chosen["selected_scale"], "development_selection": chosen,
        "metrics": rows, "actual_model_requests_including_warmups": len(generations),
        "weight_value_inspection_groups": inspection["groups"],
        "actual_edit_totals": {str(s): child["patch_audits"][str(s)]["totals"] for s in SCALES},
        "verification_performed": False, "prior_official_score": "21/24 unchanged",
        "scope": "development-only attention choice for separate visual calibration",
        "not_established": ["perfect general computer use", "Astra superiority", "tenfold performance", "checkpoint export"],
        "independent_result_review_required": True}


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2, allow_nan=False))
