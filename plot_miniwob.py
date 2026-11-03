"""Plot a saved MiniWoB comparison; never calls a model or provider."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ("click-button", "click-button-sequence", "click-checkboxes", "click-color",
            "click-link", "click-option", "click-tab", "enter-text", "enter-password",
            "login-user", "simple-arithmetic", "focus-text")
CASES = {f"{family}:seed={seed}": (f"miniwob/{family}-v1", seed)
         for family in FAMILIES for seed in (2026110317, 2026110353)}


def summarize(raw):
    outer = json.loads(raw)
    report = outer.get("child", outer)
    if report.get("planned_cases") != 24 or report.get("planned_episodes") != 48:
        raise ValueError("Expected the frozen 24-case paired experiment")
    if type(report.get("candidate_scale")) not in (int, float) or report["candidate_scale"] != .5:
        raise ValueError("Expected the frozen 0.50 candidate")
    entered = report.get("model_calls")
    if type(entered) is not int or not 0 <= entered <= 386:
        raise ValueError("Invalid total entered model-call count")
    records = report.get("generation_records", [])
    seen_calls = set()
    for call in records:
        number = call["call"]
        if type(number) is not int or not 1 <= number <= entered or number in seen_calls:
            raise ValueError("Invalid or duplicate returned-call index")
        seen_calls.add(number)
        if type(call["scale"]) not in (int, float) or call["scale"] not in (1.0, .5):
            raise ValueError("Unexpected generation scale")
        if call["scope"] == "scored_miniwob_episode":
            if call["case_id"] not in CASES:
                raise ValueError("Generation outside the frozen inventory")
        elif call["scope"] != "unscored_neutral_warmup" or call["case_id"] is not None:
            raise ValueError("Unexpected generation scope")
    variants, by_case = {}, {}
    for key in ("1.0", "0.5"):
        rows = report.get("variants", {}).get(key, [])
        if not isinstance(rows, list) or len(rows) > 24:
            raise ValueError("Invalid completed episode inventory")
        mapped = {}
        for row in rows:
            case_id = row["case_id"]
            if not isinstance(case_id, str) or case_id in mapped or type(row["success"]) is not bool:
                raise ValueError("Duplicate case or invalid outcome")
            if (case_id not in CASES or type(row["seed"]) is not int
                    or (row["env_id"], row["seed"]) != CASES[case_id]):
                raise ValueError("Episode does not join a frozen task and seed")
            for counter in ("model_calls", "policy_dispatches", "environment_steps", "invalid_actions"):
                if type(row[counter]) is not int or not 0 <= row[counter] <= 8:
                    raise ValueError("Invalid bounded episode counter")
            if (type(row["elapsed_s"]) not in (int, float) or not math.isfinite(row["elapsed_s"])
                    or row["elapsed_s"] < 0):
                raise ValueError("Invalid episode duration")
            if (type(row["terminated"]) is not bool or type(row["truncated"]) is not bool
                    or type(row["reward"]) not in (int, float)
                    or row["reward"] not in ((-1, 1) if row["terminated"] else (0,))
                    or row["success"] != (row["terminated"] and not row["truncated"] and row["reward"] == 1)):
                raise ValueError("Success flag does not match the native binary outcome")
            returned = [call for call in records if call["scope"] == "scored_miniwob_episode"
                        and call["scale"] == float(key) and call["case_id"] == case_id]
            if len(returned) != row["model_calls"]:
                raise ValueError("Completed episode model calls do not join returned records")
            mapped[case_id] = row
        calls = [call for call in records
                 if call["scale"] == float(key) and call["scope"] == "scored_miniwob_episode"]
        seconds = [call["elapsed_s"] for call in calls]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in seconds):
            raise ValueError("Invalid recorded generation duration")
        successes = sum(row["success"] for row in rows)
        variants[key] = {
            "planned_cases": 24, "completed_cases": len(rows), "successes": successes,
            "completed_unsuccessful": len(rows) - successes, "missing_cases": 24 - len(rows),
            "success_fraction_of_planned": successes / 24,
            "recorded_scored_request_returns": len(calls),
            "median_recorded_request_seconds": statistics.median(seconds) if seconds else None,
            "completed_episode_seconds": sum(row["elapsed_s"] for row in rows),
        }
        by_case[key] = mapped
    common = sorted(set(by_case["1.0"]) & set(by_case["0.5"]))
    gains = [case for case in common if by_case["0.5"][case]["success"] and not by_case["1.0"][case]["success"]]
    regressions = [case for case in common if by_case["1.0"][case]["success"] and not by_case["0.5"][case]["success"]]
    return {
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "outer_status": outer.get("status"), "child_status": report.get("status"),
        "variants": variants, "paired_completed_cases": len(common),
        "paired_gains": gains, "paired_regressions": regressions,
        "total_entered_model_calls_including_warmups": report.get("model_calls"),
        "request_metric_scope": "Recorded scored returns, including any partial final episode; excludes warmups. Full request latency, not isolated decoding speed.",
        "scope": "Fixed 12-family, two-seed subset; native timers include inference. No Astra comparison or broad capability claim.",
        "missing_case_policy": "Missing cases remain in the denominator and are unknown, not recorded failures.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.receipt.read_bytes())
    rows = [summary["variants"][key] for key in ("1.0", "0.5")]
    names = ["Original fine-tune", "Adam-1-CU · 0.50"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    fig.subplots_adjust(top=.81, bottom=.24, wspace=.38, left=.055, right=.99)
    bottom = [0, 0]
    for field, label, color in (("successes", "Native success", "#2A9D8F"),
                                ("completed_unsuccessful", "Unsuccessful", "#A3AEC1"),
                                ("missing_cases", "Missing / unknown", "#E9C46A")):
        values = [row[field] for row in rows]
        axes[0].bar(names, values, bottom=bottom, label=label, color=color, width=.65)
        bottom = [total + value for total, value in zip(bottom, values)]
    for index, row in enumerate(rows):
        axes[0].text(index, 24.5, f"{row['successes']}/24 success", ha="center", fontsize=10)
    axes[0].set_ylim(0, 34)
    axes[0].set_ylabel("Cases (24 planned per variant)")
    axes[0].legend(loc="upper left", fontsize=8)
    for ax, field, ylabel in (
        (axes[1], "median_recorded_request_seconds", "Median returned model request (seconds)"),
        (axes[2], "recorded_scored_request_returns", "Recorded scored model request returns"),
    ):
        values = [row[field] for row in rows]
        bars = ax.bar(names, [value or 0 for value in values], color=["#335C81", "#2A9D8F"], width=.65)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value or 0,
                    "Unavailable" if value is None else f"{value:.2f}" if field.endswith("seconds") else str(value),
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.margins(y=.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=12)
    completed = sum(row["completed_cases"] for row in rows)
    coverage = "All 48 episodes completed" if completed == 48 else f"Incomplete: {completed}/48 episodes recorded"
    fig.suptitle("Adam-1-CU · MiniWoB fixed 24-case comparison\n"
                 + coverage, fontsize=14, fontweight="bold", y=.98, va="top")
    fig.supxlabel(
        f"{summary['paired_completed_cases']}/24 complete pairs · {len(summary['paired_gains'])} gains · "
        f"{len(summary['paired_regressions'])} regressions. Native timers include inference. "
        "Returns include any partial episode.\nNo Astra, population-wide, or billing claim.", fontsize=9, y=.025)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = args.output.with_suffix("." + extension)
        fig.savefig(path, dpi=160)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
