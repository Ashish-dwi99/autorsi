"""Plot saved paired smoke results; never performs model or provider calls."""
import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(k, n):
    z = 1.96
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return center-half, center+half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outer = json.loads(args.receipt.read_text())
    report = outer.get("child", outer)
    variants = report["variants"]
    keys = [key for key in ("1.0", "0.5", "0.75", "0.0") if variants.get(key)]
    if not keys:
        raise ValueError("No completed task rows to plot")
    labels = {"1.0": "Original", "0.5": "Strength 0.50", "0.75": "Strength 0.75", "0.0": "Subtracted control"}
    colors = ["#335C81", "#2A9D8F", "#5C8E51", "#9A8C98"][:len(keys)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), constrained_layout=True)
    rates, errors, medians, counts = [], [[], []], [], []
    for key in keys:
        rows = variants[key]
        n, k = len(rows), sum(bool(row["success"]) for row in rows)
        rate = k / n
        low, high = wilson(k, n)
        rates.append(rate * 100)
        errors[0].append((rate-low)*100)
        errors[1].append((high-rate)*100)
        dev_calls = sum(row["native_dispatch_counters"]["actual_model_calls"] for row in rows)
        calls = [r for r in report["generation_records"] if r["scale"] == float(key)][:dev_calls]
        medians.append(statistics.median(r["elapsed_s"] for r in calls) if calls else 0)
        counts.append(len(calls))
    names = [labels[k] for k in keys]
    for ax, values, ylabel in zip(axes, (rates, medians, counts),
            ("Task success (%)", "Median full model request (seconds)", "Completed model requests")):
        bars = ax.bar(names, values, color=colors, width=.65)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0].errorbar(range(len(keys)), rates, yerr=errors, fmt="none", ecolor="#333333", capsize=4)
    axes[0].set_ylim(0, 110)
    for i, key in enumerate(keys):
        axes[0].text(i, 104, f"n={len(variants[key])}", ha="center", fontsize=9)
    fig.suptitle("BU weight-edit pilot · synthetic browser tasks", fontsize=15, fontweight="bold")
    fig.supxlabel("95% Wilson intervals describe this small fixed task set only. No Astra comparison; no general benchmark or cost claim.", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.output.with_suffix("."+extension), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
