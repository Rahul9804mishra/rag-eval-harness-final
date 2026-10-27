"""
Generates a bar chart (results_chart.png) comparing strategy accuracy.

Run from the project root:
    python scripts/plot_results.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from src.config import RESULTS_PATH, RESULTS_CHART_PATH, STRATEGY_NAMES


def main():
    if not os.path.exists(RESULTS_PATH):
        print(
            f"No results file found at {RESULTS_PATH}.\n"
            f"Run  python eval/run_eval.py  first."
        )
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Keep canonical strategy order; skip any that aren't in results.
    methods = [m for m in STRATEGY_NAMES if m in results]
    scores = [results[m] for m in methods]

    labels = [m.replace("_", "\n") for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#4A90D9", "#8E5FD9", "#E8912D", "#3CB878"]
    bars = ax.bar(labels, scores, color=colours[:len(methods)], width=0.5)

    ax.set_ylabel("Hit Rate (%)", fontsize=12)
    ax.set_title("Retrieval Accuracy by Strategy (Hit Rate @ K)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color="red", linestyle="--", linewidth=0.8, label="80% reference")
    ax.legend(fontsize=10)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{score:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(RESULTS_CHART_PATH, dpi=150)
    print(f"Chart saved → {RESULTS_CHART_PATH}")


if __name__ == "__main__":
    main()
