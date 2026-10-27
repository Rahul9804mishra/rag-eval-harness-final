"""
Reads results.json (produced by eval/run_eval.py) and exits with a
non-zero status code if the best strategy's accuracy is below
ACCURACY_THRESHOLD. GitHub Actions treats non-zero as a failed step.

Run from the project root:
    python eval/check_threshold.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_PATH, ACCURACY_THRESHOLD


def main():
    if not os.path.exists(RESULTS_PATH):
        print(
            f"ERROR: No results file at {RESULTS_PATH}.\n"
            f"Run  python eval/run_eval.py  first."
        )
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    if not results:
        print("ERROR: results.json is empty.")
        sys.exit(1)

    best_strategy = max(results, key=results.get)
    best_score = results[best_strategy]

    print("\nStrategy scores (Hit Rate %):")
    for strategy, score in results.items():
        marker = " ← best" if strategy == best_strategy else ""
        print(f"  {strategy:<22} {score:.1f}%{marker}")

    print(f"\nBest strategy : {best_strategy} ({best_score:.1f}%)")
    print(f"Threshold     : {ACCURACY_THRESHOLD}%")

    if best_score < ACCURACY_THRESHOLD:
        print("\nAccuracy is below threshold — build FAILED.\n")
        sys.exit(1)
    else:
        print("\nAccuracy meets threshold — build PASSED.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
