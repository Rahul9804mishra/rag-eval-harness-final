"""
Sanity-checks eval/ground_truth.csv before starting evaluation.

Checks performed
----------------
1. Required columns are present.
2. No missing expected answers.
3. No duplicate questions.
4. No placeholder rows.
5. Row count compared to the TARGET_ROWS target.
6. All cited source documents exist under data/documents/.

Run from the project root:
    python eval/validate_ground_truth.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import GROUND_TRUTH_PATH, DOCUMENTS_DIR

REQUIRED_COLUMNS = ["question", "expected_answer", "source_document", "source_page"]
TARGET_ROWS = 50

PLACEHOLDER_MARKERS = [
    "replace this row",
    "add your own",
    "your question here",
    "placeholder",
]


def _is_placeholder(row: pd.Series) -> bool:
    for col in ("question", "expected_answer"):
        val = str(row.get(col, "")).lower()
        if any(m in val for m in PLACEHOLDER_MARKERS):
            return True
    return False


def main():
    errors = []
    warnings = []

    # --- File existence ---
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"ERROR: Ground-truth file not found: {GROUND_TRUTH_PATH}")
        sys.exit(1)

    gt = pd.read_csv(GROUND_TRUTH_PATH)

    # --- Required columns ---
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in gt.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    # --- Placeholder rows ---
    placeholder_mask = gt.apply(_is_placeholder, axis=1)
    n_placeholders = placeholder_mask.sum()
    if n_placeholders:
        warnings.append(f"{n_placeholders} placeholder row(s) detected — they will be "
                        f"ignored during evaluation.")
    gt_valid = gt[~placeholder_mask].copy()

    # --- Empty answers ---
    empty_answers = gt_valid[
        gt_valid["expected_answer"].isna()
        | (gt_valid["expected_answer"].astype(str).str.strip() == "")
    ]
    if len(empty_answers) > 0:
        warnings.append(
            f"{len(empty_answers)} row(s) have an empty expected_answer: "
            f"rows {list(empty_answers.index)}"
        )

    # --- Duplicate questions ---
    dupes = gt_valid[gt_valid.duplicated(subset=["question"])]
    if len(dupes) > 0:
        warnings.append(
            f"{len(dupes)} duplicate question(s) found — only the first will be used."
        )

    # --- Source document existence ---
    missing_docs = []
    if "source_document" in gt_valid.columns:
        for doc_name in gt_valid["source_document"].dropna().unique():
            doc_path = os.path.join(DOCUMENTS_DIR, doc_name)
            if not os.path.exists(doc_path):
                missing_docs.append(doc_name)
    if missing_docs:
        warnings.append(
            f"Source document(s) cited in CSV but not found under {DOCUMENTS_DIR}:\n"
            + "\n".join(f"  - {d}" for d in missing_docs)
        )

    # --- Print report ---
    n_valid = len(gt_valid.drop_duplicates(subset=["question"]))
    print(f"\nGround-truth validation report")
    print(f"  Total CSV rows     : {len(gt)}")
    print(f"  Placeholder rows   : {n_placeholders}")
    print(f"  Valid questions    : {n_valid}  (target: {TARGET_ROWS})")
    print(f"  Source docs cited  : {gt_valid['source_document'].nunique()}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠  {w}")
    else:
        print("\n  No warnings.")

    if n_valid < TARGET_ROWS:
        print(f"\n  You still need {TARGET_ROWS - n_valid} more valid question(s).")
    else:
        print(f"\n  ✓ Row count target met ({n_valid} ≥ {TARGET_ROWS}).")

    if not warnings:
        print("\nValidation passed.\n")
    else:
        print("\nValidation completed with warnings (see above).\n")


if __name__ == "__main__":
    main()
