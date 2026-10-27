"""
Tests for ground-truth dataset integrity and the validate_ground_truth script.
"""
import os
import sys
import pytest
import pandas as pd
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import GROUND_TRUTH_PATH

REQUIRED_COLUMNS = ["question", "expected_answer", "source_document", "source_page"]
PLACEHOLDER_MARKERS = ["replace this row", "add your own", "placeholder"]
TARGET_ROWS = 50


def _is_placeholder(row: pd.Series) -> bool:
    for col in ("question", "expected_answer"):
        val = str(row.get(col, "")).lower()
        if any(m in val for m in PLACEHOLDER_MARKERS):
            return True
    return False


@pytest.fixture(scope="module")
def ground_truth_df():
    return pd.read_csv(GROUND_TRUTH_PATH)


class TestGroundTruthStructure:
    def test_file_exists(self):
        assert os.path.exists(GROUND_TRUTH_PATH), (
            f"Ground-truth file not found: {GROUND_TRUTH_PATH}"
        )

    def test_required_columns_present(self, ground_truth_df):
        missing = [c for c in REQUIRED_COLUMNS if c not in ground_truth_df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_no_empty_expected_answers(self, ground_truth_df):
        valid = ground_truth_df[~ground_truth_df.apply(_is_placeholder, axis=1)]
        empty = valid[
            valid["expected_answer"].isna()
            | (valid["expected_answer"].astype(str).str.strip() == "")
        ]
        assert len(empty) == 0, (
            f"{len(empty)} row(s) have empty expected_answer: {list(empty.index)}"
        )

    def test_no_duplicate_questions(self, ground_truth_df):
        valid = ground_truth_df[~ground_truth_df.apply(_is_placeholder, axis=1)]
        dupes = valid[valid.duplicated(subset=["question"])]
        assert len(dupes) == 0, (
            f"{len(dupes)} duplicate question(s) found: {list(dupes['question'])}"
        )

    def test_valid_row_count_meets_target(self, ground_truth_df):
        valid = ground_truth_df[~ground_truth_df.apply(_is_placeholder, axis=1)]
        n = len(valid.drop_duplicates(subset=["question"]))
        assert n >= TARGET_ROWS, (
            f"Only {n} valid questions; need at least {TARGET_ROWS}."
        )

    def test_source_documents_exist(self, ground_truth_df):
        from src.config import DOCUMENTS_DIR
        valid = ground_truth_df[~ground_truth_df.apply(_is_placeholder, axis=1)]
        missing = []
        for doc_name in valid["source_document"].dropna().unique():
            if not os.path.exists(os.path.join(DOCUMENTS_DIR, doc_name)):
                missing.append(doc_name)
        assert missing == [], (
            f"Source document(s) cited in CSV but missing from {DOCUMENTS_DIR}:\n"
            + "\n".join(missing)
        )


class TestGroundTruthEdgeCases:
    def test_empty_csv_raises(self, tmp_path):
        """run_eval should reject an entirely empty CSV."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("question,expected_answer,source_document,source_page\n")
        from eval.run_eval import load_valid_questions
        with pytest.raises(ValueError, match="No valid questions"):
            load_valid_questions(str(empty_csv))

    def test_missing_file_raises(self, tmp_path):
        from eval.run_eval import load_valid_questions
        with pytest.raises(FileNotFoundError):
            load_valid_questions(str(tmp_path / "nonexistent.csv"))

    def test_placeholder_only_csv_raises(self, tmp_path):
        placeholder_csv = tmp_path / "placeholder.csv"
        placeholder_csv.write_text(
            "question,expected_answer,source_document,source_page\n"
            "REPLACE THIS ROW,REPLACE THIS ROW,file.pdf,1\n"
        )
        from eval.run_eval import load_valid_questions
        with pytest.raises(ValueError, match="No valid questions"):
            load_valid_questions(str(placeholder_csv))
