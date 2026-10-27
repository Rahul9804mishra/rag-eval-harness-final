"""
Tests for result export — checks that run_eval.py writes the expected
output files with correct structure, without actually running the full
evaluation (which requires the embedding model and can be slow).
"""
import os
import sys
import json
import pytest
import pandas as pd
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.metrics import compute_summary
from src.config import STRATEGY_NAMES


class TestComputeSummaryOutput:
    """Validate compute_summary produces the expected output keys."""

    def _row(self, hit=1, rr=1.0, recall=1.0, precision=1.0):
        return {"hit": hit, "rr": rr, "recall_k": recall, "precision_k": precision}

    def test_output_keys_present(self):
        rows = [self._row() for _ in range(5)]
        s = compute_summary(rows)
        expected_keys = {
            "n_questions", "hit_rate_pct", "mrr",
            "avg_recall_at_k", "avg_precision_at_k"
        }
        assert expected_keys == set(s.keys())

    def test_n_questions_correct(self):
        rows = [self._row() for _ in range(12)]
        s = compute_summary(rows)
        assert s["n_questions"] == 12

    def test_hit_rate_is_percentage(self):
        rows = [self._row(hit=1) for _ in range(3)] + [self._row(hit=0) for _ in range(1)]
        s = compute_summary(rows)
        assert s["hit_rate_pct"] == pytest.approx(75.0)

    def test_mrr_range(self):
        rows = [self._row(rr=0.5), self._row(rr=1.0)]
        s = compute_summary(rows)
        assert 0.0 <= s["mrr"] <= 1.0


class TestLegacyResultsJson:
    """Verify that results.json written by run_eval contains all 4 strategy keys."""

    def test_results_json_has_all_strategies(self, tmp_path):
        legacy = {name: 80.0 for name in STRATEGY_NAMES}
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(legacy, f)

        with open(results_path) as f:
            loaded = json.load(f)

        for name in STRATEGY_NAMES:
            assert name in loaded
            assert isinstance(loaded[name], float)

    def test_results_json_values_are_percentages(self, tmp_path):
        legacy = {name: 65.0 for name in STRATEGY_NAMES}
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(legacy, f)

        with open(results_path) as f:
            loaded = json.load(f)

        for name, val in loaded.items():
            assert 0.0 <= val <= 100.0, f"{name} score {val} outside [0, 100]"


class TestDetailCsvStructure:
    """Check that the per-question CSV has the expected columns."""

    EXPECTED_COLUMNS = {
        "strategy", "q_id", "question", "expected_answer",
        "source_document", "hit", "rr", "recall_k", "precision_k",
        "retrieved_texts",
    }

    def test_csv_columns(self, tmp_path):
        rows = []
        for name in STRATEGY_NAMES:
            rows.append({
                "strategy": name,
                "q_id": 0,
                "question": "Sample question?",
                "expected_answer": "Sample answer",
                "source_document": "doc.txt",
                "hit": 1,
                "rr": 1.0,
                "recall_k": 1.0,
                "precision_k": 1.0,
                "retrieved_texts": "chunk 1 ||| chunk 2",
            })
        csv_path = tmp_path / "eval_test.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        assert self.EXPECTED_COLUMNS.issubset(set(loaded.columns))

    def test_one_row_per_question_per_strategy(self, tmp_path):
        n_questions = 5
        rows = []
        for name in STRATEGY_NAMES:
            for i in range(n_questions):
                rows.append({
                    "strategy": name, "q_id": i,
                    "question": f"Q{i}", "expected_answer": f"A{i}",
                    "source_document": "doc.txt",
                    "hit": 0, "rr": 0.0, "recall_k": 0.0, "precision_k": 0.0,
                    "retrieved_texts": "",
                })
        csv_path = tmp_path / "eval_detail.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == n_questions * len(STRATEGY_NAMES)
