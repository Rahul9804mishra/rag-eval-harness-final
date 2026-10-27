"""
Tests for src/metrics.py — all metrics are pure-function so no mocks needed.
"""
import pytest
from unittest.mock import MagicMock

from src.metrics import (
    hit_at_k,
    rank_of_hit,
    reciprocal_rank,
    recall_at_k,
    precision_at_k,
    score_retrieval,
    compute_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str):
    """Create a mock LangChain Document with .page_content."""
    doc = MagicMock()
    doc.page_content = text
    return doc


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_exact_match_in_first_chunk(self):
        chunks = [_doc("The answer is 75%."), _doc("Something else.")]
        assert hit_at_k(chunks, "75%") == 1

    def test_case_insensitive(self):
        chunks = [_doc("The CEO is Alexandra Reyes.")]
        assert hit_at_k(chunks, "alexandra reyes") == 1

    def test_no_match_returns_zero(self):
        chunks = [_doc("Irrelevant content."), _doc("More irrelevant content.")]
        assert hit_at_k(chunks, "specific answer") == 0

    def test_empty_chunks(self):
        assert hit_at_k([], "any answer") == 0

    def test_plain_string_chunks(self):
        chunks = ["The answer is 75%.", "Other text."]
        assert hit_at_k(chunks, "75%") == 1

    def test_score_retrieval_alias(self):
        chunks = [_doc("answer text here")]
        assert score_retrieval(chunks, "answer text") == 1


# ---------------------------------------------------------------------------
# rank_of_hit
# ---------------------------------------------------------------------------

class TestRankOfHit:
    def test_found_at_rank_1(self):
        chunks = [_doc("answer is here"), _doc("not relevant")]
        assert rank_of_hit(chunks, "answer is here") == 1

    def test_found_at_rank_2(self):
        chunks = [_doc("not relevant"), _doc("the answer is 4 weeks")]
        assert rank_of_hit(chunks, "4 weeks") == 2

    def test_not_found_returns_none(self):
        chunks = [_doc("nothing useful")]
        assert rank_of_hit(chunks, "missing answer") is None

    def test_empty_chunks(self):
        assert rank_of_hit([], "anything") is None


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_rank_1(self):
        chunks = [_doc("the answer is 90 days")]
        assert reciprocal_rank(chunks, "90 days") == pytest.approx(1.0)

    def test_rank_2(self):
        chunks = [_doc("irrelevant"), _doc("the answer is 90 days")]
        assert reciprocal_rank(chunks, "90 days") == pytest.approx(0.5)

    def test_not_found(self):
        chunks = [_doc("nothing")]
        assert reciprocal_rank(chunks, "missing") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_full_recall(self):
        chunks = [_doc("Austin Texas is the location")]
        # "Austin, Texas" → tokens {"austin,", "texas"} — comma attached to austin
        # Let's use a clean answer:
        assert recall_at_k(chunks, "Austin Texas") == pytest.approx(1.0)

    def test_partial_recall(self):
        # Answer has 4 tokens; only 2 appear in chunks.
        chunks = [_doc("The rate is 4 percent per year and other stuff")]
        result = recall_at_k(chunks, "the rate is missing")
        # tokens: the, rate, is, missing — "the", "rate", "is" present → 3/4
        assert 0.0 < result < 1.0

    def test_no_recall(self):
        chunks = [_doc("completely irrelevant")]
        assert recall_at_k(chunks, "specific unique phrase xyz") == pytest.approx(0.0)

    def test_empty_chunks(self):
        assert recall_at_k([], "any answer") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_chunks_contain_answer(self):
        chunks = [_doc("answer 75%"), _doc("answer is 75%")]
        assert precision_at_k(chunks, "75%") == pytest.approx(1.0)

    def test_half_chunks_contain_answer(self):
        chunks = [_doc("answer 75%"), _doc("something else")]
        assert precision_at_k(chunks, "75%") == pytest.approx(0.5)

    def test_no_chunks_contain_answer(self):
        chunks = [_doc("nothing here"), _doc("also nothing")]
        assert precision_at_k(chunks, "75%") == pytest.approx(0.0)

    def test_empty_chunks(self):
        assert precision_at_k([], "75%") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def _make_rows(self, hits, rrs, recalls, precisions):
        return [
            {"hit": h, "rr": r, "recall_k": rc, "precision_k": p}
            for h, r, rc, p in zip(hits, rrs, recalls, precisions)
        ]

    def test_all_hits(self):
        rows = self._make_rows([1, 1, 1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        s = compute_summary(rows)
        assert s["hit_rate_pct"] == pytest.approx(100.0)
        assert s["mrr"] == pytest.approx(1.0)

    def test_no_hits(self):
        rows = self._make_rows([0, 0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        s = compute_summary(rows)
        assert s["hit_rate_pct"] == pytest.approx(0.0)
        assert s["mrr"] == pytest.approx(0.0)

    def test_partial_hits(self):
        rows = self._make_rows([1, 0, 1, 0], [1.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.5, 0.0])
        s = compute_summary(rows)
        assert s["hit_rate_pct"] == pytest.approx(50.0)
        assert s["n_questions"] == 4

    def test_empty_input(self):
        s = compute_summary([])
        assert s["n_questions"] == 0
        assert s["hit_rate_pct"] == 0.0
