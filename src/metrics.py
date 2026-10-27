"""
Retrieval evaluation metrics.

All functions operate on lists of retrieved chunk texts and an expected
answer string (exact-match baseline). No LLM or semantic judge is used.

Metrics implemented
-------------------
hit_at_k          : 1 if the answer appears in any of the top-k chunks.
rank_of_hit       : 1-based position of the first chunk containing the answer
                    (None if not found).
reciprocal_rank   : 1 / rank_of_hit, or 0.0 if not found.
recall_at_k       : fraction of expected answer tokens found across all top-k
                    chunks (token-level recall proxy).
precision_at_k    : fraction of top-k chunks that contain the expected answer.
score_retrieval   : legacy alias → same as hit_at_k (kept for backward compat).
compute_summary   : aggregate metrics over a list of per-question results.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _chunk_texts(chunks) -> list[str]:
    """
    Accept either LangChain Document objects or plain strings and return
    a list of plain strings.
    """
    result = []
    for c in chunks:
        if hasattr(c, "page_content"):
            result.append(c.page_content)
        else:
            result.append(str(c))
    return result


# ---------------------------------------------------------------------------
# Per-question metrics
# ---------------------------------------------------------------------------

def hit_at_k(chunks, expected_answer: str) -> int:
    """
    Returns 1 if ``expected_answer`` appears verbatim (case-insensitive)
    in any of the retrieved chunks, else 0.

    This is the Hit Rate / Success@K metric.
    """
    needle = _normalise(expected_answer)
    for text in _chunk_texts(chunks):
        if needle in _normalise(text):
            return 1
    return 0


# Backward-compatible alias used by the original run_eval.py
def score_retrieval(retrieved_chunks, expected_answer: str) -> int:
    return hit_at_k(retrieved_chunks, expected_answer)


def rank_of_hit(chunks, expected_answer: str) -> Optional[int]:
    """
    Return the 1-based rank of the first chunk containing ``expected_answer``,
    or None if none do.
    """
    needle = _normalise(expected_answer)
    for rank, text in enumerate(_chunk_texts(chunks), start=1):
        if needle in _normalise(text):
            return rank
    return None


def reciprocal_rank(chunks, expected_answer: str) -> float:
    """
    Mean Reciprocal Rank contribution for a single query: 1/rank, or 0.
    """
    r = rank_of_hit(chunks, expected_answer)
    return 1.0 / r if r is not None else 0.0


def recall_at_k(chunks, expected_answer: str) -> float:
    """
    Token-level recall proxy: what fraction of the expected-answer tokens
    appear anywhere across all retrieved chunks?

    This is a soft measure useful when the exact answer string may be
    spread across multiple chunks.
    """
    answer_tokens = set(_normalise(expected_answer).split())
    if not answer_tokens:
        return 0.0
    combined_text = " ".join(_chunk_texts(chunks)).lower()
    found = sum(1 for t in answer_tokens if t in combined_text)
    return found / len(answer_tokens)


def precision_at_k(chunks, expected_answer: str) -> float:
    """
    What fraction of the retrieved chunks contain the expected answer?
    """
    texts = _chunk_texts(chunks)
    if not texts:
        return 0.0
    needle = _normalise(expected_answer)
    hits = sum(1 for t in texts if needle in _normalise(t))
    return hits / len(texts)


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def compute_summary(per_question_rows: list[dict]) -> dict:
    """
    Compute aggregate metrics from a list of per-question result dicts.

    Each dict must contain the keys produced by ``run_eval.py``:
      hit, rr (reciprocal rank), recall_k, precision_k

    Returns a dict of floats (all as percentages where applicable).
    """
    n = len(per_question_rows)
    if n == 0:
        return {
            "n_questions": 0,
            "hit_rate_pct": 0.0,
            "mrr": 0.0,
            "avg_recall_at_k": 0.0,
            "avg_precision_at_k": 0.0,
        }

    hit_rate = sum(r["hit"] for r in per_question_rows) / n * 100
    mrr = sum(r["rr"] for r in per_question_rows) / n
    avg_recall = sum(r["recall_k"] for r in per_question_rows) / n * 100
    avg_precision = sum(r["precision_k"] for r in per_question_rows) / n * 100

    return {
        "n_questions": n,
        "hit_rate_pct": round(hit_rate, 1),
        "mrr": round(mrr, 4),
        "avg_recall_at_k": round(avg_recall, 1),
        "avg_precision_at_k": round(avg_precision, 1),
    }
