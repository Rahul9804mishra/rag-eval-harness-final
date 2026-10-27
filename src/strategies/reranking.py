"""
Strategy 4: Reranking (cross-encoder).

Takes the candidate pool from hybrid search and re-scores each candidate
by passing the (query, candidate) pair through a cross-encoder. Unlike
bi-encoders (used for embedding), a cross-encoder sees both texts
together, giving more accurate relevance scores at extra compute cost.

The candidate pool is capped at RERANK_MAX_CANDIDATES to keep CPU
inference time acceptable.
"""

from sentence_transformers import CrossEncoder

from src.config import RERANK_MODEL_NAME, RERANK_MAX_CANDIDATES, TOP_K

_reranker_instance = None


def get_reranker():
    """Return a cached CrossEncoder so the model loads only once per process."""
    global _reranker_instance
    if _reranker_instance is None:
        print(f"[reranking] Loading cross-encoder: {RERANK_MODEL_NAME}")
        _reranker_instance = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker_instance


def rerank(query: str, candidates: list, k: int = TOP_K):
    """
    Re-rank ``candidates`` (list of strings) for ``query``.

    Returns a list of the top-k strings, highest relevance first.
    """
    if not candidates:
        return []

    candidates = candidates[:RERANK_MAX_CANDIDATES]
    reranker = get_reranker()
    pairs = [(query, c) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _s in ranked[:k]]
