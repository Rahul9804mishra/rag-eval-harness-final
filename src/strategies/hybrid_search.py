"""
Strategy 3: Hybrid Search (BM25 keyword search + vector similarity).

Combines exact keyword matching (BM25) with semantic similarity so that
queries containing names, numbers, or codes that pure embeddings sometimes
miss are still retrieved.

Fusion uses min-max normalisation of both score distributions before
weighted combination, avoiding the bug in the original code where raw
BM25 scores (which are unbounded) dominated the fused rank.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import TOP_K, HYBRID_CANDIDATE_MULTIPLIER, HYBRID_ALPHA


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    """Consistent lowercase tokenisation for indexing and querying."""
    return text.lower().split()


def _min_max_normalize(arr):
    """Normalise an array to [0, 1]. Returns zeros if the range is zero."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_bm25_index(chunks):
    """
    Build a BM25 index over ``chunks``.

    Returns
    -------
    bm25 : BM25Okapi
    chunks : list[Document]  — returned unchanged for convenience
    """
    tokenized = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"[hybrid_search] BM25 index built over {len(chunks)} chunks")
    return bm25, chunks


def retrieve_hybrid(
    vectorstore,
    bm25,
    chunks,
    query: str,
    k: int = TOP_K,
    alpha: float = HYBRID_ALPHA,
):
    """
    Retrieve the top-k chunks by fusing vector and BM25 scores.

    Both score distributions are min-max normalised to [0, 1] before
    weighting, so neither dominates due to scale differences.

    Returns a list of (chunk_text, fused_score) tuples, highest score first.
    """
    candidate_pool = k * HYBRID_CANDIDATE_MULTIPLIER

    # ---- Vector similarity scores ----------------------------------------
    # similarity_search_with_score returns (doc, distance) where lower
    # distance = more similar for the default L2 metric.
    vector_hits = vectorstore.similarity_search_with_score(query, k=candidate_pool)

    # Build a content -> raw_vector_score map (convert distance to similarity)
    vector_score_map = {}
    for doc, dist in vector_hits:
        # 1 / (1 + dist) maps [0, ∞) → (0, 1] monotonically
        sim = 1.0 / (1.0 + dist)
        content = doc.page_content
        if content not in vector_score_map or sim > vector_score_map[content]:
            vector_score_map[content] = sim

    # ---- BM25 scores -------------------------------------------------------
    bm25_raw = np.array(bm25.get_scores(_tokenize(query)))
    bm25_norm = _min_max_normalize(bm25_raw)

    # Top BM25 candidates
    bm25_top_idx = np.argsort(bm25_raw)[::-1][:candidate_pool]
    bm25_score_map = {
        chunks[i].page_content: float(bm25_norm[i]) for i in bm25_top_idx
    }

    # ---- Normalise vector scores -------------------------------------------
    if vector_score_map:
        vec_arr = np.array(list(vector_score_map.values()))
        vec_norm_arr = _min_max_normalize(vec_arr)
        vector_score_map = {
            k: float(v)
            for k, v in zip(vector_score_map.keys(), vec_norm_arr)
        }

    # ---- Fuse ------------------------------------------------------------------
    all_contents = set(vector_score_map) | set(bm25_score_map)
    combined = {}
    for content in all_contents:
        v_score = vector_score_map.get(content, 0.0)
        b_score = bm25_score_map.get(content, 0.0)
        combined[content] = alpha * v_score + (1.0 - alpha) * b_score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return ranked
