"""
Evaluation Harness — runs all ground-truth questions through all four
retrieval strategies and reports multiple retrieval metrics.

Metrics computed per strategy
------------------------------
Hit Rate / Success@K    : fraction of questions where the answer was found
                          in the top-k retrieved chunks.
MRR                     : Mean Reciprocal Rank.
Avg Recall@K            : token-level recall averaged over all questions.
Avg Precision@K         : fraction of top-k chunks containing the answer.

Outputs
-------
results/eval_<timestamp>.csv   — full per-question breakdown
results/summary_<timestamp>.json — aggregate metrics per strategy
results.json                    — legacy flat summary (for check_threshold.py)

Run from the project root:
    python eval/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# Allow running as `python eval/run_eval.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import (
    GROUND_TRUTH_PATH,
    RESULTS_PATH,
    RESULTS_DIR,
    STRATEGY_NAMES,
    TOP_K,
)
from src.ingest import load_documents
from src.metrics import (
    hit_at_k,
    reciprocal_rank,
    recall_at_k,
    precision_at_k,
    compute_summary,
)
from src.strategies import simple_chunking, semantic_chunking, hybrid_search, reranking


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

PLACEHOLDER_MARKERS = [
    "replace this row",
    "add your own",
    "your question here",
    "placeholder",
]


def _is_placeholder(row: pd.Series) -> bool:
    for col in ("question", "expected_answer"):
        val = str(row.get(col, "")).lower()
        if any(marker in val for marker in PLACEHOLDER_MARKERS):
            return True
    return False


def load_valid_questions(gt_path: str) -> pd.DataFrame:
    """
    Load the ground-truth CSV, drop placeholder rows, and validate the
    remainder has the required columns and non-empty answers.

    Raises
    ------
    FileNotFoundError  — if the CSV does not exist.
    ValueError         — if there are no valid rows after filtering.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Ground-truth file not found: {gt_path}\n"
            f"Run: python eval/validate_ground_truth.py"
        )

    gt = pd.read_csv(gt_path)
    required_cols = ["question", "expected_answer"]
    missing = [c for c in required_cols if c not in gt.columns]
    if missing:
        raise ValueError(
            f"Ground-truth CSV is missing required columns: {missing}\n"
            f"Expected columns: question, expected_answer, source_document, source_page"
        )

    before = len(gt)
    if len(gt) > 0:
        gt = gt[~gt.apply(_is_placeholder, axis=1)].copy()
    if "expected_answer" in gt.columns and len(gt) > 0:
        gt = gt[gt["expected_answer"].notna() & (gt["expected_answer"].astype(str) != "")]
    if "question" in gt.columns and len(gt) > 0:
        gt = gt[gt["question"].notna() & (gt["question"].astype(str) != "")]
    gt = gt.drop_duplicates(subset=["question"])
    gt = gt.reset_index(drop=True)

    filtered = before - len(gt)
    print(f"[eval] Loaded {len(gt)} valid questions "
          f"({filtered} placeholder/invalid rows removed).")

    if len(gt) == 0:
        raise ValueError(
            "No valid questions remain after filtering placeholders.\n"
            "Fill in eval/ground_truth.csv with real Q&A pairs."
        )
    return gt


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_all_indexes():
    """
    Build every vector/BM25 index once so the evaluation loop only does
    retrieval (fast) rather than rebuilding per question.
    """
    docs = load_documents()

    print("\nBuilding simple_chunking index …")
    simple_vs, simple_chunks = simple_chunking.build_simple_index(docs)

    print("\nBuilding semantic_chunking index …")
    semantic_vs, _ = semantic_chunking.build_semantic_index(docs)

    print("\nBuilding BM25 index over simple chunks …")
    bm25, chunks = hybrid_search.build_bm25_index(simple_chunks)

    return {
        "simple_vs": simple_vs,
        "semantic_vs": semantic_vs,
        "bm25": bm25,
        "chunks": chunks,
        "hybrid_vs": simple_vs,   # hybrid fuses BM25 with the same vectorstore
    }


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(gt_path: str = GROUND_TRUTH_PATH) -> dict:
    """
    Evaluate all strategies and return the summary dict.
    """
    gt = load_valid_questions(gt_path)
    idx = build_all_indexes()

    # Per-question row storage: list of dicts keyed by strategy name
    rows = {name: [] for name in STRATEGY_NAMES}

    print(f"\n[eval] Evaluating {len(gt)} questions × {len(STRATEGY_NAMES)} strategies …\n")

    for q_idx, row in gt.iterrows():
        query = str(row["question"])
        expected = str(row["expected_answer"])
        source_doc = str(row.get("source_document", ""))

        # ---- 1. Simple chunking -------------------------------------------
        simple_hits = simple_chunking.retrieve_simple(
            idx["simple_vs"], query, k=TOP_K
        )
        rows["simple_chunking"].append({
            "q_id": q_idx,
            "question": query,
            "expected_answer": expected,
            "source_document": source_doc,
            "hit": hit_at_k(simple_hits, expected),
            "rr": reciprocal_rank(simple_hits, expected),
            "recall_k": recall_at_k(simple_hits, expected),
            "precision_k": precision_at_k(simple_hits, expected),
            "retrieved_texts": " ||| ".join(
                c.page_content[:120] for c in simple_hits
            ),
        })

        # ---- 2. Semantic chunking -----------------------------------------
        semantic_hits = semantic_chunking.retrieve_semantic(
            idx["semantic_vs"], query, k=TOP_K
        )
        rows["semantic_chunking"].append({
            "q_id": q_idx,
            "question": query,
            "expected_answer": expected,
            "source_document": source_doc,
            "hit": hit_at_k(semantic_hits, expected),
            "rr": reciprocal_rank(semantic_hits, expected),
            "recall_k": recall_at_k(semantic_hits, expected),
            "precision_k": precision_at_k(semantic_hits, expected),
            "retrieved_texts": " ||| ".join(
                c.page_content[:120] for c in semantic_hits
            ),
        })

        # ---- 3. Hybrid search ---------------------------------------------
        hybrid_hits = hybrid_search.retrieve_hybrid(
            idx["hybrid_vs"], idx["bm25"], idx["chunks"],
            query, k=TOP_K
        )
        hybrid_texts = [text for text, _score in hybrid_hits]
        rows["hybrid_search"].append({
            "q_id": q_idx,
            "question": query,
            "expected_answer": expected,
            "source_document": source_doc,
            "hit": hit_at_k(hybrid_texts, expected),
            "rr": reciprocal_rank(hybrid_texts, expected),
            "recall_k": recall_at_k(hybrid_texts, expected),
            "precision_k": precision_at_k(hybrid_texts, expected),
            "retrieved_texts": " ||| ".join(t[:120] for t in hybrid_texts),
        })

        # ---- 4. Reranking -------------------------------------------------
        # Wider hybrid pool → reranker picks the best k from it
        rerank_candidates_raw = hybrid_search.retrieve_hybrid(
            idx["hybrid_vs"], idx["bm25"], idx["chunks"],
            query, k=TOP_K * 3
        )
        rerank_candidate_texts = [t for t, _s in rerank_candidates_raw]
        reranked_texts = reranking.rerank(query, rerank_candidate_texts, k=TOP_K)
        rows["reranking"].append({
            "q_id": q_idx,
            "question": query,
            "expected_answer": expected,
            "source_document": source_doc,
            "hit": hit_at_k(reranked_texts, expected),
            "rr": reciprocal_rank(reranked_texts, expected),
            "recall_k": recall_at_k(reranked_texts, expected),
            "precision_k": precision_at_k(reranked_texts, expected),
            "retrieved_texts": " ||| ".join(t[:120] for t in reranked_texts),
        })

    # ---- Compute summaries ------------------------------------------------
    summaries = {}
    for name in STRATEGY_NAMES:
        summaries[name] = compute_summary(rows[name])

    # ---- Save outputs -------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Per-question CSV
    all_rows = []
    for name in STRATEGY_NAMES:
        for r in rows[name]:
            all_rows.append({"strategy": name, **r})
    detail_csv_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.csv")
    pd.DataFrame(all_rows).to_csv(detail_csv_path, index=False)
    print(f"\n[eval] Per-question results saved → {detail_csv_path}")

    # 2. Summary JSON
    summary_json_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.json")
    with open(summary_json_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[eval] Summary JSON saved       → {summary_json_path}")

    # 3. Legacy results.json for check_threshold.py / plot_results.py
    legacy = {name: summaries[name]["hit_rate_pct"] for name in STRATEGY_NAMES}
    with open(RESULTS_PATH, "w") as f:
        json.dump(legacy, f, indent=2)
    print(f"[eval] Legacy results.json saved → {RESULTS_PATH}")

    # ---- Print to console -------------------------------------------------
    n_q = len(gt)
    print(f"\n{'='*60}")
    print(f"  Evaluation Results  ({n_q} questions, Top-{TOP_K})")
    print(f"{'='*60}")
    header = f"{'Strategy':<22} {'Hit%':>6} {'MRR':>6} {'Rec%':>6} {'Pre%':>6}"
    print(header)
    print("-" * len(header))
    for name in STRATEGY_NAMES:
        s = summaries[name]
        print(
            f"{name:<22} {s['hit_rate_pct']:>5.1f}%"
            f" {s['mrr']:>6.4f}"
            f" {s['avg_recall_at_k']:>5.1f}%"
            f" {s['avg_precision_at_k']:>5.1f}%"
        )
    print(f"{'='*60}\n")

    return summaries


if __name__ == "__main__":
    run_evaluation()
