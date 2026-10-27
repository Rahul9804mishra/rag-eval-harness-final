"""
Central configuration for the RAG Evaluation Harness project.

All tunable values live here. Other modules import from this file
rather than hardcoding constants.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "eval", "ground_truth.csv")

# Output artefacts (legacy flat summary — kept for backward compat with
# check_threshold.py and plot_results.py)
RESULTS_PATH = os.path.join(BASE_DIR, "results.json")
RESULTS_CHART_PATH = os.path.join(BASE_DIR, "results_chart.png")

# Detailed per-run outputs go into results/
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Persistent Chroma vector-store directories — separate so the two
# chunk strategies never overwrite each other.
CHROMA_SIMPLE_DIR = os.path.join(BASE_DIR, "chroma_simple")
CHROMA_SEMANTIC_DIR = os.path.join(BASE_DIR, "chroma_semantic")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# Free, local, no API key required.
# Swap for "text-embedding-3-small" (OpenAIEmbeddings) for the paid option.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
SIMPLE_CHUNK_SIZE = 400       # characters
SIMPLE_CHUNK_OVERLAP = 80     # characters
SEMANTIC_BREAKPOINT_TYPE = "percentile"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5                          # chunks to return per query
HYBRID_CANDIDATE_MULTIPLIER = 3    # BM25 candidate pool = TOP_K * this
HYBRID_ALPHA = 0.6                 # vector weight (1-alpha = BM25 weight)
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MAX_CANDIDATES = 20         # cap for CPU reranking

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# Minimum accuracy the BEST strategy must achieve for CI to pass.
ACCURACY_THRESHOLD = 60.0

STRATEGY_NAMES = [
    "simple_chunking",
    "semantic_chunking",
    "hybrid_search",
    "reranking",
]
