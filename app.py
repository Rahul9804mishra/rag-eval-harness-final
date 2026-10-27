from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# Ensure src/ is importable when Streamlit runs from the project root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "results.json"
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="RAG Document Chat", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")
st.caption(
    "Ask questions about your indexed documents, inspect retrieved sources, "
    "and review retrieval benchmark results."
)

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
# Maps the display label shown in the dropdown to an internal strategy key.
# All four keys must match the names used in results.json / STRATEGY_NAMES.
STRATEGY_OPTIONS: dict[str, str] = {
    "Simple chunking":   "simple_chunking",
    "Semantic chunking": "semantic_chunking",
    "Hybrid search":     "hybrid_search",
    "Reranking":         "reranking",
}

# Which ChromaDB persist-directories back each strategy.
# Hybrid and reranking reuse chroma_simple (same as eval/run_eval.py does).
STRATEGY_CHROMA_DIRS: dict[str, Path] = {
    "simple_chunking":   ROOT / "chroma_simple",
    "semantic_chunking": ROOT / "chroma_semantic",
    "hybrid_search":     ROOT / "chroma_simple",   # reuses simple index
    "reranking":         ROOT / "chroma_simple",   # reuses simple index
}


def _chroma_dir_ok(strategy_key: str) -> bool:
    """Return True if the required ChromaDB directory exists and is non-empty."""
    d = STRATEGY_CHROMA_DIRS.get(strategy_key)
    return d is not None and d.is_dir()


def available_strategies() -> dict[str, str]:
    """
    Return the subset of STRATEGY_OPTIONS whose backing index exists on disk.
    Always includes all four labels so the dropdown is never empty; strategies
    with missing indexes will show a clear error when selected.
    """
    return STRATEGY_OPTIONS  # show all four; guard errors at query time


# ---------------------------------------------------------------------------
# Helpers shared across strategies
# ---------------------------------------------------------------------------

def source_label(metadata: Any, index: int) -> str:
    if isinstance(metadata, dict) and metadata:
        for key in ("source", "file_path", "filename", "document", "page"):
            if metadata.get(key) is not None:
                return f"{key}: {metadata[key]}"
        return str(metadata)
    return f"Retrieved chunk {index}"


def generate_with_ollama(
    question: str,
    chunks: list[dict[str, Any]],
    model: str,
    base_url: str,
) -> str:
    context = "\n\n".join(
        f"[Source {i + 1}: {source_label(item['metadata'], i + 1)}]\n{item['text']}"
        for i, item in enumerate(chunks)
    )
    prompt = (
        "Answer the question using only the supplied context. If the context does not "
        "contain the answer, say you could not find the answer in the indexed documents. "
        "Do not invent facts. Cite relevant sources as [Source 1], [Source 2], etc.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")
    return answer


# ---------------------------------------------------------------------------
# Per-strategy retrieval
# ---------------------------------------------------------------------------

def _load_vectorstore(strategy_key: str):
    """
    Load the LangChain Chroma vectorstore for a strategy.
    Returns a Chroma instance.  Raises if the directory is missing.
    """
    from langchain_community.vectorstores import Chroma
    from src.embed_store import get_embeddings

    persist_dir = STRATEGY_CHROMA_DIRS[strategy_key]
    if not persist_dir.is_dir():
        raise FileNotFoundError(
            f"Index directory not found: {persist_dir}\n"
            "Run `python eval/run_eval.py` from the project root to build all indexes."
        )
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    if vectorstore._collection.count() == 0:
        raise ValueError(
            f"The index at {persist_dir} is empty.\n"
            "Re-run `python eval/run_eval.py` to rebuild it."
        )
    return vectorstore


def _chunks_to_display(raw_results, strategy_key: str) -> list[dict[str, Any]]:
    """
    Normalise retrieval results from any strategy into a uniform list of
    dicts with keys: text, metadata, distance.

    - simple_chunking / semantic_chunking: List[LangChain Document]
    - hybrid_search: List[(text: str, score: float)]
    - reranking: List[str]
    """
    out: list[dict[str, Any]] = []
    if strategy_key in ("simple_chunking", "semantic_chunking"):
        for doc in raw_results:
            out.append({
                "text": doc.page_content.lstrip("\ufeff").strip(),
                "metadata": doc.metadata,
                "distance": None,
            })
    elif strategy_key == "hybrid_search":
        for text, score in raw_results:
            out.append({
                "text": text.lstrip("\ufeff").strip(),
                "metadata": {},
                "distance": round(1.0 - score, 4),   # convert fused score to pseudo-distance
            })
    elif strategy_key == "reranking":
        for text in raw_results:
            out.append({
                "text": text.lstrip("\ufeff").strip(),
                "metadata": {},
                "distance": None,
            })
    return out


@st.cache_resource(show_spinner="Loading BM25 index…")
def _get_bm25_index():
    """
    Build (or return cached) BM25 index over the simple-chunking chunks.

    Uses Streamlit's resource cache so the index is only built once per
    app session.  Returns (bm25, chunks).
    """
    from src.strategies.simple_chunking import build_simple_index
    from src.strategies.hybrid_search import build_bm25_index
    from src.ingest import load_documents

    docs = load_documents()
    vs, chunks = build_simple_index(docs)
    bm25, chunks = build_bm25_index(chunks)
    return vs, bm25, chunks


def retrieve_for_strategy(
    strategy_key: str,
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Dispatch to the correct retrieval pipeline for *strategy_key*.

    Raises on missing index or import failure so the caller can surface
    a clear error rather than silently falling back.
    """
    if strategy_key == "simple_chunking":
        from src.strategies.simple_chunking import retrieve_simple
        vs = _load_vectorstore("simple_chunking")
        raw = retrieve_simple(vs, question, k=top_k)
        return _chunks_to_display(raw, strategy_key)

    if strategy_key == "semantic_chunking":
        from src.strategies.semantic_chunking import retrieve_semantic
        vs = _load_vectorstore("semantic_chunking")
        raw = retrieve_semantic(vs, question, k=top_k)
        return _chunks_to_display(raw, strategy_key)

    if strategy_key == "hybrid_search":
        from src.strategies.hybrid_search import retrieve_hybrid
        vs = _load_vectorstore("hybrid_search")
        # BM25 index is cached; we pull it from the shared session cache.
        try:
            _, bm25, chunks = _get_bm25_index()
        except Exception as exc:
            raise RuntimeError(
                f"Could not build BM25 index for hybrid search: {exc}"
            ) from exc
        raw = retrieve_hybrid(vs, bm25, chunks, question, k=top_k)
        return _chunks_to_display(raw, strategy_key)

    if strategy_key == "reranking":
        from src.strategies.hybrid_search import retrieve_hybrid
        from src.strategies.reranking import rerank
        vs = _load_vectorstore("reranking")
        try:
            _, bm25, chunks = _get_bm25_index()
        except Exception as exc:
            raise RuntimeError(
                f"Could not build BM25 index for reranking: {exc}"
            ) from exc
        # Wider candidate pool for the cross-encoder to re-rank.
        from src.config import TOP_K as CFG_TOP_K
        candidate_pool = max(top_k * 3, CFG_TOP_K * 3)
        raw_hybrid = retrieve_hybrid(vs, bm25, chunks, question, k=candidate_pool)
        candidates = [text for text, _score in raw_hybrid]
        reranked = rerank(question, candidates, k=top_k)
        return _chunks_to_display(reranked, strategy_key)

    raise ValueError(f"Unknown strategy key: {strategy_key!r}")


# ---------------------------------------------------------------------------
# Evaluation Dashboard
# ---------------------------------------------------------------------------

def load_results() -> dict[str, float]:
    """Read results.json (legacy flat summary written by run_eval.py)."""
    if not RESULTS_PATH.exists():
        return {}
    try:
        raw = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return {
            str(name): float(score)
            for name, score in raw.items()
            if isinstance(score, (int, float))
        }
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def load_latest_summary() -> dict | None:
    """
    Return the most recent summary_<timestamp>.json from the results/ folder,
    or None if no summaries exist.
    """
    if not RESULTS_DIR.is_dir():
        return None
    summaries = sorted(RESULTS_DIR.glob("summary_*.json"))
    if not summaries:
        return None
    try:
        return json.loads(summaries[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def render_results_dashboard() -> None:
    st.subheader("Retrieval benchmark")
    scores = load_results()
    if not scores:
        st.info(
            "No readable results.json was found. "
            "Run `python eval/run_eval.py` from the project root first."
        )
        return

    # ---- Top-level metric cards (one per strategy) -----------------------
    cols = st.columns(len(scores))
    for col, (strategy, score) in zip(cols, scores.items()):
        col.metric(strategy.replace("_", " ").title(), f"{score:.1f}%")

    # ---- Bar chart -------------------------------------------------------
    frame = pd.DataFrame({
        "Strategy": [name.replace("_", " ").title() for name in scores.keys()],
        "Hit Rate (%)": list(scores.values()),
    })
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(frame["Strategy"], frame["Hit Rate (%)"])
    ax.set_title("Retrieval Hit Rate by Strategy")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, max(100, max(scores.values(), default=0) + 12))
    ax.tick_params(axis="x", labelrotation=15)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ---- Detailed metrics from the latest summary JSON -------------------
    summary = load_latest_summary()
    if summary:
        st.markdown("### Detailed metrics (latest run)")
        metric_rows = []
        for strat, metrics in summary.items():
            metric_rows.append({
                "Strategy": strat.replace("_", " ").title(),
                "Questions": metrics.get("n_questions", "-"),
                "Hit Rate %": f"{metrics.get('hit_rate_pct', 0):.1f}",
                "MRR": f"{metrics.get('mrr', 0):.4f}",
                "Avg Recall %": f"{metrics.get('avg_recall_at_k', 0):.1f}",
                "Avg Precision %": f"{metrics.get('avg_precision_at_k', 0):.1f}",
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.caption(
        "Scores are read from results.json and the latest summary_*.json. "
        "Re-run `python eval/run_eval.py` to refresh."
    )


# ---------------------------------------------------------------------------
# Document Chat
# ---------------------------------------------------------------------------

def render_chat() -> None:
    st.subheader("Ask your documents")

    all_strategies = available_strategies()  # always all 4

    with st.sidebar:
        st.header("Chat settings")
        strategy_label = st.selectbox("Retrieval index", list(all_strategies.keys()))
        strategy_key = all_strategies[strategy_label]

        # Show index availability indicator
        if _chroma_dir_ok(strategy_key):
            st.success(f"Index ready: `{STRATEGY_CHROMA_DIRS[strategy_key].name}/`")
        else:
            st.warning(
                f"Index not found: `{STRATEGY_CHROMA_DIRS[strategy_key].name}/`\n\n"
                "Run `python eval/run_eval.py` to build it."
            )

        top_k = st.slider("Retrieved chunks (Top-K)", 1, 10, 4)
        st.divider()
        st.subheader("Answer generation")
        use_ollama = st.toggle("Generate an answer with Ollama", value=True)
        ollama_model = st.text_input(
            "Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.2")
        )
        ollama_url = st.text_input(
            "Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

    # Strategy description blurb shown below the subheader
    STRATEGY_DESCRIPTIONS = {
        "simple_chunking":   "Fixed-size character splitting → vector similarity search.",
        "semantic_chunking": "Meaning-boundary splitting → vector similarity search.",
        "hybrid_search":     "BM25 keyword scores fused with vector similarity (α=0.6).",
        "reranking":         "Hybrid search pool re-ranked by a cross-encoder model.",
    }
    st.info(
        f"**{strategy_label}** — {STRATEGY_DESCRIPTIONS.get(strategy_key, '')}\n\n"
        "For a natural-language answer, start Ollama locally and download a model. "
        "If Ollama is unavailable, the app still shows the retrieved evidence."
    )

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    with st.spinner(f"Searching with *{strategy_label}*…"):
        try:
            chunks = retrieve_for_strategy(strategy_key, question, top_k)
        except FileNotFoundError as exc:
            st.error(
                f"**Index not found** — {strategy_key}\n\n"
                f"{exc}\n\n"
                "Run `python eval/run_eval.py` from the project root to build "
                "all four indexes, then reload this page."
            )
            return
        except (ValueError, RuntimeError) as exc:
            st.error(f"**Retrieval error** ({strategy_key}): {exc}")
            return
        except Exception as exc:
            st.error(
                f"**Unexpected error** while retrieving with `{strategy_key}`:\n\n{exc}"
            )
            return

    if not chunks:
        st.warning("No document chunks were returned for this question.")
        return

    st.markdown(f"**Question:** {question}")
    st.caption(
        f"Strategy: **{strategy_label}** (`{strategy_key}`) · Retrieved {len(chunks)} chunk(s)"
    )

    if use_ollama:
        try:
            with st.spinner(f"Generating a grounded answer with {ollama_model}…"):
                answer = generate_with_ollama(question, chunks, ollama_model, ollama_url)
            st.markdown("### Answer")
            st.write(answer)
        except requests.RequestException as exc:
            st.warning(
                f"Could not connect to Ollama at `{ollama_url}` ({exc}). "
                "Start Ollama and make sure the model is installed, or turn off "
                "answer generation to inspect the retrieved chunks."
            )
        except (ValueError, RuntimeError) as exc:
            st.warning(f"Answer generation failed: {exc}")

    st.markdown("### Retrieved evidence")
    for i, item in enumerate(chunks, start=1):
        distance = item.get("distance")
        if strategy_key == "hybrid_search" and distance is not None:
            score_text = f" · Fused score: {1.0 - distance:.4f}"
        elif isinstance(distance, (int, float)):
            score_text = f" · Distance: {distance:.4f}"
        else:
            score_text = ""
        label = source_label(item["metadata"], i) if item["metadata"] else f"Chunk {i}"
        with st.expander(f"Source {i}: {label}{score_text}", expanded=(i == 1)):
            st.write(item["text"])
            if item["metadata"]:
                st.json(item["metadata"])


# ---------------------------------------------------------------------------
# Page router  (only runs when Streamlit executes this file as a script)
# ---------------------------------------------------------------------------

def _run_app() -> None:
    """Entry-point called by Streamlit when it executes this file."""
    with st.sidebar:
        st.caption("Local project · Streamlit + ChromaDB")
        page = st.radio("Navigation", ["💬 Document Chat", "📊 Evaluation Dashboard"])

    if page == "💬 Document Chat":
        render_chat()
    else:
        render_results_dashboard()


# Streamlit re-runs this script on every interaction.  Guard module-level
# execution so that `import app` from tests does not trigger the UI code.
# We detect a live Streamlit execution by checking for the script runner
# context.  If it is absent (plain import / pytest), skip the UI entirely.
_IN_STREAMLIT = False
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    _IN_STREAMLIT = _get_ctx() is not None
except Exception:
    pass

if _IN_STREAMLIT:
    _run_app()
