from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.json"
CHART_PATH = ROOT / "results_chart.png"

st.set_page_config(page_title="RAG Document Chat", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")
st.caption("Ask questions about your indexed documents, inspect retrieved sources, and review retrieval benchmark results.")

# ---------- Helpers ----------
def load_results() -> dict[str, float]:
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


def available_stores() -> dict[str, Path]:
    candidates = {
        "Simple chunking": ROOT / "chroma_simple",
        "Semantic chunking": ROOT / "chroma_semantic",
    }
    return {name: path for name, path in candidates.items() if path.is_dir()}


def get_collection(store_path: Path):
    client = chromadb.PersistentClient(path=str(store_path))
    collections = client.list_collections()
    if not collections:
        return None, None
    # Prefer a non-empty collection, rather than blindly selecting the first.
    for item in collections:
        collection = client.get_collection(name=item.name)
        if collection.count() > 0:
            return collection, item.name
    return client.get_collection(name=collections[0].name), collections[0].name


def retrieve(collection, query: str, top_k: int) -> list[dict[str, Any]]:
    count = collection.count()
    if count < 1:
        return []
    response = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    documents = (response.get("documents") or [[]])[0] or []
    metadatas = (response.get("metadatas") or [[]])[0] or []
    distances = (response.get("distances") or [[]])[0] or []
    found = []
    for i, document in enumerate(documents):
        if not document:
            continue
        found.append({
            "text": document.lstrip("\ufeff").strip(),
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "distance": distances[i] if i < len(distances) else None,
        })
    return found


def source_label(metadata: Any, index: int) -> str:
    if isinstance(metadata, dict) and metadata:
        for key in ("source", "file_path", "filename", "document", "page"):
            if metadata.get(key) is not None:
                return f"{key}: {metadata[key]}"
        return str(metadata)
    return f"Retrieved chunk {index}"


def generate_with_ollama(question: str, chunks: list[dict[str, Any]], model: str, base_url: str) -> str:
    context = "\n\n".join(
        f"[Source {i + 1}: {source_label(item['metadata'], i + 1)}]\n{item['text']}"
        for i, item in enumerate(chunks)
    )
    prompt = (
        "Answer the question using only the supplied context. If the context does not contain "
        "the answer, say you could not find the answer in the indexed documents. Do not invent "
        "facts. Cite relevant sources as [Source 1], [Source 2], etc.\n\n"
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


def render_results_dashboard() -> None:
    st.subheader("Retrieval benchmark")
    scores = load_results()
    if not scores:
        st.info("No readable results.json was found. Run `python eval/run_eval.py` from the project root first.")
        return

    cols = st.columns(len(scores))
    for col, (strategy, score) in zip(cols, scores.items()):
        col.metric(strategy.replace("_", " ").title(), f"{score:.1f}%")

    # Plot directly from the current JSON so the dashboard cannot show a stale PNG.
    frame = pd.DataFrame({
        "Strategy": [name.replace("_", " ") for name in scores.keys()],
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
    st.caption("Scores are read from results.json. Re-run the evaluation to refresh them.")


def render_chat() -> None:
    st.subheader("Ask your documents")
    stores = available_stores()
    if not stores:
        st.error(
            "No supported Chroma index folders were found. Expected `chroma_simple/` or "
            "`chroma_semantic/` beside app.py. Build the indexes with your backend first."
        )
        return

    with st.sidebar:
        st.header("Chat settings")
        store_name = st.selectbox("Retrieval index", list(stores.keys()))
        top_k = st.slider("Retrieved chunks (Top-K)", 1, 10, 4)
        st.divider()
        st.subheader("Answer generation")
        use_ollama = st.toggle("Generate an answer with Ollama", value=True)
        ollama_model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.2"))
        ollama_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    st.info(
        "Retrieval searches your existing Chroma index. For a natural-language answer, start Ollama "
        "locally and download a model. If Ollama is unavailable, the app will still show the retrieved evidence."
    )
    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    with st.spinner("Searching indexed documents…"):
        try:
            collection, collection_name = get_collection(stores[store_name])
            if collection is None:
                st.warning(f"The index `{store_name}` has no Chroma collections. Run your ingestion/indexing step.")
                return
            chunks = retrieve(collection, question, top_k)
        except Exception as exc:
            st.error(
                f"Could not query this index: {exc}\n\n"
                "Check that the index exists and that the app uses the same embedding function/model "
                "as the backend that created it."
            )
            return

    if not chunks:
        st.warning("No document chunks were returned for this question.")
        return

    st.markdown(f"**Question:** {question}")
    st.caption(f"Index: {store_name} · Collection: `{collection_name}` · Retrieved {len(chunks)} chunk(s)")

    if use_ollama:
        try:
            with st.spinner(f"Generating a grounded answer with {ollama_model}…"):
                answer = generate_with_ollama(question, chunks, ollama_model, ollama_url)
            st.markdown("### Answer")
            st.write(answer)
        except requests.RequestException as exc:
            st.warning(
                f"Could not connect to Ollama at `{ollama_url}` ({exc}). "
                "Start Ollama, make sure the model is installed, or turn off answer generation to inspect the retrieved chunks."
            )
        except (ValueError, RuntimeError) as exc:
            st.warning(f"Answer generation failed: {exc}")

    st.markdown("### Retrieved evidence")
    for i, item in enumerate(chunks, start=1):
        distance = item["distance"]
        distance_text = f" · Distance: {distance:.4f}" if isinstance(distance, (int, float)) else ""
        with st.expander(f"Source {i}: {source_label(item['metadata'], i)}{distance_text}", expanded=(i == 1)):
            st.write(item["text"])
            if item["metadata"]:
                st.json(item["metadata"])


with st.sidebar:
    st.caption("Local project · Streamlit + ChromaDB")
    page = st.radio("Navigation", ["💬 Document Chat", "📊 Evaluation Dashboard"])

if page == "💬 Document Chat":
    render_chat()
else:
    render_results_dashboard()
