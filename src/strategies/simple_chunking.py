"""
Strategy 1: Simple (fixed-size) Chunking — the baseline.

Splits documents into fixed-character-length chunks with a small overlap,
embeds them, and stores them in a dedicated Chroma collection.

Each chunk retains the source document metadata so evaluation can track
which document a chunk came from.
"""

import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.config import (
    SIMPLE_CHUNK_SIZE,
    SIMPLE_CHUNK_OVERLAP,
    CHROMA_SIMPLE_DIR,
    TOP_K,
)
from src.embed_store import get_embeddings


def build_simple_index(docs, persist_dir: str = CHROMA_SIMPLE_DIR):
    """
    Split ``docs`` into fixed-size chunks and build a Chroma vector store.

    Returns
    -------
    vectorstore : Chroma
        The populated vector store ready for similarity search.
    chunks : list[Document]
        The individual chunks (needed by hybrid_search to build BM25).
    """
    # Remove stale persisted data so we always work with fresh content.
    if persist_dir and __import__("os").path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SIMPLE_CHUNK_SIZE,
        chunk_overlap=SIMPLE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Tag each chunk with its strategy name for downstream diagnostics.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["strategy"] = "simple"

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_dir
    )
    print(f"[simple_chunking] indexed {len(chunks)} chunks "
          f"(chunk_size={SIMPLE_CHUNK_SIZE}, overlap={SIMPLE_CHUNK_OVERLAP})")
    return vectorstore, chunks


def retrieve_simple(vectorstore, query: str, k: int = TOP_K):
    """Return the top-k most similar chunks for ``query``."""
    return vectorstore.similarity_search(query, k=k)
