"""
Strategy 2: Semantic Chunking.

Instead of splitting every N characters, this splits on meaning
boundaries — where consecutive sentences become semantically dissimilar
according to the embedding model itself.  Produces fewer, more coherent
chunks than fixed-size splitting at the cost of slightly more ingestion time.
"""

import shutil

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma

from src.config import SEMANTIC_BREAKPOINT_TYPE, CHROMA_SEMANTIC_DIR, TOP_K
from src.embed_store import get_embeddings


def build_semantic_index(docs, persist_dir: str = CHROMA_SEMANTIC_DIR):
    """
    Split ``docs`` on semantic boundaries and build a Chroma vector store.

    Returns
    -------
    vectorstore : Chroma
    chunks : list[Document]
    """
    # Remove stale persisted data.
    if persist_dir and __import__("os").path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = get_embeddings()
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
    )
    chunks = splitter.split_documents(docs)

    # SemanticChunker can occasionally produce very large or empty chunks
    # (documented pitfall).  Filter out anything shorter than 30 characters.
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 30]

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["strategy"] = "semantic"

    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_dir
    )
    print(f"[semantic_chunking] indexed {len(chunks)} chunks")
    return vectorstore, chunks


def retrieve_semantic(vectorstore, query: str, k: int = TOP_K):
    """Return the top-k most similar chunks for ``query``."""
    return vectorstore.similarity_search(query, k=k)
