"""
Loads raw source documents (.pdf or .txt) from data/documents/.

This is the first stage of the pipeline:
  Source Documents → Chunking Layer → Vector Store → Retrieval → Eval
"""

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)

from src.config import DOCUMENTS_DIR


def load_documents(path: str = DOCUMENTS_DIR):
    """
    Load every PDF and .txt file found under ``path``.

    Returns a list of LangChain Document objects (one per PDF page or one
    per .txt file), each carrying ``.page_content`` and ``.metadata``.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no supported documents are found inside the directory.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Document folder not found: {path}\n"
            f"Create it and place your .txt/.pdf source files there.\n"
            f"  Windows:  mkdir \"{path}\""
        )

    docs = []

    # Load PDFs
    try:
        pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs.extend(pdf_loader.load())
    except Exception as exc:
        print(f"[ingest] Warning: error loading PDFs: {exc}")

    # Load plain-text files
    try:
        txt_loader = DirectoryLoader(
            path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs.extend(txt_loader.load())
    except Exception as exc:
        print(f"[ingest] Warning: error loading TXT files: {exc}")

    if not docs:
        raise ValueError(
            f"No .pdf or .txt files found in {path}.\n"
            f"Add your source documents there and retry."
        )

    # Attach a short readable source identifier to each document
    for doc in docs:
        src = doc.metadata.get("source", "")
        doc.metadata["source_filename"] = os.path.basename(src)

    print(f"[ingest] Loaded {len(docs)} document page(s)/file(s) from {path}")
    return docs


if __name__ == "__main__":
    loaded = load_documents()
    for d in loaded[:3]:
        print("---")
        print(d.metadata)
        print(d.page_content[:300])
