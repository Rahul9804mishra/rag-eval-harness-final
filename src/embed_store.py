"""
Shared embedding model helper.

Both the simple-chunking and semantic-chunking strategies need an
embedding model. Defining it in exactly one place means every strategy
uses a consistent model, and swapping models (e.g. to an OpenAI model)
only requires editing this file.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME

_embeddings_instance = None


def get_embeddings():
    """
    Return a cached HuggingFaceEmbeddings instance so the (fairly large)
    model is only loaded into memory once per process, even if multiple
    strategies call this function.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_instance
