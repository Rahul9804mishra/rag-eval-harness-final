"""
Lightweight smoke tests for each retrieval strategy.

These tests use a tiny in-memory corpus so they run quickly without
downloading any additional models (all-MiniLM-L6-v2 must already be
cached or be downloadable).  They do NOT assert specific accuracy numbers —
only that the strategies return non-empty, correctly-typed results.
"""
import os
import sys
import pytest
import tempfile
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Shared tiny corpus
# ---------------------------------------------------------------------------

TINY_CORPUS = [
    Document(
        page_content=(
            "NovaCorp Technologies was founded in 2008 and is headquartered "
            "in Austin, Texas. The CEO is Alexandra Reyes."
        ),
        metadata={"source": "handbook.txt", "source_filename": "handbook.txt"},
    ),
    Document(
        page_content=(
            "The minimum attendance percentage required for students is 75%. "
            "Remote work is allowed up to 3 days per week."
        ),
        metadata={"source": "handbook.txt", "source_filename": "handbook.txt"},
    ),
    Document(
        page_content=(
            "Annual L&D budget is $2,000 per employee. "
            "Passwords must be at least 12 characters long."
        ),
        metadata={"source": "it_policy.txt", "source_filename": "it_policy.txt"},
    ),
    Document(
        page_content=(
            "Parental leave is 16 weeks for primary caregivers. "
            "PTO accrues at 15 days per year for the first two years of service."
        ),
        metadata={"source": "benefits.txt", "source_filename": "benefits.txt"},
    ),
    Document(
        page_content=(
            "The probationary period lasts 90 days. "
            "Severance is 2 weeks per year of service up to 26 weeks maximum."
        ),
        metadata={"source": "handbook.txt", "source_filename": "handbook.txt"},
    ),
]

QUERY = "What is the minimum attendance percentage?"
EXPECTED = "75%"


# ---------------------------------------------------------------------------
# Strategy: Simple Chunking
# ---------------------------------------------------------------------------

class TestSimpleChunking:
    def test_build_and_retrieve(self, tmp_path):
        from src.strategies.simple_chunking import build_simple_index, retrieve_simple

        vs, chunks = build_simple_index(
            TINY_CORPUS, persist_dir=str(tmp_path / "chroma_simple")
        )
        assert len(chunks) > 0, "Expected at least one chunk"
        results = retrieve_simple(vs, QUERY, k=3)
        assert len(results) > 0, "Expected at least one result"
        for r in results:
            assert hasattr(r, "page_content")
            assert isinstance(r.page_content, str)
            assert len(r.page_content) > 0

    def test_retrieves_relevant_chunk(self, tmp_path):
        from src.strategies.simple_chunking import build_simple_index, retrieve_simple

        vs, _ = build_simple_index(
            TINY_CORPUS, persist_dir=str(tmp_path / "chroma_simple2")
        )
        results = retrieve_simple(vs, QUERY, k=3)
        combined = " ".join(r.page_content for r in results).lower()
        assert "75%" in combined, "Expected 75% to appear in retrieved chunks"


# ---------------------------------------------------------------------------
# Strategy: Semantic Chunking
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    def test_build_and_retrieve(self, tmp_path):
        from src.strategies.semantic_chunking import build_semantic_index, retrieve_semantic

        vs, chunks = build_semantic_index(
            TINY_CORPUS, persist_dir=str(tmp_path / "chroma_semantic")
        )
        assert len(chunks) > 0
        results = retrieve_semantic(vs, QUERY, k=3)
        assert len(results) > 0
        for r in results:
            assert hasattr(r, "page_content")

    def test_filters_tiny_chunks(self, tmp_path):
        from src.strategies.semantic_chunking import build_semantic_index

        _, chunks = build_semantic_index(
            TINY_CORPUS, persist_dir=str(tmp_path / "chroma_semantic2")
        )
        # All indexed chunks should be at least 30 chars
        for c in chunks:
            assert len(c.page_content.strip()) >= 30


# ---------------------------------------------------------------------------
# Strategy: Hybrid Search (BM25 + vector)
# ---------------------------------------------------------------------------

class TestHybridSearch:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from src.strategies.simple_chunking import build_simple_index
        from src.strategies.hybrid_search import build_bm25_index

        self.vs, self.simple_chunks = build_simple_index(
            TINY_CORPUS, persist_dir=str(tmp_path / "chroma_hybrid")
        )
        self.bm25, self.chunks = build_bm25_index(self.simple_chunks)

    def test_bm25_index_built(self):
        assert self.bm25 is not None
        assert len(self.chunks) > 0

    def test_retrieve_returns_tuples(self):
        from src.strategies.hybrid_search import retrieve_hybrid

        results = retrieve_hybrid(self.vs, self.bm25, self.chunks, QUERY, k=3)
        assert len(results) > 0
        for text, score in results:
            assert isinstance(text, str)
            assert isinstance(score, float)
            assert score >= 0.0

    def test_fused_scores_are_normalised(self):
        from src.strategies.hybrid_search import retrieve_hybrid

        results = retrieve_hybrid(self.vs, self.bm25, self.chunks, QUERY, k=3)
        for _text, score in results:
            # Fused scores should be in [0, 1] after normalisation
            assert 0.0 <= score <= 1.0 + 1e-6, (
                f"Fused score {score} is outside [0, 1]"
            )

    def test_retrieves_relevant_content(self):
        from src.strategies.hybrid_search import retrieve_hybrid

        results = retrieve_hybrid(self.vs, self.bm25, self.chunks, QUERY, k=3)
        combined = " ".join(t for t, _ in results).lower()
        assert "75%" in combined


# ---------------------------------------------------------------------------
# Strategy: Reranking
# ---------------------------------------------------------------------------

class TestReranking:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from src.strategies.simple_chunking import build_simple_index
        from src.strategies.hybrid_search import build_bm25_index, retrieve_hybrid

        vs, simple_chunks = build_simple_index(
            TINY_CORPUS, persist_dir=str(tmp_path / "chroma_rerank")
        )
        bm25, chunks = build_bm25_index(simple_chunks)
        raw = retrieve_hybrid(vs, bm25, chunks, QUERY, k=9)
        self.candidates = [t for t, _ in raw]

    def test_rerank_returns_list_of_strings(self):
        from src.strategies.reranking import rerank

        results = rerank(QUERY, self.candidates, k=3)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, str)

    def test_rerank_respects_k(self):
        from src.strategies.reranking import rerank

        results = rerank(QUERY, self.candidates, k=2)
        assert len(results) <= 2

    def test_rerank_empty_candidates(self):
        from src.strategies.reranking import rerank

        results = rerank(QUERY, [], k=3)
        assert results == []

    def test_rerank_finds_relevant_content(self):
        from src.strategies.reranking import rerank

        results = rerank(QUERY, self.candidates, k=3)
        combined = " ".join(results).lower()
        assert "75%" in combined
