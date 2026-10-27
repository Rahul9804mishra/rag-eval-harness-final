"""
Tests for the app.py strategy-routing layer.

These tests verify:
1. STRATEGY_OPTIONS exposes all four strategy keys.
2. available_strategies() always returns all four entries.
3. _chunks_to_display() normalises each strategy's raw output format
   into the uniform dict schema expected by the UI.
4. retrieve_for_strategy() raises FileNotFoundError (not a silent fallback)
   when the backing ChromaDB directory is absent.
5. STRATEGY_CHROMA_DIRS maps hybrid_search and reranking to chroma_simple,
   matching the eval harness.

These tests do NOT start a Streamlit server; they import app module members
directly.  app.py's UI rendering code is guarded behind a Streamlit runtime
context check, so it is safely skipped during imports.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Minimal Streamlit stub
# ---------------------------------------------------------------------------
# app.py calls st.set_page_config / st.title / etc. at module level for
# Streamlit to pick up.  We stub them as no-ops so the import succeeds
# without a running Streamlit server.

def _make_stub_streamlit():
    st = types.ModuleType("streamlit")
    for attr in (
        "set_page_config", "title", "caption", "subheader", "info",
        "error", "warning", "success", "write", "markdown", "spinner",
        "chat_input", "sidebar", "columns", "metric", "pyplot", "dataframe",
        "expander", "json", "divider", "radio", "selectbox", "slider",
        "text_input", "toggle", "bar_label",
    ):
        setattr(st, attr, MagicMock(return_value=None))

    # cache_resource must be a transparent decorator
    def _cache_resource(fn=None, **_kwargs):
        if fn is not None:
            return fn
        return lambda f: f

    st.cache_resource = _cache_resource

    # Sidebar / spinner need to work as context managers
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    st.sidebar = ctx
    st.spinner = ctx

    # columns() needs to return an iterable of mock columns
    st.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])

    return st


# Install stub BEFORE importing app so no real Streamlit is needed
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = _make_stub_streamlit()

# Also stub the runtime module so app.py's guard evaluates cleanly
_rt = types.ModuleType("streamlit.runtime")
_sr = types.ModuleType("streamlit.runtime.scriptrunner")
_sr.get_script_run_ctx = MagicMock(return_value=None)  # simulate "not in Streamlit"
sys.modules.setdefault("streamlit.runtime", _rt)
sys.modules.setdefault("streamlit.runtime.scriptrunner", _sr)

import app  # noqa: E402  (must come after stub install)


# ---------------------------------------------------------------------------
# 1. Strategy registry
# ---------------------------------------------------------------------------

class TestStrategyRegistry(unittest.TestCase):

    def test_all_four_strategy_keys_present(self):
        expected_keys = {
            "simple_chunking",
            "semantic_chunking",
            "hybrid_search",
            "reranking",
        }
        actual_keys = set(app.STRATEGY_OPTIONS.values())
        self.assertEqual(actual_keys, expected_keys)

    def test_all_four_display_labels_present(self):
        labels = list(app.STRATEGY_OPTIONS.keys())
        self.assertIn("Simple chunking", labels)
        self.assertIn("Semantic chunking", labels)
        self.assertIn("Hybrid search", labels)
        self.assertIn("Reranking", labels)

    def test_available_strategies_returns_all_four(self):
        strats = app.available_strategies()
        self.assertEqual(len(strats), 4)
        self.assertIn("Hybrid search", strats)
        self.assertIn("Reranking", strats)

    def test_hybrid_and_reranking_share_chroma_simple(self):
        """Hybrid and reranking must point at chroma_simple, matching eval harness."""
        simple_dir = app.STRATEGY_CHROMA_DIRS["simple_chunking"]
        self.assertEqual(app.STRATEGY_CHROMA_DIRS["hybrid_search"], simple_dir)
        self.assertEqual(app.STRATEGY_CHROMA_DIRS["reranking"], simple_dir)

    def test_semantic_points_at_chroma_semantic(self):
        semantic_dir = app.ROOT / "chroma_semantic"
        self.assertEqual(app.STRATEGY_CHROMA_DIRS["semantic_chunking"], semantic_dir)


# ---------------------------------------------------------------------------
# 2. _chunks_to_display normalisation
# ---------------------------------------------------------------------------

class TestChunksToDisplay(unittest.TestCase):

    def _make_doc(self, text: str, meta: dict):
        """Create a minimal LangChain-like Document stub."""
        doc = MagicMock()
        doc.page_content = text
        doc.metadata = meta
        return doc

    def test_simple_chunking_documents(self):
        docs = [
            self._make_doc("  Hello world  ", {"source": "a.txt"}),
            self._make_doc("Second chunk", {"source": "b.txt"}),
        ]
        result = app._chunks_to_display(docs, "simple_chunking")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Hello world")
        self.assertEqual(result[0]["metadata"], {"source": "a.txt"})
        self.assertIsNone(result[0]["distance"])

    def test_semantic_chunking_documents(self):
        docs = [self._make_doc("Semantic text", {"source": "c.txt"})]
        result = app._chunks_to_display(docs, "semantic_chunking")
        self.assertEqual(result[0]["text"], "Semantic text")

    def test_hybrid_search_tuples(self):
        raw = [("Hybrid chunk one", 0.85), ("Hybrid chunk two", 0.60)]
        result = app._chunks_to_display(raw, "hybrid_search")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Hybrid chunk one")
        self.assertEqual(result[0]["metadata"], {})
        # distance is 1 - fused_score, rounded to 4dp
        self.assertAlmostEqual(result[0]["distance"], round(1.0 - 0.85, 4), places=4)

    def test_reranking_strings(self):
        raw = ["Top reranked chunk", "Second reranked chunk"]
        result = app._chunks_to_display(raw, "reranking")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Top reranked chunk")
        self.assertEqual(result[0]["metadata"], {})
        self.assertIsNone(result[0]["distance"])

    def test_strips_bom(self):
        docs = [self._make_doc("\ufeffContent after BOM", {})]
        result = app._chunks_to_display(docs, "simple_chunking")
        self.assertFalse(result[0]["text"].startswith("\ufeff"))
        self.assertEqual(result[0]["text"], "Content after BOM")


# ---------------------------------------------------------------------------
# 3. retrieve_for_strategy raises on missing index (no silent fallback)
# ---------------------------------------------------------------------------

class TestRetrieveForStrategyErrors(unittest.TestCase):

    def test_missing_chroma_dir_raises_file_not_found(self):
        """
        When the backing ChromaDB directory does not exist, retrieve_for_strategy
        must raise FileNotFoundError — it must NOT silently switch strategy.
        """
        with patch.object(
            app,
            "_load_vectorstore",
            side_effect=FileNotFoundError("Index not found"),
        ):
            with self.assertRaises(FileNotFoundError):
                app.retrieve_for_strategy("simple_chunking", "test query", top_k=3)

    def test_unknown_strategy_raises_value_error(self):
        with self.assertRaises(ValueError):
            app.retrieve_for_strategy("nonexistent_strategy", "test", top_k=3)

    def test_hybrid_bm25_failure_raises_runtime_error(self):
        """
        If _get_bm25_index fails (e.g. no documents), hybrid_search must
        raise RuntimeError rather than returning empty results silently.
        """
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 5

        with patch.object(app, "_load_vectorstore", return_value=mock_vs), \
             patch.object(app, "_get_bm25_index", side_effect=Exception("no docs")):
            with self.assertRaises(RuntimeError):
                app.retrieve_for_strategy("hybrid_search", "query", top_k=3)

    def test_reranking_bm25_failure_raises_runtime_error(self):
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 5

        with patch.object(app, "_load_vectorstore", return_value=mock_vs), \
             patch.object(app, "_get_bm25_index", side_effect=Exception("no docs")):
            with self.assertRaises(RuntimeError):
                app.retrieve_for_strategy("reranking", "query", top_k=3)


# ---------------------------------------------------------------------------
# 4. retrieve_for_strategy calls the correct module function
# ---------------------------------------------------------------------------

class TestRetrieveDispatch(unittest.TestCase):
    """
    Verify that each strategy key dispatches to its corresponding module
    function, not to some other strategy's implementation.
    """

    def _mock_vs(self):
        vs = MagicMock()
        vs._collection.count.return_value = 10
        return vs

    def test_simple_chunking_calls_retrieve_simple(self):
        mock_doc = MagicMock()
        mock_doc.page_content = "simple result"
        mock_doc.metadata = {"source": "x.txt"}

        with patch.object(app, "_load_vectorstore", return_value=self._mock_vs()), \
             patch("src.strategies.simple_chunking.retrieve_simple",
                   return_value=[mock_doc]) as mock_fn:
            result = app.retrieve_for_strategy("simple_chunking", "q", top_k=3)
            mock_fn.assert_called_once()
            self.assertEqual(result[0]["text"], "simple result")

    def test_semantic_chunking_calls_retrieve_semantic(self):
        mock_doc = MagicMock()
        mock_doc.page_content = "semantic result"
        mock_doc.metadata = {}

        with patch.object(app, "_load_vectorstore", return_value=self._mock_vs()), \
             patch("src.strategies.semantic_chunking.retrieve_semantic",
                   return_value=[mock_doc]) as mock_fn:
            result = app.retrieve_for_strategy("semantic_chunking", "q", top_k=3)
            mock_fn.assert_called_once()
            self.assertEqual(result[0]["text"], "semantic result")

    def test_hybrid_search_calls_retrieve_hybrid(self):
        mock_bm25 = MagicMock()
        mock_chunks = [MagicMock()]

        with patch.object(app, "_load_vectorstore", return_value=self._mock_vs()), \
             patch.object(app, "_get_bm25_index",
                          return_value=(self._mock_vs(), mock_bm25, mock_chunks)), \
             patch("src.strategies.hybrid_search.retrieve_hybrid",
                   return_value=[("hybrid text", 0.75)]) as mock_fn:
            result = app.retrieve_for_strategy("hybrid_search", "q", top_k=3)
            mock_fn.assert_called_once()
            self.assertEqual(result[0]["text"], "hybrid text")

    def test_reranking_calls_rerank(self):
        mock_bm25 = MagicMock()
        mock_chunks = [MagicMock()]

        with patch.object(app, "_load_vectorstore", return_value=self._mock_vs()), \
             patch.object(app, "_get_bm25_index",
                          return_value=(self._mock_vs(), mock_bm25, mock_chunks)), \
             patch("src.strategies.hybrid_search.retrieve_hybrid",
                   return_value=[("candidate", 0.8)]), \
             patch("src.strategies.reranking.rerank",
                   return_value=["reranked text"]) as mock_fn:
            result = app.retrieve_for_strategy("reranking", "q", top_k=3)
            mock_fn.assert_called_once()
            self.assertEqual(result[0]["text"], "reranked text")


# ---------------------------------------------------------------------------
# 5. Results dashboard reads all four strategies from results.json
# ---------------------------------------------------------------------------

class TestResultsDashboard(unittest.TestCase):

    def test_load_results_reads_all_four_strategies(self):
        """load_results() must return scores for all 4 strategies."""
        import json as _json
        import tempfile

        fake_results = {
            "simple_chunking": 96.0,
            "semantic_chunking": 78.0,
            "hybrid_search": 98.0,
            "reranking": 94.0,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(fake_results, f)
            tmp_path = Path(f.name)

        try:
            # Temporarily redirect app.RESULTS_PATH to the temp file
            original = app.RESULTS_PATH
            app.RESULTS_PATH = tmp_path
            scores = app.load_results()
            self.assertEqual(set(scores.keys()), set(fake_results.keys()))
            self.assertAlmostEqual(scores["hybrid_search"], 98.0)
            self.assertAlmostEqual(scores["reranking"], 94.0)
        finally:
            app.RESULTS_PATH = original
            tmp_path.unlink(missing_ok=True)

    def test_load_results_handles_missing_file(self):
        original = app.RESULTS_PATH
        app.RESULTS_PATH = Path("/nonexistent_path_that_does_not_exist/results.json")
        try:
            scores = app.load_results()
            self.assertEqual(scores, {})
        finally:
            app.RESULTS_PATH = original

    def test_load_results_handles_corrupt_json(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{bad json")
            tmp_path = Path(f.name)

        original = app.RESULTS_PATH
        app.RESULTS_PATH = tmp_path
        try:
            scores = app.load_results()
            self.assertEqual(scores, {})
        finally:
            app.RESULTS_PATH = original
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
