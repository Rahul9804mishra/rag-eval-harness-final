"""
Tests for src/ingest.py — document loading.
"""
import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import DOCUMENTS_DIR
from src.ingest import load_documents


class TestLoadDocuments:
    def test_loads_from_default_directory(self):
        """The default documents directory should contain loadable files."""
        docs = load_documents(DOCUMENTS_DIR)
        assert len(docs) > 0

    def test_documents_have_page_content(self):
        docs = load_documents(DOCUMENTS_DIR)
        for doc in docs:
            assert hasattr(doc, "page_content"), "Document missing page_content"
            assert isinstance(doc.page_content, str)

    def test_documents_have_metadata(self):
        docs = load_documents(DOCUMENTS_DIR)
        for doc in docs:
            assert hasattr(doc, "metadata"), "Document missing metadata"
            assert isinstance(doc.metadata, dict)

    def test_documents_have_source_filename(self):
        """Our ingest.py should attach source_filename to every document."""
        docs = load_documents(DOCUMENTS_DIR)
        for doc in docs:
            assert "source_filename" in doc.metadata, (
                f"source_filename missing from {doc.metadata}"
            )

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        missing = str(tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError):
            load_documents(missing)

    def test_empty_directory_raises_value_error(self, tmp_path):
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .pdf or .txt files"):
            load_documents(str(empty_dir))

    def test_txt_files_are_loaded(self, tmp_path):
        """A directory with a single .txt file should produce at least one doc."""
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "sample.txt").write_text(
            "This is test content for the RAG pipeline.", encoding="utf-8"
        )
        docs = load_documents(str(doc_dir))
        assert len(docs) >= 1
        assert any("test content" in d.page_content for d in docs)
