"""Tests for fitz_toc — uses mock to avoid real PDF dependency."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf as fitz
import pytest


class TestGetEmbeddedToc:
    @patch("src.tree_builder.fitz_toc.fitz")
    def test_returns_toc_list(self, mock_fitz: MagicMock) -> None:
        from src.tree_builder.fitz_toc import get_embedded_toc

        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.get_toc.return_value = [(1, "Ch1", 1), (2, "Sec1.1", 5)]
        mock_fitz.open.return_value = mock_doc

        result = get_embedded_toc(Path("fake.pdf"))
        assert len(result) == 2
        assert result[0] == (1, "Ch1", 1)

    @patch("src.tree_builder.fitz_toc.fitz")
    def test_empty_toc_returns_empty(self, mock_fitz: MagicMock) -> None:
        from src.tree_builder.fitz_toc import get_embedded_toc

        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.get_toc.return_value = []
        mock_fitz.open.return_value = mock_doc

        result = get_embedded_toc(Path("fake.pdf"))
        assert result == []


class TestPageCount:
    @patch("src.tree_builder.fitz_toc.fitz")
    def test_returns_count(self, mock_fitz: MagicMock) -> None:
        from src.tree_builder.fitz_toc import page_count

        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 42
        mock_fitz.open.return_value = mock_doc

        assert page_count(Path("fake.pdf")) == 42


class TestOpenPdfErrors:
    """PDFLoadError translation — only `fitz.open` is patched (not the whole
    module) so the real fitz exception hierarchy is available to raise/catch."""

    @patch("src.tree_builder.fitz_toc.fitz.open")
    def test_empty_file_raises_pdf_load_error(self, mock_open: MagicMock) -> None:
        from src.tree_builder.fitz_toc import PDFLoadError, get_embedded_toc

        mock_open.side_effect = fitz.EmptyFileError("boom")
        with pytest.raises(PDFLoadError, match="empty"):
            get_embedded_toc(Path("fake.pdf"))

    @patch("src.tree_builder.fitz_toc.fitz.open")
    def test_corrupt_file_raises_pdf_load_error(self, mock_open: MagicMock) -> None:
        from src.tree_builder.fitz_toc import PDFLoadError, get_embedded_toc

        mock_open.side_effect = fitz.FileDataError("boom")
        with pytest.raises(PDFLoadError, match="corrupt"):
            get_embedded_toc(Path("fake.pdf"))

    @patch("src.tree_builder.fitz_toc.fitz.open")
    def test_missing_file_raises_pdf_load_error(self, mock_open: MagicMock) -> None:
        from src.tree_builder.fitz_toc import PDFLoadError, get_embedded_toc

        mock_open.side_effect = fitz.FileNotFoundError("boom")
        with pytest.raises(PDFLoadError, match="not found"):
            get_embedded_toc(Path("fake.pdf"))

    @patch("src.tree_builder.fitz_toc.fitz.open")
    def test_encrypted_file_raises_pdf_load_error(self, mock_open: MagicMock) -> None:
        from src.tree_builder.fitz_toc import PDFLoadError, get_embedded_toc, page_count

        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_open.return_value = mock_doc

        with pytest.raises(PDFLoadError, match=r"encrypted|password"):
            get_embedded_toc(Path("fake.pdf"))
        assert mock_doc.close.called

        mock_doc.close.reset_mock()
        with pytest.raises(PDFLoadError, match=r"encrypted|password"):
            page_count(Path("fake.pdf"))
        assert mock_doc.close.called

    @patch("src.tree_builder.fitz_toc.fitz.open")
    def test_non_fitz_exception_propagates_unmodified(self, mock_open: MagicMock) -> None:
        """Proves the except clauses are narrow — a generic OSError (not part of
        fitz's exception hierarchy) is not translated into PDFLoadError."""
        from src.tree_builder.fitz_toc import get_embedded_toc

        mock_open.side_effect = OSError("disk on fire")
        with pytest.raises(OSError, match="disk on fire"):
            get_embedded_toc(Path("fake.pdf"))
