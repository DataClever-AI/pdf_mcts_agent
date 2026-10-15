"""Shared fixtures for mvp_v2 tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.extraction import (
    DoclingDocument,
    DoclingPage,
    DoclingTable,
    DoclingTextBlock,
    ExtractionProvenance,
)


@pytest.fixture
def sample_bookmarks() -> list[tuple[int, str, int]]:
    """3-level bookmark hierarchy."""
    return [
        (1, "Chapter 1", 1),
        (2, "Section 1.1", 3),
        (2, "Section 1.2", 6),
        (3, "Subsection 1.2.1", 7),
        (1, "Chapter 2", 10),
        (2, "Section 2.1", 12),
    ]


@pytest.fixture
def sample_docling_doc() -> DoclingDocument:
    """DoclingDocument with text blocks on various pages."""

    def _block(idx: int, page: int, text: str) -> DoclingTextBlock:
        return DoclingTextBlock(
            block_id=f"#/texts/{idx}",
            text=text,
            label="text",
            page_no=page,
            reading_order=idx,
            depth=0,
            provenance=ExtractionProvenance(source="docling", page_no=page),
        )

    def _table(idx: int, page: int) -> DoclingTable:
        return DoclingTable(
            table_id=f"#/tables/{idx}",
            page_no=page,
            provenance=ExtractionProvenance(source="docling", page_no=page),
            row_count=3,
            col_count=2,
        )

    text_blocks = [
        _block(0, 1, "Intro text page 1"),
        _block(1, 2, "More intro page 2"),
        _block(2, 3, "Section 1.1 content"),
        _block(3, 4, "Section 1.1 continued"),
        _block(4, 5, "Section 1.1 end"),
        _block(5, 6, "Section 1.2 content"),
        _block(6, 7, "Subsection 1.2.1 content"),
        _block(7, 8, "Subsection 1.2.1 more"),
        _block(8, 9, "Section 1.2 trailing"),
        _block(9, 10, "Chapter 2 content"),
        _block(10, 11, "Chapter 2 continued"),
        _block(11, 12, "Section 2.1 content"),
        _block(12, 13, "Section 2.1 more"),
        _block(13, 14, "Section 2.1 end"),
    ]
    tables = [
        _table(0, 4),
        _table(1, 12),
    ]
    pages = {
        i: DoclingPage(page_no=i, width=612.0, height=792.0)
        for i in range(1, 15)
    }
    return DoclingDocument(
        document_id="test_doc",
        source_path="/test/doc.pdf",
        total_pages=14,
        pages=pages,
        text_blocks=text_blocks,
        tables=tables,
    )


@pytest.fixture
def malformed_bookmarks_duplicate_page() -> list[tuple[int, str, int]]:
    """Two level-2 siblings share page_no=6 (duplicate at same level)."""
    return [
        (1, "Chapter 1", 1),
        (2, "Section 1.1", 6),
        (2, "Section 1.2", 6),  # duplicate
        (1, "Chapter 2", 10),
    ]


@pytest.fixture
def malformed_bookmarks_negative_page() -> list[tuple[int, str, int]]:
    return [
        (1, "Chapter 1", -1),
        (2, "Section 1.1", 3),
    ]


@pytest.fixture
def malformed_bookmarks_out_of_range() -> list[tuple[int, str, int]]:
    """Assumes total_pages=14 in tests using this — page_no=999 exceeds it."""
    return [
        (1, "Chapter 1", 1),
        (2, "Section 1.1", 999),
    ]


@pytest.fixture
def malformed_bookmarks_out_of_order() -> list[tuple[int, str, int]]:
    """Level-2 sibling at page 10 followed by one at page 5 (goes backwards)."""
    return [
        (1, "Chapter 1", 1),
        (2, "Section 1.1", 10),
        (2, "Section 1.2", 5),  # out of order
    ]


@pytest.fixture
def malformed_bookmarks_empty_title() -> list[tuple[int, str, int]]:
    return [
        (1, "Chapter 1", 1),
        (2, "   ", 3),  # whitespace-only title
    ]


@pytest.fixture
def sample_docling_doc_no_headers() -> DoclingDocument:
    """DoclingDocument with only body text — zero section_header blocks."""
    text_blocks = [
        DoclingTextBlock(
            block_id=f"#/texts/{i}", text=f"Body text {i}", label="text",
            page_no=i + 1, reading_order=i, depth=0,
            provenance=ExtractionProvenance(source="docling", page_no=i + 1),
        )
        for i in range(5)
    ]
    return DoclingDocument(
        document_id="no_headers_doc", source_path="/test/no_headers.pdf",
        total_pages=5, text_blocks=text_blocks,
    )


@pytest.fixture
def sample_docling_doc_with_headers() -> DoclingDocument:
    """DoclingDocument with section_header blocks carrying heading_level, for
    synthesize_toc() tests. Levels assigned as HeadingHierarchyModel would from
    numbering ('1.', '1.1', etc.) — level 1 and 2 present."""
    def _header(idx: int, page: int, text: str, level: int) -> DoclingTextBlock:
        return DoclingTextBlock(
            block_id=f"#/texts/{idx}", text=text, label="section_header",
            page_no=page, reading_order=idx, depth=0,
            provenance=ExtractionProvenance(source="docling", page_no=page),
            heading_level=level,
        )

    def _body(idx: int, page: int, text: str) -> DoclingTextBlock:
        return DoclingTextBlock(
            block_id=f"#/texts/{idx}", text=text, label="text",
            page_no=page, reading_order=idx, depth=0,
            provenance=ExtractionProvenance(source="docling", page_no=page),
        )

    text_blocks = [
        _header(0, 1, "1. Introduction", 1),
        _body(1, 1, "Intro body text"),
        _header(2, 2, "1.1 Overview", 2),
        _body(3, 2, "Overview body text"),
        _header(4, 5, "2. Setup", 1),
        _body(5, 5, "Setup body text"),
    ]
    return DoclingDocument(
        document_id="with_headers_doc", source_path="/test/with_headers.pdf",
        total_pages=6, text_blocks=text_blocks,
    )
