"""Tests for synthetic_toc — building a TOC from Docling section_header blocks."""
from __future__ import annotations

from src.models.extraction import DoclingDocument, DoclingTextBlock, ExtractionProvenance
from src.tree_builder.synthetic_toc import synthesize_toc


class TestSynthesizeToc:
    def test_builds_toc_from_headers_in_reading_order(
        self, sample_docling_doc_with_headers: DoclingDocument
    ) -> None:
        toc = synthesize_toc(sample_docling_doc_with_headers)
        assert len(toc) == 3
        assert [level for level, _, _ in toc] == [1, 2, 1]
        assert [title for _, title, _ in toc] == [
            "1. Introduction", "1.1 Overview", "2. Setup",
        ]
        assert [page for _, _, page in toc] == [1, 2, 5]

    def test_no_headers_returns_empty(self, sample_docling_doc_no_headers: DoclingDocument) -> None:
        assert synthesize_toc(sample_docling_doc_no_headers) == []

    def test_missing_heading_level_defaults_to_1(self) -> None:
        block = DoclingTextBlock(
            block_id="#/texts/0", text="Untitled Heading", label="section_header",
            page_no=1, reading_order=0, depth=0,
            provenance=ExtractionProvenance(source="docling", page_no=1),
            heading_level=None,
        )
        doc = DoclingDocument(
            document_id="doc", source_path="/test/doc.pdf", total_pages=1, text_blocks=[block]
        )
        toc = synthesize_toc(doc)
        assert toc == [(1, "Untitled Heading", 1)]

    def test_blank_title_header_excluded(self) -> None:
        blank = DoclingTextBlock(
            block_id="#/texts/0", text="   ", label="section_header",
            page_no=1, reading_order=0, depth=0,
            provenance=ExtractionProvenance(source="docling", page_no=1),
            heading_level=1,
        )
        real = DoclingTextBlock(
            block_id="#/texts/1", text="Real Heading", label="section_header",
            page_no=2, reading_order=1, depth=0,
            provenance=ExtractionProvenance(source="docling", page_no=2),
            heading_level=1,
        )
        doc = DoclingDocument(
            document_id="doc", source_path="/test/doc.pdf", total_pages=2,
            text_blocks=[blank, real],
        )
        toc = synthesize_toc(doc)
        assert toc == [(1, "Real Heading", 2)]
