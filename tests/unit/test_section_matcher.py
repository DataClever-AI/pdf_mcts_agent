"""Tests for section_matcher — bookmark/content matching and hierarchy."""
from __future__ import annotations

from src.models.extraction import DoclingDocument, DoclingTextBlock, ExtractionProvenance
from src.tree_builder.section_matcher import (
    _build_hierarchy,
    _compute_page_ranges,
    match_content_to_sections,
)


class TestComputePageRanges:
    def test_single_bookmark_owns_all_pages(self) -> None:
        bookmarks = [(1, "Only Section", 1)]
        result = _compute_page_ranges(bookmarks, 50)
        assert len(result) == 1
        _, _, _, page_start, page_end = result[0]
        assert page_start == 1
        assert page_end == 50

    def test_two_l1_bookmarks_split_pages(self) -> None:
        bookmarks = [(1, "Ch1", 1), (1, "Ch2", 10)]
        result = _compute_page_ranges(bookmarks, 20)
        assert result[0][3:] == (1, 9)   # Ch1: pages 1-9
        assert result[1][3:] == (10, 20)  # Ch2: pages 10-20

    def test_nested_levels_page_ranges(self, sample_bookmarks: list[tuple[int, str, int]]) -> None:
        result = _compute_page_ranges(sample_bookmarks, 14)
        # Chapter 1: pages 1 to 9 (ends before Chapter 2 at page 10)
        assert result[0][3:] == (1, 9)
        # Section 1.1: pages 3 to 5 (ends before Section 1.2 at page 6)
        assert result[1][3:] == (3, 5)
        # Section 1.2: pages 6 to 9 (ends before Chapter 2)
        assert result[2][3:] == (6, 9)
        # Subsection 1.2.1: pages 7 to 9 (no next sibling before Chapter 2)
        assert result[3][3:] == (7, 9)
        # Chapter 2: pages 10 to 14
        assert result[4][3:] == (10, 14)
        # Section 2.1: pages 12 to 14
        assert result[5][3:] == (12, 14)

    def test_page_clamping(self) -> None:
        bookmarks = [(1, "Bad", 0), (1, "Also Bad", 999)]
        result = _compute_page_ranges(bookmarks, 10)
        assert result[0][3] == 1   # clamped from 0
        assert result[1][3] == 10  # clamped from 999


class TestBuildHierarchy:
    def test_flat_bookmarks_no_parent(self) -> None:
        ranges = [
            ("sec_0001", "A", 1, 1, 5),
            ("sec_0002", "B", 1, 6, 10),
        ]
        h = _build_hierarchy(ranges)
        assert h["sec_0001"] == (None, [])
        assert h["sec_0002"] == (None, [])

    def test_nested_parent_child(self) -> None:
        ranges = [
            ("sec_0001", "Ch1", 1, 1, 10),
            ("sec_0002", "Sec1.1", 2, 3, 5),
            ("sec_0003", "Sec1.2", 2, 6, 9),
        ]
        h = _build_hierarchy(ranges)
        assert h["sec_0001"] == (None, ["sec_0002", "sec_0003"])
        assert h["sec_0002"] == ("sec_0001", [])
        assert h["sec_0003"] == ("sec_0001", [])

    def test_deep_nesting(self) -> None:
        ranges = [
            ("sec_0001", "L1", 1, 1, 20),
            ("sec_0002", "L2", 2, 5, 15),
            ("sec_0003", "L3", 3, 7, 10),
        ]
        h = _build_hierarchy(ranges)
        assert h["sec_0001"][1] == ["sec_0002"]
        assert h["sec_0002"][1] == ["sec_0003"]
        assert h["sec_0003"] == ("sec_0002", [])


class TestMatchContentToSections:
    def test_empty_bookmarks_returns_empty(self) -> None:
        doc = DoclingDocument(document_id="x", source_path="x", total_pages=10)
        result = match_content_to_sections([], doc, 10)
        assert result == []

    def test_content_assigned_to_deepest_section(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: DoclingDocument,
    ) -> None:
        sections = match_content_to_sections(sample_bookmarks, sample_docling_doc, 14)
        sec_by_id = {s.section_id: s for s in sections}

        # Section 1.1 (pages 3-5) should have blocks on pages 3,4,5 + table on page 4
        sec_1_1 = sec_by_id["sec_0002"]
        assert len(sec_1_1.text_blocks) == 3
        assert len(sec_1_1.tables) == 1
        assert all(3 <= b.page_no <= 5 for b in sec_1_1.text_blocks)

    def test_child_content_not_in_parent(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: DoclingDocument,
    ) -> None:
        sections = match_content_to_sections(sample_bookmarks, sample_docling_doc, 14)
        sec_by_id = {s.section_id: s for s in sections}

        # Chapter 1 (pages 1-9) owns only pages 1-2 directly
        # (pages 3-5 → sec 1.1, pages 6-9 → sec 1.2 and subsec 1.2.1)
        ch1 = sec_by_id["sec_0001"]
        ch1_pages = {b.page_no for b in ch1.text_blocks}
        assert ch1_pages == {1, 2}

    def test_all_blocks_assigned(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: DoclingDocument,
    ) -> None:
        sections = match_content_to_sections(sample_bookmarks, sample_docling_doc, 14)
        total_assigned = sum(len(s.text_blocks) + len(s.tables) for s in sections)
        total_expected = len(sample_docling_doc.text_blocks) + len(sample_docling_doc.tables)
        assert total_assigned == total_expected

    def test_section_hierarchy_correct(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: DoclingDocument,
    ) -> None:
        sections = match_content_to_sections(sample_bookmarks, sample_docling_doc, 14)
        sec_by_id = {s.section_id: s for s in sections}

        # Chapter 1 has children: Section 1.1 and Section 1.2
        ch1 = sec_by_id["sec_0001"]
        assert ch1.parent_id is None
        assert "sec_0002" in ch1.child_ids
        assert "sec_0003" in ch1.child_ids

        # Subsection 1.2.1 is child of Section 1.2
        subsec = sec_by_id["sec_0004"]
        assert subsec.parent_id == "sec_0003"

    def test_section_count_matches_bookmarks(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: DoclingDocument,
    ) -> None:
        sections = match_content_to_sections(sample_bookmarks, sample_docling_doc, 14)
        assert len(sections) == len(sample_bookmarks)

    def test_same_page_siblings_do_not_collapse(self) -> None:
        """
        Regression test: three same-level sections sharing one page used to
        collapse into the first one (page-based assignment, first-candidate-
        wins). Reading-order anchoring should split content by heading block
        instead, so each sibling gets its own content.
        """
        bookmarks = [
            (3, "5.4.2 Historical Graphical Trends", 82),
            (3, "5.4.3 CVP Entry", 82),
            (3, "5.4.4 Derived Value Calculator", 82),
        ]

        def _heading(idx: int, text: str) -> DoclingTextBlock:
            return DoclingTextBlock(
                block_id=f"#/texts/{idx}",
                text=text,
                label="section_header",
                page_no=82,
                reading_order=idx,
                depth=0,
                provenance=ExtractionProvenance(source="docling", page_no=82),
            )

        def _body(idx: int, text: str) -> DoclingTextBlock:
            return DoclingTextBlock(
                block_id=f"#/texts/{idx}",
                text=text,
                label="text",
                page_no=82,
                reading_order=idx,
                depth=0,
                provenance=ExtractionProvenance(source="docling", page_no=82),
            )

        text_blocks = [
            _heading(0, "5.4.2 Historical Graphical Trends"),
            _body(1, "Historical trends content line one."),
            _body(2, "Historical trends content line two."),
            _heading(3, "5.4.3 CVP Entry"),
            _body(4, "CVP entry content line one."),
            _heading(5, "5.4.4 Derived Value Calculator"),
            _body(6, "Derived value calculator content line one."),
            _body(7, "Derived value calculator content line two."),
        ]
        doc = DoclingDocument(
            document_id="collision_test",
            source_path="/test/doc.pdf",
            total_pages=82,
            text_blocks=text_blocks,
        )

        sections = match_content_to_sections(bookmarks, doc, 82)
        sec_by_title = {s.title: s for s in sections}

        trends = sec_by_title["5.4.2 Historical Graphical Trends"]
        cvp = sec_by_title["5.4.3 CVP Entry"]
        derived = sec_by_title["5.4.4 Derived Value Calculator"]

        assert len(trends.text_blocks) == 3  # heading + 2 body lines
        assert len(cvp.text_blocks) == 2      # heading + 1 body line
        assert len(derived.text_blocks) == 3  # heading + 2 body lines
        assert all("CVP" not in b.text for b in trends.text_blocks)
        assert all("Derived" not in b.text for b in cvp.text_blocks)
