"""Tests for tree_validator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tree_builder.section_matcher import match_content_to_sections
from src.tree_builder.synthetic_toc import synthesize_toc
from src.tree_builder.tree_export import build_tree_json
from src.validation.tree_validator import validate_tree


def _build_and_write(
    bookmarks: list[tuple[int, str, int]],
    doc: object,
    total_pages: int,
) -> tuple[Path, list[dict]]:
    """Helper: build tree and write to temp file."""
    from src.models.extraction import DoclingDocument

    assert isinstance(doc, DoclingDocument)
    sections = match_content_to_sections(bookmarks, doc, total_pages)
    tree = build_tree_json(sections)
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(tree, tmp)
    tmp.close()
    return Path(tmp.name), tree


class TestCoverageCheck:
    def test_pass_when_all_blocks_assigned(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        total_blocks = len(doc.text_blocks) + len(doc.tables)
        path, _ = _build_and_write(sample_bookmarks, doc, 14)
        report = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        assert report.coverage.status == "PASS"
        assert report.coverage.details["orphaned_count"] == 0

    def test_fail_when_blocks_missing(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        # Pretend docling had more blocks than tree captured
        path, _ = _build_and_write(sample_bookmarks, doc, 14)
        report = validate_tree(path, sample_bookmarks, 999, "test.pdf")
        assert report.coverage.status == "FAIL"

    def test_expected_front_matter_excludes_pre_toc_blocks_from_orphan_count(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        """
        Blocks preceding the first bookmarked page (cover/TOC front matter)
        have no heading of their own and are correctly never assigned to any
        section — expected_front_matter excludes them from the orphan count
        instead of flagging them as a coverage gap.
        """
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        total_blocks = len(doc.text_blocks) + len(doc.tables)
        path, _ = _build_and_write(sample_bookmarks, doc, 14)
        # Simulate 3 extra front-matter blocks docling saw but no section covers
        report = validate_tree(
            path, sample_bookmarks, total_blocks + 3, "test.pdf", expected_front_matter=3
        )
        assert report.coverage.status == "PASS"
        assert report.coverage.details["orphaned_count"] == 0


class TestPrecisionCheck:
    def test_pass_when_all_blocks_in_range(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        total_blocks = len(doc.text_blocks) + len(doc.tables)
        path, _ = _build_and_write(sample_bookmarks, doc, 14)
        report = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        assert report.precision.status == "PASS"
        assert report.precision.details["misplaced_count"] == 0

    def test_fail_when_block_out_of_range(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        path, tree = _build_and_write(sample_bookmarks, doc, 14)
        # Tamper: put a node with page_no=99 into first section
        tree[0]["semantic_nodes"].append({
            "node_id": "fake_node",
            "canonical_text": "wrong page",
            "semantic_order": 999,
            "node_type": "paragraph",
            "page_no": 99,
        })
        with open(path, "w") as f:
            json.dump(tree, f)

        total_blocks = len(doc.text_blocks) + len(doc.tables) + 1
        report = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        assert report.precision.status == "FAIL"
        assert report.precision.details["misplaced_count"] >= 1

    def test_page_tolerance_allows_one_page_spillover(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        """
        section_matcher assigns content by reading-order, not page_no — a
        paragraph can legitimately land 1 page past page_end before the next
        heading appears. page_tolerance=1 accepts that as valid, not misplaced.
        """
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        path, tree = _build_and_write(sample_bookmarks, doc, 14)
        tree[0]["semantic_nodes"].append({
            "node_id": "spillover_node",
            "canonical_text": "one page past page_end",
            "semantic_order": 999,
            "node_type": "paragraph",
            "page_no": tree[0]["page_end"] + 1,
        })
        with open(path, "w") as f:
            json.dump(tree, f)

        total_blocks = len(doc.text_blocks) + len(doc.tables) + 1
        strict = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        assert strict.precision.status == "FAIL"

        tolerant = validate_tree(
            path, sample_bookmarks, total_blocks, "test.pdf", page_tolerance=1
        )
        assert tolerant.precision.status == "PASS"


class TestStructureCheck:
    def test_pass_with_correct_hierarchy(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        total_blocks = len(doc.text_blocks) + len(doc.tables)
        path, _ = _build_and_write(sample_bookmarks, doc, 14)
        report = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        assert report.structure.status == "PASS"
        assert report.structure.details["count_match"] is True
        assert report.structure.details["has_cycle"] is False

    def test_fail_when_section_count_mismatch(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        path, tree = _build_and_write(sample_bookmarks, doc, 14)
        # Remove last section
        tree.pop()
        with open(path, "w") as f:
            json.dump(tree, f)

        total_blocks = sum(len(s.get("semantic_nodes", [])) for s in tree)
        report = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        assert report.structure.status == "FAIL"
        assert report.structure.details["count_match"] is False


class TestSyntheticTocStructureCheck:
    def test_structure_check_passes_by_construction_on_synthetic_toc(
        self, sample_docling_doc_with_headers: object,
    ) -> None:
        """
        The structure check compares the tree's hierarchy back against the
        `bookmarks` list validate_tree() was given. On the embedded-TOC path
        that list is an independent oracle (real fitz bookmarks). On the
        synthetic-TOC path, the "bookmarks" ARE the synthesized list that built
        the tree — so this passing is a tautology (it can't catch inference
        quality issues), not proof the inferred structure is actually correct.
        """
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc_with_headers
        assert isinstance(doc, DoclingDocument)
        synthetic_bookmarks = synthesize_toc(doc)
        assert synthetic_bookmarks  # sanity: fixture has headers

        path, _ = _build_and_write(synthetic_bookmarks, doc, doc.total_pages)
        total_blocks = len(doc.text_blocks)
        report = validate_tree(path, synthetic_bookmarks, total_blocks, "test.pdf")
        assert report.structure.status == "PASS"
        assert report.structure.details["count_match"] is True


class TestOverallReport:
    def test_pass_overall_when_all_pass(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        total_blocks = len(doc.text_blocks) + len(doc.tables)
        path, _ = _build_and_write(sample_bookmarks, doc, 14)
        report = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        assert report.status == "PASS"

    def test_to_dict_has_all_keys(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        total_blocks = len(doc.text_blocks) + len(doc.tables)
        path, _ = _build_and_write(sample_bookmarks, doc, 14)
        report = validate_tree(path, sample_bookmarks, total_blocks, "test.pdf")
        d = report.to_dict()
        assert "pdf" in d
        assert "status" in d
        assert "coverage" in d
        assert "precision" in d
        assert "structure" in d
