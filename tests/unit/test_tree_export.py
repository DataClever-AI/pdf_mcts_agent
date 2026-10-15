"""Tests for tree_export."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tree_builder.section_matcher import match_content_to_sections
from src.tree_builder.tree_export import build_tree_json, write_tree_json


class TestBuildTreeJson:
    def test_produces_correct_section_count(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)
        tree = build_tree_json(sections)
        assert len(tree) == len(sample_bookmarks)

    def test_section_has_required_fields(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)
        tree = build_tree_json(sections)

        required = {
            "section_id", "title", "hierarchy_level", "hierarchy_path",
            "page_start", "page_end", "parent_section_id", "child_sections",
            "node_count", "confidence", "structural_source", "semantic_nodes",
            "semantic_order_start", "semantic_order_end",
        }
        for section in tree:
            assert required.issubset(section.keys()), f"Missing: {required - section.keys()}"

    def test_semantic_order_monotonic(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)
        tree = build_tree_json(sections)

        all_orders: list[int] = []
        for section in tree:
            for node in section["semantic_nodes"]:
                all_orders.append(node["semantic_order"])

        # All orders should be unique
        assert len(all_orders) == len(set(all_orders))
        # Orders should be monotonically increasing within global sequence
        assert all_orders == sorted(all_orders)

    def test_hierarchy_path_starts_with_slash(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)
        tree = build_tree_json(sections)
        for section in tree:
            assert section["hierarchy_path"].startswith("/")

    def test_confidence_always_1(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)
        tree = build_tree_json(sections)
        for section in tree:
            assert section["confidence"] == 1.0
            assert section["structural_source"] == "toc"


class TestStructuralSource:
    def test_defaults_to_toc(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)
        tree = build_tree_json(sections)
        assert all(s["structural_source"] == "toc" for s in tree)

    def test_explicit_inferred(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)
        tree = build_tree_json(sections, structural_source="inferred")
        assert all(s["structural_source"] == "inferred" for s in tree)


class TestWriteTreeJson:
    def test_roundtrip(
        self,
        sample_bookmarks: list[tuple[int, str, int]],
        sample_docling_doc: object,
    ) -> None:
        from src.models.extraction import DoclingDocument

        doc = sample_docling_doc
        assert isinstance(doc, DoclingDocument)
        sections = match_content_to_sections(sample_bookmarks, doc, 14)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.json"
            write_tree_json(sections, path)
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert len(data) == len(sample_bookmarks)
