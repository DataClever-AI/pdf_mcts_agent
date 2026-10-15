"""
Tree export — serialize MatchedSections to tree.json.

Output: flat list of section dicts with semantic_nodes.
Compatible with mtcs_ai Section model for future Part 2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.tree_builder.section_matcher import MatchedSection


def _build_hierarchy_path(
    section: MatchedSection,
    sections_by_id: dict[str, MatchedSection],
) -> str:
    """Build hierarchy path like /Chapter 1/Section 2/Subsection 3."""
    parts: list[str] = []
    current: MatchedSection | None = section
    while current is not None:
        parts.append(current.title)
        current = sections_by_id.get(current.parent_id or "")
    parts.reverse()
    return "/" + "/".join(parts)


def _section_to_dict(
    section: MatchedSection,
    sections_by_id: dict[str, MatchedSection],
    semantic_order_counter: list[int],
    structural_source: str = "toc",
) -> dict[str, Any]:
    """Convert a MatchedSection to tree.json dict format."""
    semantic_nodes: list[dict[str, Any]] = []

    # Text blocks → semantic nodes
    for block in sorted(section.text_blocks, key=lambda b: b.reading_order):
        order = semantic_order_counter[0]
        semantic_order_counter[0] += 1
        node: dict[str, Any] = {
            "node_id": block.block_id,
            "canonical_text": block.text,
            "semantic_order": order,
            "node_type": "paragraph" if block.label != "section_header" else "heading",
            "page_no": block.page_no,
        }
        if block.bbox is not None:
            node["bbox"] = {
                "x0": block.bbox.x0,
                "y0": block.bbox.y0,
                "x1": block.bbox.x1,
                "y1": block.bbox.y1,
            }
        semantic_nodes.append(node)

    # Tables → semantic nodes
    for table in sorted(section.tables, key=lambda t: t.page_no):
        order = semantic_order_counter[0]
        semantic_order_counter[0] += 1
        node = {
            "node_id": table.table_id,
            "canonical_text": (
                table.markdown or f"[Table {table.row_count}x{table.col_count}]"
            ),
            "semantic_order": order,
            "node_type": "table",
            "page_no": table.page_no,
        }
        if table.bbox is not None:
            node["bbox"] = {
                "x0": table.bbox.x0,
                "y0": table.bbox.y0,
                "x1": table.bbox.x1,
                "y1": table.bbox.y1,
            }
        semantic_nodes.append(node)

    hierarchy_path = _build_hierarchy_path(section, sections_by_id)
    semantic_orders = [n["semantic_order"] for n in semantic_nodes]

    return {
        "section_id": section.section_id,
        "title": section.title,
        "hierarchy_level": section.level,
        "hierarchy_path": hierarchy_path,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "parent_section_id": section.parent_id,
        "child_sections": section.child_ids,
        "node_count": len(semantic_nodes),
        "confidence": 1.0,
        "structural_source": structural_source,
        "semantic_order_start": min(semantic_orders) if semantic_orders else None,
        "semantic_order_end": max(semantic_orders) if semantic_orders else None,
        "semantic_nodes": semantic_nodes,
    }


def build_tree_json(
    sections: list[MatchedSection],
    structural_source: str = "toc",
) -> list[dict[str, Any]]:
    """
    Build tree.json payload from matched sections.

    Returns flat list of section dicts. Each section contains
    its semantic_nodes inline. child_sections are section_id references.

    structural_source: "toc" (embedded fitz bookmarks) or "inferred"
    (synthetic TOC built from Docling headings — see synthetic_toc.py).
    """
    sections_by_id = {s.section_id: s for s in sections}
    semantic_order_counter = [1]

    result: list[dict[str, Any]] = []
    for section in sections:
        result.append(
            _section_to_dict(section, sections_by_id, semantic_order_counter, structural_source)
        )

    return result


def write_tree_json(
    sections: list[MatchedSection],
    output_path: Path,
    structural_source: str = "toc",
) -> Path:
    """Build and write tree.json to disk."""
    payload = build_tree_json(sections, structural_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path
