"""
Tree loader — load tree.json into list[Section].

Simplified from mtcs_ai canonical_store: no graph_v1.json merge.
All provenance fields default to toc values (confidence=1.0, structural_source="toc").
Flattens nested child_sections into a flat list.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.section import Section, SemanticNode


def _parse_semantic_nodes(raw_nodes: list[dict[str, Any]]) -> list[SemanticNode]:
    """Parse semantic_nodes list from tree.json section dict."""
    nodes: list[SemanticNode] = []
    for n in raw_nodes:
        nodes.append(
            SemanticNode(
                node_id=n.get("node_id", ""),
                canonical_text=n.get("canonical_text", ""),
                semantic_order=int(n.get("semantic_order", 0)),
                node_type=str(n.get("node_type", "paragraph")),
            )
        )
    return nodes


def _parse_section(raw: dict[str, Any]) -> Section:
    """Parse a single section dict from tree.json."""
    return Section(
        section_id=raw["section_id"],
        title=raw["title"],
        hierarchy_level=int(raw.get("hierarchy_level", 1)),
        hierarchy_path=raw.get("hierarchy_path", f"/{raw['title']}"),
        semantic_nodes=_parse_semantic_nodes(raw.get("semantic_nodes", [])),
        semantic_order_start=raw.get("semantic_order_start"),
        semantic_order_end=raw.get("semantic_order_end"),
        node_count=raw.get("node_count"),
        child_sections=[str(c) for c in raw.get("child_sections", [])],
        page_start=int(raw["page_start"]) if raw.get("page_start") is not None else None,
        page_end=int(raw["page_end"]) if raw.get("page_end") is not None else None,
        # Provenance defaults — all sections come from fitz bookmarks
        confidence=float(raw.get("confidence", 1.0)),
        structural_source=str(raw.get("structural_source", "toc")),
        inference_reasons=list(raw.get("inference_reasons", [])),
        evidence_source_ids=list(raw.get("evidence_source_ids", [])),
    )


def load_tree(tree_json_path: Path) -> list[Section]:
    """
    Load tree.json into a flat list of Section objects.

    Handles both flat lists (our format) and nested structures.
    All sections are returned in the order they appear in the file.

    Args:
        tree_json_path: Path to tree.json

    Returns:
        Flat list of Section objects.

    Raises:
        FileNotFoundError: If tree.json does not exist.
        ValueError: If tree.json is malformed.
    """
    if not tree_json_path.exists():
        raise FileNotFoundError(f"tree.json not found: {tree_json_path}")

    with open(tree_json_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"tree.json must be a list of section dicts, got {type(raw_data)}")

    sections: list[Section] = []
    for raw in raw_data:
        try:
            sections.append(_parse_section(raw))
        except (KeyError, ValueError, AssertionError) as e:
            raise ValueError(f"Malformed section '{raw.get('section_id', '?')}': {e}") from e

    return sections
