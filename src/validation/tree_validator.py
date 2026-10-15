"""
Tree validator — 3 checks against source PDF.

Check 1: Coverage — every docling block appears in exactly one section
Check 2: Precision — each block's page_no falls within its section's page range
Check 3: Structure — hierarchy matches bookmarks (levels, parent/child)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    """Result of a single validation check."""

    status: str  # "PASS" | "FAIL"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Full validation report across all 3 checks."""

    pdf: str
    status: str  # "PASS" if all checks pass, else "FAIL"
    coverage: CheckResult = field(default_factory=lambda: CheckResult(status="SKIP"))
    precision: CheckResult = field(default_factory=lambda: CheckResult(status="SKIP"))
    structure: CheckResult = field(default_factory=lambda: CheckResult(status="SKIP"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf": self.pdf,
            "status": self.status,
            "coverage": {"status": self.coverage.status, **self.coverage.details},
            "precision": {"status": self.precision.status, **self.precision.details},
            "structure": {"status": self.structure.status, **self.structure.details},
        }


def _check_coverage(
    tree_sections: list[dict[str, Any]],
    total_docling_blocks: int,
    expected_front_matter: int = 0,
) -> CheckResult:
    """
    Check 1: Every content block appears in exactly one section.

    Counts all node_ids across tree sections and compares to total docling blocks.

    expected_front_matter: count of blocks that precede the first bookmarked
    section (cover page, title page, a manual's own TOC/list-of-figures pages)
    — content with no heading of its own, correctly excluded from every
    section, not a coverage gap. Defaults to 0 (strict — every existing caller
    that doesn't know this count keeps today's behavior).
    """
    all_node_ids: list[str] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []

    for section in tree_sections:
        for node in section.get("semantic_nodes", []):
            nid = node.get("node_id", "")
            all_node_ids.append(nid)
            if nid in seen_ids:
                duplicates.append(nid)
            seen_ids.add(nid)

    orphaned_count = max(0, total_docling_blocks - len(seen_ids) - expected_front_matter)

    status = "PASS" if orphaned_count == 0 and not duplicates else "FAIL"
    return CheckResult(
        status=status,
        details={
            "total_docling_blocks": total_docling_blocks,
            "matched_blocks": len(seen_ids),
            "expected_front_matter": expected_front_matter,
            "orphaned_count": orphaned_count,
            "duplicate_count": len(duplicates),
            "duplicates": duplicates[:10],
        },
    )


def _check_precision(
    tree_sections: list[dict[str, Any]],
    page_tolerance: int = 0,
) -> CheckResult:
    """
    Check 2: Each block's page_no falls within its section's page range
    (within page_tolerance pages either side).

    page_tolerance: section_matcher assigns content by reading-order interval,
    not by page_no — a paragraph can legitimately spill 1 page past a
    section's page_end before the next heading appears mid-page. Tolerance=0
    (default) preserves today's strict page-range behavior for callers that
    don't pass it explicitly.
    """
    total_checked = 0
    misplaced: list[dict[str, Any]] = []

    for section in tree_sections:
        page_start = section.get("page_start", 0)
        page_end = section.get("page_end", 0)
        section_id = section.get("section_id", "")

        for node in section.get("semantic_nodes", []):
            total_checked += 1
            page_no = node.get("page_no", 0)
            if page_no < page_start - page_tolerance or page_no > page_end + page_tolerance:
                misplaced.append({
                    "node_id": node.get("node_id", ""),
                    "page_no": page_no,
                    "section_id": section_id,
                    "section_range": f"{page_start}-{page_end}",
                })

    status = "PASS" if not misplaced else "FAIL"
    return CheckResult(
        status=status,
        details={
            "total_sections": len(tree_sections),
            "total_nodes_checked": total_checked,
            "page_tolerance": page_tolerance,
            "misplaced_count": len(misplaced),
            "misplaced_blocks": misplaced[:20],
        },
    )


def _check_structure(
    tree_sections: list[dict[str, Any]],
    bookmarks: list[tuple[int, str, int]],
) -> CheckResult:
    """
    Check 3: Hierarchy matches bookmarks.

    - bookmark count == section count
    - levels match
    - no orphan non-root sections
    - no cycles
    """
    level_mismatches: list[dict[str, Any]] = []
    orphan_sections: list[str] = []
    missing_parents: list[str] = []

    section_count = len(tree_sections)
    bookmark_count = len(bookmarks)
    count_match = section_count == bookmark_count

    # Check levels match
    section_ids = set()
    for i, section in enumerate(tree_sections):
        section_ids.add(section.get("section_id", ""))
        if i < len(bookmarks):
            expected_level = bookmarks[i][0]
            actual_level = section.get("hierarchy_level", 0)
            if actual_level != expected_level:
                level_mismatches.append({
                    "section_id": section.get("section_id", ""),
                    "expected_level": expected_level,
                    "actual_level": actual_level,
                })

    # Check parent references
    for section in tree_sections:
        parent_id = section.get("parent_section_id")
        if parent_id is not None and parent_id not in section_ids:
            missing_parents.append(section.get("section_id", ""))

    # Check no orphan non-root sections
    root_ids = {
        s.get("section_id", "")
        for s in tree_sections
        if s.get("parent_section_id") is None
    }
    for section in tree_sections:
        sid = section.get("section_id", "")
        parent = section.get("parent_section_id")
        if parent is not None and parent not in section_ids:
            orphan_sections.append(sid)

    # Check child references are valid
    invalid_children: list[dict[str, Any]] = []
    for section in tree_sections:
        sid = section.get("section_id", "")
        for child_id in section.get("child_sections", []):
            if child_id not in section_ids:
                invalid_children.append({"parent": sid, "child": child_id})

    # Cycle detection via DFS
    children_map: dict[str, list[str]] = {}
    for section in tree_sections:
        children_map[section.get("section_id", "")] = section.get("child_sections", [])

    has_cycle = False
    visited: set[str] = set()
    in_stack: set[str] = set()

    def _dfs(node_id: str) -> bool:
        if node_id in in_stack:
            return True
        if node_id in visited:
            return False
        visited.add(node_id)
        in_stack.add(node_id)
        for child in children_map.get(node_id, []):
            if _dfs(child):
                return True
        in_stack.discard(node_id)
        return False

    for sid in section_ids:
        if _dfs(sid):
            has_cycle = True
            break

    max_depth = max((s.get("hierarchy_level", 0) for s in tree_sections), default=0)

    all_ok = (
        count_match
        and not level_mismatches
        and not orphan_sections
        and not missing_parents
        and not invalid_children
        and not has_cycle
    )

    status = "PASS" if all_ok else "FAIL"
    return CheckResult(
        status=status,
        details={
            "bookmark_count": bookmark_count,
            "section_count": section_count,
            "count_match": count_match,
            "max_depth": max_depth,
            "root_count": len(root_ids),
            "level_mismatches": level_mismatches[:10],
            "orphan_sections": orphan_sections[:10],
            "missing_parents": missing_parents[:10],
            "invalid_children": invalid_children[:10],
            "has_cycle": has_cycle,
        },
    )


def validate_tree(
    tree_json_path: Path,
    bookmarks: list[tuple[int, str, int]],
    total_docling_blocks: int,
    pdf_name: str = "",
    expected_front_matter: int = 0,
    page_tolerance: int = 0,
) -> ValidationReport:
    """
    Run all 3 validation checks on a tree.json file.

    Args:
        tree_json_path: Path to tree.json
        bookmarks: Original bookmarks from fitz for structure check
        total_docling_blocks: Total text_blocks + tables from docling for coverage check
        pdf_name: PDF filename for report
        expected_front_matter: blocks preceding the first bookmarked section
            (cover/TOC pages) — excluded from the coverage check's orphan count.
        page_tolerance: pages of slack allowed either side of a section's
            page range in the precision check (reading-order-based content
            assignment can legitimately spill a page past the nominal range).

    Returns:
        ValidationReport with PASS/FAIL status per check.
    """
    with open(tree_json_path, encoding="utf-8") as f:
        tree_sections: list[dict[str, Any]] = json.load(f)

    coverage = _check_coverage(tree_sections, total_docling_blocks, expected_front_matter)
    precision = _check_precision(tree_sections, page_tolerance)
    structure = _check_structure(tree_sections, bookmarks)

    overall = "PASS" if all(
        c.status == "PASS" for c in [coverage, precision, structure]
    ) else "FAIL"

    return ValidationReport(
        pdf=pdf_name,
        status=overall,
        coverage=coverage,
        precision=precision,
        structure=structure,
    )
