"""
Section matcher — assign docling content blocks to bookmark sections.

Algorithm:
  1. Compute page ranges from bookmarks (page_start → next bookmark page - 1)
  2. Build hierarchy via level stack (parent/child from bookmark levels)
  3. Assign text blocks by reading-order interval, anchored to each bookmark's
     own heading text block — NOT by page_no alone. Page_no can't disambiguate
     sibling sections that share a page (common for short subsections): if
     three headings land on the same page, page-based assignment sends every
     block on that page to whichever section sorts first, starving the rest
     (node_count=0). Reading order — Docling's global sequential block index —
     pins each section's true start regardless of page-sharing.
  4. Assign tables by page_no (Docling doesn't give tables a reading_order,
     so this is a known coarser fallback — acceptable since tables are a
     small fraction of content and rarely collide within a shared page).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models.extraction import DoclingDocument, DoclingTable, DoclingTextBlock


@dataclass
class MatchedSection:
    """A bookmark section with its matched content blocks."""

    section_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    text_blocks: list[DoclingTextBlock] = field(default_factory=list)
    tables: list[DoclingTable] = field(default_factory=list)


def _compute_page_ranges(
    bookmarks: list[tuple[int, str, int]],
    total_pages: int,
) -> list[tuple[str, str, int, int, int]]:
    """
    Compute (section_id, title, level, page_start, page_end) for each bookmark.

    page_end for bookmark[i] = bookmark[i+1].page_no - 1 for next same-or-higher level.
    Last bookmark gets page_end = total_pages.

    Bookmarks with page_no < 1 are clamped to 1.
    Bookmarks with page_no > total_pages are clamped to total_pages.
    """
    result: list[tuple[str, str, int, int, int]] = []

    for i, (level, title, page_no) in enumerate(bookmarks):
        page_start = max(1, min(page_no, total_pages))

        # Find next bookmark at same or higher level to determine page_end
        page_end = total_pages
        for j in range(i + 1, len(bookmarks)):
            next_level, _, next_page = bookmarks[j]
            if next_level <= level:
                page_end = max(page_start, next_page - 1)
                break

        section_id = f"sec_{i + 1:04d}"
        result.append((section_id, title, level, page_start, page_end))

    return result


def _build_hierarchy(
    ranges: list[tuple[str, str, int, int, int]],
) -> dict[str, tuple[str | None, list[str]]]:
    """
    Build parent_id and child_ids from bookmark levels using a stack.

    Returns dict[section_id -> (parent_id, child_ids)].
    """
    hierarchy: dict[str, tuple[str | None, list[str]]] = {}
    # Stack of (level, section_id) — tracks current ancestor at each level
    stack: list[tuple[int, str]] = []

    for section_id, _title, level, _ps, _pe in ranges:
        # Pop stack until we find a parent (level < current)
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_id = stack[-1][1] if stack else None
        hierarchy[section_id] = (parent_id, [])

        # Register as child of parent
        if parent_id is not None and parent_id in hierarchy:
            hierarchy[parent_id][1].append(section_id)

        stack.append((level, section_id))

    return hierarchy


def _normalize_heading(text: str) -> str:
    """Normalize heading text for tolerant matching (case/whitespace/punctuation)."""
    text = re.sub(r"[^\w\s]", "", text.strip().lower())
    return " ".join(text.split())


def _find_heading_anchor(
    title: str,
    page_no: int,
    blocks_by_page: dict[int, list[DoclingTextBlock]],
) -> int | None:
    """
    Find the reading_order of the text block that IS this bookmark's own
    heading, by matching normalized text among blocks on the bookmark's page.
    Prefers blocks Docling classified as "section_header"; falls back to any
    block with matching text (headings are sometimes mislabeled as body text).
    """
    norm_title = _normalize_heading(title)
    if not norm_title:
        return None
    candidates = blocks_by_page.get(page_no, [])

    for block in candidates:
        if block.label != "section_header":
            continue
        norm_text = _normalize_heading(block.text)
        if norm_text == norm_title or norm_title in norm_text or norm_text in norm_title:
            return block.reading_order

    for block in candidates:
        if _normalize_heading(block.text) == norm_title:
            return block.reading_order

    return None


def _compute_reading_order_ranges(
    bookmarks: list[tuple[int, str, int]],
    page_ranges: list[tuple[str, str, int, int, int]],
    doc: DoclingDocument,
) -> list[tuple[int, int]]:
    """
    Compute (reading_order_start, reading_order_end) per bookmark.

    Anchor = reading_order of the bookmark's own matched heading block, or —
    if no heading text match is found on its page — the first block on that
    page, or — if the page has no content at all — the previous section's
    anchor + 1. The max() clamp keeps anchors monotonic non-decreasing so
    ranges never invert even if a heading match lands out of expected order.
    """
    blocks_by_page: dict[int, list[DoclingTextBlock]] = {}
    for b in doc.text_blocks:
        blocks_by_page.setdefault(b.page_no, []).append(b)
    for page_blocks in blocks_by_page.values():
        page_blocks.sort(key=lambda b: b.reading_order)

    anchors: list[int] = []
    prev_anchor = -1
    for _section_id, title, _level, page_start, _page_end in page_ranges:
        anchor = _find_heading_anchor(title, page_start, blocks_by_page)
        if anchor is None:
            page_blocks = blocks_by_page.get(page_start, [])
            anchor = page_blocks[0].reading_order if page_blocks else prev_anchor + 1
        anchor = max(anchor, prev_anchor + 1)
        anchors.append(anchor)
        prev_anchor = anchor

    max_reading_order = max((b.reading_order for b in doc.text_blocks), default=0)

    ranges: list[tuple[int, int]] = []
    for i, (level, _title, _page) in enumerate(bookmarks):
        end = max_reading_order
        for j in range(i + 1, len(bookmarks)):
            next_level = bookmarks[j][0]
            if next_level <= level:
                end = anchors[j] - 1
                break
        ranges.append((anchors[i], max(anchors[i], end)))

    return ranges


def _assign_content(
    sections: list[MatchedSection],
    reading_order_ranges: dict[str, tuple[int, int]],
    doc: DoclingDocument,
) -> None:
    """
    Assign text_blocks by reading-order interval containment (the section
    with the tightest/most specific containing interval wins — equivalent
    to "deepest" without relying on level matching exactly).

    Assign tables by page_no (see module docstring — known coarser fallback).
    """
    def _tightest(containing: list[MatchedSection]) -> MatchedSection:
        return min(
            containing,
            key=lambda s: reading_order_ranges[s.section_id][1]
            - reading_order_ranges[s.section_id][0],
        )

    for block in doc.text_blocks:
        containing = [
            s
            for s in sections
            if reading_order_ranges[s.section_id][0]
            <= block.reading_order
            <= reading_order_ranges[s.section_id][1]
        ]
        if containing:
            _tightest(containing).text_blocks.append(block)

    page_sections: dict[int, list[MatchedSection]] = {}
    for sec in sections:
        for p in range(sec.page_start, sec.page_end + 1):
            page_sections.setdefault(p, []).append(sec)
    for p in page_sections:
        page_sections[p].sort(key=lambda s: s.level, reverse=True)

    for table in doc.tables:
        candidates = page_sections.get(table.page_no, [])
        if candidates:
            candidates[0].tables.append(table)


def match_content_to_sections(
    bookmarks: list[tuple[int, str, int]],
    doc: DoclingDocument,
    total_pages: int,
) -> list[MatchedSection]:
    """
    Match docling content blocks to bookmark sections.

    Args:
        bookmarks: list of (level, title, page_no) from fitz.get_toc()
        doc: DoclingDocument with text_blocks and tables
        total_pages: total page count of the PDF

    Returns:
        list of MatchedSection, one per bookmark, with content assigned.
    """
    if not bookmarks:
        return []

    ranges = _compute_page_ranges(bookmarks, total_pages)
    hierarchy = _build_hierarchy(ranges)
    reading_order_ranges = dict(
        zip(
            (section_id for section_id, *_ in ranges),
            _compute_reading_order_ranges(bookmarks, ranges, doc),
            strict=True,
        )
    )

    sections: list[MatchedSection] = []
    for section_id, title, level, page_start, page_end in ranges:
        parent_id, child_ids = hierarchy[section_id]
        sections.append(
            MatchedSection(
                section_id=section_id,
                title=title,
                level=level,
                page_start=page_start,
                page_end=page_end,
                parent_id=parent_id,
                child_ids=list(child_ids),
            )
        )

    _assign_content(sections, reading_order_ranges, doc)

    return sections
