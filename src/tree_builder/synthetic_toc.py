"""
Synthetic TOC — build a (level, title, page_no) bookmark list from Docling's
SECTION_HEADER-labeled blocks, for PDFs with no embedded fitz bookmarks.

Uses DoclingTextBlock.heading_level (assigned by docling's HeadingHierarchyModel,
see docling_extract.py) as the level signal. Falls back to level=1 for any heading
whose level wasn't inferred — still a valid flat single-level TOC, strictly better
than no TOC at all.
"""
from __future__ import annotations

from src.models.extraction import DoclingDocument


def synthesize_toc(doc: DoclingDocument) -> list[tuple[int, str, int]]:
    """
    Build a synthetic bookmark list from Docling's section_header blocks.

    Returns [] if Docling found zero section_header blocks — caller must treat
    this as a hard failure, there is no structure signal to build a tree from.
    """
    headers = [b for b in doc.text_blocks if b.label == "section_header"]
    headers.sort(key=lambda b: b.reading_order)

    toc: list[tuple[int, str, int]] = []
    for block in headers:
        title = block.text.strip()
        if not title:
            continue
        level = block.heading_level if block.heading_level is not None else 1
        toc.append((level, title, block.page_no))
    return toc
