"""
Typed immutable extraction entities for tree building.

Stripped from pdf_tree domain/extraction.py — keeps only what's needed:
BoundingBox, ExtractionProvenance, DoclingTextBlock, DoclingTable, DoclingPage, DoclingDocument.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class BoundingBox:
    """PDF region coordinates with origin convention."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_no: int
    coordinate_origin: Literal["topleft", "bottomleft"] = "topleft"


@dataclass(frozen=True)
class ExtractionProvenance:
    """Where a content block came from."""

    source: Literal["docling", "fitz"]
    page_no: int
    confidence: float = 1.0
    bbox: BoundingBox | None = None


@dataclass(frozen=True)
class DoclingTextBlock:
    """A single text block extracted by Docling."""

    block_id: str
    text: str
    label: str
    page_no: int
    reading_order: int
    depth: int
    provenance: ExtractionProvenance
    bbox: BoundingBox | None = None
    heading_level: int | None = None  # SectionHeaderItem.level; None for non-headings


@dataclass(frozen=True)
class DoclingTable:
    """A table extracted by Docling."""

    table_id: str
    page_no: int
    provenance: ExtractionProvenance
    bbox: BoundingBox | None = None
    row_count: int = 0
    col_count: int = 0
    markdown: str = ""  # rendered cell content — empty if Docling gave no cells


@dataclass(frozen=True)
class DoclingPage:
    """Per-page metadata from Docling extraction."""

    page_no: int
    width: float
    height: float
    text_block_count: int = 0
    table_count: int = 0


@dataclass
class DoclingDocument:
    """Typed Docling extraction result."""

    document_id: str
    source_path: str
    total_pages: int
    pages: dict[int, DoclingPage] = field(default_factory=dict)
    text_blocks: list[DoclingTextBlock] = field(default_factory=list)
    tables: list[DoclingTable] = field(default_factory=list)
    extraction_elapsed_s: float = 0.0
    merged_dict: dict[str, Any] | None = field(default=None, repr=False)
