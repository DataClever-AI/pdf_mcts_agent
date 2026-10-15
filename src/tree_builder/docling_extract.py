"""
Docling extraction engine — windowed PDF content extraction.

Adapted from pdf_tree infrastructure. Extracts text blocks and tables
with page numbers, bounding boxes, and reading order.

Optional dependency: docling must be installed separately.
  uv add 'docling[gpu]'  (with GPU)
  uv add docling          (CPU-only)
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.models.extraction import (
    BoundingBox,
    DoclingDocument,
    DoclingPage,
    DoclingTable,
    DoclingTextBlock,
    ExtractionProvenance,
)

try:
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        HeadingHierarchyOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False


CANONICAL_COLLECTIONS: tuple[str, ...] = (
    "texts", "groups", "tables", "pictures", "key_value_items",
)


def _render_table_markdown(data: dict[str, Any]) -> str:
    """
    Render Docling table_cells into a pipe-delimited text grid.

    Without this, table content (e.g. an alarm-color legend) is invisible to
    retrieval/VLM — tree_export previously wrote only "[Table RxC]" placeholders.
    Cell spans are collapsed to their start offset; good enough for retrieval
    text, not meant to reproduce exact layout.
    """
    num_rows = int(data.get("num_rows", 0))
    num_cols = int(data.get("num_cols", 0))
    cells = data.get("table_cells") or []
    if num_rows <= 0 or num_cols <= 0 or not cells:
        return ""

    grid: list[list[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells:
        text = (cell.get("text") or "").strip()
        if not text:
            continue
        r = cell.get("start_row_offset_idx", 0)
        c = cell.get("start_col_offset_idx", 0)
        if 0 <= r < num_rows and 0 <= c < num_cols:
            grid[r][c] = text

    lines = [" | ".join(row) for row in grid if any(cell for cell in row)]
    return "\n".join(lines)


def _rewrite_ref(ref_payload: Any, *, offsets: dict[str, int]) -> Any:
    """Rewrite a $ref pointer using global collection offsets."""
    if isinstance(ref_payload, dict):
        rv = ref_payload.get("$ref")
        if not isinstance(rv, str):
            return ref_payload
        return {"$ref": _rewrite_ref(rv, offsets=offsets)}
    if not isinstance(ref_payload, str):
        return ref_payload
    if not ref_payload.startswith("#/"):
        return ref_payload
    parts = ref_payload.split("/")
    if len(parts) != 3:
        return ref_payload
    cn = parts[1]
    if cn not in offsets:
        return ref_payload
    try:
        local_idx = int(parts[2])
    except ValueError:
        return ref_payload
    return f"#/{cn}/{offsets[cn] + local_idx}"


def _rewrite_recursive(obj: Any, *, offsets: dict[str, int]) -> Any:
    """Recursively rewrite all $ref pointers in a document object."""
    if isinstance(obj, dict):
        r: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "children" and isinstance(v, list):
                r[k] = [_rewrite_ref(c, offsets=offsets) for c in v]
            elif isinstance(v, dict) and "$ref" in v:
                r[k] = _rewrite_ref(v, offsets=offsets)
            else:
                r[k] = _rewrite_recursive(v, offsets=offsets)
        return r
    if isinstance(obj, list):
        return [_rewrite_recursive(i, offsets=offsets) for i in obj]
    return obj


def _build_pipeline_options() -> Any:
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.images_scale = 1.0
    opts.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU,
    )
    # Numbering-based heading levels — used only by the synthetic-TOC fallback
    # (see synthetic_toc.py); no effect on the normal embedded-bookmark path,
    # which builds hierarchy from fitz bookmarks, not these levels.
    opts.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=True, use_numbering=True, use_style=False,
    )
    return opts


def _build_converter() -> Any:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=_build_pipeline_options(),
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


def _generate_windows(
    total_pages: int,
    window_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Generate overlapping 1-indexed page windows."""
    windows: list[tuple[int, int]] = []
    start = 1
    while start <= total_pages:
        end = min(start + window_size - 1, total_pages)
        windows.append((start, end))
        if end >= total_pages:
            break
        start = end - overlap + 1
    return windows


def _trim_overlap_pages(doc_dict: dict[str, Any], min_page: int) -> dict[str, Any]:
    """
    Drop text/table entries whose page falls in a window's leading overlap
    region (already fully captured by the previous window's conversion).

    Without this, every window boundary produces duplicate blocks for the
    overlapped pages. Since reading_order is assigned positionally over the
    merged, concatenated batches (see _build_typed_document), the duplicate
    copy contributed by the later window ends up with a higher reading_order
    than content that comes after it in the actual document — corrupting
    section_matcher's reading-order-anchored content assignment right at
    every window boundary.
    """
    def _page_of(node: dict[str, Any]) -> int:
        prov = node.get("prov") or [{}]
        return int(prov[0].get("page_no") or 0)

    trimmed = dict(doc_dict)
    for cn in ("texts", "tables"):
        items = doc_dict.get(cn) or []
        trimmed[cn] = [n for n in items if _page_of(n) >= min_page]
    return trimmed


def _merge_batch_dicts(
    batch_dicts: list[dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "body": {"self_ref": "#/body", "children": [], "parent": None},
        "furniture": {"self_ref": "#/furniture", "children": [], "parent": None},
        "pages": {},
        "texts": [],
        "groups": [],
        "tables": [],
        "pictures": [],
        "key_value_items": [],
    }
    global_offsets: dict[str, int] = dict.fromkeys(CANONICAL_COLLECTIONS, 0)

    for dd in batch_dicts:
        offsets_snap = deepcopy(global_offsets)
        for cn in CANONICAL_COLLECTIONS:
            coll = dd.get(cn) or []
            merged[cn].extend(
                [_rewrite_recursive(obj, offsets=offsets_snap) for obj in coll]
            )
            global_offsets[cn] += len(coll)
        body = dd.get("body", {})
        merged["body"]["children"].extend(
            [_rewrite_ref(c, offsets=offsets_snap) for c in body.get("children", [])]
        )
        for pg_no, pg_payload in (dd.get("pages", {}) or {}).items():
            merged["pages"][str(pg_no)] = pg_payload

    logger.info(
        "merged texts=%d tables=%d pages=%d",
        len(merged["texts"]),
        len(merged["tables"]),
        len(merged["pages"]),
    )
    return merged


def _bbox_from_dict(
    bbox_dict: dict[str, Any] | None,
    page_no: int,
) -> BoundingBox | None:
    if not bbox_dict:
        return None
    return BoundingBox(
        x0=float(bbox_dict.get("l", 0)),
        y0=float(bbox_dict.get("t", 0)),
        x1=float(bbox_dict.get("r", 0)),
        y1=float(bbox_dict.get("b", 0)),
        page_no=page_no,
        coordinate_origin="bottomleft",
    )


def _build_typed_document(
    document_id: str,
    source_path: str,
    merged: dict[str, Any],
    elapsed_s: float,
) -> DoclingDocument:
    """Convert merged dict into typed DoclingDocument."""

    pages: dict[int, DoclingPage] = {}
    for pg_no_str, pg_data in (merged.get("pages") or {}).items():
        try:
            pg_no = int(pg_no_str)
        except ValueError:
            continue
        size = (pg_data or {}).get("size") or {}
        pages[pg_no] = DoclingPage(
            page_no=pg_no,
            width=float(size.get("width", 0)),
            height=float(size.get("height", 0)),
            text_block_count=sum(
                1 for t in merged.get("texts", [])
                if ((t.get("prov") or [{}])[0]).get("page_no") == pg_no
            ),
            table_count=sum(
                1 for t in merged.get("tables", [])
                if ((t.get("prov") or [{}])[0]).get("page_no") == pg_no
            ),
        )

    text_blocks: list[DoclingTextBlock] = []
    for idx, node in enumerate(merged.get("texts", [])):
        prov_raw = (node.get("prov") or [{}])[0]
        pg = prov_raw.get("page_no") or 0
        bbox = _bbox_from_dict(prov_raw.get("bbox"), pg)
        prov = ExtractionProvenance(source="docling", page_no=pg, bbox=bbox)
        label = node.get("label") or "text"
        heading_level = node.get("level") if label == "section_header" else None
        text_blocks.append(
            DoclingTextBlock(
                block_id=f"#/texts/{idx}",
                text=(node.get("text") or "").strip(),
                label=label,
                page_no=pg,
                reading_order=idx,
                depth=0,
                provenance=prov,
                bbox=bbox,
                heading_level=heading_level,
            )
        )

    tables: list[DoclingTable] = []
    for idx, node in enumerate(merged.get("tables", [])):
        prov_raw = (node.get("prov") or [{}])[0]
        pg = prov_raw.get("page_no") or 0
        bbox = _bbox_from_dict(prov_raw.get("bbox"), pg)
        prov = ExtractionProvenance(source="docling", page_no=pg, bbox=bbox)
        data = node.get("data") or {}
        tables.append(
            DoclingTable(
                table_id=f"#/tables/{idx}",
                page_no=pg,
                bbox=bbox,
                provenance=prov,
                row_count=int(data.get("num_rows", 0)),
                col_count=int(data.get("num_cols", 0)),
                markdown=_render_table_markdown(data),
            )
        )

    return DoclingDocument(
        document_id=document_id,
        source_path=source_path,
        total_pages=len(pages),
        pages=pages,
        text_blocks=text_blocks,
        tables=tables,
        extraction_elapsed_s=elapsed_s,
        merged_dict=merged,
    )


class DoclingExtractionEngine:
    """
    Windowed Docling extraction engine.

    Lazy-initializes the DocumentConverter on first call.
    """

    def __init__(self) -> None:
        self._converter: Any = None

    @property
    def is_available(self) -> bool:
        return _DOCLING_AVAILABLE

    def _get_converter(self) -> Any:
        if not _DOCLING_AVAILABLE:
            raise ImportError(
                "docling is not installed.\n"
                "Install: uv add 'docling[gpu]'  (GPU)  or  uv add docling  (CPU)"
            )
        if self._converter is None:
            self._converter = _build_converter()
        return self._converter

    def extract(
        self,
        pdf_path: Path,
        logger: logging.Logger,
        *,
        window_size: int = 60,
        window_overlap: int = 6,
        on_window_complete: Any | None = None,
    ) -> DoclingDocument:
        """
        Extract DoclingDocument from PDF via windowed batches.

        on_window_complete(window_idx, total_windows, elapsed_s) called after each.
        """
        converter = self._get_converter()

        from src.tree_builder.fitz_toc import _open_pdf

        doc_fitz = _open_pdf(pdf_path)
        total_pages: int = doc_fitz.page_count
        doc_fitz.close()

        windows = _generate_windows(total_pages, window_size, window_overlap)
        logger.info("%d windows for %d pages", len(windows), total_pages)
        batch_dicts: list[dict[str, Any]] = []
        t_start = time.perf_counter()

        for w_idx, (start, end) in enumerate(windows, start=1):
            logger.info("window %d/%d pages %d-%d", w_idx, len(windows), start, end)
            try:
                result = converter.convert(str(pdf_path), page_range=(start, end))
                doc_dict = result.document.export_to_dict()
                if w_idx > 1:
                    prev_end = windows[w_idx - 2][1]
                    doc_dict = _trim_overlap_pages(doc_dict, min_page=prev_end + 1)
                batch_dicts.append(doc_dict)
            except Exception as exc:
                logger.warning("window %d/%d failed — %s", w_idx, len(windows), exc)
            if on_window_complete is not None:
                try:
                    on_window_complete(w_idx, len(windows), time.perf_counter() - t_start)
                except Exception:
                    pass

        elapsed_s = time.perf_counter() - t_start

        if not batch_dicts:
            raise RuntimeError("All extraction windows failed")

        merged = _merge_batch_dicts(batch_dicts, logger)
        document_id = hashlib.sha256(str(pdf_path.resolve()).encode()).hexdigest()[:24]

        return _build_typed_document(
            document_id=document_id,
            source_path=str(pdf_path),
            merged=merged,
            elapsed_s=elapsed_s,
        )
