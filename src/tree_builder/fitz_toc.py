"""
PyMuPDF (fitz) bookmark, page count, and page rendering.

Thin wrapper — no business rules.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


class PDFLoadError(Exception):
    """A PDF could not be opened or read. Message distinguishes the cause."""


def _open_pdf(pdf_path: Path) -> fitz.Document:
    """Open a PDF, translating fitz-specific failures into PDFLoadError."""
    try:
        doc = fitz.open(pdf_path)
    except fitz.EmptyFileError as exc:
        raise PDFLoadError(f"PDF is empty or zero-length: {pdf_path}") from exc
    except fitz.FileDataError as exc:
        raise PDFLoadError(f"PDF is corrupt or has invalid structure: {pdf_path}") from exc
    except fitz.FileNotFoundError as exc:
        raise PDFLoadError(f"PDF file not found: {pdf_path}") from exc
    if doc.needs_pass:
        doc.close()
        raise PDFLoadError(f"PDF is password-protected/encrypted: {pdf_path}")
    return doc


def get_embedded_toc(pdf_path: Path) -> list[tuple[int, str, int]]:
    """
    Extract embedded bookmarks via fitz.get_toc().

    Returns list of (level, title, page_no). 1-indexed page numbers.
    Returns empty list if PDF has no bookmarks.
    """
    doc = _open_pdf(pdf_path)
    toc: list[tuple[int, str, int]] = doc.get_toc()
    doc.close()
    return toc if toc else []


def page_count(pdf_path: Path) -> int:
    """Return total page count of a PDF."""
    doc = _open_pdf(pdf_path)
    n: int = doc.page_count
    doc.close()
    return n


def render_pages(
    pdf_path: Path,
    page_numbers: list[int],
    output_dir: Path,
    dpi: int = 150,
) -> dict[int, Path]:
    """
    Render specific PDF pages to PNG files. 1-indexed page numbers.

    Skips pages already rendered (cached). Returns {page_no: png_path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    doc = _open_pdf(pdf_path)
    total = doc.page_count
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    for page_no in sorted(set(page_numbers)):
        if page_no < 1 or page_no > total:
            continue
        out_path = output_dir / f"page_{page_no:04d}.png"
        if not out_path.exists():
            pix = doc[page_no - 1].get_pixmap(matrix=mat, alpha=False)
            out_path.write_bytes(pix.tobytes("png"))
        result[page_no] = out_path

    doc.close()
    return result
