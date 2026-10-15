#!/usr/bin/env python3
"""
CLI: Build tree.json from a PDF, with or without bookmarks.

Usage:
    uv run python scripts/build_tree.py <pdf_path> [-o output_path]

Requires: pymupdf (always), docling (optional, for content extraction and for
the synthetic-TOC fallback when the PDF has no embedded bookmarks).
Without docling: builds tree structure from bookmarks only (no semantic_nodes),
and cannot fall back when bookmarks are missing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tree_builder.bookmark_sanity import check_bookmark_sanity
from src.tree_builder.fitz_toc import PDFLoadError, get_embedded_toc, page_count
from src.tree_builder.section_matcher import match_content_to_sections
from src.tree_builder.synthetic_toc import synthesize_toc
from src.tree_builder.tree_export import write_tree_json
from src.validation.tree_validator import validate_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tree.json from PDF bookmarks + docling")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output tree.json path")
    parser.add_argument("--no-docling", action="store_true", help="Skip docling extraction")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger("build_tree")

    pdf_path: Path = args.pdf_path
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    # Step 1: Extract bookmarks + page count (guarded against corrupt/encrypted/empty PDFs)
    try:
        bookmarks = get_embedded_toc(pdf_path)
        total = page_count(pdf_path)
    except PDFLoadError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("PDF: %s | %d pages | %d bookmarks", pdf_path.name, total, len(bookmarks))

    if bookmarks:
        sanity = check_bookmark_sanity(bookmarks, total)
        for issue in sanity.issues:
            logger.warning("Bookmark issue [%s]: %s", issue.kind, issue.detail)
        if sanity.has_hard_failures:
            logger.error(
                "Aborting: bookmark structure has %d hard-failure issue(s) "
                "(out-of-order page numbers). Fix the source PDF's TOC "
                "or remove/edit the offending bookmarks.",
                sum(1 for i in sanity.issues if i.kind == "out_of_order"),
            )
            sys.exit(1)

        levels = sorted(set(b[0] for b in bookmarks))
        logger.info("Bookmark levels: %s", levels)

    # Step 2: Docling extraction — now runs unconditionally, since it's needed
    # either way (semantic_nodes content, and the synthetic-TOC fallback below).
    doc = None
    if not args.no_docling:
        try:
            from src.tree_builder.docling_extract import DoclingExtractionEngine

            engine = DoclingExtractionEngine()
            if not engine.is_available:
                logger.warning("Docling not installed — building tree without content")
            else:
                logger.info("Running docling extraction...")
                doc = engine.extract(pdf_path, logger)
                logger.info(
                    "Docling: %d text_blocks, %d tables",
                    len(doc.text_blocks),
                    len(doc.tables),
                )
        except ImportError:
            logger.warning("Docling not available — building tree without content")

    if doc is None:
        from src.models.extraction import DoclingDocument

        doc = DoclingDocument(
            document_id="empty",
            source_path=str(pdf_path),
            total_pages=total,
        )

    # Step 3: Synthetic-TOC fallback when the PDF has no embedded bookmarks.
    if bookmarks:
        structural_source = "toc"
    else:
        if doc.document_id == "empty":
            logger.error(
                "No embedded bookmarks and docling unavailable — cannot build tree structure."
            )
            sys.exit(1)

        synthetic = synthesize_toc(doc)
        if not synthetic:
            logger.error(
                "No embedded bookmarks and Docling found no section headings — cannot "
                "infer document structure. This PDF may lack both a TOC and visually "
                "distinguishable headings."
            )
            sys.exit(1)

        logger.info(
            "No embedded bookmarks — synthesized %d-entry TOC from Docling headings",
            len(synthetic),
        )
        sanity = check_bookmark_sanity(synthetic, total)
        for issue in sanity.issues:
            logger.warning("Synthetic bookmark issue [%s]: %s", issue.kind, issue.detail)

        bookmarks = synthetic
        structural_source = "inferred"

    # Step 4: Match content to sections
    sections = match_content_to_sections(bookmarks, doc, total)
    sections_with_content = sum(1 for s in sections if s.text_blocks or s.tables)
    logger.info("Matched: %d sections, %d with content", len(sections), sections_with_content)

    # Step 5: Export
    output = args.output or pdf_path.with_suffix(".tree.json")
    write_tree_json(sections, output, structural_source=structural_source)
    logger.info("Written: %s", output)

    # Step 6: Validate (never aborts the build — see robustness-layer plan for rationale)
    total_docling_blocks = len(doc.text_blocks) + len(doc.tables)
    # Content before the first bookmarked page (cover/TOC/front matter) has no
    # heading of its own — legitimately unassigned, not a coverage gap.
    min_page_start = min((s.page_start for s in sections), default=1)
    expected_front_matter = sum(
        1 for b in doc.text_blocks if b.page_no < min_page_start
    ) + sum(1 for t in doc.tables if t.page_no < min_page_start)
    report = validate_tree(
        output,
        bookmarks,
        total_docling_blocks,
        pdf_name=pdf_path.name,
        expected_front_matter=expected_front_matter,
        page_tolerance=1,
    )
    if report.status == "PASS":
        logger.info("Validation: PASS (coverage/precision/structure all OK)")
    else:
        logger.warning(
            "Validation: FAIL — see details below. tree.json was still written; "
            "inspect before using downstream."
        )
        for check_name, check in (
            ("coverage", report.coverage),
            ("precision", report.precision),
            ("structure", report.structure),
        ):
            if check.status == "FAIL":
                logger.warning("  %s FAIL: %s", check_name, check.details)


if __name__ == "__main__":
    main()
