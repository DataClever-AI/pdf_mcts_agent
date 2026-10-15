#!/usr/bin/env python3
"""
CLI: Validate tree.json against source PDF.

Usage:
    uv run python scripts/validate_tree.py <tree_json> <pdf_path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tree_builder.fitz_toc import get_embedded_toc
from src.validation.tree_validator import validate_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tree.json against source PDF")
    parser.add_argument("tree_json", type=Path, help="Path to tree.json")
    parser.add_argument("pdf_path", type=Path, help="Path to source PDF")
    parser.add_argument("--docling-blocks", type=int, default=0,
                        help="Total docling blocks (text+tables) for coverage check. "
                             "If 0, coverage check uses tree node count as baseline.")
    args = parser.parse_args()

    if not args.tree_json.exists():
        print(f"ERROR: tree.json not found: {args.tree_json}", file=sys.stderr)
        sys.exit(1)
    if not args.pdf_path.exists():
        print(f"ERROR: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    bookmarks = get_embedded_toc(args.pdf_path)

    # If docling block count not provided, use tree's own node count as baseline
    total_blocks = args.docling_blocks
    if total_blocks == 0:
        with open(args.tree_json, encoding="utf-8") as f:
            tree_data = json.load(f)
        total_blocks = sum(
            len(s.get("semantic_nodes", []))
            for s in tree_data
        )

    report = validate_tree(
        tree_json_path=args.tree_json,
        bookmarks=bookmarks,
        total_docling_blocks=total_blocks,
        pdf_name=args.pdf_path.name,
    )

    print(json.dumps(report.to_dict(), indent=2))
    sys.exit(0 if report.status == "PASS" else 1)


if __name__ == "__main__":
    main()
