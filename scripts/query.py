#!/usr/bin/env python3
"""
CLI: Query a tree.json with MCTS RAG pipeline.

Usage:
    uv run python scripts/query.py --tree data/forester_tree_full.json \
        --query "What are the safety procedures for door panel removal?"
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import DocumentReasoner
from src.persistence.database import Database
from src.rag.loader import load_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS RAG query over tree.json")
    parser.add_argument("--tree", type=Path, required=True, help="Path to tree.json")
    parser.add_argument("--query", "-q", required=True, help="Query string")
    parser.add_argument("--db", type=Path, default=Path("data/rag.db"), help="SQLite DB path")
    parser.add_argument("--thread-id", default=None, help="Conversation thread ID")
    parser.add_argument("--no-vlm-mcts", action="store_true",
                        help="Disable VLM in MCTS (use heuristic only)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.tree.exists():
        print(f"ERROR: tree.json not found: {args.tree}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tree: {args.tree}")
    sections = load_tree(args.tree)
    print(f"  {len(sections)} sections loaded")

    # Setup DB
    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = Database(args.db)
    db.create_tables()

    from src.models.mcts import MCTSConfig
    mcts_config = MCTSConfig(
        use_vlm=not args.no_vlm_mcts,
        n_simulations=50,
        max_vlm_concurrency=10,
    )

    reasoner = DocumentReasoner(sections=sections, db=db, mcts_config=mcts_config)
    print(f"\nQuery: {args.query}\n")

    async def _run() -> None:
        result = await reasoner.run(args.query, thread_id=args.thread_id)

        if result.get("blocked"):
            print("=" * 60)
            print(f"QUERY BLOCKED [{result.get('block_category', '?').upper()}]")
            print("=" * 60)
            print(result.get("answer") or "[blocked]")
            return

        if result.get("answer_blocked"):
            print("=" * 60)
            print("ANSWER BLOCKED [SAFETY]")
            print("=" * 60)
            print(result.get("answer") or "[blocked]")
            judge = result.get("judge_result") or {}
            print(f"Safety score: {judge.get('safety', '?')}")
            return

        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result.get("answer") or "[No answer]")
        print()

        judge = result.get("judge_result") or {}
        print(f"Judge:      {judge.get('verdict', '?')} (overall={judge.get('overall', '?'):.2f})")
        print(f"Confidence: {result.get('answer_confidence', '?')}")
        print(f"Fallback:   {result.get('fallback_used', False)}")
        print(f"MCTS path:  {result.get('mcts_path') or []}")
        print(f"Sims:       {result.get('mcts_n_simulations', 0)}")

        logs = result.get("logs") or []
        if logs:
            print("\nAudit trail:")
            for line in logs:
                print(f"  {line}")

        if judge.get("issues"):
            print(f"\nIssues: {judge['issues']}")

    asyncio.run(_run())
    db.close()


if __name__ == "__main__":
    main()
