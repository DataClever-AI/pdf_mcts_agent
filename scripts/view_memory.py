#!/usr/bin/env python3
"""
CLI: View conversation memory stored in data/mvp.db.

The LangGraph `messages`/`summary` state (AgentState, state.py) lives only in
Streamlit's st.session_state for the duration of a browser session — it is
never written to disk (the LangGraph checkpointer exists in graph.py but is
never wired up; app.py always passes thread_id=None). The durable record of
"what was asked, what was answered, how it was judged/rated" is the
`feedback` table in data/mvp.db, keyed by conversation_id. This script reads
that table back out in chronological order, i.e. the same shape the agent's
own conversation memory would have if it were persisted.

Usage:
    uv run python scripts/view_memory.py                     # list conversations
    uv run python scripts/view_memory.py <conversation_id>    # show full history
    uv run python scripts/view_memory.py <conversation_id> --with-corrections
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _list_conversations(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT conversation_id,
               COUNT(*) AS turns,
               MIN(created_at) AS first_seen,
               MAX(created_at) AS last_seen
        FROM feedback
        GROUP BY conversation_id
        ORDER BY last_seen DESC
        """
    ).fetchall()
    if not rows:
        print("No conversations recorded yet.")
        return
    print(f"{'conversation_id':<34} {'turns':>5}  {'first_seen':<19}  {'last_seen':<19}")
    print("-" * 80)
    for r in rows:
        print(f"{r['conversation_id'] or '(none)':<34} {r['turns']:>5}  "
              f"{r['first_seen']:<19}  {r['last_seen']:<19}")


def _show_conversation(
    conn: sqlite3.Connection, conversation_id: str, with_corrections: bool
) -> None:
    rows = conn.execute(
        """
        SELECT id, question, expanded_query, answer, judge_verdict, judge_overall,
               rating, correction, fallback_used, created_at
        FROM feedback
        WHERE conversation_id = ?
        ORDER BY created_at ASC
        """,
        (conversation_id,),
    ).fetchall()
    if not rows:
        print(f"No turns found for conversation_id={conversation_id!r}")
        return

    for i, r in enumerate(rows, 1):
        print("=" * 80)
        print(f"Turn {i}  [{r['created_at']}]  feedback_id={r['id']}")
        print(f"Q: {r['question']}")
        if r["expanded_query"] and r["expanded_query"] != r["question"]:
            print(f"   expanded: {r['expanded_query']}")
        answer = r["answer"] or "(no answer)"
        print(f"A: {answer}")
        flags = [f"judge={r['judge_verdict']}({r['judge_overall']})"]
        if r["fallback_used"]:
            flags.append("fallback_used")
        if r["rating"] is not None:
            flags.append(f"rating={r['rating']}/5")
        print("   " + " | ".join(flags))
        if r["correction"]:
            print(f"   human correction: {r['correction']}")

    if with_corrections:
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        corrections = conn.execute(
            f"""
            SELECT feedback_id, correction_text, expert_answer, verified, created_at
            FROM human_corrections
            WHERE feedback_id IN ({placeholders})
            ORDER BY created_at ASC
            """,
            ids,
        ).fetchall()
        if corrections:
            print("=" * 80)
            print("Human corrections:")
            for c in corrections:
                verified = "verified" if c["verified"] else "unverified"
                print(f"  feedback_id={c['feedback_id']} ({verified}, {c['created_at']})")
                print(f"    correction: {c['correction_text']}")
                if c["expert_answer"]:
                    print(f"    expert_answer: {c['expert_answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View LangGraph agent's conversation memory (data/mvp.db feedback table)"
    )
    parser.add_argument(
        "conversation_id", nargs="?", default=None,
        help="Show full turn-by-turn history for this conversation_id. "
             "Omit to list all conversation_ids with turn counts.",
    )
    parser.add_argument(
        "--db", type=Path, default=Path("data/mvp.db"),
        help="Path to SQLite DB (default: data/mvp.db)",
    )
    parser.add_argument(
        "--with-corrections", action="store_true",
        help="Also print linked human_corrections rows.",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        if args.conversation_id:
            _show_conversation(conn, args.conversation_id, args.with_corrections)
        else:
            _list_conversations(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
