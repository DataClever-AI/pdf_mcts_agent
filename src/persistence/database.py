"""
SQLite persistence — users, feedback, MCTS runs, retrieval patterns.

Synchronous SQLite (not aiosqlite) — called from store_memory node
which runs in an executor if needed. The LangGraph checkpointer
uses its own separate aiosqlite connection for graph state.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from pathlib import Path
from typing import Any


class Database:
    """SQLite database for feedback and MCTS run storage."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._path, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                username    TEXT UNIQUE NOT NULL,
                email       TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                salt        TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id     TEXT,
                question            TEXT NOT NULL,
                expanded_query      TEXT,
                retrieved_section_ids TEXT,
                mcts_path           TEXT,
                answer              TEXT,
                judge_verdict       TEXT,
                judge_overall       REAL,
                rating              INTEGER CHECK (rating BETWEEN 1 AND 5),
                correction          TEXT,
                fallback_used       INTEGER DEFAULT 0,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS human_corrections (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id         INTEGER REFERENCES feedback(id),
                original_question   TEXT,
                wrong_sections      TEXT,
                correction_text     TEXT,
                expert_answer       TEXT,
                verified            INTEGER DEFAULT 0,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS retrieval_patterns (
                query_pattern       TEXT PRIMARY KEY,
                successful_section_ids TEXT,
                success_count       INTEGER DEFAULT 1,
                last_updated        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS mcts_runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id     TEXT,
                query               TEXT,
                best_path           TEXT,
                n_simulations       INTEGER,
                score               REAL,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    def insert_feedback(
        self,
        *,
        conversation_id: str | None,
        question: str,
        expanded_query: str | None = None,
        retrieved_section_ids: list[str] | None = None,
        mcts_path: list[str] | None = None,
        answer: str | None = None,
        judge_verdict: str | None = None,
        judge_overall: float | None = None,
        rating: int | None = None,
        correction: str | None = None,
        fallback_used: bool = False,
    ) -> int:
        """Insert a feedback record. Returns inserted row id."""
        cursor = self._conn.execute(
            """INSERT INTO feedback
               (conversation_id, question, expanded_query, retrieved_section_ids,
                mcts_path, answer, judge_verdict, judge_overall,
                rating, correction, fallback_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id,
                question,
                expanded_query,
                json.dumps(retrieved_section_ids or []),
                json.dumps(mcts_path or []),
                answer,
                judge_verdict,
                judge_overall,
                rating,
                correction,
                int(fallback_used),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def update_feedback_rating(
        self,
        feedback_id: int,
        rating: int,
        correction: str | None = None,
    ) -> None:
        """
        Attach a human rating (and optional free-text correction) to an
        existing feedback row — human-in-the-loop review after the answer
        was already shown, not part of the original insert_feedback call.
        """
        self._conn.execute(
            "UPDATE feedback SET rating = ?, correction = ? WHERE id = ?",
            (rating, correction, feedback_id),
        )
        self._conn.commit()

    def get_feedback_by_id(self, feedback_id: int) -> dict[str, Any] | None:
        """Return a single feedback row by id, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
        return dict(row) if row else None

    def insert_human_correction(
        self,
        *,
        feedback_id: int,
        original_question: str,
        wrong_sections: list[str],
        correction_text: str,
        expert_answer: str | None = None,
    ) -> int:
        """Insert a human correction linked to a feedback record."""
        cursor = self._conn.execute(
            """INSERT INTO human_corrections
               (feedback_id, original_question, wrong_sections, correction_text, expert_answer)
               VALUES (?, ?, ?, ?, ?)""",
            (
                feedback_id,
                original_question,
                json.dumps(wrong_sections),
                correction_text,
                expert_answer,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def insert_mcts_run(
        self,
        *,
        conversation_id: str | None,
        query: str,
        best_path: list[str],
        n_simulations: int,
        score: float,
    ) -> int:
        """Insert an MCTS run record."""
        cursor = self._conn.execute(
            """INSERT INTO mcts_runs
               (conversation_id, query, best_path, n_simulations, score)
               VALUES (?, ?, ?, ?, ?)""",
            (
                conversation_id,
                query,
                json.dumps(best_path),
                n_simulations,
                score,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def upsert_retrieval_pattern(
        self,
        *,
        query_pattern: str,
        section_ids: list[str],
    ) -> None:
        """Upsert a successful retrieval pattern."""
        self._conn.execute(
            """INSERT INTO retrieval_patterns
               (query_pattern, successful_section_ids, success_count, last_updated)
               VALUES (?, ?, 1, CURRENT_TIMESTAMP)
               ON CONFLICT(query_pattern) DO UPDATE SET
                 successful_section_ids = excluded.successful_section_ids,
                 success_count = success_count + 1,
                 last_updated = CURRENT_TIMESTAMP""",
            (query_pattern, json.dumps(section_ids)),
        )
        self._conn.commit()

    def get_retrieval_pattern(self, query_pattern: str) -> dict[str, Any] | None:
        """Look up a retrieval pattern by normalized query."""
        row = self._conn.execute(
            "SELECT * FROM retrieval_patterns WHERE query_pattern = ?",
            (query_pattern,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_recent_feedback(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return most recent feedback records (global, all sessions)."""
        rows = self._conn.execute(
            """SELECT * FROM feedback
               WHERE answer IS NOT NULL AND answer != ''
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]  # oldest first for readability

    def get_recent_feedback_by_session(
        self, conversation_id: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Return most recent feedback for a specific conversation session."""
        rows = self._conn.execute(
            """SELECT * FROM feedback
               WHERE conversation_id = ?
                 AND answer IS NOT NULL AND answer != ''
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]  # oldest first

    # ---------------------------------------------------------------------------
    # User CRUD
    # ---------------------------------------------------------------------------

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """Hash password with PBKDF2-HMAC-SHA256 + salt."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=260_000,
        ).hex()

    def create_user(
        self,
        username: str,
        password: str,
        email: str | None = None,
    ) -> str:
        """
        Create a new user. Returns the new user_id (UUID).

        Raises ValueError if username or email already exists.
        """
        # Check uniqueness
        if self._conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone():
            raise ValueError(f"Username '{username}' already exists.")
        if email and self._conn.execute(
            "SELECT 1 FROM users WHERE email = ?", (email,)
        ).fetchone():
            raise ValueError(f"Email '{email}' already in use.")

        user_id = secrets.token_hex(16)
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)

        self._conn.execute(
            "INSERT INTO users (id, username, email, password_hash, salt) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, username, email, password_hash, salt),
        )
        self._conn.commit()
        return user_id

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Return user dict by username, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Return user dict by user_id, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None

    def verify_password(self, username: str, password: str) -> str | None:
        """
        Verify username + password. Returns user_id on success, None on failure.
        Constant-time comparison to prevent timing attacks.
        """
        user = self.get_user_by_username(username)
        if not user:
            # Run hash anyway to prevent timing oracle
            self._hash_password(password, "dummy_salt_000000")
            return None
        expected = self._hash_password(password, user["salt"])
        if secrets.compare_digest(expected, user["password_hash"]):
            return str(user["id"])
        return None

    def update_password(self, user_id: str, new_password: str) -> None:
        """Update a user's password."""
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(new_password, salt)
        self._conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
            (password_hash, salt, user_id),
        )
        self._conn.commit()

    def delete_user(self, user_id: str) -> None:
        """Delete user account and all associated feedback."""
        self._conn.execute("DELETE FROM feedback WHERE conversation_id = ?", (user_id,))
        self._conn.execute("DELETE FROM mcts_runs WHERE conversation_id = ?", (user_id,))
        self._conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self._conn.commit()

    def list_users(self) -> list[dict[str, Any]]:
        """Return all users (id, username, email, created_at) — no password data."""
        rows = self._conn.execute(
            "SELECT id, username, email, created_at FROM users ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
