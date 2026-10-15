"""Tests for Database — feedback persistence and human-in-the-loop rating."""
from __future__ import annotations

from src.persistence.database import Database


def _db() -> Database:
    db = Database(":memory:")
    db.create_tables()
    return db


class TestFeedbackRating:
    def test_insert_feedback_returns_usable_id(self) -> None:
        db = _db()
        feedback_id = db.insert_feedback(
            conversation_id="conv1", question="How do I replace the battery?",
        )
        assert feedback_id > 0
        row = db.get_feedback_by_id(feedback_id)
        assert row is not None
        assert row["question"] == "How do I replace the battery?"
        assert row["rating"] is None

    def test_update_feedback_rating_attaches_rating_and_correction(self) -> None:
        db = _db()
        feedback_id = db.insert_feedback(conversation_id="conv1", question="q")
        db.update_feedback_rating(feedback_id, rating=2, correction="missed step 3")
        row = db.get_feedback_by_id(feedback_id)
        assert row is not None
        assert row["rating"] == 2
        assert row["correction"] == "missed step 3"

    def test_update_feedback_rating_without_correction(self) -> None:
        db = _db()
        feedback_id = db.insert_feedback(conversation_id="conv1", question="q")
        db.update_feedback_rating(feedback_id, rating=5)
        row = db.get_feedback_by_id(feedback_id)
        assert row is not None
        assert row["rating"] == 5
        assert row["correction"] is None

    def test_get_feedback_by_id_missing_returns_none(self) -> None:
        db = _db()
        assert db.get_feedback_by_id(999) is None

    def test_insert_human_correction_links_to_feedback(self) -> None:
        db = _db()
        feedback_id = db.insert_feedback(conversation_id="conv1", question="q")
        correction_id = db.insert_human_correction(
            feedback_id=feedback_id,
            original_question="q",
            wrong_sections=["sec_0001"],
            correction_text="the answer skipped a step",
        )
        assert correction_id > 0
