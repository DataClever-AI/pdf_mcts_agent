"""Tests for bookmark_sanity — pre-check for malformed bookmark lists."""
from __future__ import annotations

from src.tree_builder.bookmark_sanity import check_bookmark_sanity


class TestOutOfRange:
    def test_flags_out_of_range(
        self, malformed_bookmarks_out_of_range: list[tuple[int, str, int]]
    ) -> None:
        report = check_bookmark_sanity(malformed_bookmarks_out_of_range, total_pages=14)
        kinds = {i.kind for i in report.issues}
        assert "out_of_range" in kinds
        assert report.has_hard_failures is False


class TestNegative:
    def test_flags_negative(
        self, malformed_bookmarks_negative_page: list[tuple[int, str, int]]
    ) -> None:
        report = check_bookmark_sanity(malformed_bookmarks_negative_page, total_pages=14)
        kinds = {i.kind for i in report.issues}
        assert "negative" in kinds
        assert report.has_hard_failures is False


class TestDuplicatePage:
    def test_flags_duplicate_but_not_hard_failure(
        self, malformed_bookmarks_duplicate_page: list[tuple[int, str, int]]
    ) -> None:
        """Duplicate page_no across same-level siblings is common in real manuals
        (short section, next heading on same page) — section_matcher's
        reading-order anchoring already handles it correctly, so this is warn-only."""
        report = check_bookmark_sanity(malformed_bookmarks_duplicate_page, total_pages=14)
        kinds = {i.kind for i in report.issues}
        assert "duplicate_page" in kinds
        assert report.has_hard_failures is False


class TestOutOfOrder:
    def test_flags_out_of_order_and_is_hard_failure(
        self, malformed_bookmarks_out_of_order: list[tuple[int, str, int]]
    ) -> None:
        report = check_bookmark_sanity(malformed_bookmarks_out_of_order, total_pages=14)
        kinds = {i.kind for i in report.issues}
        assert "out_of_order" in kinds
        assert report.has_hard_failures is True


class TestEmptyTitle:
    def test_flags_empty_title(
        self, malformed_bookmarks_empty_title: list[tuple[int, str, int]]
    ) -> None:
        report = check_bookmark_sanity(malformed_bookmarks_empty_title, total_pages=14)
        kinds = {i.kind for i in report.issues}
        assert "empty_title" in kinds
        assert report.has_hard_failures is False


class TestCleanBookmarks:
    def test_no_issues_on_happy_path(self, sample_bookmarks: list[tuple[int, str, int]]) -> None:
        report = check_bookmark_sanity(sample_bookmarks, total_pages=14)
        assert report.issues == []
        assert report.has_hard_failures is False


class TestMultipleIssues:
    def test_combined_issues_all_detected(self) -> None:
        bookmarks = [
            (1, "Chapter 1", -1),  # negative
            (2, "   ", 3),  # empty title
            (2, "Section 1.2", 999),  # out of range
        ]
        report = check_bookmark_sanity(bookmarks, total_pages=14)
        kinds = {i.kind for i in report.issues}
        assert {"negative", "empty_title", "out_of_range"} <= kinds
