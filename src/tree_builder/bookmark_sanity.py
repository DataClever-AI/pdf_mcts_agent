"""
Bookmark sanity pre-check — inspects raw (level, title, page_no) bookmarks
BEFORE section_matcher touches them, and reports (does not silently fix)
structural issues.

Policy (applied by callers, not enforced here):
- negative page_no, out-of-range page_no, empty title: existing clamping in
  section_matcher._compute_page_ranges already handles these safely — warn only.
- duplicate page_no among same-level siblings: warn only. Common and legitimate
  in real manuals (a short section ends and the next heading lands on the same
  page) — section_matcher's reading-order-anchored content assignment already
  handles this correctly (validated against real multi-hundred-section manuals,
  including one with 52 such same-page sibling pairs). It only yields a
  degenerate 1-page range, not a corrupted one.
- out-of-order page_no among same-level siblings: hard failure. Violates
  _compute_page_ranges's monotonicity assumption in a way reading-order
  anchoring cannot compensate for — produces overlapping/inverted ranges,
  not just a narrow one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_HARD_FAILURE_KINDS: frozenset[str] = frozenset({"out_of_order"})


@dataclass
class BookmarkIssue:
    kind: str  # "out_of_range" | "negative" | "duplicate_page" | "out_of_order" | "empty_title"
    index: int  # index into the bookmarks list
    detail: str


@dataclass
class BookmarkSanityReport:
    issues: list[BookmarkIssue] = field(default_factory=list)

    @property
    def has_hard_failures(self) -> bool:
        return any(i.kind in _HARD_FAILURE_KINDS for i in self.issues)


def check_bookmark_sanity(
    bookmarks: list[tuple[int, str, int]],
    total_pages: int,
) -> BookmarkSanityReport:
    """
    Inspect raw bookmarks for structural problems.

    Does not modify bookmarks or apply clamping — that still happens downstream
    in section_matcher._compute_page_ranges. This only detects and reports so
    the caller can decide whether to proceed or abort.
    """
    report = BookmarkSanityReport()
    seen_pages_by_level: dict[int, set[int]] = {}
    prev_page_by_level: dict[int, int] = {}

    for i, (level, title, page_no) in enumerate(bookmarks):
        if page_no < 1:
            report.issues.append(BookmarkIssue(
                "negative", i, f"bookmark[{i}] '{title}' has page_no={page_no} (<1)"
            ))
        elif page_no > total_pages:
            report.issues.append(BookmarkIssue(
                "out_of_range", i,
                f"bookmark[{i}] '{title}' has page_no={page_no} > total_pages={total_pages}"
            ))

        if not title or not title.strip():
            report.issues.append(BookmarkIssue(
                "empty_title", i, f"bookmark[{i}] has empty/whitespace title (page_no={page_no})"
            ))

        siblings = seen_pages_by_level.setdefault(level, set())
        if page_no in siblings:
            report.issues.append(BookmarkIssue(
                "duplicate_page", i,
                f"bookmark[{i}] '{title}' duplicates page_no={page_no} "
                f"at level={level} (already seen at this level)"
            ))
        siblings.add(page_no)

        prev_page = prev_page_by_level.get(level)
        if prev_page is not None and page_no < prev_page:
            report.issues.append(BookmarkIssue(
                "out_of_order", i,
                f"bookmark[{i}] '{title}' page_no={page_no} is before previous "
                f"sibling's page_no={prev_page} at level={level}"
            ))
        prev_page_by_level[level] = page_no

    return report
