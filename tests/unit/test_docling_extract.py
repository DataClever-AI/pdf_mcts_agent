"""Tests for docling_extract's pure-dict helpers (no docling dependency needed)."""
from __future__ import annotations

from src.tree_builder.docling_extract import (
    _generate_windows,
    _merge_batch_dicts,
    _trim_overlap_pages,
)


class TestGenerateWindows:
    def test_single_window_when_under_size(self) -> None:
        assert _generate_windows(total_pages=40, window_size=60, overlap=6) == [(1, 40)]

    def test_overlapping_windows(self) -> None:
        windows = _generate_windows(total_pages=240, window_size=60, overlap=6)
        assert windows == [(1, 60), (55, 114), (109, 168), (163, 222), (217, 240)]


class TestTrimOverlapPages:
    def _text_node(self, page_no: int, text: str) -> dict:
        return {"text": text, "label": "text", "prov": [{"page_no": page_no, "bbox": {}}]}

    def test_drops_pages_below_min_page(self) -> None:
        doc_dict = {
            "texts": [
                self._text_node(55, "overlap page"),
                self._text_node(60, "overlap page 2"),
                self._text_node(61, "new content"),
            ],
            "tables": [],
        }
        trimmed = _trim_overlap_pages(doc_dict, min_page=61)
        assert [n["text"] for n in trimmed["texts"]] == ["new content"]

    def test_keeps_all_when_min_page_is_lowest(self) -> None:
        doc_dict = {"texts": [self._text_node(5, "a"), self._text_node(6, "b")], "tables": []}
        trimmed = _trim_overlap_pages(doc_dict, min_page=1)
        assert len(trimmed["texts"]) == 2


class TestMergeBatchDictsNoDuplicateReadingOrder:
    """
    Regression test for the window-overlap bug: without trimming, the same
    page's content appears twice across window boundaries, and the duplicate
    copy from the later window gets a higher reading_order than content that
    actually follows it in the document. Confirms trimming (as done in
    DoclingExtractionEngine.extract before appending to batch_dicts) fixes this.
    """

    def _text_node(self, page_no: int, text: str) -> dict:
        return {"text": text, "label": "text", "prov": [{"page_no": page_no, "bbox": {}}]}

    def test_trimmed_windows_produce_correct_order(self, caplog: object) -> None:
        import logging

        window1 = {
            "texts": [self._text_node(1, "page1"), self._text_node(2, "page2 (overlap)")],
            "tables": [],
        }
        window2_raw = {
            "texts": [
                self._text_node(2, "page2 (overlap, duplicate)"),
                self._text_node(3, "page3"),
            ],
            "tables": [],
        }
        window2_trimmed = _trim_overlap_pages(window2_raw, min_page=3)

        merged = _merge_batch_dicts([window1, window2_trimmed], logging.getLogger("test"))
        texts_in_order = [t["text"] for t in merged["texts"]]
        assert texts_in_order == ["page1", "page2 (overlap)", "page3"]
