"""Tests for bm25_retrieval — BM25Index and reciprocal_rank_fusion."""
from __future__ import annotations

from src.models.derived import ScoredSection
from src.models.section import Section, SemanticNode
from src.rag.bm25_retrieval import BM25Index, reciprocal_rank_fusion


def _section(section_id: str, title: str, text: str) -> Section:
    return Section(
        section_id=section_id,
        title=title,
        hierarchy_level=1,
        hierarchy_path=f"/{title}",
        semantic_nodes=[
            SemanticNode(node_id=f"{section_id}_n1", canonical_text=text,
                         semantic_order=1, node_type="paragraph"),
        ],
        semantic_order_start=1, semantic_order_end=1, node_count=1,
        child_sections=[], page_start=1, page_end=1,
        confidence=1.0, structural_source="toc", inference_reasons=[], evidence_source_ids=[],
    )


def _scored(section: Section, score: float) -> ScoredSection:
    return ScoredSection(
        section=section, retrieval_score=score,
        title_score=score, path_score=0.0, content_score=0.0,
    )


class TestBM25Index:
    def test_retrieve_ranks_matching_section_first(self) -> None:
        sections = [
            _section("sec_0001", "Battery Installation", "Insert the new battery."),
            _section("sec_0002", "Maintenance Schedule", "Perform maintenance every 6 months."),
        ]
        index = BM25Index(sections)
        results = index.retrieve("how do I replace the battery", top_k=2)
        assert results[0].section.section_id == "sec_0001"

    def test_empty_index_returns_empty(self) -> None:
        index = BM25Index([])
        assert index.retrieve("anything") == []


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        sec_a = _section("a", "A", "text a")
        sec_b = _section("b", "B", "text b")
        ranked = [_scored(sec_a, 5.0), _scored(sec_b, 1.0)]
        fused = reciprocal_rank_fusion([ranked])
        assert [s.section.section_id for s in fused] == ["a", "b"]

    def test_section_ranked_high_in_multiple_lists_wins(self) -> None:
        sec_a = _section("a", "A", "text a")
        sec_b = _section("b", "B", "text b")
        sec_c = _section("c", "C", "text c")
        # "a" is #2 in both lists; "b" is #1 in one list but absent from the
        # other; "c" is #1 in the other but absent from the first. RRF should
        # rank "a" above both "b" and "c" because it appears near the top of
        # every sub-query's results, not just one.
        list1 = [_scored(sec_b, 9.0), _scored(sec_a, 5.0)]
        list2 = [_scored(sec_c, 9.0), _scored(sec_a, 5.0)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert fused[0].section.section_id == "a"

    def test_fused_score_is_positive_and_sorted_descending(self) -> None:
        sec_a = _section("a", "A", "text a")
        sec_b = _section("b", "B", "text b")
        fused = reciprocal_rank_fusion([[_scored(sec_a, 1.0), _scored(sec_b, 0.5)]])
        assert all(s.retrieval_score > 0 for s in fused)
        assert fused[0].retrieval_score >= fused[1].retrieval_score

    def test_empty_lists_return_empty(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []
