"""retrieve_sections: BM25-only, original query + joined sub-questions."""
from __future__ import annotations

import asyncio

from src.agent.nodes.retrieval import retrieve_sections
from src.models.section import Section, SemanticNode
from src.rag.bm25_retrieval import BM25Index


def _sec(section_id: str, title: str, text: str) -> Section:
    return Section(
        section_id=section_id, title=title, hierarchy_level=1, hierarchy_path=f"/{title}",
        semantic_nodes=[SemanticNode(node_id=f"{section_id}_n", canonical_text=text,
                                     semantic_order=1, node_type="paragraph")],
        semantic_order_start=1, semantic_order_end=1, node_count=1, child_sections=[],
        page_start=1, page_end=1, confidence=1.0, structural_source="toc",
        inference_reasons=[], evidence_source_ids=[],
    )


def _index() -> BM25Index:
    return BM25Index([
        _sec("sec_use", "Intended Use Statement", "The monitor is intended for hemodynamic monitoring."),
        _sec("sec_noise", "Safety Precautions", "Safety precautions to avoid potential issues when using."),
    ])


def test_original_query_keeps_the_right_section_when_subquestions_drift() -> None:
    state = {
        "query": "monitor intended use",
        "original_query": "monitor intended use",
        # drifted sub-questions that share nothing with the target title
        "sub_queries": ["What safety precautions avoid potential issues when using it?"],
        "logs": [],
    }

    out = asyncio.run(retrieve_sections(state, bm25_index=_index()))  # type: ignore[arg-type]

    assert out["candidate_sections"][0].section.section_id == "sec_use"


def test_works_without_subqueries() -> None:
    state = {"query": "intended use", "original_query": "intended use", "sub_queries": [], "logs": []}

    out = asyncio.run(retrieve_sections(state, bm25_index=_index()))  # type: ignore[arg-type]

    assert out["candidate_sections"][0].section.section_id == "sec_use"
