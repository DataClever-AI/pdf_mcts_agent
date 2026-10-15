"""Nodes 1-4: expand_query, retrieve_sections, rank_sections, expand_evidence."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.agent.nodes._shared import _is_error, _log
from src.agent.prompts import (
    build_query_expansion_prompt,
    build_section_ranking_prompt,
    parse_expanded_queries,
)
from src.agent.state import AgentState
from src.models.derived import RankedSection, ScoredSection
from src.rag.bm25_retrieval import BM25Index, reciprocal_rank_fusion
from src.rag.evidence_expander import EvidenceExpander
from src.vlm.client import VLMError, call_vlm, parse_float_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1: expand_query
# ---------------------------------------------------------------------------

async def expand_query(state: AgentState) -> dict[str, Any]:
    """
    Reformulate query into 3 sub-questions via VLM.

    Only uses messages + summary when is_followup is True (set by guard_query)
    — otherwise treats the query as fully standalone. Without this gate, a
    stale topic from earlier in the conversation can leak into sub-question
    generation on a completely unrelated new question (observed directly in
    testing: "historical trends" pulled in "HPI" from 3 turns earlier despite
    no pronoun/reference to it), and from there into what gets retrieved.
    """
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    query = state.get("query", "")
    is_followup = state.get("is_followup", False)
    messages: list[dict[str, Any]] = list(state.get("messages") or []) if is_followup else []
    summary = state.get("summary") if is_followup else None
    try:
        response = await call_vlm(
            prompt=build_query_expansion_prompt(
                query, messages, summary,
                manual_category=state.get("manual_category"),
            ),
            task="query_understanding",
            max_tokens=256,
            temperature=0.0,
        )
        sub_queries = parse_expanded_queries(response.content)
        expanded = " ".join(sub_queries) if sub_queries else query
        detail = f"{len(sub_queries)} sub-queries generated"
    except VLMError as e:
        logger.warning("expand_query VLM fallback: %s", e)
        sub_queries = [query]
        expanded = query
        detail = f"VLM error, fallback to original query: {e}"

    return {
        "expanded_query": expanded,
        "original_query": query,
        "sub_queries": sub_queries,
        **_log("expand_query", detail, t0),
    }


# ---------------------------------------------------------------------------
# Node 2: retrieve_sections
# ---------------------------------------------------------------------------

async def retrieve_sections(
    state: AgentState,
    *,
    bm25_index: BM25Index,
) -> dict[str, Any]:
    """
    Lexical (BM25) retrieval. No VLM, no embeddings.

    Two ranked lists fused via Reciprocal Rank Fusion: the user's ORIGINAL
    query, and the sub-questions from expand_query joined into one string.
    The original query keeps the user's exact terms (VLM sub-questions drift:
    "monitor's intended use" expanded into questions that buried the right
    section at rank 27); the joined sub-questions add vocabulary the user
    didn't use. Fusing one list per sub-question instead was measured worse
    (top-1 14/20 vs 18/20) — each short sub-question adds its own noise.

    BM25 only seeds candidates and the MCTS prior. Sections it misses (a
    paraphrase with no shared words) are still reachable: run_mcts searches
    the whole section tree, not just these candidates.
    """
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    original = state.get("original_query") or state.get("query", "")
    sub_queries = state.get("sub_queries") or []
    joined = " ".join(sub_queries)
    queries = [original] + ([joined] if joined and joined != original else [])
    try:
        ranked_lists = [bm25_index.retrieve(q, top_k=30) for q in queries]
        scored = (
            ranked_lists[0] if len(ranked_lists) == 1
            else reciprocal_rank_fusion(ranked_lists)[:30]
        )
        top = (
            f"{scored[0].section.section_id} score={scored[0].retrieval_score:.4f}"
            if scored else "none"
        )
        detail = (
            f"{len(scored)} candidates from {len(queries)} bm25 "
            f"{'query' if len(queries) == 1 else 'queries'}, top={top}"
        )
        return {
            "candidate_sections": scored,
            **_log("retrieve_sections", detail, t0),
        }
    except Exception as e:
        logger.error("retrieve_sections failed: %s", e)
        return {
            "error": f"retrieve_sections: {e}", "error_layer": "bm25",
            **_log("retrieve_sections", f"FAILED: {e}", t0),
        }


# ---------------------------------------------------------------------------
# Node 3: rank_sections (VLM parallel batch)
# ---------------------------------------------------------------------------

async def rank_sections(state: AgentState) -> dict[str, Any]:
    """Hybrid ranking: VLM relevance (parallel) + BM25 + structural signals."""
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    candidates: list[ScoredSection] = state.get("candidate_sections") or []
    # Rank against what the user actually asked — the VLM sub-questions can
    # drift and the ranking prompt truncates long queries.
    expanded_query = state.get("original_query") or state.get("query", "")

    if not candidates:
        return {"ranked_sections": [], **_log("rank_sections", "no candidates to rank", t0)}

    # Normalise BM25 scores to [0, 1]
    max_bm25 = max((c.retrieval_score for c in candidates), default=1.0)
    if max_bm25 == 0.0:
        max_bm25 = 1.0

    # Parallel VLM scoring (semaphore=10)
    sem = asyncio.Semaphore(10)

    async def _vlm_score(scored: ScoredSection) -> float:
        async with sem:
            try:
                r = await call_vlm(
                    prompt=build_section_ranking_prompt(
                        expanded_query, scored.section
                    ),
                    task="section_ranking",
                    max_tokens=8,
                    temperature=0.0,
                )
                return parse_float_response(r.content, default=0.5)
            except VLMError:
                return 0.5

    vlm_scores = await asyncio.gather(*[_vlm_score(c) for c in candidates])

    # Depth bonus: relative to the max level of the FULL document corpus.
    # Uses state["max_hierarchy_level"] computed from all sections at load time —
    # NOT from candidates (top-30 BM25 may not include the deepest sections).
    # Works for any manual depth: L1-only, L1-L2, L1-Ln.
    #   depth_bonus = level / max_level  → [1/max_level ... 1.0]
    # L1 in flat manual (max=1): 1.0      (it's the only/deepest level)
    # L1 in 4-level manual:      0.25     (structural container)
    # L4 in 4-level manual:      1.0      (deepest procedural detail)
    # Empty containers (node_count=0): 0.1 regardless of level.
    max_level = max(state.get("max_hierarchy_level") or 1, 1)

    ranked: list[RankedSection] = []
    for scored, vlm_score in zip(candidates, vlm_scores, strict=True):
        sec = scored.section
        norm_bm25 = scored.retrieval_score / max_bm25
        node_count = sec.node_count or 0

        if node_count == 0:
            depth_bonus = 0.1  # empty structural container — penalized
        else:
            depth_bonus = sec.hierarchy_level / max_level  # relative depth [0,1]

        composite = (
            norm_bm25        * 0.35
            + vlm_score      * 0.35
            + sec.confidence * 0.20
            + depth_bonus    * 0.10
        )
        ranked.append(
            RankedSection(
                section=sec,
                retrieval_score=scored.retrieval_score,
                vlm_score=vlm_score,
                ranking_score=composite,
                rank=0,
            )
        )

    ranked.sort(key=lambda r: r.ranking_score, reverse=True)
    top_n = ranked[:12]   # top_k=30 input → keep top 12 ranked
    for i, r in enumerate(top_n):
        object.__setattr__(r, "rank", i + 1)

    top1 = (
        f"{top_n[0].section.section_id} composite={top_n[0].ranking_score:.3f} "
        f"(bm25_raw={top_n[0].retrieval_score:.2f} vlm={top_n[0].vlm_score:.2f})"
        if top_n else "none"
    )
    # Exposed so run_mcts can reuse these VLM relevance scores instead of
    # asking the VLM the same section+query relevance question a second time
    # via rollout_scoring.
    rank_vlm_scores = {r.section.section_id: r.vlm_score for r in top_n}
    return {
        "ranked_sections": top_n,
        "rank_vlm_scores": rank_vlm_scores,
        **_log("rank_sections", f"{len(top_n)} kept, top1={top1}", t0),
    }


# ---------------------------------------------------------------------------
# Node 4: expand_evidence
# ---------------------------------------------------------------------------

async def expand_evidence(
    state: AgentState,
    *,
    expander: EvidenceExpander,
) -> dict[str, Any]:
    """Expand ranked sections into EvidenceBundle (parent/sibling/child)."""
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    ranked: list[RankedSection] = state.get("ranked_sections") or []
    query = state.get("original_query") or state.get("query", "")

    if not ranked:
        return {
            "evidence": None,
            **_log("expand_evidence", "no ranked sections, empty evidence", t0),
        }

    try:
        bundle = expander.expand(ranked, query)
        detail = (
            f"primary={len(bundle.primary_sections)} parent={len(bundle.parent_sections)} "
            f"sibling={len(bundle.sibling_sections)} child={len(bundle.child_sections)} "
            f"related={len(bundle.related_nodes)} nodes={bundle.total_node_count}"
        )
        return {"evidence": bundle, **_log("expand_evidence", detail, t0)}
    except Exception as e:
        logger.error("expand_evidence failed: %s", e)
        return {
            "error": f"expand_evidence: {e}", "error_layer": "expander",
            **_log("expand_evidence", f"FAILED: {e}", t0),
        }
