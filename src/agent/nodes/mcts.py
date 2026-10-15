"""Node 5: run_mcts, Node 6: refine_path (judge FAIL -> penalize weak sections -> re-MCTS)."""
from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.nodes._shared import _is_error, _log
from src.agent.state import AgentState
from src.models.evidence import EvidenceBundle
from src.rag.mcts_explorer import MCTSExplorer

logger = logging.getLogger(__name__)

# Only strong MCTS finds are added to the evidence, and few of them: a weak or
# huge section (a 138-node glossary) costs prompt budget and distracts the answer.
_MIN_ADD_SCORE = 0.8
_MAX_ADD = 3
_MAX_ADD_NODES = 60


def _bm25_priors(state: AgentState) -> dict[str, float]:
    """Candidate BM25/RRF scores normalised to [0, 1], used as exploration priors."""
    candidates = state.get("candidate_sections") or []
    top = max((c.retrieval_score for c in candidates), default=0.0)
    if top <= 0:
        return {}
    return {c.section.section_id: c.retrieval_score / top for c in candidates}


def _merge_found_sections(
    bundle: EvidenceBundle, search_state: Any, explorer: MCTSExplorer
) -> tuple[EvidenceBundle | None, list[str]]:
    """Prepend MCTS top sections missing from the bundle to its primary sections."""
    present = {s.section_id for s in bundle.all_sections}
    known = explorer.sections
    new = [
        known[sid] for sid in search_state.top_ids[:_MAX_ADD]
        if sid in known and sid not in present
        and search_state.score_cache.get(sid, 0.0) >= _MIN_ADD_SCORE
        and (known[sid].node_count or 0) <= _MAX_ADD_NODES
    ]
    if not new:
        return None, []
    primary = [*new, *bundle.primary_sections]
    merged = EvidenceBundle(
        primary_sections=primary,
        parent_sections=bundle.parent_sections,
        sibling_sections=bundle.sibling_sections,
        child_sections=bundle.child_sections,
        related_nodes=bundle.related_nodes,
        total_node_count=bundle.total_node_count + sum(s.node_count or 0 for s in new),
        expansion_trace=[*bundle.expansion_trace, *(f"mcts_found: {s.section_id}" for s in new)],
        query=bundle.query,
    )
    return merged, [s.section_id for s in new]


# ---------------------------------------------------------------------------
# Node 5: run_mcts
# ---------------------------------------------------------------------------

async def run_mcts(
    state: AgentState,
    *,
    explorer: MCTSExplorer,
    score_cache_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Vectorless MCTS over the full section tree (see MCTSExplorer).

    The search starts at the document's chapters and descends by VLM-scored
    relevance, so it can reach sections BM25 never surfaced (e.g. a paraphrase
    sharing no words with the section). BM25 candidate scores are passed only
    as an exploration prior. rank_sections' VLM scores are reused for leaf
    sections it already scored.

    Sections the search finds that the ranker's bundle lacks are added to the
    front of the evidence (the answer prompt otherwise never sees them).
    """
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    bundle: EvidenceBundle | None = state.get("evidence")
    query = state.get("original_query") or state.get("query", "")

    if bundle is None:
        return {
            "mcts_path": None, "mcts_score_cache": {}, "mcts_n_simulations": 0,
            **_log("run_mcts", "no evidence bundle, skipped", t0),
        }

    try:
        search_state = await explorer.search(
            bundle, query, score_cache_override=score_cache_override,
            precomputed_scores=state.get("rank_vlm_scores"),
            priors=_bm25_priors(state),
        )
        merged, added = _merge_found_sections(bundle, search_state, explorer)
        out: dict[str, Any] = {
            "mcts_path": search_state.best_path,
            "mcts_score_cache": search_state.score_cache,
            "mcts_n_simulations": search_state.completed_simulations,
            **_log(
                "run_mcts",
                f"best_path={search_state.best_path} top={search_state.top_ids[:3]} "
                f"sims={search_state.completed_simulations}/{search_state.n_simulations} "
                f"expansions={search_state.expansions} vlm_calls={search_state.vlm_calls} "
                f"fallbacks={search_state.vlm_fallbacks} added_to_evidence={added}",
                t0,
            ),
        }
        if merged is not None:
            out["evidence"] = merged
        return out
    except Exception as e:
        logger.error("run_mcts failed: %s", e)
        return {
            "error": f"run_mcts: {e}", "error_layer": "mcts",
            **_log("run_mcts", f"FAILED: {e}", t0),
        }


# ---------------------------------------------------------------------------
# Node 6: refine_path (judge FAIL → penalize weak sections → re-MCTS)
# ---------------------------------------------------------------------------

async def refine_path(
    state: AgentState,
    *,
    explorer: MCTSExplorer,
) -> dict[str, Any]:
    """
    Penalize weak sections from judge FAIL, re-run MCTS with patched cache.

    No-op for all known judge FAIL issues (gap_admission_false_negative,
    ungrounded, incomplete_steps) — weak_section_ids is deliberately left
    empty for each (see build_gap_check_prompt / build_grounding_check_prompt
    / answer_is_incomplete), so there's nothing here to penalize. All three
    failure modes are corrected downstream instead, in generate_answer's
    retry_issue path — see its docstring.

    Feeds generate_answer directly (see graph.py edge) — NOT run_mcts again.
    This function already runs its own explorer.search() and returns
    mcts_path/mcts_score_cache itself; routing back through run_mcts would
    immediately overwrite that with a fresh, non-patched rescore, discarding
    this function's result one node later (this was the actual prior wiring
    — confirmed by direct test, tracked as a real bug, not a hypothetical).
    """
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    judge_result = state.get("judge_result") or {}
    weak_ids = set(judge_result.get("weak_section_ids") or [])
    existing_cache = dict(state.get("mcts_score_cache") or {})

    # Penalize weak sections to 0.1
    patched_cache = {
        sid: (0.1 if sid in weak_ids else score)
        for sid, score in existing_cache.items()
    }

    bundle: EvidenceBundle | None = state.get("evidence")
    query = state.get("original_query") or state.get("query", "")

    if bundle is None:
        return _log("refine_path", "no evidence, skipped", t0)

    try:
        search_state = await explorer.search(
            bundle, query, score_cache_override=patched_cache
        )
        return {
            "mcts_path": search_state.best_path,
            "mcts_score_cache": search_state.score_cache,
            "mcts_n_simulations": search_state.completed_simulations,
            **_log(
                "refine_path",
                f"weak={len(weak_ids)} sections penalized "
                f"(0 means this retry is currently a no-op — see docstring)",
                t0,
            ),
        }
    except Exception as e:
        logger.warning("refine_path failed, keeping original path: %s", e)
        return _log("refine_path", f"FAILED, kept original path: {e}", t0)
