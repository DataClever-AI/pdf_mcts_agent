"""
LangGraph agent graph — wires all nodes into a compiled StateGraph.

Topology (2026 canonical memory pattern):
  START → guard_query
        → [blocked: END]
        → summarize_if_needed   ← compresses messages when > 10
        → expand_query          ← uses messages + summary for follow-ups
        → retrieve_sections → rank_sections
        → expand_evidence → run_mcts → generate_answer
        → judge_answer → [refine_path → generate_answer → judge_answer | done]
        → store_memory          ← appends new Q&A to messages

  refine_path feeds generate_answer directly, NOT run_mcts again — it
  already runs its own explorer.search() internally and sets mcts_path.
  routing back through run_mcts would immediately discard that result with
  a fresh, non-patched rescore (confirmed via direct test: this was the
  actual prior wiring, and it doubled MCTS's VLM cost on every retry for a
  result that was thrown away one node later).
        → END

Error routing: every fallible node → handle_error → store_memory → END.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    expand_evidence,
    expand_query,
    generate_answer,
    guard_query,
    handle_error,
    judge_answer,
    rank_sections,
    refine_path,
    retrieve_sections,
    route_after_guard,
    route_after_judge,
    route_on_error,
    run_mcts,
    store_memory,
    summarize_if_needed,
)
from src.agent.state import AgentState
from src.models.mcts import MCTSConfig
from src.rag.bm25_retrieval import BM25Index
from src.rag.evidence_expander import EvidenceExpander
from src.rag.mcts_explorer import MCTSExplorer


def build_graph(
    bm25_index: BM25Index,
    expander: EvidenceExpander,
    explorer: MCTSExplorer,
    db: Any,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the LangGraph pipeline graph.

    Args:
        bm25_index:      BM25 index over loaded sections.
        expander:        EvidenceExpander for parent/sibling/child expansion.
        explorer:        MCTSExplorer (VLM tree search over the full section tree).
        db:              Database for store_memory persistence.
        checkpointer:    Optional AsyncSqliteSaver for conversation memory.

    Returns:
        Compiled LangGraph graph ready for ainvoke().
    """
    workflow: Any = StateGraph(AgentState)

    # --- Node closures (inject dependencies) ---

    async def _expand_query(state: AgentState) -> dict[str, Any]:
        return await expand_query(state)

    async def _retrieve(state: AgentState) -> dict[str, Any]:
        return await retrieve_sections(state, bm25_index=bm25_index)

    async def _rank(state: AgentState) -> dict[str, Any]:
        return await rank_sections(state)

    async def _expand_ev(state: AgentState) -> dict[str, Any]:
        return await expand_evidence(state, expander=expander)

    async def _mcts(state: AgentState) -> dict[str, Any]:
        return await run_mcts(state, explorer=explorer)

    async def _refine(state: AgentState) -> dict[str, Any]:
        return await refine_path(state, explorer=explorer)

    async def _store(state: AgentState) -> dict[str, Any]:
        return await store_memory(state, db=db)

    # --- Register nodes ---
    workflow.add_node("guard_query", guard_query)
    workflow.add_node("summarize_if_needed", summarize_if_needed)
    workflow.add_node("expand_query", _expand_query)
    workflow.add_node("retrieve_sections", _retrieve)
    workflow.add_node("rank_sections", _rank)
    workflow.add_node("expand_evidence", _expand_ev)
    workflow.add_node("run_mcts", _mcts)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("judge_answer", judge_answer)
    workflow.add_node("refine_path", _refine)
    workflow.add_node("store_memory", _store)
    workflow.add_node("handle_error", handle_error)

    # --- Edges ---
    workflow.add_edge(START, "guard_query")
    # blocked → END (no memory, no retrieval, no summarization)
    # ok      → summarize_if_needed → expand_query → full pipeline
    workflow.add_conditional_edges(
        "guard_query",
        route_after_guard,
        {"blocked": END, "ok": "summarize_if_needed"},
    )
    workflow.add_edge("summarize_if_needed", "expand_query")

    # Fallible nodes with error routing
    _fallible: list[tuple[str, str]] = [
        ("expand_query", "retrieve_sections"),
        ("retrieve_sections", "rank_sections"),
        ("rank_sections", "expand_evidence"),
        ("expand_evidence", "run_mcts"),
        ("run_mcts", "generate_answer"),
        ("generate_answer", "judge_answer"),
    ]
    for src, ok_dst in _fallible:
        workflow.add_conditional_edges(
            src,
            route_on_error,
            {"error": "handle_error", "ok": ok_dst},
        )

    # Judge routing: refine on FAIL (up to 2 attempts) or finish
    workflow.add_conditional_edges(
        "judge_answer",
        route_after_judge,
        {"refine": "refine_path", "done": "store_memory"},
    )

    # Refine path already ran its own explorer.search() and set mcts_path —
    # feed generate_answer directly instead of re-running run_mcts (which
    # would discard refine_path's result with a fresh, unpatched rescore).
    workflow.add_edge("refine_path", "generate_answer")

    # handle_error always routes to store_memory for persistence
    workflow.add_edge("handle_error", "store_memory")
    workflow.add_edge("store_memory", END)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


@asynccontextmanager
async def get_checkpointer(db_path: Path) -> AsyncGenerator[Any, None]:
    """Context manager for AsyncSqliteSaver checkpointer."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        yield checkpointer


class DocumentReasoner:
    """Ergonomic facade over the compiled LangGraph graph."""

    def __init__(
        self,
        sections: list[Any],
        db: Any,
        mcts_config: MCTSConfig | None = None,
        checkpointer: Any = None,
    ) -> None:
        """
        Build and wire all pipeline components.

        Args:
            sections:    All Section objects from load_tree().
            db:          Initialized Database instance.
            mcts_config: Optional MCTS configuration override.
            checkpointer: Optional LangGraph checkpointer for conversation memory.
        """
        sections_by_id = {s.section_id: s for s in sections}
        config = mcts_config or MCTSConfig(use_vlm=True, n_simulations=50)
        # Compute max_hierarchy_level from FULL corpus — not from retrieval candidates.
        # This normalizes depth_bonus correctly regardless of what BM25 retrieves.
        self._max_hierarchy_level = max(
            (s.hierarchy_level for s in sections), default=1
        )

        bm25_index = BM25Index(sections)
        expander = EvidenceExpander(sections_by_id)
        explorer = MCTSExplorer(config, sections=sections)
        self._db = db

        self._graph = build_graph(
            bm25_index=bm25_index,
            expander=expander,
            explorer=explorer,
            db=db,
            checkpointer=checkpointer,
        )

    async def run(
        self,
        query: str,
        messages: list[dict[str, Any]] | None = None,
        summary: str | None = None,
        conversation_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """
        Execute the full pipeline for a single query.

        Args:
            query:           User query string.
            messages:        Accumulated Q&A pairs from this session (canonical memory).
            summary:         Rolling VLM summary of older messages.
            conversation_id: Session ID for SQLite feedback scoping.
            thread_id:       LangGraph checkpointer thread ID (optional).

        Returns:
            Final AgentState. Read state["messages"] and state["summary"] after
            the call to carry memory into the next query.
        """
        initial: AgentState = {
            "max_hierarchy_level": self._max_hierarchy_level,
            "manual_category": None,
            "query": query,
            "expanded_query": None,
            "original_query": None,
            "sub_queries": [],
            "candidate_sections": [],
            "ranked_sections": [],
            "rank_vlm_scores": {},
            "evidence": None,
            "mcts_path": None,
            "mcts_score_cache": {},
            "mcts_n_simulations": 0,
            "blocked": False,
            "block_category": None,
            "is_followup": False,
            "answer": None,
            "answer_blocked": False,
            "answer_confidence": None,
            "fallback_used": False,
            "judge_result": None,
            "judge_attempts": 0,
            "feedback": None,
            "messages": list(messages or []),
            "summary": summary,
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "store_memory_done": False,
            "feedback_id": None,
            "error": None,
            "error_layer": None,
            "logs": [],
            "last_answer_prompt": None,
            "last_answer_response": None,
            "last_judge_prompt": None,
            "last_judge_response": None,
        }

        config: dict[str, Any] = {}
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        raw: Any = await self._graph.ainvoke(initial, config=config or None)
        result: AgentState = raw
        return result
