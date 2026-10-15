"""
LangGraph node functions — one function per pipeline step.

Split into one module per pipeline stage (guard, memory, retrieval, mcts,
answer, judge, persistence, routing) — was a single 900+ line file mixing
orchestration for all 11 nodes plus their private heuristics. This __init__
re-exports the same public names the old flat module did, so every existing
import (`from src.agent.nodes import guard_query, ...` in graph.py, tests,
scripts) keeps working unchanged.

Each node receives AgentState, returns a partial state update dict.
Nodes skip with pass-through if state["error"] is already set,
except handle_error which consumes it.

VLM tasks:
  expand_query     → query_understanding (fallback: original query)
  rank_sections    → section_ranking x20 parallel (fallback: BM25 scores)
  run_mcts         → rollout_scoring xN parallel (fallback: heuristic)
  generate_answer  → answer_generation (fallback: raw evidence)
  judge_answer     → judge_answer (fallback: WARN)
"""
from __future__ import annotations

from src.agent.nodes.answer import generate_answer
from src.agent.nodes.guard import guard_query
from src.agent.nodes.judge import judge_answer
from src.agent.nodes.mcts import refine_path, run_mcts
from src.agent.nodes.memory import summarize_if_needed
from src.agent.nodes.persistence import handle_error, store_memory
from src.agent.nodes.retrieval import (
    expand_evidence,
    expand_query,
    rank_sections,
    retrieve_sections,
)
from src.agent.nodes.routing import route_after_guard, route_after_judge, route_on_error

__all__ = [
    "expand_evidence",
    "expand_query",
    "generate_answer",
    "guard_query",
    "handle_error",
    "judge_answer",
    "rank_sections",
    "refine_path",
    "retrieve_sections",
    "route_after_guard",
    "route_after_judge",
    "route_on_error",
    "run_mcts",
    "store_memory",
    "summarize_if_needed",
]
