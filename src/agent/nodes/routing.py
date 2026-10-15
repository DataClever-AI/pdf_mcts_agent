"""Conditional-edge routing predicates — one place to see all graph branching logic."""
from __future__ import annotations

from src.agent.state import AgentState


def route_after_guard(state: AgentState) -> str:
    """Route after guard_query: 'blocked' if query was blocked, else 'ok'."""
    return "blocked" if state.get("blocked") else "ok"


def route_on_error(state: AgentState) -> str:
    """Route to 'error' if state has an error, else 'ok'."""
    return "error" if state.get("error") else "ok"


def route_after_judge(state: AgentState) -> str:
    """
    Route after judge_answer:
    - Safety hard-block (answer_blocked=True) → 'done' (never refine unsafe answers)
    - FAIL + attempts < 2 → 'refine' (quality issue, try different MCTS path)
    - else → 'done'
    """
    if state.get("answer_blocked"):
        return "done"  # safety block — no retry, return refusal
    result = state.get("judge_result") or {}
    attempts = state.get("judge_attempts") or 0
    if result.get("verdict") == "FAIL" and attempts < 2:
        return "refine"
    return "done"
