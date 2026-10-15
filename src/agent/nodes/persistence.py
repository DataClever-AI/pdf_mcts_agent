"""Node 9: store_memory, Node 10: handle_error."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.agent.nodes._shared import _log
from src.agent.prompts import format_evidence_as_fallback
from src.agent.state import AgentState
from src.models.evidence import EvidenceBundle

logger = logging.getLogger(__name__)

_LOGS_DIR = Path("logs")


def _write_audit_log(state: AgentState, query: str, judge: dict[str, Any]) -> str:
    """
    Flush this query's audit trail (state["logs"]) to a .log file, plus a
    .json sidecar with the (prompt, completion) pairs for the two highest
    fine-tuning-value VLM calls: final answer and quality judge. Guard,
    query-expansion, section-ranking, and MCTS-rollout calls are NOT
    captured here — high call volume, low value as fine-tuning examples for
    a final-answer model — scoped down deliberately, not an oversight.

    Returns a short status string for the caller's own log line. Never
    raises — a logging failure must not affect the pipeline's real output.
    """
    try:
        _LOGS_DIR.mkdir(exist_ok=True)
        conv_id = state.get("conversation_id") or "anon"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"{conv_id}_{stamp}"

        header = (
            f"Query: {query}\n"
            f"Blocked: {state.get('blocked')} ({state.get('block_category')})\n"
            f"Confidence: {state.get('answer_confidence')}\n"
            f"Judge: {judge.get('verdict')} overall={judge.get('overall')}\n"
            + "-" * 60 + "\n"
        )
        (_LOGS_DIR / f"{base}.log").write_text(
            header + "\n".join(state.get("logs") or []) + "\n",
            encoding="utf-8",
        )

        sidecar = {
            "query": query,
            "expanded_query": state.get("expanded_query"),
            "sub_queries": state.get("sub_queries"),
            "answer": state.get("answer"),
            "judge_result": judge,
            "answer_prompt": state.get("last_answer_prompt"),
            "answer_response": state.get("last_answer_response"),
            "judge_prompt": state.get("last_judge_prompt"),
            "judge_response": state.get("last_judge_response"),
        }
        (_LOGS_DIR / f"{base}.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
        return f"flushed → logs/{base}.log (+.json)"
    except OSError as e:
        logger.warning("store_memory: failed to write log files: %s", e)
        return f"log write failed: {e}"


async def store_memory(
    state: AgentState,
    *,
    db: Any,
) -> dict[str, Any]:
    """
    Persist query+answer+judge to SQLite, and the full audit trail + VLM
    (prompt, completion) pairs to ./logs/ for review and fine-tuning dataset
    extraction. Always runs — never skipped.

    Stores:
    - feedback row (every query, including blocked/fallback)
    - mcts_run row (only when mcts_path exists)
    - {conversation_id}_{timestamp}.log  — human-readable audit trail
    - {conversation_id}_{timestamp}.json — answer/judge prompt+completion pairs
    """
    query = state.get("query", "")
    answer = state.get("answer") or ""
    judge = state.get("judge_result") or {}

    # Always persist feedback
    feedback_id: int | None = None
    try:
        feedback_id = db.insert_feedback(
            conversation_id=state.get("conversation_id"),
            question=query,
            expanded_query=state.get("expanded_query"),
            retrieved_section_ids=[
                getattr(getattr(s, "section", None), "section_id", "?")
                for s in (state.get("ranked_sections") or [])
            ],
            mcts_path=state.get("mcts_path") or [],
            answer=answer[:2000] if answer else None,
            judge_verdict=judge.get("verdict"),
            judge_overall=float(judge.get("overall", 0.5)) if judge else None,
            fallback_used=bool(state.get("fallback_used")),
        )
    except Exception as e:
        logger.warning("store_memory feedback failed: %s", e)

    # Persist MCTS run if path exists
    try:
        mcts_path = state.get("mcts_path") or []
        if mcts_path:
            db.insert_mcts_run(
                conversation_id=state.get("conversation_id"),
                query=query,
                best_path=mcts_path,
                n_simulations=state.get("mcts_n_simulations") or 0,
                score=float(judge.get("overall", 0.5)),
            )
    except Exception as e:
        logger.warning("store_memory mcts_run failed: %s", e)

    # Append Q&A pair to messages (canonical LangGraph pattern)
    # Blocked or no-answer queries don't add to messages
    new_messages = list(state.get("messages") or [])
    if answer and not state.get("blocked") and not state.get("answer_blocked"):
        new_messages.append({"role": "user", "content": query})
        new_messages.append({"role": "assistant", "content": answer[:1000]})

    log_status = _write_audit_log(state, query, judge)
    return {
        "store_memory_done": True,
        "messages": new_messages,
        "feedback_id": feedback_id,
        **_log("store_memory", log_status),
    }


async def handle_error(state: AgentState) -> dict[str, Any]:
    """Produce fallback answer from partial state. Clears error."""
    error = state.get("error") or "unknown error"
    layer = state.get("error_layer") or "unknown"
    bundle: EvidenceBundle | None = state.get("evidence")

    fallback = (
        format_evidence_as_fallback(bundle)
        if bundle else
        "No evidence retrieved before the error occurred."
    )
    answer = f"[Pipeline error in {layer}] {fallback}"
    logger.error("handle_error: layer=%s error=%s", layer, error)

    return {
        "answer": answer,
        "fallback_used": True,
        "answer_confidence": "LOW",
        "judge_result": {"verdict": "FAIL", "overall": 0.0, "issues": [error]},
        "error": None,
        "error_layer": None,
        **_log("handle_error", f"layer={layer} error={error}"),
    }
