"""Node 0: guard_query — runs first, blocks off-topic / harmful queries fast."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from src.agent.nodes._shared import _binary_reply_is, _is_error, _log
from src.agent.prompts import (
    BLOCK_CATEGORIES,
    build_followup_check_prompt,
    build_guard_prompt,
    get_refusal_message,
)
from src.agent.state import AgentState
from src.vlm.client import VLMError, call_vlm

logger = logging.getLogger(__name__)


async def _check_is_followup(query: str, messages: list[dict[str, Any]] | None) -> bool:
    """
    Does the current query reference/continue the prior conversation?

    Gates conversation-context use in expand_query and generate_answer (see
    state.py module docstring). Fail-safe default is False (treat as an
    isolated new question) — safer for isolation than assuming FOLLOWUP, and
    consistent with "when in doubt, choose NEW" in the prompt itself.
    """
    if not messages:
        return False
    try:
        response = await call_vlm(
            prompt=build_followup_check_prompt(query, messages),
            task="query_guard",
            max_tokens=5,
            temperature=0.0,
        )
        return _binary_reply_is(response.content, "FOLLOWUP")
    except VLMError:
        return False


async def guard_query(state: AgentState) -> dict[str, Any]:
    """
    Safety guardrail — classify query before any retrieval.

    Calls VLM with a binary PASS/BLOCK classification prompt.
    Response format: "PASS" or "BLOCK:<category>".
    If BLOCK → set answer to refusal message, set blocked=True.
    Fallback on VLMError: conservative keyword check only.
    Fast path: <3s, no BM25/MCTS/evidence.

    Also computes is_followup (run concurrently with the guard classification)
    — this is the one place in the graph where it's cheap to compute once and
    thread through to every downstream node that might need conversation
    context (expand_query, generate_answer).
    """
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    query = state.get("query", "")
    messages = state.get("messages")

    # Fast keyword heuristics — catch obvious cases without VLM.
    # Word-boundary match only (not substring) — "malware" as a bare substring
    # would also match inside unrelated words. "kill", "poison", "hack", and
    # "virus" are deliberately excluded even as whole words: they have common
    # legitimate equipment-manual usage ("kill switch", "poison index" on a
    # coolant label, cybersecurity "hacking" questions — already an explicit
    # PASS example below, "antivirus" software questions) that a blind
    # keyword hit would hard-block with no VLM review. Only kept here are
    # terms with no plausible safe technical-manual meaning; the VLM guard
    # classification below (context + domain-aware) is the right judge for
    # the ambiguous ones.
    q_lower = query.lower().strip()
    _dangerous_kws = {
        "bomb", "explosive", "weapon", "suicide", "malware",
        "ransomware", "drug synthesis", "self-harm",
    }
    if any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in _dangerous_kws):
        return {
            "blocked": True,
            "block_category": "dangerous",
            "is_followup": False,
            "answer": get_refusal_message("dangerous"),
            "answer_confidence": "HIGH",
            "judge_result": {
                "verdict": "BLOCKED",
                "overall": 0.0,
                "issues": ["query_guard:dangerous_keyword"],
                "weak_section_ids": [],
            },
            **_log("guard_query", "BLOCK:dangerous (keyword fastpath)", t0),
        }

    # VLM classification (guard verdict + is_followup run concurrently)
    guard_call = call_vlm(
        prompt=build_guard_prompt(
            query,
            manual_category=state.get("manual_category"),
            messages=messages,
        ),
        task="query_guard",
        max_tokens=20,
        temperature=0.0,
    )
    is_followup = await _check_is_followup(query, messages)

    try:
        response = await guard_call
        verdict = response.content.strip().upper()
    except VLMError as e:
        logger.warning("guard_query VLM failed, allowing query: %s", e)
        return {
            "blocked": False, "block_category": None, "is_followup": is_followup,
            **_log("guard_query", f"VLM error, allowed (fail-open): {e}", t0),
        }

    if verdict == "PASS" or not verdict.startswith("BLOCK"):
        return {
            "blocked": False, "block_category": None, "is_followup": is_followup,
            **_log("guard_query", f"PASS is_followup={is_followup}", t0),
        }

    # Parse BLOCK:<category>
    parts = verdict.split(":", 1)
    category = parts[1].strip().lower() if len(parts) > 1 else "off_topic"
    # Normalise to known categories
    if category not in BLOCK_CATEGORIES:
        category = "off_topic"

    logger.info("guard_query BLOCKED: category=%s query=%r", category, query[:80])
    return {
        "blocked": True,
        "block_category": category,
        "is_followup": is_followup,
        "answer": get_refusal_message(category),
        "answer_confidence": "HIGH",
        "judge_result": {
            "verdict": "BLOCKED",
            "overall": 0.0,
            "issues": [f"query_guard:{category}"],
            "weak_section_ids": [],
        },
        **_log("guard_query", f"BLOCK:{category}", t0),
    }
