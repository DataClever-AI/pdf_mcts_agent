"""Node 8: judge_answer — two-layer output guardrail + quality judge."""
from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.nodes._shared import _binary_reply_is, _is_error, _log
from src.agent.prompts import (
    answer_has_dangerous_keywords,
    answer_is_gap_admission,
    answer_is_incomplete,
    build_answer_safety_prompt,
    build_gap_check_prompt,
    build_grounding_check_prompt,
    get_safety_block_message,
)
from src.agent.state import AgentState
from src.models.evidence import EvidenceBundle
from src.vlm.client import VLMError, call_vlm

logger = logging.getLogger(__name__)


async def _check_answer_safety(answer: str, query: str) -> bool:
    """
    Two-layer output safety check. Returns True if answer is UNSAFE.

    Layer 1 — keyword heuristic (0ms): catch obvious dangerous content.
    Layer 2 — binary VLM check: simple SAFE/UNSAFE question, not part of judge JSON.

    Fail-open: if VLM fails → treat as SAFE (don't block valid technical answers).
    """
    # Layer 1: fast keyword check
    if answer_has_dangerous_keywords(answer):
        logger.warning("answer_safety: keyword block triggered")
        return True

    # Layer 2: binary VLM check
    try:
        response = await call_vlm(
            prompt=build_answer_safety_prompt(query, answer),
            task="query_guard",   # reuse query_guard task for binary classification
            max_tokens=5,
            temperature=0.0,
        )
        verdict = response.content.strip().upper()
        is_unsafe = verdict.startswith("UNSAFE")
        if is_unsafe:
            logger.warning(
                "answer_safety: VLM binary check UNSAFE for query=%r", query[:60]
            )
        return is_unsafe
    except VLMError as e:
        logger.warning("answer_safety VLM failed, treating as SAFE: %s", e)
        return False  # fail-open — don't block valid technical answers


async def judge_answer(state: AgentState) -> dict[str, Any]:
    """
    Two-layer output guardrail + quality judge.

    Layer A — Answer Safety (runs first, separate from judge):
      - Keyword check on answer content (0ms)
      - Binary VLM SAFE/UNSAFE check (~2s)
      - UNSAFE → hard-block, replace answer, set answer_blocked=True
      - No retry on safety block (never refine dangerous content)

    Layer B — Quality Judge (runs only if answer passed safety):
      - Gap-admission answers ("manual does not cover X"): single YES/NO VLM
        check — does any section's listed CONTENT (not just its title)
        actually answer the question?
      - Committed answers with a numbered procedure: deterministic regex
        completeness check first (answer_is_incomplete, no VLM call) — did
        the answer drop numbered steps present in its source section? Only
        if complete does it get the GROUNDED/UNGROUNDED VLM check: does the
        answer contradict the evidence? (Two focused binary VLM calls
        instead of one compound scored-JSON call — the compound version
        produced self-contradictory output on the 7B judge model, e.g.
        quoting identical answer/reference text as a "contradiction".)
      - FAIL + attempts < 2 → refine_path → generate_answer (issue-specific
        correction injected, see generate_answer's retry_issue)

    Fail-open: VLM errors → WARN/SAFE (don't block valid technical answers).
    """
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    answer = state.get("answer") or ""
    bundle: EvidenceBundle | None = state.get("evidence")
    query = state.get("query", "")
    attempts = (state.get("judge_attempts") or 0) + 1

    if not answer or bundle is None:
        return {
            "judge_result": {"verdict": "WARN", "overall": 0.5, "issues": ["no_answer"]},
            "judge_attempts": attempts,
            "answer_blocked": False,
            **_log("judge_answer", "no answer/evidence, WARN", t0),
        }

    # --- LAYER A: ANSWER SAFETY ---
    is_unsafe = await _check_answer_safety(answer, query)
    if is_unsafe:
        return {
            "judge_result": {
                "verdict": "FAIL",
                "grounding": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "overall": 0.0,
                "issues": ["answer_safety_block"],
                "weak_section_ids": [],
            },
            "judge_attempts": attempts,
            "answer": get_safety_block_message(),
            "answer_blocked": True,
            "answer_confidence": "LOW",
            "fallback_used": True,
            **_log("judge_answer", "Layer A UNSAFE — answer blocked", t0),
        }

    # --- LAYER B: QUALITY JUDGE (two focused binary calls + 1 deterministic) ---
    manual_category = state.get("manual_category")
    judge_result: dict[str, Any]
    try:
        if answer_is_gap_admission(answer):
            judge_prompt = build_gap_check_prompt(query, bundle, manual_category=manual_category)
            response = await call_vlm(
                prompt=judge_prompt,
                task="judge_answer",
                max_tokens=5,
                temperature=0.0,
            )
            judge_response_text = response.content
            evidence_actually_answers_it = _binary_reply_is(response.content, "YES")
            judge_result = (
                {
                    "verdict": "FAIL",
                    "overall": 0.2,
                    "issues": ["gap_admission_false_negative"],
                    "weak_section_ids": [],
                }
                if evidence_actually_answers_it
                else {"verdict": "PASS", "overall": 0.85, "issues": [], "weak_section_ids": []}
            )
        elif answer_is_incomplete(answer, bundle):
            # Deterministic regex check (no VLM call) — see answer_is_incomplete
            # docstring for why this is more reliable than asking the 7B judge
            # to count steps in two texts.
            judge_prompt = None
            judge_response_text = "DETERMINISTIC:incomplete_steps (no VLM call)"
            judge_result = {
                "verdict": "FAIL", "overall": 0.3,
                "issues": ["incomplete_steps"], "weak_section_ids": [],
            }
        else:
            judge_prompt = build_grounding_check_prompt(
                query, answer, bundle, manual_category=manual_category
            )
            response = await call_vlm(
                prompt=judge_prompt,
                task="judge_answer",
                max_tokens=5,
                temperature=0.0,
            )
            judge_response_text = response.content
            is_grounded = _binary_reply_is(response.content, "GROUNDED")
            judge_result = (
                {"verdict": "PASS", "overall": 0.8, "issues": [], "weak_section_ids": []}
                if is_grounded
                else {"verdict": "FAIL", "overall": 0.2, "issues": ["ungrounded"],
                      "weak_section_ids": []}
            )

        overall = judge_result["overall"]
        confidence = "HIGH" if overall >= 0.75 else ("MEDIUM" if overall >= 0.5 else "LOW")
        issue = (judge_result.get("issues") or ["none"])[0]
        return {
            "judge_result": judge_result,
            "judge_attempts": attempts,
            "answer_blocked": False,
            "answer_confidence": confidence,
            "last_judge_prompt": judge_prompt,
            "last_judge_response": judge_response_text,
            **_log(
                "judge_answer",
                f"{judge_result['verdict']} overall={overall:.2f} attempt={attempts} issue={issue}",
                t0,
            ),
        }
    except VLMError as e:
        logger.warning("judge_answer quality VLM fallback: %s", e)
        return {
            "judge_result": {
                "verdict": "WARN",
                "overall": 0.5,
                "issues": [f"judge_vlm_error: {e}"],
                "weak_section_ids": [],
            },
            "judge_attempts": attempts,
            "answer_blocked": False,
            "answer_confidence": "MEDIUM",
            **_log("judge_answer", f"quality VLM error, WARN fail-open: {e}", t0),
        }
