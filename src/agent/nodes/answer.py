"""Node 7: generate_answer."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.agent.nodes._shared import _is_error, _log
from src.agent.prompts import (
    build_answer_prompt,
    build_section_relevance_prompt,
    ensure_citations,
    format_evidence_as_fallback,
)
from src.agent.state import AgentState
from src.models.evidence import EvidenceBundle
from src.models.section import Section
from src.vlm.client import VLMError, call_vlm

logger = logging.getLogger(__name__)

_MAX_PRIMARY = 12        # sections shown in full when the gate doesn't narrow the evidence
_GATE_CANDIDATES = 6     # top sections checked by the relevance gate
_GATE_MAX_KEEP = 4       # sections that passed the gate and are sent to the generator
_RETRY_PRIMARY = 3       # sections kept when retrying a refusal the gate couldn't confirm


async def _relevant_sections(query: str, bundle: EvidenceBundle) -> list[Section] | None:
    """
    Top sections that individually answer the query, in ranked order.

    Returns [] when none do, and None when the check itself failed (VLM error)
    — the caller then falls back to the full evidence, never blocks on it.
    """
    candidates = bundle.primary_sections[:_GATE_CANDIDATES]
    try:
        replies = await asyncio.gather(*[
            call_vlm(
                prompt=build_section_relevance_prompt(query, sec),
                task="section_ranking",
                max_tokens=5,
                temperature=0.0,
            )
            for sec in candidates
        ])
    except VLMError as e:
        logger.warning("relevance gate VLM failed, using full evidence: %s", e)
        return None
    return [
        sec for sec, r in zip(candidates, replies, strict=True)
        if r.content.strip().upper().startswith("YES")
    ]


async def generate_answer(state: AgentState) -> dict[str, Any]:
    """
    Synthesize answer via VLM. Fallback: return raw evidence.

    Only includes messages + summary as conversation context when is_followup
    is True (set by guard_query) — otherwise generates from evidence alone.
    Without this gate, an unrelated question's wording/accuracy can drift
    purely because of what was discussed earlier in the session (observed
    directly in testing: the same standalone question produced a different
    answer, and a different judge verdict, depending on unrelated prior
    conversation history).
    """
    if _is_error(state):
        return {}
    t0 = time.monotonic()
    bundle: EvidenceBundle | None = state.get("evidence")
    mcts_path: list[str] | None = state.get("mcts_path")
    query = state.get("query", "")

    if bundle is None:
        return {
            "answer": "No evidence available.", "fallback_used": True,
            **_log("generate_answer", "no evidence bundle", t0),
        }

    is_followup = state.get("is_followup", False)
    messages: list[dict[str, Any]] = list(state.get("messages") or []) if is_followup else []
    summary = state.get("summary") if is_followup else None

    # Retry after refine_path: for ALL known judge FAIL issues
    # (gap_admission_false_negative, ungrounded, incomplete_steps),
    # weak_section_ids is deliberately left empty (see judge_answer), so
    # penalizing sections is a no-op — the evidence doesn't change, so
    # without a correction this retry reproduces the identical output at
    # temperature=0.0 (observed directly on all three issue types). Inject
    # an issue-specific correction into this regeneration instead.
    prior_judge = state.get("judge_result") or {}
    prior_issues = prior_judge.get("issues") or []
    retry_issue = next(
        (
            i for i in prior_issues
            if i in ("gap_admission_false_negative", "ungrounded", "incomplete_steps")
        ),
        None,
    )

    # Relevance gate: ask one short YES/NO per top section ("does THIS section
    # answer the question?") and generate only from the sections that pass.
    # Left to decide inside one long prompt, the 7B sometimes says "the manual
    # does not cover X" while the top section does (and sometimes pads with
    # invented steps); the per-section check is reliable and removes the
    # crowding that causes both. No section passes -> keep the full evidence
    # and let the answer prompt refuse as before (it already refuses well on
    # questions the manual doesn't cover).
    relevant = await _relevant_sections(query, bundle)
    if relevant:
        prompt_bundle = bundle.only(relevant[:_GATE_MAX_KEEP], "relevance gate: YES sections only")
        gate = f"gate={[s.section_id for s in relevant[:_GATE_MAX_KEEP]]}"
    elif retry_issue == "gap_admission_false_negative":
        # Judge says the evidence does answer it but no section passed the gate:
        # retry on the best few sections instead of the same crowded evidence.
        prompt_bundle = bundle.focused(_RETRY_PRIMARY)
        gate = "gate=none, retry on top sections"
    else:
        prompt_bundle = bundle
        gate = "gate=none" if relevant == [] else "gate=skipped"
    narrowed = prompt_bundle is not bundle
    answer_prompt = build_answer_prompt(
        query, prompt_bundle, mcts_path, messages, summary,
        manual_category=state.get("manual_category"),
        retry_issue=retry_issue,
        max_primary=len(prompt_bundle.primary_sections) if narrowed else _MAX_PRIMARY,
        evidence_verified=bool(relevant),
    )
    try:
        response = await call_vlm(
            prompt=answer_prompt,
            task="answer_generation",
            max_tokens=2048,  # model supports 128k context — allow longer answers
            temperature=0.0,  # deterministic — 0.1 caused same evidence to flip
                              # between grounded answer and false "not discussed" refusal
            frequency_penalty=0.4,  # breaks greedy-decoding repetition loops
                                    # (7B model got stuck repeating one line to max_tokens)
        )
        final_answer = ensure_citations(response.content, bundle)
        return {
            "answer": final_answer,
            "fallback_used": False,
            "answer_confidence": "HIGH",
            "last_answer_prompt": answer_prompt,
            "last_answer_response": response.content,
            **_log(
                "generate_answer",
                f"{len(final_answer)} chars, retry_issue={retry_issue or 'none'}, {gate}",
                t0,
            ),
        }
    except VLMError as e:
        logger.warning("generate_answer VLM fallback: %s", e)
        return {
            "answer": f"[VLM unavailable] {format_evidence_as_fallback(bundle)}",
            "fallback_used": True,
            "answer_confidence": "LOW",
            **_log("generate_answer", f"VLM unavailable, evidence-only fallback: {e}", t0),
        }
