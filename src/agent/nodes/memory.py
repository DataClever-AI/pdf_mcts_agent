"""Node 0.5: summarize_if_needed — runs after guard_query, before expand_query."""
from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.nodes._shared import _is_error, _log
from src.agent.state import AgentState
from src.vlm.client import VLMError, call_vlm

logger = logging.getLogger(__name__)

# Conversation memory window constants (LangGraph canonical pattern)
_MESSAGES_KEEP_WINDOW = 4          # keep last 2 Q&A pairs (4 messages) after summarization
_MESSAGES_SUMMARIZE_THRESHOLD = 10  # trigger summarization when > 5 Q&A pairs (10 messages)


async def summarize_if_needed(state: AgentState) -> dict[str, Any]:
    """
    Summarize old messages when conversation grows beyond the keep window.

    Canonical LangGraph 2026 pattern:
      - messages: list of recent Q&A pairs (role/content dicts)
      - summary: rolling VLM summary of older messages
      - When len(messages) > threshold → summarize messages[:-keep_window]
        and prepend to existing summary → trim messages to keep_window

    Runs BEFORE expand_query so VLM sees compressed history within context limit.
    Skips if not enough messages. Never raises — failure just keeps messages as-is.
    """
    if _is_error(state):
        return {}

    messages: list[dict[str, Any]] = list(state.get("messages") or [])
    if len(messages) <= _MESSAGES_SUMMARIZE_THRESHOLD:
        return _log(
            "summarize_if_needed",
            f"skipped (n={len(messages)}<={_MESSAGES_SUMMARIZE_THRESHOLD})",
        )

    t0 = time.monotonic()
    to_summarize = messages[:-_MESSAGES_KEEP_WINDOW]
    to_keep = messages[-_MESSAGES_KEEP_WINDOW:]
    existing_summary = state.get("summary") or ""

    # Build summarization prompt
    convo_text = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:400]}"
        for m in to_summarize
    )
    prior = f"Prior summary:\n{existing_summary}\n\n" if existing_summary else ""
    prompt = (
        f"{prior}Summarize these technical manual Q&A exchanges in 3-5 sentences. "
        "Focus on topics discussed, sections referenced, and key findings. "
        "Be concise.\n\n"
        f"{convo_text}\n\nSummary:"
    )

    try:
        response = await call_vlm(
            prompt=prompt,
            task="feedback_summarization",
            max_tokens=300,
            temperature=0.0,
        )
        new_summary = response.content.strip()
        logger.info(
            "summarize_if_needed: compressed %d → %d messages",
            len(messages), len(to_keep),
        )
        return {
            "summary": new_summary, "messages": to_keep,
            **_log(
                "summarize_if_needed",
                f"compressed {len(messages)}→{len(to_keep)} messages", t0,
            ),
        }
    except VLMError as e:
        logger.warning("summarize_if_needed VLM failed, keeping messages: %s", e)
        return _log("summarize_if_needed", f"VLM error, kept messages as-is: {e}", t0)
