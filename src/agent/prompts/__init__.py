"""
Prompt builders for all VLM tasks.

Split into one module per prompt family (guard, query_expansion, ranking,
answer, judge, safety, feedback, display) — was a single 1000+ line file
mixing prompt text with parsing/citation/heuristic logic for every VLM task.
This __init__ re-exports the same public names the old flat module did, so
every existing import (`from src.agent.prompts import build_guard_prompt,
...` in src/agent/nodes/*.py) keeps working unchanged.

Design principles applied throughout:
  1. Output format stated FIRST (small 7B models anchor to first instruction)
  2. One clear job per prompt — no competing directives
  3. Concrete examples over abstract rules
  4. Scoring anchors wherever numeric output is required
  5. Token efficiency — no repetition, no padding
"""
from __future__ import annotations

from src.agent.prompts.answer import (
    build_answer_prompt,
    build_section_relevance_prompt,
    ensure_citations,
)
from src.agent.prompts.display import format_evidence_as_fallback, format_evidence_summary
from src.agent.prompts.feedback import build_feedback_summary_prompt
from src.agent.prompts.guard import (
    BLOCK_CATEGORIES,
    build_followup_check_prompt,
    build_guard_prompt,
    get_refusal_message,
)
from src.agent.prompts.judge import (
    SAFETY_BLOCK_THRESHOLD,
    answer_is_gap_admission,
    answer_is_incomplete,
    build_gap_check_prompt,
    build_grounding_check_prompt,
    get_safety_block_message,
)
from src.agent.prompts.query_expansion import (
    build_query_expansion_prompt,
    parse_expanded_queries,
)
from src.agent.prompts.ranking import build_section_ranking_prompt
from src.agent.prompts.safety import answer_has_dangerous_keywords, build_answer_safety_prompt

__all__ = [
    "BLOCK_CATEGORIES",
    "SAFETY_BLOCK_THRESHOLD",
    "answer_has_dangerous_keywords",
    "answer_is_gap_admission",
    "answer_is_incomplete",
    "build_answer_prompt",
    "build_section_relevance_prompt",
    "build_answer_safety_prompt",
    "build_feedback_summary_prompt",
    "build_followup_check_prompt",
    "build_gap_check_prompt",
    "build_grounding_check_prompt",
    "build_guard_prompt",
    "build_query_expansion_prompt",
    "build_section_ranking_prompt",
    "ensure_citations",
    "format_evidence_as_fallback",
    "format_evidence_summary",
    "get_refusal_message",
    "get_safety_block_message",
    "parse_expanded_queries",
]
