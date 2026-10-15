"""AgentState TypedDict — the shared state flowing through LangGraph nodes.

Memory architecture (LangGraph 2026 canonical):
  messages: list[dict]  — structured Q&A pairs (role/content), persisted per session
  summary: str | None   — rolling VLM summary of messages older than keep window
  Together these replace the old `conversation_history` string approach.

  Flow per query:
    1. guard_query: classifies is_followup (does the CURRENT query reference/
       continue the prior conversation, or is it a standalone new topic?).
       guard_query itself always sees messages (needs them to correctly judge
       ambiguous pronoun-only follow-ups), but is_followup gates every OTHER
       node's use of conversation history.
    2. summarize_if_needed: if len(messages) > 10 → summarize oldest, keep last 4
       (runs regardless of is_followup — this manages buffer size, not per-turn
       relevance).
    3. expand_query: only uses messages/summary if is_followup — otherwise
       treats the query as fully standalone, preventing stale topics from an
       earlier conversation leaking into sub-question generation and, from
       there, into retrieval.
    4. generate_answer: only includes messages/summary as conversation context
       if is_followup — otherwise generates from evidence alone, preventing
       an unrelated question's wording/accuracy from drifting based on what
       was discussed earlier in the session.
    5. store_memory: appends new Q&A pair to messages (regardless of
       is_followup — every turn still gets recorded for future follow-ups).
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class FeedbackRecord(TypedDict, total=False):
    """Human feedback submitted after answer display."""

    rating: int
    correction: str
    expert_answer: str


class AgentState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""

    # --- Document corpus metadata ---
    max_hierarchy_level: int   # max level across ALL sections in the loaded document
    manual_category: str | None  # category key from CATEGORY_OPTIONS (e.g. "medical")

    # --- Query ---
    query: str
    expanded_query: str | None
    original_query: str | None
    sub_queries: list[str]

    # --- Retrieval ---
    candidate_sections: list[Any]
    ranked_sections: list[Any]
    rank_vlm_scores: dict[str, float]  # section_id -> rank_sections' VLM relevance
                                        # score, reused by run_mcts instead of
                                        # re-scoring the same section+query twice
    evidence: Any | None

    # --- MCTS ---
    mcts_path: list[str] | None
    mcts_score_cache: dict[str, float]
    mcts_n_simulations: int

    # --- Guard ---
    blocked: bool
    block_category: str | None
    is_followup: bool  # does query reference/continue prior conversation? gates
                       # conversation-context use in expand_query/generate_answer

    # --- Answer ---
    answer: str | None
    answer_blocked: bool
    answer_confidence: str | None
    fallback_used: bool

    # --- Judge ---
    judge_result: dict[str, Any] | None
    judge_attempts: int

    # --- Human ---
    feedback: FeedbackRecord | None

    # --- Conversation memory (LangGraph canonical) ---
    messages: list[dict[str, Any]]   # Q&A pairs: {"role": "user"|"assistant", "content": "..."}
    summary: str | None              # VLM summary of messages older than keep window
    conversation_id: str | None      # Session ID (scopes messages to one browser session)
    thread_id: str | None

    # --- Persistence ---
    store_memory_done: bool
    feedback_id: int | None  # SQLite feedback.id from this turn's insert_feedback —
                              # lets the UI attach a human rating/correction afterward

    # --- Diagnostics ---
    error: str | None
    error_layer: str | None

    # --- Audit log (LangGraph reducer pattern) ---
    # Annotated[list[str], add] — each node returns only its own delta line(s)
    # in "logs"; LangGraph concatenates them via the `add` reducer into one
    # running audit trail for the whole query. This replaced a prior
    # node_trace/step_metrics pair that mutated `state` in place without
    # returning the key — confirmed via direct repro that such mutations are
    # discarded by LangGraph, so those two fields never actually accumulated.
    logs: Annotated[list[str], add]

    # Raw (prompt, completion) for the two highest fine-tuning-value VLM
    # calls — final answer and quality judge. Written to a JSON sidecar next
    # to the .log file by store_memory. Guard/rank/rollout calls are not
    # captured (high volume, low training value) — scoped down deliberately.
    last_answer_prompt: str | None
    last_answer_response: str | None
    last_judge_prompt: str | None
    last_judge_response: str | None
