"""5. feedback_summarization prompt — summarize human corrections into reusable patterns."""
from __future__ import annotations


def build_feedback_summary_prompt(
    question: str,
    wrong_sections: list[str],
    correction: str,
    expert_answer: str | None,
) -> str:
    """Summarize human correction into a reusable retrieval pattern."""
    expert_part = f"\nExpert answer: {expert_answer}" if expert_answer else ""
    return (
        "Summarize this human correction in 2-3 sentences for future retrieval.\n"
        "Focus on: what was wrong, which topic is actually covered, and where.\n\n"
        f"Question: {question}\n"
        f"Wrong sections: {wrong_sections}\n"
        f"Correction: {correction}"
        f"{expert_part}\n\n"
        "Summary:"
    )
