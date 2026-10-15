"""1. query_understanding prompt — reformulate query into 3 sub-questions."""
from __future__ import annotations

import re

from src.agent.prompts._shared import _format_messages_block


def build_query_expansion_prompt(
    query: str,
    messages: list[dict[str, str]] | None = None,
    summary: str | None = None,
    manual_category: str | None = None,
) -> str:
    """
    Reformulate current query into 3 specific sub-questions.
    Each sub-question must cover a DIFFERENT ASPECT of the topic.
    """
    from src.agent.categories import get_category

    cat = get_category(manual_category or "general")
    domain_line = f"Domain: {cat['domain_hint']}.\n" if manual_category else ""
    context_block = _format_messages_block(messages, summary)

    context_section = (
        f"\n\nConversation context (use ONLY to resolve ambiguous references "
        f"like 'it', 'that', 'this feature' in the current question above):\n"
        f"{context_block}"
    ) if context_block else ""

    return (
        f"You are a technical manual search assistant. {domain_line}"
        "Generate exactly 3 sub-questions that together fully cover the current "
        "question below.\n\n"
        "RULES:\n"
        "1. ALL 3 sub-questions MUST be about the CURRENT QUESTION topic only.\n"
        "   Do NOT use topics from prior conversation unless the current question\n"
        "   uses a pronoun or follow-up phrase ('what about X?', 'and the Y?').\n"
        "2. Each sub-question must cover a DIFFERENT ASPECT:\n"
        "   - One about HOW (procedure, steps)\n"
        "   - One about WHAT (specification, conditions, requirements)\n"
        "   - One about SAFETY or TROUBLESHOOTING related to the topic\n"
        "   Do NOT ask the same question 3 times with different wording.\n"
        "3. Use precise technical terminology for the domain.\n"
        "4. Each sub-question must be a complete, standalone question.\n"
        "5. Output format: numbered list only. No preamble, no explanation.\n\n"
        f"CURRENT QUESTION: {query}"
        f"{context_section}\n\n"
        "3 sub-questions:"
    )


def parse_expanded_queries(content: str) -> list[str]:
    """Parse the 3 numbered sub-questions from VLM expansion output."""
    matches = re.findall(r"\d+\.\s*(.+?)(?=\n\d+\.|$)", content.strip(), re.DOTALL)
    questions = [m.strip() for m in matches if m.strip()]
    if len(questions) >= 3:
        return questions[:3]
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    return lines[:3] if lines else [content.strip()]
