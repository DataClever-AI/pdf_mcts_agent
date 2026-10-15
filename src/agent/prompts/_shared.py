"""Text-processing helpers shared across prompt builders."""
from __future__ import annotations

import re

from src.models.section import Section

_CITATION_STOPWORDS = {
    "what", "how", "do", "does", "the", "a", "an", "is", "are", "you", "i",
    "we", "in", "of", "to", "for", "and", "or", "should", "could", "when",
    "where", "why", "please", "this", "that", "these", "those", "will",
    "can", "with", "on", "at", "be", "it", "as", "if", "not", "no",
}


def _tokenize_for_overlap(text: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", "", text.lower()).split()) - _CITATION_STOPWORDS


def _content_nodes(sec: Section, max_nodes: int = 8) -> list[str]:
    """
    Extract meaningful content nodes, skipping PDF navigation artifacts.

    Filters out:
    - Very short text (< 20 chars) — page numbers, single words
    - Section title repeated verbatim
    - Chapter header patterns like "3 Installation and Setup"
    - Short lines without punctuation (product name headers, bare refs)
    """
    title_lower = sec.title.lower().strip()
    results: list[str] = []
    for node in sec.semantic_nodes:
        text = node.canonical_text.strip()
        if not text or len(text) < 20:
            continue
        if text.lower() == title_lower:
            continue
        # Skip section-header patterns: "3 Installation", "3.3.2 Battery Installation"
        # Pattern: starts with digits and dots, then a letter
        if re.match(r"^[\d\.]+\s+[A-Za-z]", text) and len(text) < 70:
            continue
        if len(text) < 50 and not any(c in text for c in ".,:;-()"):
            continue
        results.append(text[:350])
        if len(results) >= max_nodes:
            break
    return results


def _format_messages_block(
    messages: list[dict[str, str]] | None,
    summary: str | None,
    user_only: bool = False,
) -> str:
    """
    Format messages + summary into a context block for prompts.

    user_only=True: show only USER questions, not assistant answers.
    Used in answer_generation to prevent the model from copying previous answers.
    """
    parts: list[str] = []
    if summary:
        parts.append(f"[Earlier conversation summary]\n{summary}")
    if messages:
        recent = messages[-6:]  # last 3 Q&A pairs max
        lines = []
        for m in recent:
            role = m.get("role", "")
            if user_only and role != "user":
                continue  # skip assistant answers to prevent copy-paste
            label = "User" if role == "user" else "Assistant"
            content = str(m.get("content", ""))[:300]
            lines.append(f"{label}: {content}")
        if lines:
            parts.append("[Prior questions for context]\n" + "\n".join(lines))
    return "\n\n".join(parts)


__all__: list[str] = []  # internal helpers — not part of the package's public API
