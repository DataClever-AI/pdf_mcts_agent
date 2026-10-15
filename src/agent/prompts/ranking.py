"""2. section_ranking prompt — score section relevance 0.0-1.0."""
from __future__ import annotations

import re

from src.agent.prompts._shared import _content_nodes
from src.models.section import Section


def _best_preview(section: Section, query: str, max_chars: int = 250) -> str:
    """Find the content node most relevant to the query (highest keyword overlap)."""
    query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    stopwords = {"what", "how", "do", "does", "the", "a", "an", "is", "are",
                 "you", "i", "we", "in", "of", "to", "for", "and", "or",
                 "should", "could", "when", "where", "why", "please"}
    query_words -= stopwords

    best_node = ""
    best_overlap = -1
    first_nontrivial = ""

    for node in section.semantic_nodes:
        text = node.canonical_text.strip()
        if len(text) < 25:
            continue
        if not first_nontrivial:
            first_nontrivial = text
        node_words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
        overlap = len(query_words & node_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_node = text

    return (best_node or first_nontrivial or "")[:max_chars]


def build_section_ranking_prompt(
    expanded_query: str,
    section: Section,
) -> str:
    """
    Score section relevance 0.0-1.0.
    Scoring anchors stated explicitly so 7B model calibrates correctly.
    Query truncated to avoid token waste on long expanded queries.
    """
    content_nodes = _content_nodes(section, max_nodes=4)
    content_preview = (
        " | ".join(c[:120] for c in content_nodes[:3])
        if content_nodes else "(no content)"
    )
    node_count = section.node_count or 0
    # Truncate expanded query (3 joined sub-questions can be 200+ words)
    query_short = expanded_query[:200] + "..." if len(expanded_query) > 200 else expanded_query

    return (
        "OUTPUT: reply with ONLY a decimal number from 0.0 to 1.0.\n\n"
        "Rate how relevant this manual section is to the query.\n"
        "Base your score primarily on the SECTION TITLE and HIERARCHY PATH.\n\n"
        "Scoring anchors:\n"
        "  1.0 = section title/path directly matches the query topic\n"
        "  0.7 = section is clearly related but not the primary source\n"
        "  0.5 = section mentions the topic tangentially\n"
        "  0.2 = section is about a different topic in the same domain\n"
        "  0.0 = section is completely unrelated\n\n"
        f"Query: {query_short}\n"
        f"Section title: {section.title}\n"
        f"Hierarchy path: {section.hierarchy_path}\n"
        f"Content nodes: {node_count}\n"
        f"Content preview: {content_preview}\n\n"
        "Score:"
    )
