"""
Evidence expander — build EvidenceBundle from ranked sections.

Expands along 4 axes: parent, sibling (max 2), child (all), related nodes.
No VLM. No embeddings. Uses section_id references in child_sections.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from src.models.derived import RankedSection
from src.models.evidence import EvidenceBundle
from src.models.section import Section, SemanticNode


class ExpansionConfig(BaseModel):
    """Evidence expansion configuration."""

    max_siblings_per_section: int = 2
    max_related_nodes: int = 5
    max_expansion_factor: int = 6
    min_keyword_overlap: int = 2


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[/\-_]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def _deduplicate(sections: list[Section]) -> list[Section]:
    seen: set[str] = set()
    result: list[Section] = []
    for s in sections:
        if s.section_id not in seen:
            seen.add(s.section_id)
            result.append(s)
    return result


class EvidenceExpander:
    """Expand ranked sections into a contextualised EvidenceBundle."""

    def __init__(
        self,
        sections_by_id: dict[str, Section],
        config: ExpansionConfig | None = None,
    ) -> None:
        """
        Args:
            sections_by_id: All loaded sections keyed by section_id.
            config:         Expansion configuration.
        """
        self._store = sections_by_id
        self.config = config or ExpansionConfig()

    def _get_parent(self, section: Section) -> Section | None:
        """Find parent section via reverse lookup."""
        for candidate in self._store.values():
            if section.section_id in candidate.child_sections:
                return candidate
        return None

    def _get_children(self, section: Section) -> list[Section]:
        """Get all direct children from child_sections references."""
        return [
            self._store[cid]
            for cid in section.child_sections
            if cid in self._store
        ]

    def _get_siblings(self, section: Section, parent: Section | None) -> list[Section]:
        """Get siblings via parent's child_sections, excluding self."""
        if parent is None:
            return []
        return [
            self._store[cid]
            for cid in parent.child_sections
            if cid in self._store and cid != section.section_id
        ]

    def expand(
        self,
        ranked_sections: list[RankedSection],
        query: str,
    ) -> EvidenceBundle:
        """
        Build an EvidenceBundle from top ranked sections.

        Args:
            ranked_sections: Top-N sections from the ranker.
            query:           Original query for keyword overlap.

        Returns:
            EvidenceBundle with 4-axis expansion and audit trace.
        """
        primary = [r.section for r in ranked_sections]
        primary_ids = {s.section_id for s in primary}
        trace: list[str] = []

        for r in ranked_sections:
            trace.append(
                f"primary: {r.section.section_id} "
                f"(rank={r.rank}, score={r.ranking_score:.3f})"
            )

        # Parent expansion
        parents: list[Section] = []
        for sec in primary:
            parent = self._get_parent(sec)
            if parent and parent.section_id not in primary_ids:
                parents.append(parent)
                trace.append(f"parent: {parent.section_id} ← {sec.section_id}")

        # Sibling expansion (top 2 per primary, by confidence)
        siblings: list[Section] = []
        occupied = primary_ids | {s.section_id for s in parents}
        for sec in primary:
            parent = self._get_parent(sec)
            raw_siblings = self._get_siblings(sec, parent)
            raw_siblings.sort(key=lambda s: s.confidence, reverse=True)
            added = 0
            for sib in raw_siblings:
                if added >= self.config.max_siblings_per_section:
                    break
                if sib.section_id not in occupied:
                    siblings.append(sib)
                    occupied.add(sib.section_id)
                    trace.append(
                        f"sibling: {sib.section_id} ← {sec.section_id} "
                        f"(conf={sib.confidence:.2f})"
                    )
                    added += 1

        # Child expansion (all direct children)
        children: list[Section] = []
        for sec in primary:
            for child in self._get_children(sec):
                if child.section_id not in occupied:
                    children.append(child)
                    occupied.add(child.section_id)
                    trace.append(f"child: {child.section_id} ← {sec.section_id}")

        # Related node expansion (keyword overlap)
        query_tokens = set(_tokenize(query))
        related: list[SemanticNode] = []
        for sec in self._store.values():
            if sec.section_id in occupied:
                continue
            content = " ".join(n.canonical_text for n in sec.semantic_nodes)
            content_tokens = set(_tokenize(content))
            overlap = query_tokens & content_tokens
            if len(overlap) >= self.config.min_keyword_overlap:
                for node in sec.semantic_nodes[:2]:
                    related.append(node)
                    trace.append(
                        f"related_node: {node.node_id} "
                        f"(overlap: {', '.join(list(overlap)[:3])})"
                    )
                if len(related) >= self.config.max_related_nodes:
                    break

        related = related[: self.config.max_related_nodes]
        parents = _deduplicate(parents)
        siblings = _deduplicate(siblings)
        children = _deduplicate(children)

        all_sections = primary + parents + siblings + children
        total_node_count = sum(s.node_count or 0 for s in all_sections)

        # Size guard
        max_total = len(primary) * self.config.max_expansion_factor
        if len(all_sections) > max_total:
            trace.append(
                f"WARN: bundle size {len(all_sections)} > {max_total} (trimming)"
            )
            children = children[: max(0, max_total - len(primary) - len(parents) - len(siblings))]
            all_sections = primary + parents + siblings + children

        return EvidenceBundle(
            primary_sections=primary,
            parent_sections=parents,
            sibling_sections=siblings,
            child_sections=children,
            related_nodes=related,
            total_node_count=total_node_count,
            expansion_trace=trace,
            query=query,
        )
