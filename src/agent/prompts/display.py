"""Evidence display helpers — used for debug/summary output, not sent to the VLM."""
from __future__ import annotations

from src.models.evidence import EvidenceBundle


def format_evidence_summary(bundle: EvidenceBundle) -> str:
    """Short summary of EvidenceBundle for display."""
    return (
        f"Primary: {len(bundle.primary_sections)} sections, "
        f"Parents: {len(bundle.parent_sections)}, "
        f"Siblings: {len(bundle.sibling_sections)}, "
        f"Children: {len(bundle.child_sections)}, "
        f"Nodes: {bundle.total_node_count}"
    )


def format_evidence_as_fallback(bundle: EvidenceBundle) -> str:
    """Format raw evidence as a plain-text fallback answer when VLM unavailable."""
    lines = ["[Evidence summary — VLM answer unavailable]\n"]
    for sec in bundle.primary_sections[:3]:
        lines.append(f"\n[{sec.section_id}] {sec.title}")
        for node in sec.semantic_nodes[:2]:
            lines.append(f"  {node.canonical_text[:300]}")
    return "\n".join(lines)
