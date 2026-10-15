"""EvidenceBundle — expanded context for MCTS + answer generation."""

from functools import cached_property

from pydantic import BaseModel

from src.models.section import Section, SemanticNode


class EvidenceBundle(BaseModel):
    """Sections expanded from ranked primary set along 4 axes."""

    primary_sections: list[Section]
    parent_sections: list[Section]
    sibling_sections: list[Section]
    child_sections: list[Section]
    related_nodes: list[SemanticNode]
    total_node_count: int
    expansion_trace: list[str]
    query: str

    @cached_property
    def all_sections(self) -> list[Section]:
        """All sections across primary, parent, sibling, and child axes."""
        return (
            self.primary_sections
            + self.parent_sections
            + self.sibling_sections
            + self.child_sections
        )

    def focused(self, n: int) -> "EvidenceBundle":
        """Only the top-n primary sections: no parent/sibling/child/related context."""
        return self.only(self.primary_sections[:n], f"focused: top-{n} primary only")

    def only(self, primary: list[Section], note: str = "restricted evidence") -> "EvidenceBundle":
        """A bundle holding exactly `primary`: no parent/sibling/child/related context."""
        return EvidenceBundle(
            primary_sections=primary,
            parent_sections=[],
            sibling_sections=[],
            child_sections=[],
            related_nodes=[],
            total_node_count=sum(s.node_count or 0 for s in primary),
            expansion_trace=[*self.expansion_trace, note],
            query=self.query,
        )

    @property
    def total_section_count(self) -> int:
        """Total sections across all axes."""
        return len(self.all_sections)
