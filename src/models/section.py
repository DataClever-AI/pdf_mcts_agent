"""Core document models: SemanticNode and Section (immutable)."""

from pydantic import BaseModel, ConfigDict, field_validator


class SemanticNode(BaseModel):
    """A single content node within a document section."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    canonical_text: str
    semantic_order: int
    node_type: str  # "paragraph" | "table" | "heading"


class Section(BaseModel):
    """Immutable document section produced by tree.json loader."""

    model_config = ConfigDict(frozen=True)

    section_id: str
    title: str
    hierarchy_level: int
    hierarchy_path: str

    semantic_nodes: list[SemanticNode]
    semantic_order_start: int | None
    semantic_order_end: int | None
    node_count: int | None
    child_sections: list[str]

    # Page location (from tree.json — None if tree was built without page data)
    page_start: int | None = None
    page_end: int | None = None

    confidence: float
    structural_source: str
    inference_reasons: list[str]
    evidence_source_ids: list[str]

    @field_validator("hierarchy_level")
    @classmethod
    def level_must_be_positive(cls, v: int) -> int:
        """Hierarchy level must be >= 1."""
        assert v >= 1, "hierarchy_level must be >= 1"
        return v

    @field_validator("hierarchy_path")
    @classmethod
    def path_must_start_with_slash(cls, v: str) -> str:
        """Hierarchy path must start with /."""
        assert v.startswith("/"), "hierarchy_path must start with /"
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_range(cls, v: float) -> float:
        """Confidence must be between 0.0 and 1.0."""
        assert 0.0 <= v <= 1.0, "confidence must be between 0.0 and 1.0"
        return v

    @field_validator("section_id", "title")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        """Section ID and title must not be blank."""
        assert len(v.strip()) > 0, "section_id and title must not be empty"
        return v
