"""Derived section models carrying retrieval metadata."""

from typing import Any

from pydantic import BaseModel, field_validator

from src.models.section import Section


def _coerce_section(v: Any) -> Any:
    """Normalise a Section from any import path (Streamlit cache workaround)."""
    if isinstance(v, dict):
        return v
    if hasattr(v, "model_dump") and hasattr(v, "section_id"):
        return v.model_dump()
    return v


class ScoredSection(BaseModel):
    """Output of BM25 retrieval."""

    section: Section
    retrieval_score: float
    title_score: float
    path_score: float
    content_score: float

    @field_validator("section", mode="before")
    @classmethod
    def normalise_section(cls, v: Any) -> Any:
        """Accept Section objects from any import path."""
        return _coerce_section(v)


class RankedSection(BaseModel):
    """Output of section ranking (hybrid BM25 + VLM + structural)."""

    section: Section
    retrieval_score: float
    vlm_score: float
    ranking_score: float
    rank: int  # 1-based

    @field_validator("section", mode="before")
    @classmethod
    def normalise_section(cls, v: Any) -> Any:
        """Accept Section objects from any import path."""
        return _coerce_section(v)
