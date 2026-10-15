"""
Tests for judge.py's deterministic (non-VLM) heuristics — gap-admission
detection and the completeness check added in the Tier 2 review fix.
"""
from __future__ import annotations

from src.agent.prompts import answer_is_gap_admission, answer_is_incomplete
from src.models.evidence import EvidenceBundle
from src.models.section import Section, SemanticNode


def _section_with_steps(section_id: str, title: str, steps: list[str]) -> Section:
    """
    One SemanticNode per step (matching real docling extraction — each step
    is its own node, not concatenated text). Step text must be long enough
    (>=20 chars) and not match _content_nodes' "digit-prefix short line"
    header-skip pattern (a short "1. X" line under 70 chars is filtered out
    as a chapter-header artifact) — the fixture text is deliberately full
    sentences for this reason, not terse "1. X" fragments.
    """
    nodes = [
        SemanticNode(node_id=f"{section_id}_n{i}", canonical_text=f"{i}. {step}",
                     semantic_order=i, node_type="paragraph")
        for i, step in enumerate(steps, 1)
    ]
    return Section(
        section_id=section_id, title=title, hierarchy_level=1, hierarchy_path=f"/{title}",
        semantic_nodes=nodes,
        semantic_order_start=1, semantic_order_end=len(steps), node_count=len(steps),
        child_sections=[], page_start=1, page_end=1,
        confidence=1.0, structural_source="toc", inference_reasons=[], evidence_source_ids=[],
    )


_BATTERY_STEPS = [
    "Power off the device completely and wait at least ten seconds before proceeding.",
    "Remove the four screws securing the battery cover using a Phillips screwdriver.",
    "Insert the new battery, carefully matching the polarity markings inside the compartment.",
]


def _bundle(sections: list[Section]) -> EvidenceBundle:
    return EvidenceBundle(
        primary_sections=sections, parent_sections=[], sibling_sections=[],
        child_sections=[], related_nodes=[],
        total_node_count=sum(s.node_count or 0 for s in sections),
        expansion_trace=[], query="q",
    )


class TestAnswerIsGapAdmission:
    def test_pure_gap_phrase_is_admission(self) -> None:
        assert answer_is_gap_admission("The manual does not discuss this topic.")

    def test_gap_phrase_with_full_procedure_is_not_admission(self) -> None:
        # Vestigial framing followed by a real 3+ step procedure — the answer
        # actually committed, so this must NOT be routed to the gap check
        # (see docstring: it should hit grounding check instead).
        answer = (
            "The manual does not discuss this directly, but here are the steps:\n"
            "1. Power off the device.\n"
            "2. Remove the cover.\n"
            "3. Replace the part.\n"
        )
        assert not answer_is_gap_admission(answer)

    def test_no_gap_phrase_is_not_admission(self) -> None:
        assert not answer_is_gap_admission("1. Do this.\n2. Do that.\n")

    def test_gap_phrase_with_fewer_than_3_steps_is_still_admission(self) -> None:
        answer = "The manual does not cover this. 1. partial step only"
        assert answer_is_gap_admission(answer)


class TestAnswerIsIncomplete:
    def test_answer_missing_a_step_is_incomplete(self) -> None:
        sec = _section_with_steps("s1", "Battery", _BATTERY_STEPS)
        bundle = _bundle([sec])
        answer = "1. Power off.\n2. Remove cover.\n"  # dropped step 3
        assert answer_is_incomplete(answer, bundle)

    def test_answer_with_all_steps_is_complete(self) -> None:
        sec = _section_with_steps("s1", "Battery", _BATTERY_STEPS)
        bundle = _bundle([sec])
        answer = "1. Power off.\n2. Remove cover.\n3. Insert battery.\n"
        assert not answer_is_incomplete(answer, bundle)

    def test_prose_only_answer_is_not_flagged_incomplete(self) -> None:
        sec = _section_with_steps("s1", "Battery", _BATTERY_STEPS)
        bundle = _bundle([sec])
        # Zero numbered steps in the answer — not a committed procedure,
        # handled by the gap-admission path instead, not this check.
        assert not answer_is_incomplete("The battery can be replaced easily.", bundle)

    def test_answer_with_more_steps_than_source_is_not_incomplete(self) -> None:
        sec = _section_with_steps("s1", "Battery", _BATTERY_STEPS[:2])
        bundle = _bundle([sec])
        answer = "1. Power off.\n2. Remove cover.\n3. Extra step not in source.\n"
        assert not answer_is_incomplete(answer, bundle)
