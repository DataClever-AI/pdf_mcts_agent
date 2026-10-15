"""generate_answer relevance gate: per-section YES/NO decides what the generator sees."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent.nodes import answer as answer_node
from src.models.evidence import EvidenceBundle
from src.models.section import Section, SemanticNode
from src.vlm.client import VLMClientError


def _sec(section_id: str, title: str, text: str) -> Section:
    return Section(
        section_id=section_id, title=title, hierarchy_level=1, hierarchy_path=f"/{title}",
        semantic_nodes=[SemanticNode(node_id=f"{section_id}_n", canonical_text=text,
                                     semantic_order=1, node_type="paragraph")],
        semantic_order_start=1, semantic_order_end=1, node_count=1, child_sections=[],
        page_start=1, page_end=1, confidence=1.0, structural_source="toc",
        inference_reasons=[], evidence_source_ids=[],
    )


DISPOSAL = _sec("sec_disposal", "F.6 Monitor Disposal",
                "Disinfect the monitor and cables according to your country's laws before disposal.")
CLEANING = _sec("sec_cleaning", "F.2 Cleaning the Monitor",
                "Wipe the exterior with a soft cloth dampened with approved cleaner.")
PARENT = _sec("sec_parent", "System Care", "Chapter about care and service of the system.")


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        primary_sections=[CLEANING, DISPOSAL], parent_sections=[PARENT], sibling_sections=[],
        child_sections=[], related_nodes=[], total_node_count=3, expansion_trace=[],
        query="How do I dispose of the monitor?",
    )


class _FakeVLM:
    """Answers the relevance gate by section title; records the answer prompt."""

    def __init__(self, yes_titles: set[str], fail_gate: bool = False) -> None:
        self.yes_titles = yes_titles
        self.fail_gate = fail_gate
        self.answer_prompts: list[str] = []

    async def __call__(self, *, prompt: str, task: str, **_: Any) -> Any:
        if task == "section_ranking":  # the gate
            if self.fail_gate:
                raise VLMClientError("down")
            section_line = next(ln for ln in prompt.splitlines() if ln.startswith("SECTION:"))
            yes = any(t in section_line for t in self.yes_titles)
            return SimpleNamespace(content="YES" if yes else "NO")
        self.answer_prompts.append(prompt)
        return SimpleNamespace(content="Disinfect the monitor. **F.6 Monitor Disposal (p.1)**")


def _run(fake: _FakeVLM, monkeypatch: pytest.MonkeyPatch, **state: Any) -> dict[str, Any]:
    monkeypatch.setattr(answer_node, "call_vlm", fake)
    base = {"query": "How do I dispose of the monitor?", "evidence": _bundle(),
            "mcts_path": [], "is_followup": False, "messages": [], "logs": []}
    return asyncio.run(answer_node.generate_answer({**base, **state}))  # type: ignore[arg-type]


class TestRelevantSections:
    def test_returns_only_yes_sections_in_ranked_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(answer_node, "call_vlm", _FakeVLM({"Disposal"}))

        out = asyncio.run(answer_node._relevant_sections("q", _bundle()))

        assert [s.section_id for s in out or []] == ["sec_disposal"]

    def test_empty_list_when_nothing_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(answer_node, "call_vlm", _FakeVLM(set()))

        assert asyncio.run(answer_node._relevant_sections("q", _bundle())) == []

    def test_none_when_the_check_itself_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(answer_node, "call_vlm", _FakeVLM(set(), fail_gate=True))

        assert asyncio.run(answer_node._relevant_sections("q", _bundle())) is None


class TestGenerateAnswerGate:
    def test_generator_sees_only_the_sections_that_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeVLM({"Disposal"})

        out = _run(fake, monkeypatch)

        prompt = fake.answer_prompts[0]
        assert "country's laws" in prompt            # the verified section is there
        assert "soft cloth" not in prompt            # the one that didn't pass is not
        assert "System Care" not in prompt           # nor expansion context
        assert "was checked and contains information" in prompt
        assert "gate=['sec_disposal']" in " ".join(out["logs"])

    def test_no_yes_keeps_the_full_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeVLM(set())

        _run(fake, monkeypatch)

        prompt = fake.answer_prompts[0]
        assert "country's laws" in prompt and "soft cloth" in prompt
        assert "was checked and contains information" not in prompt

    def test_gate_failure_falls_back_to_full_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeVLM(set(), fail_gate=True)

        out = _run(fake, monkeypatch)

        assert "soft cloth" in fake.answer_prompts[0]
        assert out["fallback_used"] is False

    def test_retry_without_a_yes_section_narrows_to_top_sections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeVLM(set())

        _run(fake, monkeypatch, judge_result={"issues": ["gap_admission_false_negative"]})

        prompt = fake.answer_prompts[0]
        assert "System Care" not in prompt                  # expansion context dropped
        assert "was narrowed to the sections" in prompt


class TestEvidenceBundleOnly:
    def test_only_drops_all_expansion_context(self) -> None:
        narrowed = _bundle().only([DISPOSAL], "test")

        assert [s.section_id for s in narrowed.all_sections] == ["sec_disposal"]
        assert narrowed.expansion_trace[-1] == "test"
