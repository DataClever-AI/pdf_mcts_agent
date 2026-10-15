"""4. judge_answer prompts — grounding check, gap-admission check, completeness check."""
from __future__ import annotations

import re

from src.agent.prompts._shared import _content_nodes
from src.models.evidence import EvidenceBundle

_SAFETY_UNSAFE_MESSAGE = (
    "This answer cannot be displayed because it may contain dangerous or inappropriate "
    "information. Please rephrase your question focusing on safe procedures "
    "described in the manual."
)

SAFETY_BLOCK_THRESHOLD = 0.5


def _build_judge_evidence_text(bundle: EvidenceBundle) -> str:
    """Shared evidence formatting for the two judge binary checks."""
    # Judge gets ALL primary sections with FULL content (no truncation).
    # 32k context window verified via API.
    evidence_excerpts: list[str] = []
    for sec in bundle.primary_sections:
        nodes = _content_nodes(sec, max_nodes=50)  # all content nodes
        if nodes:
            content = "\n  ".join(nodes)
            evidence_excerpts.append(
                f"[{sec.section_id}] {sec.title}:\n  {content}"
            )
        else:
            evidence_excerpts.append(
                f"[{sec.section_id}] {sec.title}: (no content)"
            )
    return "\n\n".join(evidence_excerpts) or "No evidence available."


_GAP_ADMISSION_PATTERNS = [
    "does not discuss", "does not cover", "does not address",
    "not discussed in the manual", "not covered in the manual",
    "not available in the manual", "cannot be determined from",
    "manual does not", "no information about", "does not mention",
    "does not contain any information", "does not specify",
]

_NUMBERED_STEP_RE = re.compile(r"(?:^|\n)\s*[1-9]\d?\.\s")


def answer_is_gap_admission(answer: str) -> bool:
    """
    Heuristic: does the answer claim the manual doesn't cover the topic?

    A gap-admission PHRASE can appear as vestigial framing in an answer that
    then goes on to give a real numbered procedure anyway (seen directly: a
    retry-corrected answer opened with "the manual does not discuss X" but
    then gave a full 9-step procedure). That's not a gap-admission anymore —
    it's a committed answer, and its real risk is fabricated/ungrounded
    steps, which the grounding check (not the gap check) is built to catch.
    Routing it to the gap check instead mislabels the failure and lets
    invented steps slip past unevaluated. 3+ numbered steps is treated as
    "this answer committed" regardless of any hedging phrase present.
    """
    a = answer.lower()
    if not any(p in a for p in _GAP_ADMISSION_PATTERNS):
        return False
    return len(_NUMBERED_STEP_RE.findall(answer)) < 3


def _max_section_numbered_steps(bundle: EvidenceBundle) -> int:
    """Highest numbered-step count found in any single primary section's content."""
    max_count = 0
    for sec in bundle.primary_sections:
        content = "\n".join(_content_nodes(sec, max_nodes=50))
        max_count = max(max_count, len(_NUMBERED_STEP_RE.findall(content)))
    return max_count


def answer_is_incomplete(answer: str, bundle: EvidenceBundle) -> bool:
    """
    Deterministic completeness check: did the answer drop numbered steps
    that are present in its source section?

    build_answer_prompt's COMPLETENESS RULE tells the model never to skip a
    numbered step, but — like every other instruction-only rule this
    session — a 7B model doesn't reliably follow it, and until now nothing
    verified it: only grounding (fabricated content) had a post-hoc check,
    not omission. This is cheaper and more reliable than asking the VLM to
    count steps in two texts: a plain regex count on both sides.

    Only fires when the answer itself commits to a numbered procedure (>=1
    step) — an answer with zero numbered steps is either prose-only (nothing
    to compare) or a gap-admission, already handled by
    answer_is_gap_admission/build_gap_check_prompt, not this check.
    """
    answer_steps = len(_NUMBERED_STEP_RE.findall(answer))
    if answer_steps == 0:
        return False
    return answer_steps < _max_section_numbered_steps(bundle)


def build_grounding_check_prompt(
    query: str,
    answer: str,
    bundle: EvidenceBundle,
    manual_category: str | None = None,
) -> str:
    """
    Binary grounding check — used when the answer makes substantive claims
    (not a gap-admission). Single yes/no question, same shape as the proven
    answer-safety binary check, to avoid the compound-task failures a single
    multi-rule scored-JSON call produced on a 7B model (self-contradictory
    quotes, answer text echoed back as its own "issue").
    """
    from src.agent.categories import get_category

    cat = get_category(manual_category or "general")
    domain_line = f" for a {cat['label']} manual" if manual_category else ""

    evidence_text = _build_judge_evidence_text(bundle)

    return (
        "You are a fact-checker" + domain_line + ".\n"
        "Reply with EXACTLY ONE WORD: GROUNDED or UNGROUNDED\n\n"
        "CHECK EACH CLAIM SEPARATELY: if ANSWER is a list of bullets/steps/items,\n"
        "evaluate every item on its own — do not average across the list. One\n"
        "grounded item does NOT make the whole answer GROUNDED if even one other\n"
        "item is fabricated. Reply UNGROUNDED if ANY item fails the check below,\n"
        "even if most of the list is fine.\n\n"
        "GROUNDED = every specific factual claim in ANSWER is supported by\n"
        "  REFERENCE SECTIONS, even if paraphrased, reordered, or renumbered.\n"
        "  REFERENCE SECTIONS are EXCERPTS — the answer may correctly cite a\n"
        "  detail from a part of a section not shown here; do not mark that\n"
        "  UNGROUNDED just because you cannot verify it in the excerpt.\n\n"
        "UNGROUNDED = the answer states a specific fact that DIRECTLY\n"
        "  CONTRADICTS a sentence actually present in REFERENCE SECTIONS, or\n"
        "  invents confident-sounding specifics with no support at all. This\n"
        "  INCLUDES:\n"
        "  - naming or comparing against another product/technology/manufacturer\n"
        "    that REFERENCE SECTIONS never names,\n"
        "  - extra generic industry-standard items (audits, best practices,\n"
        "    training programs, checklists) added to round out an otherwise-real\n"
        "    list, when REFERENCE SECTIONS don't contain those specific items,\n"
        "  - a structured process/framework the answer presents as prescribed\n"
        "    procedure when REFERENCE SECTIONS only contain looser, unstructured\n"
        "    information on the topic.\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "REFERENCE SECTIONS:\n"
        f"{evidence_text}\n\n"
        "Reply with EXACTLY ONE WORD:"
    )


def build_gap_check_prompt(
    query: str,
    bundle: EvidenceBundle,
    manual_category: str | None = None,
) -> str:
    """
    Binary gap-admission check — used only when the answer claims the manual
    doesn't cover the topic. Checks whether that claim is actually true by
    asking a single focused yes/no question, instead of asking a 7B model to
    simultaneously judge title-matching, content-matching, and empty-section
    exclusion inside one compound scored-JSON call.
    """
    from src.agent.categories import get_category

    cat = get_category(manual_category or "general")
    domain_line = f" for a {cat['label']} manual" if manual_category else ""

    evidence_text = _build_judge_evidence_text(bundle)

    return (
        "You are a fact-checker" + domain_line + ".\n"
        "Reply with EXACTLY ONE WORD: YES or NO\n\n"
        "An answer claimed the manual does not cover this question's topic.\n"
        "Look ONLY at the LISTED CONTENT of each section below (the text after\n"
        "the section title) — ignore section titles alone, and ignore any\n"
        "section marked '(no content)'.\n\n"
        "QUESTION: " + query + "\n\n"
        "Does the LISTED CONTENT of any section below actually answer this\n"
        "question? Reply YES only if real content (not just a matching title)\n"
        "visibly addresses it. Reply NO if no section's content addresses it.\n\n"
        "REFERENCE SECTIONS:\n"
        f"{evidence_text}\n\n"
        "Reply with EXACTLY ONE WORD:"
    )


def get_safety_block_message() -> str:
    """Return the hard-block message for unsafe answers."""
    return _SAFETY_UNSAFE_MESSAGE
