"""Answer-safety keyword check + binary SAFE/UNSAFE VLM prompt (judge_answer Layer A)."""
from __future__ import annotations

_ANSWER_DANGER_PATTERNS = [
    "how to make a bomb", "explosive device", "synthesis of", "synthesize drugs",
    "lethal dose", "how to harm",
]


def answer_has_dangerous_keywords(answer: str) -> bool:
    """
    Fast keyword check on answer content. Returns True if dangerous.

    Deliberately excludes phrase patterns that occur verbatim in legitimate,
    grounded manual answers regardless of negation: "ignore warnings" matches
    inside "do NOT ignore warnings"; "disable safety" / "override safety" /
    "bypass safety lock" match inside standard lockout-tagout maintenance
    instructions; "how to kill" matches inside "how to kill the engine using
    the kill switch". This substring check has no negation awareness, so it
    hard-blocks the answer with no VLM review (see judge_answer Layer A) —
    the VLM safety check in Layer B is the correct judge for these, since its
    prompt already states manual procedures are ALWAYS SAFE regardless of
    wording. Only patterns with no plausible safe manual usage stay here.
    """
    a = answer.lower()
    return any(pattern in a for pattern in _ANSWER_DANGER_PATTERNS)


def build_answer_safety_prompt(query: str, answer: str) -> str:
    """
    Binary safety check: SAFE or UNSAFE.
    Technical manual procedures (welding, chemicals, electrical) are ALWAYS SAFE.
    Only UNSAFE if answer invents dangerous instructions NOT from any manual.
    """
    return (
        "You are a safety reviewer for a technical manual assistant.\n"
        "Reply with EXACTLY ONE WORD: SAFE or UNSAFE\n\n"
        "SAFE = the answer describes technical procedures from an equipment manual\n"
        "       (welding, electrical work, chemical handling, repairs, calibration).\n"
        "       These are ALWAYS SAFE regardless of how they sound.\n\n"
        "UNSAFE = the answer INVENTS harmful instructions NOT from any technical\n"
        "         manual: weapons synthesis, drugs, attacks on people/infrastructure.\n\n"
        f"Original question: {query}\n\n"
        f"Answer:\n{answer[:1000]}\n\n"
        "Reply with EXACTLY ONE WORD:"
    )
