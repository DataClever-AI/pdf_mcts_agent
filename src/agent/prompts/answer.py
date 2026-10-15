"""3. answer_generation prompt — evidence formatting, citation enforcement, main prompt."""
from __future__ import annotations

import re

from src.agent.prompts._shared import (
    _content_nodes,
    _format_messages_block,
    _tokenize_for_overlap,
)
from src.models.evidence import EvidenceBundle
from src.models.section import Section


# Evidence budget for answer generation. Shrinking these (5 full sections, 24k
# chars) and adding a "quote before refusing" rule were both measured against
# the unchanged baseline and did NOT reduce false "not covered" refusals, so
# the defaults stay at the original values. The fix for those refusals is the
# per-section relevance gate in generate_answer, not prompt tuning.
_MAX_PRIMARY = 12       # primary sections shown in full
_MAX_CHARS = 96_000     # ~24k tokens - leaves room for overhead + output in a 32k window
_SUPPORT_NODES = 50     # content nodes kept per supporting-context section


def format_evidence(
    bundle: EvidenceBundle,
    mcts_path: list[str] | None = None,
    *,
    max_primary: int = _MAX_PRIMARY,
    max_chars: int = _MAX_CHARS,
    support_nodes: int = _SUPPORT_NODES,
) -> str:
    """
    Format EvidenceBundle for VLM answer generation.

    Priority order (most relevant first):
    1. TOP RANKED — the best `max_primary` primary sections, all content nodes
    2. MCTS path sections (if different from primary)
    3. SUPPORTING CONTEXT — remaining primary sections, then parents, siblings
       and children, trimmed to `support_nodes` nodes each, added until the
       `max_chars` budget is reached
    """
    lines: list[str] = []
    all_by_id = {s.section_id: s for s in bundle.all_sections}
    shown: set[str] = set()
    used = 0

    # 1. TOP RANKED sections — always included fully (high precision)
    top = bundle.primary_sections[:max_primary]
    if top:
        lines.append("=== TOP RANKED SECTIONS ===")
        for sec in top:
            lines.append(f"\n[{sec.section_id}] {sec.title}")
            for text in _content_nodes(sec, max_nodes=50):
                lines.append(f"  • {text}")
            shown.add(sec.section_id)
        used = sum(len(line) for line in lines)

    # 2. MCTS path sections (additional exploration perspective)
    path_sections = [
        all_by_id[sid] for sid in (mcts_path or [])
        if sid in all_by_id and sid not in shown
    ]
    if path_sections:
        lines.append("\n=== MCTS EXPLORATION PATH ===")
        for sec in path_sections:
            lines.append(f"\n[{sec.section_id}] {sec.title}")
            for text in _content_nodes(sec, max_nodes=support_nodes):
                lines.append(f"  • {text}")
            shown.add(sec.section_id)
        used = sum(len(line) for line in lines)

    # 3. Supporting context — add until the budget is reached
    supporting = (
        bundle.primary_sections[max_primary:]
        + bundle.parent_sections
        + bundle.sibling_sections
        + bundle.child_sections
    )
    if supporting:
        lines.append("\n=== SUPPORTING CONTEXT ===")
        for sec in supporting:
            if sec.section_id in shown or used >= max_chars:
                continue
            entry = f"\n[{sec.section_id}] {sec.title}\n"
            for text in _content_nodes(sec, max_nodes=support_nodes):
                entry += f"  • {text}\n"
            lines.append(entry)
            shown.add(sec.section_id)
            used += len(entry)

    return "\n".join(lines)


def _build_citation_map(bundle: EvidenceBundle) -> str:
    """Build section reference guide — only primary + path sections (not all 30+)."""
    lines: list[str] = []
    # Only show primary sections in citation map — reduces length, improves precision
    for sec in bundle.primary_sections[:12]:
        page_str = ""
        if sec.page_start is not None:
            page_str = (
                f"p.{sec.page_start}"
                if sec.page_start == sec.page_end
                else f"p.{sec.page_start}-{sec.page_end}"
            )
        label = f"{sec.title}" + (f" ({page_str})" if page_str else "")
        lines.append(f"  {sec.section_id} = {label}")
    return "\n".join(lines)


def _format_page_str(sec: Section) -> str:
    if sec.page_start is None:
        return ""
    return (
        f"p.{sec.page_start}" if sec.page_start == sec.page_end
        else f"p.{sec.page_start}-{sec.page_end}"
    )


def _section_relevant_to_answer(sec: Section, answer_words: set[str]) -> bool:
    """
    Does this section's title or content share meaningful vocabulary with
    the answer? Deliberately does NOT trust mcts_path or the VLM ranking
    score as the filter — both have been observed to pick a suboptimal or
    outright wrong path/section this session. Grounding the filter in the
    ANSWER TEXT itself (not an upstream search algorithm's pick) means a
    bad MCTS run can't silently produce a wrong-but-confident citation list;
    worst case here is a topically-unrelated section slips through because
    it happens to share incidental vocabulary, not that a genuinely relevant
    section gets excluded because search picked a different "best" path.

    Content-level overlap requires >=5-char words specifically — short common
    words ("set", "has", "each", "stopped") pass the stopword filter but
    still coincidentally overlap between ANY two unrelated technical texts
    (observed directly: "Setting Up Timers" matched on {set, has, each} with
    an alarm-silencing answer). Domain-specific terms in these manuals run
    long (arrhythmia, indicator, timeout, calibration) — a longer minimum
    length is a better distinctiveness proxy than any fixed stopword list.

    Title overlap requires >=2 shared words, not 1 — a single shared word is
    often just the document's own recurring subject ("Monitoring", "Setup",
    "Alarm"), not a topical match. Observed directly: "BIS Monitoring Setup"
    and "Monitoring Paced Patients" both passed a >=1 threshold against an
    EEG setup answer purely on the word "monitoring", which is in most
    section titles of any patient-monitor manual and says nothing about
    which specific section is relevant.
    """
    if len(_tokenize_for_overlap(sec.title) & answer_words) >= 2:
        return True
    content_words: set[str] = set()
    for text in _content_nodes(sec, max_nodes=10):
        content_words |= _tokenize_for_overlap(text)
    distinctive_overlap = {w for w in (content_words & answer_words) if len(w) >= 5}
    return len(distinctive_overlap) >= 3


_SOURCES_HEADING_RE = re.compile(
    r"\n\s*\*{0,2}(?:Sources|References)\*{0,2}:?\s*\n", re.IGNORECASE
)


def _strip_model_sources_block(answer: str) -> str:
    """
    Drop any 'Sources:'/'References:' block the model appended on its own.
    build_answer_prompt now tells the model not to add one (see CITATION
    RULE), but a 7B model doesn't reliably follow that either — same pattern
    as every other instruction-only rule this session. If we don't strip it,
    two problems compound: (1) the model's own list is never filtered by
    _section_relevant_to_answer, so it can cite as many irrelevant sections
    as it wants (observed directly: an EEG setup answer citing "Calibrating
    the Transducer" and "BIS Monitoring Setup"), and (2) those extra section
    titles get tokenized into answer_words below, inflating overlap scores
    for exactly the sections that shouldn't be there.
    """
    match = _SOURCES_HEADING_RE.search(answer)
    return answer[: match.start()].rstrip() if match else answer


def ensure_citations(answer: str, bundle: EvidenceBundle) -> str:
    """
    Guarantee every answer is traceable to source sections, and that the
    Sources list is always the deterministic, filtered one — never whatever
    the model produced on its own. The CITATION RULE in build_answer_prompt
    is only a natural-language instruction; like every other instruction-only
    rule this session, a 7B model doesn't reliably follow it (it drops the
    instruction not to self-cite a list, or self-cites too broadly). Rather
    than hope harder, this always strips any model-authored Sources/
    References block and always rebuilds the footer from
    bundle.primary_sections filtered to sections that actually share
    vocabulary with the answer BODY (not the model's own citation list).

    Without the filter, this footer used to dump ALL primary_sections
    (rank_sections always keeps exactly 12 candidates, regardless of whether
    the bottom few are actually relevant) — observed directly: a yellow
    arrhythmia alarm answer citing "Setting Up Timers" and "QT Baseline",
    sections the VLM ranker itself had already scored near-zero relevance,
    just because they filled out the top-12 slots. Falls back to the full
    unfiltered list only if filtering would leave zero sections (better to
    over-cite than cite nothing).
    """
    body = _strip_model_sources_block(answer)
    answer_words = _tokenize_for_overlap(body)
    candidates = [
        sec for sec in bundle.primary_sections
        if _section_relevant_to_answer(sec, answer_words)
    ] or bundle.primary_sections  # fallback: don't end up with zero citations

    seen: set[str] = set()
    lines = ["\n\n**Sources:**"]
    for sec in candidates:
        if sec.title in seen:
            continue
        seen.add(sec.title)
        page_str = _format_page_str(sec)
        lines.append(f"- **{sec.title}**" + (f" ({page_str})" if page_str else ""))
    return body + "\n".join(lines)


def build_section_relevance_prompt(query: str, section: Section) -> str:
    """
    Binary check for ONE section: does it contain information that answers the query?

    One short question about one section is something the 7B answers reliably
    (0/5 false YES on questions the manual does not cover, YES on 35/40 target
    sections), unlike the free-form "answer or refuse" decision it makes inside
    the long answer prompt.
    """
    body = "\n".join(f"- {t}" for t in _content_nodes(section, max_nodes=30))[:2500]
    return (
        "You check whether ONE section of a technical manual contains information that\n"
        "answers a question.\n\n"
        f"QUESTION: {query}\n\n"
        f"SECTION: {section.title}\n{body}\n\n"
        "Does this section contain information that answers the question - even only\n"
        "in part, even if it uses different words (e.g. 'prior to disposal' answers\n"
        "'how do I dispose of it')? A section that only mentions the topic in passing\n"
        "or is about something else does NOT.\n"
        "Reply with ONLY one word: YES or NO."
    )


def build_answer_prompt(
    query: str,
    bundle: EvidenceBundle,
    mcts_path: list[str] | None = None,
    messages: list[dict[str, str]] | None = None,
    summary: str | None = None,
    manual_category: str | None = None,
    retry_issue: str | None = None,
    *,
    max_primary: int = _MAX_PRIMARY,
    max_chars: int = _MAX_CHARS,
    evidence_verified: bool = False,
) -> str:
    """
    Answer generation with anti-hallucination guardrail.
    Structure: context → anti-hallucination → citation rule → reference → evidence → question

    retry_issue: set to the previous attempt's judge issue
    ("gap_admission_false_negative", "ungrounded", or "incomplete_steps")
    when this is a refine_path retry. Penalizing sections and re-running
    MCTS is a no-op for all three failure modes — weak_section_ids is
    deliberately left empty for each (see judge_answer), so without this,
    the retry gets the exact same prompt and — since temperature=0.0 and
    frequency_penalty are both deterministic — deterministically reproduces
    the identical wrong answer. This injects an issue-specific correction
    instead of silently repeating the retry loop.
    """
    from src.agent.categories import get_category

    cat = get_category(manual_category or "general")
    expert_role = cat["expert_role"]

    evidence_text = format_evidence(
        bundle, mcts_path, max_primary=max_primary, max_chars=max_chars
    )
    citation_map = _build_citation_map(bundle)
    context_block = _format_messages_block(messages, summary, user_only=True)
    context_section = (
        f"\nPRIOR QUESTIONS IN THIS CONVERSATION (for context only):\n"
        f"{context_block}\n"
    ) if context_block else ""

    return (
        f"You are a {expert_role}.\n\n"
        "ANTI-HALLUCINATION — CHECK THIS FIRST:\n"
        "Step 1: Do the evidence sections contain content ABOUT the question topic?\n"
        "  UNIVERSAL MANUAL PATTERN: Technical manuals describe procedures from\n"
        "  the manufacturer's perspective (installation, setup, connection) but\n"
        "  users ask from their perspective (replace, configure, troubleshoot).\n"
        "  These cross-perspective pairs always refer to the SAME procedure:\n"
        "    install = replace = swap = fit = mount = put in\n"
        "    set up = configure = program = enter = adjust = set\n"
        "    inspect = check = verify = test = confirm\n"
        "    fault = error = failure = problem = issue = alarm\n"
        "    review = view = display = examine = monitor = look at\n"
        "  Example: question asks 'replace the battery' → section titled\n"
        "  'Battery Installation' IS the answer — present its full content.\n"
        "  Use the SECTION TITLES as final arbiter: if a title clearly names\n"
        "  the same component or action as the question, answer from it.\n"
        "  If a TOP RANKED SECTION title shares the question's key nouns —\n"
        "  regardless of domain — that is a MATCH — you MUST answer YES from\n"
        "  it, even if other evidence sections in the bundle look unrelated.\n"
        "  YES → Answer using those specific facts. Present all steps found.\n"
        "        Even partial procedures should be shown explicitly.\n"
        "  NO  → (evidence is truly about different topics) Write:\n"
        "        'The manual does not discuss [topic]. It covers [what it DOES\n"
        "        cover], but does not address [the specific gap].'\n"
        "        Do NOT invent. Do NOT write generic filler.\n"
        "        Do NOT fall back on your own general/pretrained knowledge of\n"
        "        this type of equipment to fill the gap — even topics you know\n"
        "        well in general (e.g. common cybersecurity risks, standard\n"
        "        safety practices) must come from THIS evidence, not from what\n"
        "        you already know about the subject. Evidence being off-topic\n"
        "        is not permission to switch to general knowledge — it is the\n"
        "        signal to say the manual doesn't cover it.\n\n"
        "SCOPE GUARDRAIL — applies to EVERY claim, including ones inside a list:\n"
        "  Some questions ask for things a single equipment manual cannot contain:\n"
        "  comparisons to other manufacturers'/competing products, independent\n"
        "  scientific evaluation of the evidence, or a designed process/framework\n"
        "  (e.g. a training curriculum, a root-cause-analysis methodology, a\n"
        "  cybersecurity best-practices checklist) that the manual itself does not\n"
        "  lay out. For these:\n"
        "  - NEVER name or compare against another product, technology, or\n"
        "    manufacturer unless the manual itself names it.\n"
        "  - NEVER pad a real, grounded list with extra generic industry-standard\n"
        "    items (audits, penetration testing, staff training programs, incident\n"
        "    response plans, etc.) just to make the answer feel complete. If the\n"
        "    manual gives 3 grounded points, give exactly those 3 — do not add a\n"
        "    4th, 5th, 6th invented to round it out.\n"
        "  - Check EACH bullet/step individually against the evidence, not just the\n"
        "    overall topic. One grounded item does not license adding an ungrounded\n"
        "    one next to it in the same list.\n"
        "  - If the question asks for something beyond what specific evidence\n"
        "    supports, answer ONLY the grounded part, then explicitly state what\n"
        "    the manual does not cover — do not blend real and invented content\n"
        "    into one confident-sounding list.\n\n"
        "COMPLETENESS RULE — do not silently drop items:\n"
        "  If EVIDENCE contains a numbered procedure (1. 2. 3. ...), your answer\n"
        "  MUST include EVERY numbered step, in the same order, with nothing\n"
        "  skipped, merged into another step, or summarized away. This applies\n"
        "  equally to a warning, caution, or explanation attached to a step — do\n"
        "  not drop 'why' text or a caution sentence just because the step's\n"
        "  action is the part that feels most important. Losing one step out of\n"
        "  eight, or losing a caution attached to a step, is a real defect, not\n"
        "  an acceptable simplification. Do not shorten a procedure for brevity.\n\n"
        "NO GENERIC CLOSING SENTENCE:\n"
        "  Do not end your answer with a summary sentence that makes a claim not\n"
        "  explicitly in the evidence — e.g. 'these steps ensure proper setup',\n"
        "  'this follows [manufacturer]'s guidelines for safe operation', 'this\n"
        "  ensures accurate and reliable results'. That kind of sentence is not\n"
        "  in the manual; it is invented meta-commentary. End on the last\n"
        "  concrete fact or step instead of adding a wrap-up claim.\n\n"
        "ISOLATION RULE — CRITICAL:\n"
        "You are answering the CURRENT QUESTION at the bottom of this prompt.\n"
        "Do NOT copy, repeat, or reuse any previous answer from this session.\n"
        "Each question gets a FRESH answer based solely on the evidence below.\n\n"
        "CITATION RULE:\n"
        "Cite sections as: **Section Title (p.X)** — use the reference list below.\n"
        "NEVER write [sec_0001] or any section ID directly in your answer.\n"
        "Do NOT add your own 'Sources:'/'References:' list at the end — the\n"
        "system appends an accurate one automatically after your answer.\n\n"
        + (
            "NOTE - READ BEFORE ANSWERING:\n"
            "The EVIDENCE below was narrowed to the sections most relevant to this\n"
            "question, and at least one of them answers it. Answer from that\n"
            "section's own words. If a detail the question asks for is not stated,\n"
            "say only that detail is not specified - do not say the topic is\n"
            "missing.\n\n"
            if retry_issue == "gap_admission_false_negative" else ""
        )
        + (
            "NOTE - READ BEFORE ANSWERING:\n"
            "Every section in EVIDENCE below was checked and contains information\n"
            "that answers this question. Answer from the sections' own words. If a\n"
            "specific detail the question asks for is not stated, say only that\n"
            "detail is not specified - do not say the topic is missing.\n\n"
            if evidence_verified and retry_issue != "gap_admission_false_negative" else ""
        )
        + (
            "CORRECTION — READ BEFORE ANSWERING:\n"
            "A previous attempt at THIS EXACT QUESTION included a specific claim\n"
            "(a number, duration, step, or detail) that was NOT grounded in the\n"
            "evidence below, or that CONTRADICTED another part of the same answer\n"
            "(e.g. stating two different durations for the same event). Before\n"
            "writing anything, go claim by claim: for every specific number, time,\n"
            "or step you are about to state, find the exact sentence in EVIDENCE\n"
            "that supports it. If you cannot find it, do not state it — say the\n"
            "manual does not specify that detail instead of guessing or repeating\n"
            "a plausible-sounding number from memory. Do NOT introduce a 'summary'\n"
            "or 'in short' section that restates facts with different specifics\n"
            "than the body of your answer already gave — this is exactly how the\n"
            "previous attempt contradicted itself.\n\n"
            if retry_issue == "ungrounded" else ""
        )
        + (
            "CORRECTION — READ BEFORE ANSWERING:\n"
            "A previous attempt at THIS EXACT QUESTION dropped one or more numbered\n"
            "steps that are present in the source section below. Before answering,\n"
            "count the numbered steps (1. 2. 3. ...) in the EVIDENCE section your\n"
            "answer is based on, and make sure your answer includes EVERY one of\n"
            "them, in the same order, with nothing skipped, merged into another\n"
            "step, or summarized away.\n\n"
            if retry_issue == "incomplete_steps" else ""
        )
        + "SECTION REFERENCE LIST:\n"
        f"{citation_map}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        f"CURRENT QUESTION: {query}"
        f"{context_section}\n"
        "ANSWER (for the CURRENT QUESTION above, not for any prior question):"
    )
