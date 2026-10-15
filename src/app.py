"""
MVP v2 — Complete single-page Streamlit app.

Stage 1: Upload PDF → extract bookmarks (fitz) + content (docling) → build tree
Stage 2: Tree explorer — browse sections, content, hierarchy
Stage 3: Query — MCTS RAG pipeline with full agent debug output
"""
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="MVP v2 — Document Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        # Auth
        "logged_in": False,
        "user_id": None,
        "username": None,
        # App stages
        "stage": "upload",
        "pdf_name": None,
        "pdf_path": None,
        "pdf_source_path": None,
        "pdf_stem": None,
        "pages_dir": None,
        "bookmarks": [],
        "total_pages": 0,
        "tree_json_path": None,
        "sections": [],
        "pipeline_events": [],
        "last_query": "",
        # Canonical LangGraph conversation memory
        "messages_history": [],
        "messages_summary": None,
        "conversation_id": None,  # = user_id when logged in
        # Manual category
        "manual_category": None,
        "running_query": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _a(obj: Any, key: str, default: Any = "?") -> Any:
    return getattr(obj, key, default)

def _sec(row: Any) -> Any:
    return getattr(row, "section", None)

def _verdict_icon(verdict: str) -> str:
    return {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴",
            "BLOCKED": "⛔", "BLOCK": "⛔"}.get(verdict.upper(), "⚪")

def _score_bar(score: float, width: int = 15) -> str:
    filled = int(score * width)
    return f"{'█' * filled}{'░' * (width - filled)} {score:.2f}"

def _reset() -> None:
    """Clear active document + conversation memory (both are document-scoped)."""
    for k in ["stage", "pdf_name", "pdf_path", "bookmarks", "total_pages",
              "tree_json_path", "sections", "pipeline_events",
              "pdf_stem", "pdf_source_path", "pages_dir", "manual_category",
              "messages_history", "messages_summary", "last_query"]:
        if k in st.session_state:
            del st.session_state[k]
    _init_state()
    st.cache_resource.clear()

# ---------------------------------------------------------------------------
# Stage 1 helpers — PDF → Tree
# ---------------------------------------------------------------------------

def _extract_bookmarks(pdf_path: Path) -> tuple[list[tuple[int, str, int]], int]:
    from src.tree_builder.fitz_toc import get_embedded_toc, page_count
    bm = get_embedded_toc(pdf_path)
    pages = page_count(pdf_path)
    return bm, pages


def _doc_stem(pdf_name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(pdf_name).stem)[:40]


def _user_docs_root(user_id: str) -> Path:
    """Per-user document storage root — isolates one user's trees from another's."""
    return Path("data") / "users" / user_id


def _build_tree(
    pdf_path: Path,
    bookmarks: list[tuple[int, str, int]],
    total_pages: int,
    use_docling: bool,
    progress_placeholder: Any,
    pdf_name: str,
    owner_user_id: str,
    manual_category: str | None = None,
) -> tuple[Path, Path, Path]:
    """
    Returns (tree_json_path, source_pdf_path, pages_dir).

    Documents are stored per-user at data/users/{owner_user_id}/{stem}/ —
    isolation is enforced by the storage path itself, not just by filtering
    the picker, so two users uploading a same-named PDF can never share or
    silently reuse each other's tree.

    Skips Docling/rebuild entirely if this exact user already processed this
    document — tree.json persists to disk permanently, so re-uploading the
    same PDF reuses it rather than re-running the expensive extraction. See
    also _list_processed_documents() for the picker that lets a user jump
    straight to one of their own existing trees without uploading anything.
    """
    import json
    import shutil
    import time

    from src.models.extraction import DoclingDocument
    from src.tree_builder.section_matcher import match_content_to_sections
    from src.tree_builder.tree_export import write_tree_json

    stem = _doc_stem(pdf_name)
    doc_dir = _user_docs_root(owner_user_id) / stem
    doc_dir.mkdir(parents=True, exist_ok=True)

    source_path = doc_dir / "source.pdf"
    pages_dir = doc_dir / "pages"
    tree_path = doc_dir / "tree.json"

    if tree_path.exists():
        progress_placeholder.progress(1.0, text="Reusing previously built tree ✓")
        return tree_path, source_path, pages_dir

    # Save permanent copy of the PDF
    if not source_path.exists():
        shutil.copy2(pdf_path, source_path)
    pages_dir.mkdir(exist_ok=True)

    doc: DoclingDocument | None = None

    if use_docling:
        try:
            import logging as _log

            from src.tree_builder.docling_extract import DoclingExtractionEngine
            engine = DoclingExtractionEngine()
            if engine.is_available:
                logger = _log.getLogger("docling_extract")
                logger.setLevel(_log.WARNING)

                total_wins: list[int] = [1]

                def _on_window(w_idx: int, total: int, elapsed: float) -> None:
                    total_wins[0] = total
                    progress_placeholder.progress(
                        w_idx / max(total, 1),
                        text=f"Docling: window {w_idx}/{total} ({elapsed:.0f}s elapsed)"
                    )

                progress_placeholder.progress(0.0, text="Starting Docling extraction...")
                doc = engine.extract(source_path, logger, on_window_complete=_on_window)
                progress_placeholder.progress(1.0, text="Docling complete ✓")
        except Exception as e:
            st.warning(f"Docling extraction failed ({e}) — building structure-only tree")

    if doc is None:
        from src.models.extraction import DoclingDocument as DD
        doc = DD(document_id="empty", source_path=str(source_path), total_pages=total_pages)

    sections = match_content_to_sections(bookmarks, doc, total_pages)
    write_tree_json(sections, tree_path)

    # Sidecar metadata — lets the document picker list/label this tree and
    # restores manual_category (needed by guard/prompts) without re-upload.
    meta = {
        "pdf_name": pdf_name,
        "owner_user_id": owner_user_id,
        "manual_category": manual_category,
        "bookmarks_count": len(bookmarks),
        "total_pages": total_pages,
        "section_count": len(sections),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (doc_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return tree_path, source_path, pages_dir


def _list_processed_documents(owner_user_id: str) -> list[dict[str, Any]]:
    """
    Scan data/users/{owner_user_id}/*/tree.json for this user's already-built
    documents. Isolation is by storage path — this glob physically cannot see
    another user's directory, so there's no metadata check to forget/bypass.
    """
    import json as _json

    docs: list[dict[str, Any]] = []
    for tree_path in sorted(_user_docs_root(owner_user_id).glob("*/tree.json")):
        doc_dir = tree_path.parent
        meta_path = doc_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        else:
            meta = {}
        docs.append({
            "stem": doc_dir.name,
            "tree_path": str(tree_path),
            "pdf_name": meta.get("pdf_name", doc_dir.name),
            "manual_category": meta.get("manual_category"),
            "section_count": meta.get("section_count"),
            "built_at": meta.get("built_at", "unknown"),
        })
    return docs


@st.cache_resource(max_entries=10, ttl=3600)
def _load_sections_cached(tree_path: str) -> list[Any]:
    from src.rag.loader import load_tree
    return load_tree(Path(tree_path))


@st.cache_resource
def _get_auth_db() -> Any:
    """Shared database for authentication and feedback — one instance per process."""
    from src.persistence.database import Database
    Path("data").mkdir(exist_ok=True)
    db = Database("data/mvp.db")
    db.create_tables()
    return db


@st.cache_resource(max_entries=5, ttl=3600)
def _build_reasoner_cached(tree_path: str, use_vlm_mcts: bool) -> Any:
    """
    Cached per (tree_path, use_vlm_mcts) — each entry holds a BM25 index and
    MCTS explorer for one document. max_entries=5 caps memory growth across a
    long-running process with many documents opened in one session;
    ttl=3600 reclaims idle documents after an hour. _get_auth_db is
    intentionally NOT capped — it's a true singleton (one DB connection for
    the whole app), not a per-document cache.
    """
    from src.agent.graph import DocumentReasoner
    from src.models.mcts import MCTSConfig

    sections = _load_sections_cached(tree_path)
    db = _get_auth_db()
    config = MCTSConfig(use_vlm=use_vlm_mcts, n_simulations=50, max_vlm_concurrency=10)
    return DocumentReasoner(sections=sections, db=db, mcts_config=config)


# ---------------------------------------------------------------------------
# Stage 2 — Tree renderer
# ---------------------------------------------------------------------------

def _render_tree(tree_path: str) -> None:
    import json
    with open(tree_path, encoding="utf-8") as f:
        tree: list[dict[str, Any]] = json.load(f)

    sec_by_id = {s["section_id"]: s for s in tree}
    roots = [s for s in tree if s.get("parent_section_id") is None]

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    stats_col1.metric("Total sections", len(tree))
    stats_col2.metric("Root sections", len(roots))
    max_level = max((s.get("hierarchy_level", 1) for s in tree), default=1)
    stats_col3.metric("Max depth", max_level)
    total_nodes = sum(s.get("node_count", 0) or 0 for s in tree)
    stats_col4.metric("Content nodes", total_nodes)

    st.divider()

    def _render_node(sec: dict[str, Any], indent: int = 0) -> None:
        lvl = sec.get("hierarchy_level", 1)
        title = sec.get("title", "?")
        sid = sec.get("section_id", "?")
        node_count = sec.get("node_count", 0) or 0
        page_start = sec.get("page_start", "?")
        page_end = sec.get("page_end", "?")
        children = sec.get("child_sections", [])

        icon = "📁" if children else "📄"
        label = (
            f"{'  ' * indent}{icon} **{title}** `{sid}` "
            f"— {node_count} nodes | p.{page_start}-{page_end}"
        )

        with st.expander(label, expanded=(lvl <= 2)):
            if node_count > 0:
                nodes = sec.get("semantic_nodes", [])
                for node in nodes[:3]:
                    text = node.get("canonical_text", "")[:200]
                    ntype = node.get("node_type", "")
                    page = node.get("page_no", "")
                    st.caption(f"[{ntype}] p.{page}  {text}")
                if len(nodes) > 3:
                    st.caption(f"… +{len(nodes) - 3} more nodes")
            else:
                st.caption("_(no direct content — structural container)_")

            if children:
                for cid in children:
                    child = sec_by_id.get(cid)
                    if child:
                        _render_node(child, indent + 1)

    for root in roots:
        _render_node(root, 0)


# ---------------------------------------------------------------------------
# Stage 3 — Pipeline node renderers
# ---------------------------------------------------------------------------

def render_guard(upd: dict[str, Any]) -> None:
    if upd.get("blocked"):
        cat = (upd.get("block_category") or "unknown").upper()
        st.error(f"⛔ BLOCKED [{cat}]")
        st.write(upd.get("answer", ""))
    else:
        st.success("✅ PASS — query accepted")


def render_expand(upd: dict[str, Any]) -> None:
    original = upd.get("original_query") or upd.get("query") or ""
    sub_queries: list[str] = upd.get("sub_queries") or []
    messages: list[dict[str, Any]] = list(st.session_state.get("messages_history") or [])
    summary: str | None = st.session_state.get("messages_summary")

    # Show conversation context from session memory
    if messages or summary:
        with st.expander(
            f"📚 Conversation memory ({len(messages)//2} prior Q&A"
            + (", summary active" if summary else "")
            + ")", expanded=False
        ):
            if summary:
                st.caption(f"**Summary of older exchanges:**\n{summary}")
            for m in messages[-6:]:
                role = "🙋 User" if m.get("role") == "user" else "🤖 Assistant"
                st.caption(f"**{role}:** {str(m.get('content',''))[:200]}")

    st.caption("Original query")
    st.info(original)

    if sub_queries:
        st.caption(f"Reformulated into {len(sub_queries)} sub-questions for retrieval")
        for i, q in enumerate(sub_queries, 1):
            st.success(f"**{i}.** {q}")
    else:
        st.caption("Expanded terms (fallback)")
        st.success(upd.get("expanded_query") or original)


def render_retrieve(upd: dict[str, Any]) -> None:
    candidates = upd.get("candidate_sections") or []
    if not candidates:
        st.warning("No sections retrieved")
        return
    st.caption(f"{len(candidates)} sections retrieved")
    rows = [
        {
            "section_id": _a(_sec(s), "section_id"),
            "title": str(_a(_sec(s), "title"))[:50],
            "level": _a(_sec(s), "hierarchy_level"),
            "score": f"{_a(s, 'retrieval_score', 0.0):.3f}",
        }
        for s in candidates[:20]
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def render_rank(upd: dict[str, Any]) -> None:
    ranked = upd.get("ranked_sections") or []
    if not ranked:
        st.warning("No ranked sections")
        return
    st.caption(f"{len(ranked)} sections after hybrid ranking")
    rows = [
        {
            "rank": _a(r, "rank"),
            "section_id": _a(_sec(r), "section_id"),
            "title": str(_a(_sec(r), "title"))[:45],
            "BM25": f"{_a(r, 'retrieval_score', 0.0):.3f}",
            "VLM": f"{_a(r, 'vlm_score', 0.0):.2f}",
            "composite": f"{_a(r, 'ranking_score', 0.0):.3f}",
        }
        for r in ranked
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def render_evidence(upd: dict[str, Any]) -> None:
    bundle = upd.get("evidence")
    if bundle is None:
        st.warning("No evidence bundle")
        return
    primary = getattr(bundle, "primary_sections", [])
    parents = getattr(bundle, "parent_sections", [])
    siblings = getattr(bundle, "sibling_sections", [])
    children = getattr(bundle, "child_sections", [])
    related = getattr(bundle, "related_nodes", [])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Primary", len(primary))
    c2.metric("Parents", len(parents))
    c3.metric("Siblings", len(siblings))
    c4.metric("Children", len(children))
    c5.metric("Related nodes", len(related))

    with st.expander("Expansion trace"):
        for t in getattr(bundle, "expansion_trace", []):
            st.code(t, language=None)


def render_mcts(upd: dict[str, Any]) -> None:
    path = upd.get("mcts_path") or []
    sims = upd.get("mcts_n_simulations", 0)
    cache = upd.get("mcts_score_cache") or {}

    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Simulations", sims)
        st.metric("Path length", len(path))
    with c2:
        st.caption("Best path")
        st.code(" → ".join(path) if path else "none", language=None)

    if cache:
        with st.expander(f"Section scores ({len(cache)} total)"):
            top = sorted(cache.items(), key=lambda x: x[1], reverse=True)[:15]
            for sid, score in top:
                st.progress(score, text=f"{sid}  {score:.3f}")


def render_answer(upd: dict[str, Any]) -> None:
    answer = upd.get("answer") or ""
    fallback = upd.get("fallback_used", False)
    confidence = upd.get("answer_confidence") or "?"
    icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    st.caption(f"Confidence: {icons.get(confidence,'⚪')} {confidence} "
               f"| Fallback: {'⚠️ Yes' if fallback else '✅ No'}")
    st.markdown(answer)


def render_judge(upd: dict[str, Any]) -> None:
    result = upd.get("judge_result") or {}
    answer_blocked = upd.get("answer_blocked", False)
    verdict = (result.get("verdict") or "?").upper()
    overall = float(result.get("overall", 0))

    fn = {"PASS": st.success, "WARN": st.warning, "FAIL": st.error}.get(verdict, st.info)
    fn(f"{_verdict_icon(verdict)} **{verdict}** — {_score_bar(overall)}")

    if answer_blocked:
        st.error("⛔ Answer blocked by safety guardrail")

    issues = result.get("issues") or []
    if issues:
        for issue in issues:
            st.caption(f"• {issue}")


def render_summarize(upd: dict[str, Any]) -> None:
    msgs = (upd or {}).get("messages")
    summary = (upd or {}).get("summary")
    existing = list(st.session_state.get("messages_history") or [])
    n_qa = len(existing) // 2
    threshold = 5  # matches _MESSAGES_SUMMARIZE_THRESHOLD // 2

    if summary:
        st.success(f"✅ Summary generated — {len(msgs) // 2 if msgs else 0} Q&A kept")
        st.caption(summary)
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Q&A in context", n_qa)
        col2.metric("Threshold", threshold)
        col3.metric("Status", "✅ OK" if n_qa < threshold else "⚠️ At limit")
        if len(existing) >= 2:
            last_q = existing[-2].get("content", "")[:80]
            st.caption(f"Last exchange: \"{last_q}...\"" )


_NODE_META: dict[str, tuple[str, Any]] = {
    "guard_query":          ("🛡️ Guard — Safety Check",           render_guard),
    "summarize_if_needed":  ("🧠 Memory — Summarize if needed",   render_summarize),
    "expand_query":         ("🔍 Expand — Query Understanding",    render_expand),
    "retrieve_sections": ("📦 Retrieve — BM25 Top-20",       render_retrieve),
    "rank_sections":     ("📊 Rank — BM25 + VLM Hybrid",    render_rank),
    "expand_evidence":   ("🌳 Evidence Bundle",              render_evidence),
    "run_mcts":          ("🎲 MCTS — Path Search",           render_mcts),
    "refine_path":       ("🔄 Refine Path",                  render_mcts),
    "generate_answer":   ("💬 Answer Generation",            render_answer),
    "judge_answer":      ("⚖️ Judge — Quality & Safety",    render_judge),
    "store_memory":      ("💾 Memory — Persisted to SQLite",
                          lambda u: st.success(
                              "✅ Query, answer, and judge verdict saved to database"
                              if u.get("store_memory_done")
                              else "⚠️ No state to store"
                          )),
    "handle_error":      ("❌ Error", lambda u: st.error(str(u))),
}


def _render_referenced_pages(events: list[tuple[str, dict[str, Any]]]) -> None:
    """Render PDF pages cited by the MCTS best path + ranked sections."""
    source_path_str = st.session_state.get("pdf_source_path")
    pages_dir_str = st.session_state.get("pages_dir")
    if not source_path_str or not pages_dir_str:
        return

    source_path = Path(source_path_str)
    pages_dir = Path(pages_dir_str)

    if not source_path.exists():
        return

    # Collect section_ids from MCTS path and top ranked sections
    mcts_path: list[str] = []
    ranked_sections: list[Any] = []
    for node_name, upd in events:
        if node_name == "run_mcts":
            mcts_path = upd.get("mcts_path") or []
        if node_name == "rank_sections":
            ranked_sections = upd.get("ranked_sections") or []

    if not mcts_path and not ranked_sections:
        return

    # Load sections to get page numbers
    tree_path = st.session_state.get("tree_json_path")
    if not tree_path:
        return

    import json
    with open(tree_path, encoding="utf-8") as f:
        tree: list[dict[str, Any]] = json.load(f)
    sec_map = {s["section_id"]: s for s in tree}

    # Collect pages: MCTS path sections first, then top-3 ranked
    page_sources: list[tuple[int, str, str]] = []  # (page_no, section_id, title)
    seen_pages: set[int] = set()

    target_ids = list(mcts_path)
    for r in ranked_sections[:5]:
        sid = _a(_sec(r), "section_id")
        if sid and sid not in target_ids:
            target_ids.append(sid)

    for sid in target_ids:
        sec = sec_map.get(sid)
        if not sec:
            continue
        p_start = sec.get("page_start")
        p_end = sec.get("page_end")
        title = sec.get("title", sid)
        if p_start is None:
            continue
        for p in range(p_start, min(p_start + 3, (p_end or p_start) + 1)):
            if p not in seen_pages:
                page_sources.append((p, sid, title))
                seen_pages.add(p)

    if not page_sources:
        return

    # Render pages (on-demand, cached on disk)
    page_numbers = [p for p, _, _ in page_sources]
    from src.tree_builder.fitz_toc import render_pages
    with st.spinner(f"Rendering {len(page_numbers)} referenced page(s)..."):
        rendered = render_pages(source_path, page_numbers, pages_dir, dpi=150)

    if not rendered:
        return

    # Display
    st.divider()
    st.subheader("📄 Referenced Pages")
    st.caption("Pages from sections on MCTS best path, rendered from source PDF")

    # Group pages by section
    by_section: dict[str, list[int]] = {}
    sid_titles: dict[str, str] = {}
    for p, sid, title in page_sources:
        by_section.setdefault(sid, []).append(p)
        sid_titles[sid] = title

    for sid, pages in by_section.items():
        title = sid_titles[sid]
        badge = "🏆 MCTS best path" if sid in mcts_path else "📊 Top ranked"
        with st.expander(f"{badge} — **{title}**", expanded=True):
            img_cols = st.columns(min(len(pages), 3))
            for i, page_no in enumerate(pages):
                png_path = rendered.get(page_no)
                if png_path and png_path.exists():
                    with img_cols[i % 3]:
                        st.image(str(png_path), caption=f"Page {page_no}", width="stretch")


def _render_feedback_form(events: list[tuple[str, dict[str, Any]]], query: str) -> None:
    """
    Human-in-the-loop: let a person rate/correct this answer after seeing it.

    Writes directly to the same `feedback` row store_memory already inserted
    (via feedback_id, carried in the store_memory event) — this is a
    human judgment attached after the fact, separate from judge_answer's
    automated verdict, not a replacement for it. A non-empty correction also
    creates a human_corrections row (richer record than a bare rating,
    intended for future review/fine-tuning use — see Database docstrings).
    """
    feedback_id = None
    for node_name, upd in events:
        if node_name == "store_memory" and upd:
            feedback_id = upd.get("feedback_id")
            break
    if not feedback_id:
        return

    st.divider()
    st.subheader("📝 Rate this answer")
    st.caption("😞 Very bad → 😀 Excellent")

    sentiment = st.feedback("faces", key=f"fb_faces_{feedback_id}")
    correction = st.text_area(
        "Comments — what's wrong, or what's the correct answer? (optional)",
        placeholder="Leave blank if the rating alone is enough feedback.",
        key=f"fb_comment_{feedback_id}",
    )
    submitted = st.button("Submit feedback", key=f"fb_submit_{feedback_id}")

    if submitted:
        if sentiment is None:
            st.warning("Pick a rating before submitting.")
        else:
            rating = sentiment + 1  # st.feedback("faces"): 0-4 → DB rating 1-5
            db = _get_auth_db()
            db.update_feedback_rating(feedback_id, rating, correction or None)
            if correction.strip():
                db.insert_human_correction(
                    feedback_id=feedback_id,
                    original_question=query,
                    wrong_sections=[],
                    correction_text=correction.strip(),
                )
            st.success("Thanks — feedback saved.")


async def _stream(
    reasoner: Any,
    query: str,
    messages: list[dict[str, Any]] | None = None,
    summary: str | None = None,
    conversation_id: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Stream LangGraph events per node.

    Passes canonical messages + summary memory (not manual DB loading).
    After the call, read state["messages"] and state["summary"] from the
    store_memory event to update session state for the next query.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    initial: dict[str, Any] = {
        "max_hierarchy_level": getattr(reasoner, "_max_hierarchy_level", 1),
        "manual_category": st.session_state.get("manual_category"),
        "query": query, "expanded_query": None, "original_query": None,
        "sub_queries": [],
        "candidate_sections": [], "ranked_sections": [], "rank_vlm_scores": {},
        "evidence": None,
        "mcts_path": None, "mcts_score_cache": {}, "mcts_n_simulations": 0,
        "blocked": False, "block_category": None, "answer": None,
        "answer_blocked": False, "answer_confidence": None, "fallback_used": False,
        "judge_result": None, "judge_attempts": 0, "feedback": None,
        "messages": list(messages or []),
        "summary": summary,
        "conversation_id": conversation_id, "thread_id": None,
        "store_memory_done": False,
        "feedback_id": None,
        "error": None, "error_layer": None, "logs": [],
        "last_answer_prompt": None, "last_answer_response": None,
        "last_judge_prompt": None, "last_judge_response": None,
    }
    async for chunk in reasoner._graph.astream(initial, stream_mode="updates"):
        for node_name, upd in chunk.items():
            if not node_name.startswith("__"):
                events.append((node_name, dict(upd) if upd else {}))
    return events


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def _render_auth() -> bool:
    """
    Show login / signup form. Returns True when user is logged in.
    Sets st.session_state.logged_in, user_id, username on success.
    """
    db = _get_auth_db()

    st.title("🔬 MVP v2 — Document Intelligence")
    st.markdown("### Welcome — please sign in or create an account")

    _left, col_form, _right = st.columns([1, 2, 1])

    tab_login, tab_signup = col_form.tabs(["🔐 Login", "✨ Create Account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")

        if submitted:
            with col_form:
                if not username or not password:
                    st.error("Username and password are required.")
                else:
                    user_id = db.verify_password(username, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.session_state.conversation_id = user_id
                        st.success(f"Welcome back, **{username}**!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_user = st.text_input("Username", key="su_user")
            new_email = st.text_input("Email (optional)", key="su_email")
            new_pass = st.text_input("Password", type="password", key="su_pass")
            confirm_pass = st.text_input("Confirm password", type="password", key="su_confirm")
            submitted2 = st.form_submit_button("Create account", type="primary")

        if submitted2:
            with col_form:
                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        user_id = db.create_user(
                            username=new_user,
                            password=new_pass,
                            email=new_email or None,
                        )
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = new_user
                        st.session_state.conversation_id = user_id
                        st.success(f"Account created! Welcome, **{new_user}**!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    return False


def main() -> None:
    # ── Auth gate ────────────────────────────────────────────────────────────
    if not st.session_state.get("logged_in"):
        _render_auth()
        return

    # ── Header (logged in) ───────────────────────────────────────────────────
    col_spacer, col_user = st.columns([4, 1])
    with col_user:
        username = st.session_state.get("username", "?")
        st.markdown(f"👤 **{username}**")
        col_logout, col_reset = st.columns(2)
        with col_logout:
            if st.button("Logout", width="stretch"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        with col_reset:
            if st.button("🔄 Reset", width="stretch"):
                _reset()
                st.rerun()

    st.title("🔬 MVP v2 — Document Intelligence")
    st.divider()

    stage = st.session_state.stage

    # ════════════════════════════════════════════════════════════════════════
    # STAGE: UPLOAD
    # ════════════════════════════════════════════════════════════════════════
    if stage == "upload":
        from src.agent.categories import CATEGORY_OPTIONS, get_category

        processed_docs = _list_processed_documents(st.session_state.user_id)
        if processed_docs:
            st.markdown("### 📂 Your previously processed documents")
            st.caption("Reuse a tree already built — no re-upload or Docling needed.")

            doc_cols = [4, 3, 2, 3, 1]
            h_name, h_cat, h_sections, h_date, h_use = st.columns(doc_cols)
            h_name.markdown("**Document**")
            h_cat.markdown("**Category**")
            h_sections.markdown("**Sections**")
            h_date.markdown("**Built**")
            h_use.markdown("**Action**")
            st.divider()

            for doc in processed_docs:
                cat_label = (
                    get_category(doc["manual_category"])["label"]
                    if doc["manual_category"] else "—"
                )
                col_name, col_cat, col_sections, col_date, col_use = st.columns(doc_cols)
                col_name.markdown(f"**{doc['pdf_name']}**")
                col_cat.markdown(cat_label)
                col_sections.markdown(str(doc["section_count"] or "?"))
                col_date.markdown(str(doc["built_at"]))
                with col_use:
                    if st.button("▶ Use", key=f"use_doc_{doc['stem']}", width="stretch"):
                        doc_dir = Path(doc["tree_path"]).parent
                        st.session_state.tree_json_path = doc["tree_path"]
                        st.session_state.pdf_source_path = str(doc_dir / "source.pdf")
                        st.session_state.pages_dir = str(doc_dir / "pages")
                        st.session_state.pdf_stem = doc["stem"]
                        st.session_state.pdf_name = doc["pdf_name"]
                        st.session_state.manual_category = doc["manual_category"]
                        # Conversation memory is scoped to the active document —
                        # switching documents without resetting it leaked topics
                        # from one manual's conversation into another's sub-question
                        # expansion (observed directly: HemoSphere calibration/alarm
                        # topics leaking into a Philips brightness question).
                        st.session_state.messages_history = []
                        st.session_state.messages_summary = None
                        st.session_state.stage = "ready"
                        st.rerun()
            st.divider()

        st.markdown("### Step 1 — Upload a PDF with bookmarks")
        st.caption("Bookmarks (Table of Contents) are required. "
                   "Docling extracts content; without it only structure is built.")

        col_upload, col_cat = st.columns([3, 2])
        with col_upload:
            uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        with col_cat:
            cat_labels = [label for _, label in CATEGORY_OPTIONS]
            cat_keys = [key for key, _ in CATEGORY_OPTIONS]
            cat_idx = st.selectbox(
                "Manual category",
                options=range(len(CATEGORY_OPTIONS)),
                format_func=lambda i: cat_labels[i],
                index=0,
                help="Improves guard accuracy and query terminology for this domain",
            )
            selected_category = cat_keys[cat_idx]

        if uploaded is None:
            return

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(uploaded.read())
        tmp.close()
        pdf_path = Path(tmp.name)

        with st.spinner("Reading bookmarks..."):
            bookmarks, total_pages = _extract_bookmarks(pdf_path)

        if not bookmarks:
            st.error("❌ No bookmarks found in this PDF. "
                     "MVP v2 requires a PDF with an embedded Table of Contents.")
            pdf_path.unlink(missing_ok=True)
            return

        levels = sorted(set(b[0] for b in bookmarks))
        st.success(f"✅ {len(bookmarks)} bookmarks found | {total_pages} pages | "
                   f"Levels: {levels} | Category: {cat_labels[cat_idx]}")

        with st.expander("Preview bookmarks", expanded=True):
            for level, title, page in bookmarks[:30]:
                indent = "  " * (level - 1)
                st.text(f"{indent}L{level}  {title}  (p.{page})")
            if len(bookmarks) > 30:
                st.caption(f"... +{len(bookmarks) - 30} more")

        use_docling = st.toggle(
            "Extract content with Docling",
            value=True,
            help="Adds text + tables to tree. Takes 1-5 min for large PDFs."
        )

        if st.button("▶ Build Tree", type="primary"):
            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_path = str(pdf_path)
            st.session_state.bookmarks = bookmarks
            st.session_state.total_pages = total_pages
            st.session_state.use_docling = use_docling
            st.session_state.manual_category = selected_category
            st.session_state.stage = "building"
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE: BUILDING
    # ════════════════════════════════════════════════════════════════════════
    elif stage == "building":
        pdf_path = Path(st.session_state.pdf_path)
        bookmarks = st.session_state.bookmarks
        total_pages = st.session_state.total_pages
        use_docling = st.session_state.get("use_docling", True)

        st.markdown(f"### Building tree for **{st.session_state.pdf_name}**")

        progress_ph = st.empty()
        with st.spinner("Building tree..."):
            tree_path, source_path, pages_dir = _build_tree(
                pdf_path, bookmarks, total_pages,
                use_docling, progress_ph,
                st.session_state.pdf_name or "document",
                owner_user_id=st.session_state.user_id,
                manual_category=st.session_state.get("manual_category"),
            )

        progress_ph.empty()
        st.success(f"✅ Tree built → `{tree_path}`")
        st.session_state.tree_json_path = str(tree_path)
        st.session_state.pdf_source_path = str(source_path)
        st.session_state.pages_dir = str(pages_dir)
        st.session_state.pdf_stem = source_path.parent.name
        # Conversation memory is scoped to the active document — see the
        # matching reset in the document-picker "Use" button above.
        st.session_state.messages_history = []
        st.session_state.messages_summary = None
        st.session_state.stage = "ready"
        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE: READY (tree + query)
    # ════════════════════════════════════════════════════════════════════════
    elif stage == "ready":
        tree_path = st.session_state.tree_json_path
        pdf_name = st.session_state.pdf_name or "document"
        manual_cat = st.session_state.get("manual_category") or "general"

        from src.agent.categories import get_category as _get_cat
        cat_label = _get_cat(manual_cat)["label"]
        st.info(f"📄 **{pdf_name}** · {cat_label}")

        # ── Tabs: Tree | Query ───────────────────────────────────────────
        tab_tree, tab_query = st.tabs(["🌳 Document Tree", "💬 Query"])

        # ── TAB 1: Tree ──────────────────────────────────────────────────
        with tab_tree:
            st.subheader(f"Tree — {pdf_name}")
            _render_tree(tree_path)

        # ── TAB 2: Query ─────────────────────────────────────────────────
        with tab_query:
            st.subheader("Query Pipeline")

            # Config row
            cfg_col1, cfg_col2 = st.columns([3, 1])
            with cfg_col2:
                use_vlm_mcts = st.toggle("VLM MCTS scoring", value=False,
                                          help="Slower but better path selection")

            with cfg_col1:
                query = st.text_input(
                    "Ask a question about the document",
                    placeholder="e.g. What are the safety precautions for welding?",
                    key="query_input",
                )

            run_btn = st.button("▶ Run", type="primary", disabled=not query.strip())

            if run_btn and query.strip():
                # conversation_id = user_id (set at login) — scopes memory to user
                if not st.session_state.conversation_id:
                    st.session_state.conversation_id = st.session_state.get("user_id")

                reasoner = _build_reasoner_cached(tree_path, use_vlm_mcts)

                with st.spinner("Running agent pipeline..."):
                    events = asyncio.run(_stream(
                        reasoner,
                        query.strip(),
                        messages=list(st.session_state.messages_history),
                        summary=st.session_state.messages_summary,
                        conversation_id=st.session_state.conversation_id,
                    ))

                st.session_state.pipeline_events = events
                st.session_state.last_query = query.strip()

                # Extract updated messages+summary from store_memory event
                for node_name, upd in events:
                    if node_name == "store_memory" and upd:
                        if "messages" in upd:
                            st.session_state.messages_history = upd["messages"]
                        if "summary" in upd:
                            st.session_state.messages_summary = upd["summary"]
                        break

            # Render events
            events = st.session_state.pipeline_events
            if events:
                st.divider()
                blocked = any(
                    upd.get("blocked") for name, upd in events
                    if name == "guard_query"
                )

                for node_name, upd in events:
                    label, render_fn = _NODE_META.get(
                        node_name, (f"⚙️ {node_name}", lambda u: st.json(u))
                    )
                    expanded = not (blocked and node_name in
                                    {"retrieve_sections", "rank_sections",
                                     "expand_evidence", "run_mcts"})
                    with st.expander(label, expanded=expanded):
                        render_fn(upd)  # always call — render_fn handles empty dict

                # ── Human-in-the-loop feedback ────────────────────────────
                if not blocked:
                    _render_feedback_form(events, st.session_state.get("last_query", ""))

                # ── Referenced Pages ─────────────────────────────────────
                if not blocked:
                    _render_referenced_pages(events)


if __name__ == "__main__":
    main()
