"""
MCTS Explorer — vectorless tree search over the document's section tree.

The search navigates the table-of-contents tree top-down, the way a person
would: score the chapters, descend into the promising ones, score their
subsections, and so on until it reaches the sections that hold the answer.

Per simulation:
  1. SELECT   descend from a virtual root with PUCT. An unvisited child is
              valued by its own relevance score (not "infinity"), so the
              search is guided by what was actually measured instead of by
              list order.
  2. EXPAND   at the first node whose children have not been scored yet, score
              ALL of its children (one concurrent VLM batch). Chapters are
              judged by title + subsection titles, leaves by title + content.
  3. REWARD   best score found at the expanded node (its own score or any
              child's) — how promising this subtree is.
  4. BACKPROP update visits / total reward up to the root.

No embeddings anywhere. The only non-VLM signal is an optional lexical prior
(BM25) that biases exploration order; it never prunes a subtree.

Cost is bounded by config.max_expansions / config.max_vlm_calls. When the VLM
fails for a node, that node falls back to a lexical heuristic score.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from src.models.evidence import EvidenceBundle
from src.models.mcts import MCTSConfig, MCTSNode, SearchState
from src.models.section import Section
from src.vlm.client import VLMError, call_vlm

VIRTUAL_ROOT_ID = "__mcts_virtual_root__"

# A parent defers to a child that scores within this margin of it: the child is
# the more specific location for the same answer.
_SPECIFICITY_MARGIN = 0.1
_SPECIFICITY_PENALTY = 0.15
# A chapter whose subsections were never scored is only a lead, not an answer.
_UNRESOLVED_PENALTY = 0.25
# A chapter with (almost) no content of its own — just its heading — is never
# the answer; it exists to be navigated through.
_CONTAINER_MAX_NODES = 3
_CONTAINER_PENALTY = 0.3


@dataclass
class _Ctx:
    """Per-search working state."""

    query: str
    query_tokens: set[str]
    sec_map: dict[str, Section]
    nodes: dict[str, MCTSNode]
    max_node_count: int
    values: dict[str, float]
    override: dict[str, float] | None
    precomputed: dict[str, float]
    sem: asyncio.Semaphore
    vlm_calls: int = 0
    vlm_fallbacks: int = 0
    expansions: int = 0


class MCTSExplorer:
    """Vectorless MCTS over the section tree, scored by the VLM."""

    def __init__(
        self,
        config: MCTSConfig | None = None,
        sections: list[Section] | None = None,
    ) -> None:
        """
        Args:
            config:   Search configuration.
            sections: The FULL document's sections. With them the search runs
                      over the whole tree and can reach sections retrieval
                      never surfaced. Without them it runs over the evidence
                      bundle only.
        """
        self.config = config or MCTSConfig()
        self._tree: dict[str, Section] = (
            {s.section_id: s for s in sections} if sections else {}
        )

    @property
    def sections(self) -> dict[str, Section]:
        """Full-document sections known to this explorer (may be empty)."""
        return self._tree

    async def search(
        self,
        bundle: EvidenceBundle,
        query: str,
        score_cache_override: dict[str, float] | None = None,
        precomputed_scores: dict[str, float] | None = None,
        priors: dict[str, float] | None = None,
    ) -> SearchState:
        """
        Run MCTS and return SearchState.

        Args:
            bundle:               Evidence bundle (defines the tree when the
                                  explorer was built without full sections).
            query:                User query.
            score_cache_override: Pre-built scores (refine_path). No VLM is
                                  called; sections missing from it get 0.5.
            precomputed_scores:   Relevance scores already computed for
                                  sections (rank_sections). Reused for leaf
                                  sections only — a chapter needs the
                                  subsection-aware prompt, not a title score.
            priors:               section_id → lexical prior in [0, 1] (e.g.
                                  normalised BM25). Propagated up to ancestors
                                  as the max over their subtree.

        Returns:
            SearchState with best_path, top_ids, score_cache and stats.
        """
        if not bundle.all_sections:
            return SearchState(
                query=query, nodes={}, root_ids=[], best_path=[],
                n_simulations=self.config.n_simulations,
                completed_simulations=0, score_cache={},
            )

        sec_map: dict[str, Section] = dict(self._tree)
        for s in bundle.all_sections:
            sec_map.setdefault(s.section_id, s)

        nodes, root_ids = self._build_tree(sec_map)
        self._apply_priors(nodes, priors or {})

        ctx = _Ctx(
            query=query,
            query_tokens=_tokens(query),
            sec_map=sec_map,
            nodes=nodes,
            max_node_count=max((s.node_count or 0 for s in sec_map.values()), default=1),
            values=dict(score_cache_override) if score_cache_override is not None else {},
            override=score_cache_override,
            precomputed=precomputed_scores or {},
            sem=asyncio.Semaphore(self.config.max_vlm_concurrency),
        )

        completed = 0
        for _ in range(self.config.n_simulations):
            path = self._select(ctx)
            reward = await self._expand_and_reward(ctx, path)
            if reward is None:
                break
            self._backpropagate(nodes, path, reward)
            completed += 1

        top_ids = self._top_ids(nodes, sec_map)
        return SearchState(
            query=query,
            nodes=nodes,
            root_ids=root_ids,
            best_path=self._path_to(nodes, top_ids[0]) if top_ids else [],
            n_simulations=self.config.n_simulations,
            completed_simulations=completed,
            score_cache=ctx.values,
            top_ids=top_ids,
            vlm_calls=ctx.vlm_calls,
            vlm_fallbacks=ctx.vlm_fallbacks,
            expansions=ctx.expansions,
        )

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def _build_tree(
        self, sec_map: dict[str, Section]
    ) -> tuple[dict[str, MCTSNode], list[str]]:
        """Build the node tree (document order) under a virtual root."""
        nodes: dict[str, MCTSNode] = {
            sid: MCTSNode(
                section_id=sid,
                children_ids=[c for c in s.child_sections if c in sec_map and c != sid],
            )
            for sid, s in sec_map.items()
        }
        for sid, node in nodes.items():
            for cid in node.children_ids:
                if nodes[cid].parent_id is None:  # first parent wins; keeps it a tree
                    nodes[cid].parent_id = sid
        # a child claimed by another parent must not stay in this parent's list
        for node in nodes.values():
            node.children_ids = [c for c in node.children_ids if nodes[c].parent_id == node.section_id]

        natural_roots = [sid for sid, n in nodes.items() if n.parent_id is None]
        nodes[VIRTUAL_ROOT_ID] = MCTSNode(
            section_id=VIRTUAL_ROOT_ID, children_ids=natural_roots
        )
        for rid in natural_roots:
            nodes[rid].parent_id = VIRTUAL_ROOT_ID
        return nodes, [VIRTUAL_ROOT_ID]

    def _apply_priors(self, nodes: dict[str, MCTSNode], priors: dict[str, float]) -> None:
        """Set each node's prior to the max lexical prior over its subtree."""
        for sid, p in priors.items():
            current = sid
            while current in nodes and current != VIRTUAL_ROOT_ID:
                node = nodes[current]
                node.prior = max(node.prior, min(1.0, max(0.0, p)))
                current = node.parent_id or VIRTUAL_ROOT_ID

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, ctx: _Ctx) -> list[str]:
        """Descend by PUCT through expanded nodes; stop at the first one that isn't."""
        nodes = ctx.nodes
        path = [VIRTUAL_ROOT_ID]
        current = VIRTUAL_ROOT_ID
        while True:
            node = nodes[current]
            if not node.expanded or not node.children_ids:
                return path
            nxt = max(
                node.children_ids,
                key=lambda c: nodes[c].puct(
                    node.visits, self.config.exploration_c, self.config.prior_weight
                ),
            )
            if nxt in path:
                return path
            path.append(nxt)
            current = nxt

    async def _expand_and_reward(self, ctx: _Ctx, path: list[str]) -> float | None:
        """Expand the path's last node (score its children) and return the reward."""
        nodes = ctx.nodes
        node = nodes[path[-1]]
        can_expand = (
            not node.expanded
            and bool(node.children_ids)
            and ctx.expansions < self.config.max_expansions
        )
        if can_expand:
            await self._evaluate(ctx, node.children_ids)
            node.expanded = True
            ctx.expansions += 1
            child_values = [nodes[c].value or 0.0 for c in node.children_ids]
            own = node.value if node.value is not None else 0.0
            return max([own, *child_values])
        if node.section_id == VIRTUAL_ROOT_ID:
            return None  # nothing to search
        return node.value if node.value is not None else 0.0

    def _backpropagate(
        self, nodes: dict[str, MCTSNode], path: list[str], reward: float
    ) -> None:
        """Update visits + total_score for every node in path."""
        for sid in path:
            node = nodes[sid]
            node.visits += 1
            if sid != VIRTUAL_ROOT_ID:
                node.total_score += reward

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def _evaluate(self, ctx: _Ctx, ids: list[str]) -> None:
        """Score every not-yet-scored node in `ids` concurrently."""
        todo = [i for i in ids if i not in ctx.values]
        if todo:
            scores = await asyncio.gather(*[self._score_node(ctx, i) for i in todo])
            for sid, score in zip(todo, scores, strict=True):
                ctx.values[sid] = score
        for sid in ids:
            ctx.nodes[sid].value = ctx.values.get(sid, 0.5)

    async def _score_node(self, ctx: _Ctx, section_id: str) -> float:
        """VLM relevance for one node; lexical heuristic if the VLM is unavailable."""
        if ctx.override is not None:
            return 0.5  # override mode never calls the VLM; unlisted nodes are neutral
        sec = ctx.sec_map[section_id]
        node = ctx.nodes[section_id]
        children = [ctx.sec_map[c] for c in node.children_ids]

        if (sec.node_count or 0) == 0 and not children:
            return 0.05  # empty leaf — nothing to find here
        if not children and section_id in ctx.precomputed:
            return ctx.precomputed[section_id]

        if self.config.use_vlm and ctx.vlm_calls < self.config.max_vlm_calls:
            ctx.vlm_calls += 1
            try:
                parent = ctx.sec_map.get(node.parent_id or "")
                async with ctx.sem:
                    response = await call_vlm(
                        prompt=_build_rollout_prompt(
                            query=ctx.query,
                            section=sec,
                            children=children,
                            parent_title=parent.title if parent else None,
                            query_tokens=ctx.query_tokens,
                        ),
                        task="rollout_scoring",
                        max_tokens=16,
                        temperature=0.0,
                    )
                score = _parse_score(response.content)
                if children and section_id in ctx.precomputed:
                    # Two independent VLM views of the same section (navigation
                    # prompt + rank_sections' title prompt): averaging cuts the
                    # run-to-run noise of a single 7B score.
                    score = 0.5 * score + 0.5 * ctx.precomputed[section_id]
                return score
            except VLMError:
                ctx.vlm_fallbacks += 1

        return _heuristic_score(
            sec, ctx.query_tokens, ctx.max_node_count,
            extra_text=" ".join(c.title for c in children),
        )

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _top_ids(
        self, nodes: dict[str, MCTSNode], sec_map: dict[str, Section]
    ) -> list[str]:
        """
        Best-scoring sections, most relevant first.

        Ranked by own relevance score, with chapters yielding to the sections
        under them: a chapter's score means "the answer is somewhere below",
        so a chapter whose subsections were never scored is unresolved
        (penalised), and one whose child scores about as high defers to that
        child (the child is where the answer actually lives). Without this,
        a generic chapter the VLM likes ("Specifications") outranks the real
        leaf. Ties break on search consensus (visits), then prior.
        """
        ranked: list[tuple[float, int, float, str]] = []
        for sid, node in nodes.items():
            if sid == VIRTUAL_ROOT_ID or node.value is None:
                continue
            score = node.value
            if node.children_ids:
                kids = [nodes[c].value for c in node.children_ids if nodes[c].value is not None]
                penalty = 0.0
                if not kids:
                    penalty = _UNRESOLVED_PENALTY
                elif max(kids) >= score - _SPECIFICITY_MARGIN:  # type: ignore[type-var]
                    penalty = _SPECIFICITY_PENALTY
                if (sec_map[sid].node_count or 0) <= _CONTAINER_MAX_NODES:
                    penalty = max(penalty, _CONTAINER_PENALTY)  # title-only chapter
                score -= penalty
            ranked.append((round(score, 3), node.visits, node.prior, sid))
        ranked.sort(reverse=True)
        return [sid for *_, sid in ranked[: self.config.top_k]]

    def _path_to(self, nodes: dict[str, MCTSNode], section_id: str) -> list[str]:
        """Chain from the top-level chapter down to `section_id` (max_depth long)."""
        chain: list[str] = []
        current: str | None = section_id
        while current and current != VIRTUAL_ROOT_ID and current in nodes:
            chain.append(current)
            current = nodes[current].parent_id
        chain.reverse()
        return chain[-self.config.max_depth:]


# ---------------------------------------------------------------------------
# Prompt + score helpers
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def _preview(section: Section, query_tokens: set[str], max_chars: int) -> str:
    """Content node that best matches the query (first non-trivial node as fallback)."""
    best, best_overlap, first = "", -1, ""
    for n in section.semantic_nodes:
        text = n.canonical_text.strip()
        if len(text) < 25:
            continue
        first = first or text
        overlap = len(query_tokens & _tokens(text))
        if overlap > best_overlap:
            best, best_overlap = text, overlap
    return (best or first)[:max_chars]


def _build_rollout_prompt(
    query: str,
    section: Section,
    children: list[Section],
    parent_title: str | None,
    query_tokens: set[str],
) -> str:
    """
    Navigation prompt: how likely is the answer to be in this section or below it?

    A chapter is judged by its subsection titles (like reading a table of
    contents); a leaf by its content. Output format first, anchored scale.
    """
    parts = [
        "OUTPUT: reply with ONLY a decimal number from 0.0 to 1.0.",
        "",
        "You are searching a manual's table of contents for the place that answers the query.",
        "Score how likely the answer is in this section or in one of its subsections.",
        "",
        "Scoring anchors:",
        "  1.0 = this is exactly where the answer is",
        "  0.7 = the answer is probably in this section or one of its subsections",
        "  0.4 = related topic; the answer may be nearby but not here",
        "  0.1 = a different topic",
        "  0.0 = unrelated",
        "",
        f"Query: {query}",
        f"Section: {section.title}",
        f"Path: {section.hierarchy_path}",
    ]
    if parent_title:
        parts.append(f"Parent section: {parent_title}")
    if children:
        shown = "; ".join(c.title[:60] for c in children[:12])
        more = f" (+{len(children) - 12} more)" if len(children) > 12 else ""
        parts.append(f"Subsections ({len(children)}): {shown}{more}")
    preview = _preview(section, query_tokens, 300 if not children else 160)
    if preview:
        parts.append(f"Content: {preview}")
    parts.append("\nScore:")
    return "\n".join(parts)


def _parse_score(content: str) -> float:
    """Parse float from VLM response, clamped to [0.0, 1.0]. Default 0.5."""
    try:
        return max(0.0, min(1.0, float(content.strip())))
    except ValueError:
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", content)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
    return 0.5


def _heuristic_score(
    section: Section,
    query_tokens: set[str],
    max_node_count: int,
    extra_text: str = "",
) -> float:
    """Heuristic: overlap(0.65) + confidence(0.20) + density_capped(0.15).

    `extra_text` (subsection titles) lets a chapter with no content of its own
    still be scored by what it contains. Density is capped at 40% of max so
    large sections can't dominate regardless of query relevance.
    """
    node_count = section.node_count or 0
    if node_count == 0 and not extra_text:
        return 0.05  # empty leaf — no direct content

    content = " ".join(
        [section.title, extra_text, *(n.canonical_text for n in section.semantic_nodes)]
    )
    content_tokens = _tokens(content)
    overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
    density = min(node_count / max(max_node_count, 1), 0.4)
    return min(1.0, overlap * 0.65 + section.confidence * 0.20 + density * 0.15)
