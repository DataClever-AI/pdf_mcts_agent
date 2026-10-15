"""
Tests for MCTSExplorer — heuristic scoring path (use_vlm=False), fully
deterministic and VLM-free. The VLM rollout path is exercised via the e2e
mock tests elsewhere (call_vlm is an external dependency, not unit-testable
in isolation without mocking the whole client).
"""
from __future__ import annotations

import asyncio

from src.models.evidence import EvidenceBundle
from src.models.mcts import MCTSConfig
from src.models.section import Section, SemanticNode
from src.rag.mcts_explorer import MCTSExplorer


def _section(
    section_id: str, title: str, text: str, *, children: list[str] | None = None,
) -> Section:
    node_count = 1 if text else 0
    return Section(
        section_id=section_id, title=title, hierarchy_level=1, hierarchy_path=f"/{title}",
        semantic_nodes=(
            [SemanticNode(node_id=f"{section_id}_n1", canonical_text=text,
                          semantic_order=1, node_type="paragraph")]
            if text else []
        ),
        semantic_order_start=1, semantic_order_end=1, node_count=node_count,
        child_sections=children or [], page_start=1, page_end=1,
        confidence=0.9, structural_source="toc", inference_reasons=[], evidence_source_ids=[],
    )


def _bundle(sections: list[Section], query: str) -> EvidenceBundle:
    return EvidenceBundle(
        primary_sections=sections, parent_sections=[], sibling_sections=[],
        child_sections=[], related_nodes=[],
        total_node_count=sum(s.node_count or 0 for s in sections),
        expansion_trace=[], query=query,
    )


class TestMCTSExplorerHeuristic:
    def test_relevant_section_scores_higher_than_unrelated(self) -> None:
        relevant = _section("sec_battery", "Battery Installation",
                             "Remove the old battery and insert the new battery.")
        unrelated = _section("sec_clean", "Cleaning Instructions",
                              "Wipe the exterior with a soft cloth.")
        bundle = _bundle([relevant, unrelated], "how do I replace the battery")
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=30))

        search_state = asyncio.run(explorer.search(bundle, "how do I replace the battery"))

        assert search_state.score_cache["sec_battery"] > search_state.score_cache["sec_clean"]
        assert search_state.best_path  # a path was found
        assert search_state.completed_simulations > 0

    def test_structural_container_scores_near_zero_regardless_of_query(self) -> None:
        empty_container = _section("sec_chapter", "Chapter 3", "")  # node_count=0
        bundle = _bundle([empty_container], "chapter 3")
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=10))

        search_state = asyncio.run(explorer.search(bundle, "chapter 3"))

        assert search_state.score_cache["sec_chapter"] == 0.05

    def test_precomputed_scores_are_reused_not_recomputed(self) -> None:
        sec = _section("sec_battery", "Battery Installation", "Insert the new battery.")
        bundle = _bundle([sec], "battery")
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=5))

        # A precomputed score wildly different from what the heuristic would
        # compute — if it's actually reused verbatim, the cache must show
        # exactly this value, not a heuristic recomputation.
        search_state = asyncio.run(
            explorer.search(bundle, "battery", precomputed_scores={"sec_battery": 0.42})
        )

        assert search_state.score_cache["sec_battery"] == 0.42

    def test_score_cache_override_bypasses_all_scoring(self) -> None:
        sec_a = _section("sec_a", "A", "text a")
        sec_b = _section("sec_b", "B", "text b")
        bundle = _bundle([sec_a, sec_b], "query")
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=10))

        override = {"sec_a": 0.9, "sec_b": 0.1}
        search_state = asyncio.run(
            explorer.search(bundle, "query", score_cache_override=override)
        )

        assert search_state.score_cache == override
        # Best path should favor the higher-scored section.
        assert search_state.best_path[0] == "sec_a"

    def test_empty_bundle_returns_empty_search_state(self) -> None:
        bundle = _bundle([], "query")
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False))

        search_state = asyncio.run(explorer.search(bundle, "query"))

        assert search_state.best_path == []
        assert search_state.completed_simulations == 0

    def test_parent_child_tree_built_from_child_sections(self) -> None:
        child = _section("sec_child", "Subsection", "Detailed procedure content here.")
        parent = _section("sec_parent", "Section", "", children=["sec_child"])
        bundle = _bundle([parent, child], "detailed procedure")
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=30))

        search_state = asyncio.run(explorer.search(bundle, "detailed procedure"))

        # The child (real content, matches query) should be findable and
        # score higher than its empty structural parent.
        assert search_state.score_cache["sec_child"] > search_state.score_cache["sec_parent"]


class TestMCTSTreeSearch:
    """Vectorless tree navigation (heuristic scoring — deterministic, no VLM)."""

    def _tree(self) -> tuple[Section, Section, Section]:
        chapter = _section("sec_ch", "Battery Chapter", "", children=["sec_a", "sec_b"])
        battery = _section("sec_a", "Battery Installation",
                           "Remove the old battery and insert the new battery.")
        cleaning = _section("sec_b", "Cleaning", "Wipe the exterior with a soft cloth.")
        return chapter, battery, cleaning

    def test_descends_chapter_to_relevant_child(self) -> None:
        chapter, battery, cleaning = self._tree()
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=20))

        state = asyncio.run(explorer.search(
            _bundle([chapter, battery, cleaning], "replace the battery"), "replace the battery",
        ))

        assert state.top_ids[0] == "sec_a"
        assert state.best_path == ["sec_ch", "sec_a"]

    def test_finds_section_outside_the_bundle_via_full_tree(self) -> None:
        chapter, battery, cleaning = self._tree()
        # The evidence bundle only holds the irrelevant section; the explorer
        # knows the whole document and must still reach the battery section.
        explorer = MCTSExplorer(
            MCTSConfig(use_vlm=False, n_simulations=20), sections=[chapter, battery, cleaning],
        )

        state = asyncio.run(explorer.search(_bundle([cleaning], "replace the battery"),
                                            "replace the battery"))

        assert state.top_ids[0] == "sec_a"

    def test_title_only_chapter_never_outranks_its_content(self) -> None:
        chapter = _section("sec_ch", "Battery Replacement Guide", "", children=["sec_n"])
        notes = _section("sec_n", "Notes", "misc notes battery")
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=20))

        state = asyncio.run(explorer.search(
            _bundle([chapter, notes], "battery replacement"), "battery replacement",
        ))

        assert state.score_cache["sec_ch"] > state.score_cache["sec_n"]  # chapter scores higher...
        assert state.top_ids[0] == "sec_n"                                # ...but isn't the answer

    def test_expansions_are_capped(self) -> None:
        kids = [_section(f"sec_{i}", f"Topic {i}", f"text {i}") for i in range(6)]
        parents = [
            _section(f"par_{i}", f"Parent {i}", "", children=[kids[i].section_id])
            for i in range(6)
        ]
        explorer = MCTSExplorer(
            MCTSConfig(use_vlm=False, n_simulations=50, max_expansions=3),
        )

        state = asyncio.run(explorer.search(_bundle([*parents, *kids], "topic"), "topic"))

        assert state.expansions <= 3

    def test_priors_propagate_to_ancestors(self) -> None:
        chapter, battery, cleaning = self._tree()
        explorer = MCTSExplorer(MCTSConfig(use_vlm=False, n_simulations=5))

        state = asyncio.run(explorer.search(
            _bundle([chapter, battery, cleaning], "battery"), "battery",
            priors={"sec_a": 0.8},
        ))

        assert state.nodes["sec_a"].prior == 0.8
        assert state.nodes["sec_ch"].prior == 0.8   # ancestor inherits max over subtree
        assert state.nodes["sec_b"].prior == 0.0

    def test_unvisited_child_is_valued_by_its_own_score_not_infinity(self) -> None:
        from src.models.mcts import MCTSNode

        strong = MCTSNode(section_id="a", value=0.9)
        weak = MCTSNode(section_id="b", value=0.1)

        assert strong.puct(10, 0.5) > weak.puct(10, 0.5)
        assert strong.puct(10, 0.5) != float("inf")


class TestMergeFoundSections:
    def _state(self, top_ids: list[str], scores: dict[str, float]):  # type: ignore[no-untyped-def]
        from src.models.mcts import SearchState
        return SearchState(
            query="q", nodes={}, root_ids=[], best_path=[], n_simulations=1,
            score_cache=scores, top_ids=top_ids,
        )

    def test_adds_strong_missing_sections_and_skips_weak_or_huge(self) -> None:
        from src.agent.nodes.mcts import _merge_found_sections

        have = _section("sec_have", "Have", "x")
        good = _section("sec_good", "Good", "y")
        weak = _section("sec_weak", "Weak", "z")
        huge = Section(**{**_section("sec_huge", "Huge", "w").model_dump(), "node_count": 138})
        explorer = MCTSExplorer(sections=[have, good, weak, huge])
        state = self._state(
            ["sec_good", "sec_weak", "sec_huge"],
            {"sec_good": 0.9, "sec_weak": 0.4, "sec_huge": 0.9},
        )

        merged, added = _merge_found_sections(_bundle([have], "q"), state, explorer)

        assert added == ["sec_good"]
        assert merged is not None
        assert [s.section_id for s in merged.primary_sections] == ["sec_good", "sec_have"]

    def test_returns_none_when_nothing_to_add(self) -> None:
        from src.agent.nodes.mcts import _merge_found_sections

        have = _section("sec_have", "Have", "x")
        explorer = MCTSExplorer(sections=[have])
        state = self._state(["sec_have"], {"sec_have": 0.9})

        merged, added = _merge_found_sections(_bundle([have], "q"), state, explorer)

        assert merged is None and added == []
