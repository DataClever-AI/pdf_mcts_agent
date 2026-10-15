"""Models for MCTS Explorer."""

import math

from pydantic import BaseModel, ConfigDict, field_validator


class MCTSNode(BaseModel):
    """A node in the MCTS search tree."""

    section_id: str
    parent_id: str | None = None
    children_ids: list[str] = []
    visits: int = 0
    total_score: float = 0.0
    value: float | None = None  # own relevance score; None until evaluated
    expanded: bool = False      # children have been evaluated
    prior: float = 0.0          # lexical prior in [0, 1] (max over subtree)

    @property
    def avg_score(self) -> float:
        """Average score across all visits."""
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits

    def ucb1(self, parent_visits: int, exploration_c: float = math.sqrt(2)) -> float:
        """UCB1 score — unvisited nodes return inf (always explored first)."""
        if self.visits == 0:
            return float("inf")
        if parent_visits == 0:
            return self.avg_score
        return self.avg_score + exploration_c * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def puct(
        self, parent_visits: int, exploration_c: float, prior_weight: float = 0.0
    ) -> float:
        """
        Prior-weighted selection score.

        Q is the observed mean reward once visited; before that it is the
        node's own evaluated value (optimistic estimate), so an unvisited but
        highly scored child is explored on merit instead of in list order.
        The lexical prior adds `prior_weight * prior` to Q — a noisy VLM score
        for one chapter can't bury a subtree BM25 is confident about — and
        scales the exploration bonus. A floor keeps every subtree reachable
        (the prior must never hide the right section).
        """
        q = self.avg_score if self.visits > 0 else (self.value or 0.0)
        prior_w = 0.25 + 0.75 * self.prior
        return (
            q
            + prior_weight * self.prior
            + exploration_c * prior_w * math.sqrt(max(parent_visits, 1)) / (1 + self.visits)
        )


class SearchState(BaseModel):
    """Completed MCTS search result."""

    query: str
    nodes: dict[str, MCTSNode]
    root_ids: list[str]
    best_path: list[str]
    n_simulations: int
    completed_simulations: int = 0
    score_cache: dict[str, float] = {}  # section_id → relevance score (for refine_path)
    top_ids: list[str] = []             # best-scoring sections, most relevant first
    vlm_calls: int = 0
    vlm_fallbacks: int = 0
    expansions: int = 0


class MCTSConfig(BaseModel):
    """Configuration for MCTSExplorer."""

    model_config = ConfigDict(frozen=True)

    n_simulations: int = 50
    exploration_c: float = 0.5
    prior_weight: float = 0.25  # how much the lexical (BM25) prior adds to a node's Q
    use_vlm: bool = False  # heuristic default; set True for VLM scoring
    max_vlm_concurrency: int = 10
    max_depth: int = 5
    max_expansions: int = 15  # nodes whose children get scored (bounds VLM cost)
    max_vlm_calls: int = 150  # hard cap on VLM scoring calls per search
    top_k: int = 5

    @field_validator(
        "max_vlm_concurrency", "n_simulations", "max_depth",
        "max_expansions", "max_vlm_calls", "top_k",
    )
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        """All integer configs must be >= 1."""
        assert v >= 1
        return v

    @field_validator("exploration_c", "prior_weight")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        """Exploration constant and prior weight must be >= 0.0."""
        assert v >= 0.0
        return v
