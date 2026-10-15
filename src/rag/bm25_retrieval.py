"""
BM25 retrieval — high-recall section retrieval.

Indexes 3 fields: title (3.0), hierarchy_path (1.5), canonical_text (1.0).
No VLM. No confidence filtering. top_k default = 20.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from src.models.derived import ScoredSection
from src.models.section import Section


class BM25Config(BaseModel):
    """BM25 index configuration."""

    top_k: int = 30
    title_weight: float = 3.0
    path_weight: float = 1.5
    content_weight: float = 1.0


def _stem(token: str) -> list[str]:
    """
    Light suffix stemming — produces extra stem variants for a token.
    Avoids heavy NLP deps; just removes common English inflectional endings.
    Returns the stems to ADD (the original token is always kept separately).
    """
    stems: list[str] = []
    t = token
    if len(t) < 4:
        return stems
    # Plurals / third-person: failures → failure, displays → display
    if t.endswith("ies") and len(t) > 5:
        stems.append(t[:-3] + "y")
    elif t.endswith("es") and len(t) > 5 and not t.endswith("ses"):
        stems.append(t[:-2])
    elif t.endswith("s") and len(t) > 4 and not t.endswith("ss"):
        stems.append(t[:-1])
    # Past tense: zeroed → zero, calibrated → calibrat
    if t.endswith("ed") and len(t) > 5:
        stems.append(t[:-2])
        if t.endswith("ied"):
            stems.append(t[:-3] + "y")
    # Gerund: zeroing → zero, monitoring → monitor
    if t.endswith("ing") and len(t) > 6:
        stems.append(t[:-3])
        stems.append(t[:-3] + "e")
    # -er/-ers: transducer → transduc  (light)
    if t.endswith("ers") and len(t) > 6:
        stems.append(t[:-3])
    elif t.endswith("er") and len(t) > 5:
        stems.append(t[:-2])
    return [s for s in stems if len(s) >= 3]


def tokenize(text: str, *, bigrams: bool = True, stemming: bool = True) -> list[str]:
    """
    Robust tokenizer for BM25 indexing and querying.

    Improvements over basic tokenizer:
    1. Light stemming — produces stem variants so "zeroed/zeroing" match "zero",
       "failures" matches "failure", etc.
    2. Bigram indexing — consecutive word pairs ("system fault", "alarm limit")
       are indexed as single tokens, enabling phrase-level matching.

    These two improvements address the compound-term tokenization misses
    identified in retrieval diagnosis (re-zeroed, STAT CO, calibration failures,
    system fault → section not found).
    """
    text = text.lower()
    text = re.sub(r"[/\-_]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()

    all_tokens: list[str] = list(tokens)

    # Add stems
    if stemming:
        for t in tokens:
            all_tokens.extend(_stem(t))

    # Add bigrams: "system fault" → token "system_fault"
    if bigrams and len(tokens) > 1:
        for i in range(len(tokens) - 1):
            all_tokens.append(f"{tokens[i]}_{tokens[i + 1]}")

    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for tok in all_tokens:
        if tok not in seen and tok:
            seen.add(tok)
            result.append(tok)
    return result


def compute_retrieval_score(
    title_score: float,
    path_score: float,
    content_score: float,
    *,
    title_weight: float = 3.0,
    path_weight: float = 1.5,
    content_weight: float = 1.0,
) -> float:
    """Composite BM25 score from three field scores."""
    return (
        title_score * title_weight
        + path_score * path_weight
        + content_score * content_weight
    )


class BM25Index:
    """BM25 retrieval index over a section corpus."""

    def __init__(
        self,
        sections: list[Section],
        config: BM25Config | None = None,
    ) -> None:
        """
        Build the BM25 index from all sections.

        Args:
            sections: All sections to index.
            config:   Optional config override.
        """
        self._sections = sections
        self.config = config or BM25Config()

        if not sections:
            self._title_index: Any = None
            self._path_index: Any = None
            self._content_index: Any = None
            return

        # Title + path: bigrams=True — exact phrase matching ("battery installation",
        # "system fault", "alarm priority") in short structured text.
        # Content: bigrams=False — long text creates too many bigrams, pollutes IDF
        # and hurts precision. Stemming alone is enough for content matching.
        title_corpus = [
            tokenize(s.title, bigrams=True) or ["_empty_"] for s in sections
        ]
        path_corpus = [
            tokenize(s.hierarchy_path, bigrams=True) or ["_empty_"] for s in sections
        ]
        content_corpus = [
            tokenize(
                " ".join(n.canonical_text for n in s.semantic_nodes),
                bigrams=False,   # no bigrams in long content — too much IDF noise
            ) or ["_empty_"]
            for s in sections
        ]
        self._title_index = BM25Okapi(title_corpus)
        self._path_index = BM25Okapi(path_corpus)
        self._content_index = BM25Okapi(content_corpus)

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredSection]:
        """
        Retrieve top-K sections matching query using composite BM25 scores.

        Args:
            query: The search query string.
            top_k: Max results (defaults to config.top_k = 20).

        Returns:
            List of ScoredSection sorted by retrieval_score descending.
        """
        k = top_k if top_k is not None else self.config.top_k
        if k < 1:
            raise ValueError(f"top_k must be >= 1, got {k}")
        if not self._sections:
            return []

        # Query: bigrams=False — query bigrams like "could_you", "how_to" are
        # narrative artifacts that never match section titles/paths. Only the
        # INDEX has bigrams (for title phrase matching). Stemming still applied.
        tokens = tokenize(query, bigrams=False)
        title_scores: Any = self._title_index.get_scores(tokens)
        path_scores: Any = self._path_index.get_scores(tokens)
        content_scores: Any = self._content_index.get_scores(tokens)

        scored: list[ScoredSection] = [
            ScoredSection(
                section=self._sections[i],
                retrieval_score=compute_retrieval_score(
                    float(title_scores[i]),
                    float(path_scores[i]),
                    float(content_scores[i]),
                    title_weight=self.config.title_weight,
                    path_weight=self.config.path_weight,
                    content_weight=self.config.content_weight,
                ),
                title_score=float(title_scores[i]),
                path_score=float(path_scores[i]),
                content_score=float(content_scores[i]),
            )
            for i in range(len(self._sections))
        ]

        scored.sort(key=lambda x: x.retrieval_score, reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredSection]],
    *,
    k: int = 60,
) -> list[ScoredSection]:
    """
    Fuse multiple ranked ScoredSection lists into one via Reciprocal Rank
    Fusion: score(doc) = sum over lists containing it of 1/(k + rank).

    Used to combine one BM25 pass per query-expansion sub-question, instead
    of joining all sub-questions into a single bag-of-words string before
    retrieval — joining dilutes each sub-question's own term weighting into
    one BM25 query. RRF only needs rank position, not comparable score
    magnitudes, which sidesteps BM25 scores not being on a shared scale
    across different queries. k=60 is the standard RRF constant from the
    original paper (Cormack et al.) — large enough that rank differences
    beyond the first few positions barely move the fused score.
    """
    rrf_scores: dict[str, float] = {}
    best_by_id: dict[str, ScoredSection] = {}
    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            sid = scored.section.section_id
            rrf_scores[sid] = rrf_scores.get(sid, 0.0) + 1.0 / (k + rank)
            if sid not in best_by_id or scored.retrieval_score > best_by_id[sid].retrieval_score:
                best_by_id[sid] = scored

    fused = [
        ScoredSection(
            section=best_by_id[sid].section,
            retrieval_score=rrf_scores[sid],
            title_score=best_by_id[sid].title_score,
            path_score=best_by_id[sid].path_score,
            content_score=best_by_id[sid].content_score,
        )
        for sid in rrf_scores
    ]
    fused.sort(key=lambda s: s.retrieval_score, reverse=True)
    return fused
