"""Retrieval and ranking contracts."""

from __future__ import annotations

from typing import Protocol

from memory_core.domain.models import MemoryRecord, RetrievalQuery, RetrievalResult, RetrievedMemory, ScoreBreakdown, SessionSnapshot


class RankingService(Protocol):
    """Scores memories for hybrid retrieval."""

    def score(
        self,
        query: RetrievalQuery,
        memory: MemoryRecord,
        session: SessionSnapshot | None = None,
    ) -> ScoreBreakdown:
        """Return a weighted score breakdown for a memory item."""


class RetrievalService(Protocol):
    """Coordinates semantic shortlist, graph expansion, and reranking."""

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Return fully ranked memories for a retrieval query."""

    def semantic_shortlist(self, query: RetrievalQuery) -> list[RetrievedMemory]:
        """Return the initial semantic retrieval shortlist."""

    def expand_with_graph_context(self, items: list[RetrievedMemory]) -> list[RetrievedMemory]:
        """Augment results with graph-neighbor context."""

    def rerank(self, query: RetrievalQuery, items: list[RetrievedMemory]) -> list[RetrievedMemory]:
        """Return the final reranked retrieval result set."""
