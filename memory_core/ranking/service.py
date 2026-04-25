"""Hybrid ranking implementation for retrieval results."""

from __future__ import annotations

from datetime import datetime, timezone

from memory_core.domain.enums import MemoryType
from memory_core.domain.models import MemoryRecord, RetrievalQuery, ScoreBreakdown, SessionSnapshot
from memory_core.interfaces.retrieval import RankingService

from .models import RankingWeights


class DefaultRankingService(RankingService):
    """Scores memory records using fixed hybrid ranking weights."""

    def __init__(self, weights: RankingWeights | None = None) -> None:
        self.weights = weights or RankingWeights()

    def score(
        self,
        query: RetrievalQuery,
        memory: MemoryRecord,
        session: SessionSnapshot | None = None,
    ) -> ScoreBreakdown:
        metadata = memory.metadata or {}

        semantic_score = self._clamp(float(metadata.get("semantic_score", 0.0)))
        recency_score = self._compute_recency_score(memory)
        importance_score = self._clamp(float(memory.importance))
        continuity_score = self._compute_continuity_score(query, memory, session)
        graph_score = self._clamp(float(metadata.get("graph_score", 0.0)))
        type_score = self._compute_type_score(query, memory)

        total_weight = (
            self.weights.semantic_weight
            + self.weights.recency_weight
            + self.weights.importance_weight
            + self.weights.continuity_weight
            + self.weights.graph_weight
            + self.weights.type_weight
        )
        final_score = (
            semantic_score * self.weights.semantic_weight
            + recency_score * self.weights.recency_weight
            + importance_score * self.weights.importance_weight
            + continuity_score * self.weights.continuity_weight
            + graph_score * self.weights.graph_weight
            + type_score * self.weights.type_weight
        ) / total_weight

        return ScoreBreakdown(
            semantic_score=semantic_score,
            recency_score=recency_score,
            importance_score=importance_score,
            continuity_score=continuity_score,
            graph_score=graph_score,
            type_score=type_score,
            final_score=self._clamp(final_score),
        )

    def _compute_recency_score(self, memory: MemoryRecord) -> float:
        baseline = memory.updated_at or memory.created_at
        if baseline.tzinfo is None:
            baseline = baseline.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_days = max((now - baseline).total_seconds() / 86400.0, 0.0)
        return self._clamp(1.0 / (1.0 + (age_days / 30.0)))

    def _compute_continuity_score(
        self,
        query: RetrievalQuery,
        memory: MemoryRecord,
        session: SessionSnapshot | None,
    ) -> float:
        score = 0.0
        if query.session_id and memory.session_id == query.session_id:
            score += 0.6
        if query.conversation_id and memory.conversation_id == query.conversation_id:
            score += 0.3
        if session is not None and session.rolling_summary:
            score += 0.1
        return self._clamp(score)

    def _compute_type_score(self, query: RetrievalQuery, memory: MemoryRecord) -> float:
        if query.memory_types and memory.memory_type in query.memory_types:
            return 1.0

        default_scores = {
            MemoryType.DOCUMENT: 0.70,
            MemoryType.DOCUMENT_CHUNK: 0.75,
            MemoryType.FACT: 0.80,
            MemoryType.EPISODE: 0.65,
            MemoryType.PREFERENCE: 0.90,
            MemoryType.TASK: 0.85,
            MemoryType.WORKFLOW_RULE: 0.90,
            MemoryType.SUMMARY: 0.60,
        }
        return default_scores.get(memory.memory_type, 0.5)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
