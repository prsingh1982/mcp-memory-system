"""Ranking configuration models."""

from pydantic import BaseModel


class RankingWeights(BaseModel):
    """Fixed weights for hybrid retrieval scoring."""

    semantic_weight: float = 0.40
    recency_weight: float = 0.15
    importance_weight: float = 0.10
    continuity_weight: float = 0.15
    graph_weight: float = 0.10
    type_weight: float = 0.10
