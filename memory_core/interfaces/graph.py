"""Vector and graph storage contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from memory_core.domain.models import MemoryRecord


class VectorIndex(Protocol):
    """Abstraction over semantic vector indexing and search."""

    def upsert(self, vector_id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Insert or replace an indexed embedding vector."""

    def delete(self, vector_id: str) -> None:
        """Remove an indexed embedding vector."""

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return vector search hits and associated metadata."""


class GraphStore(Protocol):
    """Abstraction over graph persistence for memory relationships and provenance."""

    def upsert_memory_node(self, memory: MemoryRecord) -> None:
        """Insert or update a graph node representing the memory."""

    def create_relationship(
        self,
        from_node_id: str,
        relation_type: str,
        to_node_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a relationship between two graph nodes."""

    def get_related_nodes(self, node_id: str, depth: int = 1) -> list[dict[str, Any]]:
        """Return related nodes for graph expansion during retrieval."""

    def mark_deleted(self, node_id: str, deleted_at: datetime) -> None:
        """Mark a graph node as soft-deleted."""
