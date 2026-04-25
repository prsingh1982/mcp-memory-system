"""Application service contracts used by UI and MCP adapters."""

from __future__ import annotations

from typing import Any, Protocol

from memory_core.domain.enums import ReviewDecision
from memory_core.domain.models import (
    CandidateMemory,
    ChunkReference,
    IngestionJob,
    MemoryRecord,
    SessionMessage,
    SessionSnapshot,
    SourceReference,
)


class IngestionService(Protocol):
    """Coordinates ingestion jobs from source intake through extraction."""

    def ingest_source(self, source: SourceReference) -> IngestionJob:
        """Create and register a new ingestion job for a source."""

    def process_job(self, job_id: str) -> IngestionJob:
        """Run the ingestion workflow for an existing job."""

    def parse_source(self, source_id: str) -> str:
        """Parse a stored source into normalized text."""

    def chunk_source(self, source_id: str, text: str) -> list[ChunkReference]:
        """Chunk normalized source text for indexing and extraction."""

    def extract_candidate_memories(self, source_id: str, chunks: list[ChunkReference]) -> list[CandidateMemory]:
        """Extract reviewable candidate memories from source chunks."""


class ReviewService(Protocol):
    """Handles review decisions for automatically extracted memories."""

    def list_candidates(self, source_id: str | None = None) -> list[CandidateMemory]:
        """Return candidate memories awaiting review."""

    def apply_decision(
        self,
        candidate_id: str,
        decision: ReviewDecision,
        target_memory_id: str | None = None,
    ) -> MemoryRecord | None:
        """Apply a review decision and return the resulting memory, if any."""


class MemoryService(Protocol):
    """Manages canonical memory lifecycle operations."""

    def store_memory(self, memory: MemoryRecord) -> MemoryRecord:
        """Persist a new memory."""

    def update_memory(self, memory_id: str, updates: dict[str, Any]) -> MemoryRecord:
        """Update an existing memory with version-aware semantics."""

    def merge_memory(self, source_memory_id: str, target_memory_id: str) -> MemoryRecord:
        """Merge one memory into another and return the updated target."""

    def delete_memory(self, memory_id: str, reason: str) -> None:
        """Soft-delete a memory and record the deletion reason."""

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        """Return a memory by id, if present."""

    def get_memory_history(self, memory_id: str) -> list[MemoryRecord]:
        """Return the version history for a memory lineage."""


class SessionService(Protocol):
    """Maintains recent conversational state and summary memory."""

    def append_message(self, session_id: str, message: SessionMessage) -> None:
        """Append a message to session continuity state."""

    def get_context_snapshot(self, session_id: str) -> SessionSnapshot | None:
        """Return the current session snapshot, if present."""

    def summarize_session(self, session_id: str) -> MemoryRecord | None:
        """Promote session context into a durable summary memory when appropriate."""


class ChatService(Protocol):
    """Coordinates retrieval, preference application, and response generation."""

    def answer(self, query: str, session_id: str, conversation_id: str) -> dict[str, Any]:
        """Return an answer payload with citations and retrieval metadata."""
