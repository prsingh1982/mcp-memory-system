"""Repository contracts for durable metadata, job state, and audit storage."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from memory_core.domain.enums import AuditEventType, MemoryType
from memory_core.domain.models import (
    AuditEvent,
    CandidateMemory,
    ChunkReference,
    IngestionJob,
    MemoryRecord,
    SessionMessage,
    SessionSnapshot,
    SourceReference,
)


class MemoryRepository(Protocol):
    """Persistence contract for canonical memory records."""

    def create_memory(self, memory: MemoryRecord) -> MemoryRecord:
        """Persist a new memory record."""

    def update_memory(self, memory: MemoryRecord) -> MemoryRecord:
        """Persist updates to an existing memory record."""

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        """Return a memory record by id, if present."""

    def list_memories(self, memory_types: list[MemoryType] | None = None) -> list[MemoryRecord]:
        """Return stored memories, optionally filtered by type."""

    def soft_delete_memory(self, memory_id: str, deleted_at: datetime) -> None:
        """Mark a memory record as deleted without removing history."""

    def get_memory_versions(self, memory_id: str) -> list[MemoryRecord]:
        """Return all known versions for a memory lineage."""


class SourceRepository(Protocol):
    """Persistence contract for ingested source metadata."""

    def create_source(self, source: SourceReference) -> SourceReference:
        """Persist source metadata."""

    def get_source(self, source_id: str) -> SourceReference | None:
        """Return source metadata by id, if present."""

    def list_sources(self) -> list[SourceReference]:
        """Return all known sources."""


class ChunkRepository(Protocol):
    """Persistence contract for parsed document chunks."""

    def upsert_chunks(self, chunks: list[ChunkReference]) -> list[ChunkReference]:
        """Persist document chunks and return the stored items."""

    def get_chunk(self, chunk_id: str) -> ChunkReference | None:
        """Return a chunk by id, if present."""

    def list_chunks(self, document_id: str) -> list[ChunkReference]:
        """Return all stored chunks for a document."""


class CandidateMemoryRepository(Protocol):
    """Persistence contract for reviewable extracted memory candidates."""

    def save_candidates(self, candidates: list[CandidateMemory]) -> list[CandidateMemory]:
        """Persist extracted candidate memories and return the stored items."""

    def get_candidate(self, candidate_id: str) -> CandidateMemory | None:
        """Return a candidate memory by id, if present."""

    def list_candidates(self, source_id: str | None = None) -> list[CandidateMemory]:
        """Return candidate memories, optionally filtered by source."""

    def delete_candidate(self, candidate_id: str) -> None:
        """Delete a candidate memory after review resolution."""


class JobRepository(Protocol):
    """Persistence contract for ingestion job state."""

    def create_job(self, job: IngestionJob) -> IngestionJob:
        """Persist a new ingestion job."""

    def update_job(self, job: IngestionJob) -> IngestionJob:
        """Persist updates to ingestion job state."""

    def get_job(self, job_id: str) -> IngestionJob | None:
        """Return job state by id, if present."""


class AuditRepository(Protocol):
    """Persistence contract for full lifecycle audit events."""

    def record_event(self, event: AuditEvent) -> AuditEvent:
        """Persist an audit event."""

    def list_events(
        self,
        event_type: AuditEventType | None = None,
        memory_id: str | None = None,
        source_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return audit events filtered by event or object identity."""


class SessionRepository(Protocol):
    """Persistence contract for session continuity and summaries."""

    def append_message(self, session_id: str, message: SessionMessage) -> None:
        """Append a message to the session timeline."""

    def get_session(self, session_id: str) -> SessionSnapshot | None:
        """Return the session snapshot, if present."""

    def save_summary(self, session_id: str, summary: str) -> None:
        """Persist or update the rolling session summary."""
