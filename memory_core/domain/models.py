"""Validated domain models used across the memory system."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .enums import AuditEventType, JobStatus, MemoryStatus, MemoryType, SourceType


class SourceReference(BaseModel):
    source_id: str
    source_type: SourceType
    title: str | None = None
    file_path: str | None = None
    original_filename: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    external_uri: str | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkReference(BaseModel):
    chunk_id: str
    document_id: str
    sequence_index: int
    text: str
    token_count: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_title: str | None = None
    page_number: int | None = None


class CitationReference(BaseModel):
    citation_id: str
    source_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    quote: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    provenance_chain: list[str] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    memory_id: str
    memory_type: MemoryType
    status: MemoryStatus
    content: str
    summary: str | None = None
    source_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    confidence: float = 0.0
    importance: float = 0.0
    tags: list[str] = Field(default_factory=list)
    version: int = 1
    parent_memory_id: str | None = None
    supersedes_memory_id: str | None = None
    embedding_ref: str | None = None
    graph_node_ref: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateMemory(BaseModel):
    candidate_id: str
    proposed_memory: MemoryRecord
    extraction_reason: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float
    suggested_action: Literal["create", "merge", "update"]
    existing_memory_id: str | None = None
    created_at: datetime


class RetrievalQuery(BaseModel):
    query: str
    session_id: str | None = None
    conversation_id: str | None = None
    memory_types: list[MemoryType] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    top_k: int = 10
    include_deleted: bool = False
    require_citations: bool = True


class ScoreBreakdown(BaseModel):
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    continuity_score: float = 0.0
    graph_score: float = 0.0
    type_score: float = 0.0
    final_score: float = 0.0


class RetrievedMemory(BaseModel):
    memory: MemoryRecord
    score: ScoreBreakdown
    citations: list[CitationReference] = Field(default_factory=list)
    matched_chunk_ids: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class RetrievalResult(BaseModel):
    query: str
    items: list[RetrievedMemory]
    applied_preferences: list[MemoryRecord] = Field(default_factory=list)
    session_summary: str | None = None
    audit_event_id: str | None = None


class SessionMessage(BaseModel):
    message_id: str
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionSnapshot(BaseModel):
    session_id: str
    conversation_id: str
    recent_messages: list[SessionMessage] = Field(default_factory=list)
    rolling_summary: str | None = None
    last_active_at: datetime


class IngestionJob(BaseModel):
    job_id: str
    source_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditEvent(BaseModel):
    event_id: str
    event_type: AuditEventType
    actor_id: str | None = None
    memory_id: str | None = None
    source_id: str | None = None
    session_id: str | None = None
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)
