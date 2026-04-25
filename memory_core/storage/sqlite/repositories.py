"""SQLite repository implementations for canonical metadata storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from memory_core.domain.enums import AuditEventType, JobStatus, MemoryStatus, MemoryType, SourceType
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

from .database import SQLiteDatabase

ModelT = TypeVar("ModelT", bound=BaseModel)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _deserialize_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _model_to_json_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[call-arg]
    return json.loads(model.json())


def _model_validate(model_class: type[ModelT], payload: dict[str, Any]) -> ModelT:
    if hasattr(model_class, "model_validate"):
        return model_class.model_validate(payload)  # type: ignore[attr-defined]
    return model_class.parse_obj(payload)


def _model_copy(model: ModelT, update: dict[str, Any]) -> ModelT:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)  # type: ignore[attr-defined]
    return model.copy(update=update)


class SQLiteRepository:
    """Base helper for SQLite-backed repositories."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database


class SQLiteSourceRepository(SQLiteRepository):
    """Stores source metadata for ingested inputs."""

    def create_source(self, source: SourceReference) -> SourceReference:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO sources (
                    source_id,
                    source_type,
                    title,
                    file_path,
                    original_filename,
                    mime_type,
                    checksum,
                    external_uri,
                    created_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    title = excluded.title,
                    file_path = excluded.file_path,
                    original_filename = excluded.original_filename,
                    mime_type = excluded.mime_type,
                    checksum = excluded.checksum,
                    external_uri = excluded.external_uri,
                    created_at = excluded.created_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    source.source_id,
                    source.source_type.value,
                    source.title,
                    source.file_path,
                    source.original_filename,
                    source.mime_type,
                    source.checksum,
                    source.external_uri,
                    _serialize_datetime(source.created_at),
                    _json_dumps(source.metadata),
                ),
            )
        return source

    def get_source(self, source_id: str) -> SourceReference | None:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return self._row_to_source(row) if row else None

    def list_sources(self) -> list[SourceReference]:
        with self.database.connection() as conn:
            rows = conn.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()
        return [self._row_to_source(row) for row in rows]

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> SourceReference:
        payload = {
            "source_id": row["source_id"],
            "source_type": SourceType(row["source_type"]),
            "title": row["title"],
            "file_path": row["file_path"],
            "original_filename": row["original_filename"],
            "mime_type": row["mime_type"],
            "checksum": row["checksum"],
            "external_uri": row["external_uri"],
            "created_at": _deserialize_datetime(row["created_at"]),
            "metadata": _json_loads(row["metadata_json"], {}),
        }
        return _model_validate(SourceReference, payload)


class SQLiteChunkRepository(SQLiteRepository):
    """Stores parsed chunks for ingested documents."""

    def upsert_chunks(self, chunks: list[ChunkReference]) -> list[ChunkReference]:
        if not chunks:
            return []

        with self.database.connection() as conn:
            conn.executemany(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    sequence_index,
                    text,
                    token_count,
                    char_start,
                    char_end,
                    section_title,
                    page_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    sequence_index = excluded.sequence_index,
                    text = excluded.text,
                    token_count = excluded.token_count,
                    char_start = excluded.char_start,
                    char_end = excluded.char_end,
                    section_title = excluded.section_title,
                    page_number = excluded.page_number
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.sequence_index,
                        chunk.text,
                        chunk.token_count,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.section_title,
                        chunk.page_number,
                    )
                    for chunk in chunks
                ],
            )
        return chunks

    def get_chunk(self, chunk_id: str) -> ChunkReference | None:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM document_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def list_chunks(self, document_id: str) -> list[ChunkReference]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_chunks
                WHERE document_id = ?
                ORDER BY sequence_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ChunkReference:
        payload = {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "sequence_index": row["sequence_index"],
            "text": row["text"],
            "token_count": row["token_count"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "section_title": row["section_title"],
            "page_number": row["page_number"],
        }
        return _model_validate(ChunkReference, payload)


class SQLiteCandidateMemoryRepository(SQLiteRepository):
    """Stores reviewable candidate memories extracted during ingestion."""

    def save_candidates(self, candidates: list[CandidateMemory]) -> list[CandidateMemory]:
        if not candidates:
            return []

        with self.database.connection() as conn:
            conn.executemany(
                """
                INSERT INTO candidate_memories (
                    candidate_id,
                    source_id,
                    proposed_memory_json,
                    extraction_reason,
                    source_chunk_ids_json,
                    confidence,
                    suggested_action,
                    existing_memory_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    proposed_memory_json = excluded.proposed_memory_json,
                    extraction_reason = excluded.extraction_reason,
                    source_chunk_ids_json = excluded.source_chunk_ids_json,
                    confidence = excluded.confidence,
                    suggested_action = excluded.suggested_action,
                    existing_memory_id = excluded.existing_memory_id,
                    created_at = excluded.created_at
                """,
                [
                    (
                        candidate.candidate_id,
                        candidate.proposed_memory.source_id,
                        _json_dumps(_model_to_json_dict(candidate.proposed_memory)),
                        candidate.extraction_reason,
                        _json_dumps(candidate.source_chunk_ids),
                        candidate.confidence,
                        candidate.suggested_action,
                        candidate.existing_memory_id,
                        _serialize_datetime(candidate.created_at),
                    )
                    for candidate in candidates
                ],
            )
        return candidates

    def get_candidate(self, candidate_id: str) -> CandidateMemory | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM candidate_memories WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def list_candidates(self, source_id: str | None = None) -> list[CandidateMemory]:
        query = "SELECT * FROM candidate_memories"
        params: list[Any] = []
        if source_id is not None:
            query += " WHERE source_id = ?"
            params.append(source_id)
        query += " ORDER BY created_at DESC"

        with self.database.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def delete_candidate(self, candidate_id: str) -> None:
        with self.database.connection() as conn:
            conn.execute("DELETE FROM candidate_memories WHERE candidate_id = ?", (candidate_id,))

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> CandidateMemory:
        payload = {
            "candidate_id": row["candidate_id"],
            "proposed_memory": json.loads(row["proposed_memory_json"]),
            "extraction_reason": row["extraction_reason"],
            "source_chunk_ids": _json_loads(row["source_chunk_ids_json"], []),
            "confidence": row["confidence"],
            "suggested_action": row["suggested_action"],
            "existing_memory_id": row["existing_memory_id"],
            "created_at": _deserialize_datetime(row["created_at"]),
        }
        return _model_validate(CandidateMemory, payload)


class SQLiteMemoryRepository(SQLiteRepository):
    """Stores canonical memories and version snapshots."""

    def create_memory(self, memory: MemoryRecord) -> MemoryRecord:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id,
                    memory_type,
                    status,
                    content,
                    summary,
                    source_id,
                    session_id,
                    conversation_id,
                    confidence,
                    importance,
                    tags_json,
                    version,
                    parent_memory_id,
                    supersedes_memory_id,
                    embedding_ref,
                    graph_node_ref,
                    created_at,
                    updated_at,
                    deleted_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    memory_type = excluded.memory_type,
                    status = excluded.status,
                    content = excluded.content,
                    summary = excluded.summary,
                    source_id = excluded.source_id,
                    session_id = excluded.session_id,
                    conversation_id = excluded.conversation_id,
                    confidence = excluded.confidence,
                    importance = excluded.importance,
                    tags_json = excluded.tags_json,
                    version = excluded.version,
                    parent_memory_id = excluded.parent_memory_id,
                    supersedes_memory_id = excluded.supersedes_memory_id,
                    embedding_ref = excluded.embedding_ref,
                    graph_node_ref = excluded.graph_node_ref,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    deleted_at = excluded.deleted_at,
                    metadata_json = excluded.metadata_json
                """,
                self._memory_params(memory),
            )
            self._record_version_snapshot(conn, memory)
        return memory

    def update_memory(self, memory: MemoryRecord) -> MemoryRecord:
        with self.database.connection() as conn:
            existing = conn.execute(
                "SELECT 1 FROM memories WHERE memory_id = ?",
                (memory.memory_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Memory not found: {memory.memory_id}")

            conn.execute(
                """
                UPDATE memories
                SET
                    memory_type = ?,
                    status = ?,
                    content = ?,
                    summary = ?,
                    source_id = ?,
                    session_id = ?,
                    conversation_id = ?,
                    confidence = ?,
                    importance = ?,
                    tags_json = ?,
                    version = ?,
                    parent_memory_id = ?,
                    supersedes_memory_id = ?,
                    embedding_ref = ?,
                    graph_node_ref = ?,
                    created_at = ?,
                    updated_at = ?,
                    deleted_at = ?,
                    metadata_json = ?
                WHERE memory_id = ?
                """,
                (
                    memory.memory_type.value,
                    memory.status.value,
                    memory.content,
                    memory.summary,
                    memory.source_id,
                    memory.session_id,
                    memory.conversation_id,
                    memory.confidence,
                    memory.importance,
                    _json_dumps(memory.tags),
                    memory.version,
                    memory.parent_memory_id,
                    memory.supersedes_memory_id,
                    memory.embedding_ref,
                    memory.graph_node_ref,
                    _serialize_datetime(memory.created_at),
                    _serialize_datetime(memory.updated_at),
                    _serialize_datetime(memory.deleted_at),
                    _json_dumps(memory.metadata),
                    memory.memory_id,
                ),
            )
            self._record_version_snapshot(conn, memory)
        return memory

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(self, memory_types: list[MemoryType] | None = None) -> list[MemoryRecord]:
        query = "SELECT * FROM memories"
        params: list[Any] = []
        if memory_types:
            placeholders = ", ".join("?" for _ in memory_types)
            query += f" WHERE memory_type IN ({placeholders})"
            params.extend(memory_type.value for memory_type in memory_types)
        query += " ORDER BY updated_at DESC"

        with self.database.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def soft_delete_memory(self, memory_id: str, deleted_at: datetime) -> None:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
            if row is None:
                raise KeyError(f"Memory not found: {memory_id}")

            current_memory = self._row_to_memory(row)
            deleted_memory = _model_copy(
                current_memory,
                update={
                    "status": MemoryStatus.DELETED,
                    "deleted_at": deleted_at,
                    "updated_at": deleted_at,
                    "version": current_memory.version + 1,
                }
            )

            conn.execute(
                """
                UPDATE memories
                SET status = ?, version = ?, updated_at = ?, deleted_at = ?
                WHERE memory_id = ?
                """,
                (
                    deleted_memory.status.value,
                    deleted_memory.version,
                    _serialize_datetime(deleted_memory.updated_at),
                    _serialize_datetime(deleted_memory.deleted_at),
                    memory_id,
                ),
            )
            self._record_version_snapshot(conn, deleted_memory)

    def get_memory_versions(self, memory_id: str) -> list[MemoryRecord]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT snapshot_json
                FROM memory_versions
                WHERE memory_id = ?
                ORDER BY version ASC
                """,
                (memory_id,),
            ).fetchall()
        return [
            _model_validate(MemoryRecord, json.loads(row["snapshot_json"]))
            for row in rows
        ]

    def _record_version_snapshot(self, conn: sqlite3.Connection, memory: MemoryRecord) -> None:
        conn.execute(
            """
            INSERT INTO memory_versions (memory_id, version, snapshot_json, recorded_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_id, version) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                recorded_at = excluded.recorded_at
            """,
            (
                memory.memory_id,
                memory.version,
                _json_dumps(_model_to_json_dict(memory)),
                _serialize_datetime(memory.updated_at),
            ),
        )

    @staticmethod
    def _memory_params(memory: MemoryRecord) -> tuple[Any, ...]:
        return (
            memory.memory_id,
            memory.memory_type.value,
            memory.status.value,
            memory.content,
            memory.summary,
            memory.source_id,
            memory.session_id,
            memory.conversation_id,
            memory.confidence,
            memory.importance,
            _json_dumps(memory.tags),
            memory.version,
            memory.parent_memory_id,
            memory.supersedes_memory_id,
            memory.embedding_ref,
            memory.graph_node_ref,
            _serialize_datetime(memory.created_at),
            _serialize_datetime(memory.updated_at),
            _serialize_datetime(memory.deleted_at),
            _json_dumps(memory.metadata),
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        payload = {
            "memory_id": row["memory_id"],
            "memory_type": MemoryType(row["memory_type"]),
            "status": MemoryStatus(row["status"]),
            "content": row["content"],
            "summary": row["summary"],
            "source_id": row["source_id"],
            "session_id": row["session_id"],
            "conversation_id": row["conversation_id"],
            "confidence": row["confidence"],
            "importance": row["importance"],
            "tags": _json_loads(row["tags_json"], []),
            "version": row["version"],
            "parent_memory_id": row["parent_memory_id"],
            "supersedes_memory_id": row["supersedes_memory_id"],
            "embedding_ref": row["embedding_ref"],
            "graph_node_ref": row["graph_node_ref"],
            "created_at": _deserialize_datetime(row["created_at"]),
            "updated_at": _deserialize_datetime(row["updated_at"]),
            "deleted_at": _deserialize_datetime(row["deleted_at"]),
            "metadata": _json_loads(row["metadata_json"], {}),
        }
        return _model_validate(MemoryRecord, payload)


class SQLiteJobRepository(SQLiteRepository):
    """Stores ingestion job state and outcomes."""

    def create_job(self, job: IngestionJob) -> IngestionJob:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id,
                    source_id,
                    status,
                    created_at,
                    updated_at,
                    error_message,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    source_id = excluded.source_id,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    error_message = excluded.error_message,
                    metadata_json = excluded.metadata_json
                """,
                (
                    job.job_id,
                    job.source_id,
                    job.status.value,
                    _serialize_datetime(job.created_at),
                    _serialize_datetime(job.updated_at),
                    job.error_message,
                    _json_dumps(job.metadata),
                ),
            )
        return job

    def update_job(self, job: IngestionJob) -> IngestionJob:
        with self.database.connection() as conn:
            existing = conn.execute(
                "SELECT 1 FROM ingestion_jobs WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Job not found: {job.job_id}")

            conn.execute(
                """
                UPDATE ingestion_jobs
                SET source_id = ?, status = ?, created_at = ?, updated_at = ?, error_message = ?, metadata_json = ?
                WHERE job_id = ?
                """,
                (
                    job.source_id,
                    job.status.value,
                    _serialize_datetime(job.created_at),
                    _serialize_datetime(job.updated_at),
                    job.error_message,
                    _json_dumps(job.metadata),
                    job.job_id,
                ),
            )
        return job

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> IngestionJob:
        payload = {
            "job_id": row["job_id"],
            "source_id": row["source_id"],
            "status": JobStatus(row["status"]),
            "created_at": _deserialize_datetime(row["created_at"]),
            "updated_at": _deserialize_datetime(row["updated_at"]),
            "error_message": row["error_message"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }
        return _model_validate(IngestionJob, payload)


class SQLiteAuditRepository(SQLiteRepository):
    """Stores full lifecycle audit events."""

    def record_event(self, event: AuditEvent) -> AuditEvent:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    event_id,
                    event_type,
                    actor_id,
                    memory_id,
                    source_id,
                    session_id,
                    timestamp,
                    details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    event_type = excluded.event_type,
                    actor_id = excluded.actor_id,
                    memory_id = excluded.memory_id,
                    source_id = excluded.source_id,
                    session_id = excluded.session_id,
                    timestamp = excluded.timestamp,
                    details_json = excluded.details_json
                """,
                (
                    event.event_id,
                    event.event_type.value,
                    event.actor_id,
                    event.memory_id,
                    event.source_id,
                    event.session_id,
                    _serialize_datetime(event.timestamp),
                    _json_dumps(event.details),
                ),
            )
        return event

    def list_events(
        self,
        event_type: AuditEventType | None = None,
        memory_id: str | None = None,
        source_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        query = "SELECT * FROM audit_events"
        conditions: list[str] = []
        params: list[Any] = []

        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        if memory_id is not None:
            conditions.append("memory_id = ?")
            params.append(memory_id)
        if source_id is not None:
            conditions.append("source_id = ?")
            params.append(source_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.database.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> AuditEvent:
        payload = {
            "event_id": row["event_id"],
            "event_type": AuditEventType(row["event_type"]),
            "actor_id": row["actor_id"],
            "memory_id": row["memory_id"],
            "source_id": row["source_id"],
            "session_id": row["session_id"],
            "timestamp": _deserialize_datetime(row["timestamp"]),
            "details": _json_loads(row["details_json"], {}),
        }
        return _model_validate(AuditEvent, payload)


class SQLiteSessionRepository(SQLiteRepository):
    """Stores recent session messages and rolling summaries."""

    def append_message(self, session_id: str, message: SessionMessage) -> None:
        with self.database.connection() as conn:
            existing = conn.execute(
                "SELECT conversation_id, rolling_summary FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            conversation_id = str(
                message.metadata.get("conversation_id")
                or (existing["conversation_id"] if existing else None)
                or session_id
            )
            rolling_summary = existing["rolling_summary"] if existing else None

            conn.execute(
                """
                INSERT INTO session_state (session_id, conversation_id, rolling_summary, last_active_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    last_active_at = excluded.last_active_at
                """,
                (
                    session_id,
                    conversation_id,
                    rolling_summary,
                    _serialize_datetime(message.created_at),
                ),
            )
            conn.execute(
                """
                INSERT INTO session_messages (
                    message_id,
                    session_id,
                    role,
                    content,
                    created_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    role = excluded.role,
                    content = excluded.content,
                    created_at = excluded.created_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    message.message_id,
                    session_id,
                    message.role,
                    message.content,
                    _serialize_datetime(message.created_at),
                    _json_dumps(message.metadata),
                ),
            )

    def get_session(self, session_id: str) -> SessionSnapshot | None:
        with self.database.connection() as conn:
            state_row = conn.execute(
                "SELECT * FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if state_row is None:
                return None

            message_rows = conn.execute(
                """
                SELECT * FROM session_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()

        messages = [self._row_to_session_message(row) for row in message_rows]
        payload = {
            "session_id": state_row["session_id"],
            "conversation_id": state_row["conversation_id"],
            "recent_messages": messages,
            "rolling_summary": state_row["rolling_summary"],
            "last_active_at": _deserialize_datetime(state_row["last_active_at"]),
        }
        return _model_validate(SessionSnapshot, payload)

    def save_summary(self, session_id: str, summary: str) -> None:
        now = datetime.utcnow()
        with self.database.connection() as conn:
            existing = conn.execute(
                "SELECT conversation_id FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            conversation_id = existing["conversation_id"] if existing else session_id
            conn.execute(
                """
                INSERT INTO session_state (session_id, conversation_id, rolling_summary, last_active_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    rolling_summary = excluded.rolling_summary,
                    last_active_at = excluded.last_active_at
                """,
                (
                    session_id,
                    conversation_id,
                    summary,
                    _serialize_datetime(now),
                ),
            )

    @staticmethod
    def _row_to_session_message(row: sqlite3.Row) -> SessionMessage:
        payload = {
            "message_id": row["message_id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": _deserialize_datetime(row["created_at"]),
            "metadata": _json_loads(row["metadata_json"], {}),
        }
        return _model_validate(SessionMessage, payload)
