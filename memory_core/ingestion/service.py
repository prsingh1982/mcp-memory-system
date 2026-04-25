"""Ingestion service orchestration for source parsing, chunking, and candidate extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from memory_core.domain.enums import AuditEventType, JobStatus, MemoryStatus, MemoryType
from memory_core.domain.models import AuditEvent, CandidateMemory, ChunkReference, IngestionJob, MemoryRecord, SourceReference
from memory_core.interfaces.llm import LLMClient
from memory_core.interfaces.parsing import Chunker
from memory_core.interfaces.services import IngestionService, MemoryService
from memory_core.interfaces.storage import (
    AuditRepository,
    CandidateMemoryRepository,
    ChunkRepository,
    JobRepository,
    SourceRepository,
)

from .parsers import ParserRegistry

ModelT = TypeVar("ModelT", bound=BaseModel)


def _model_copy(model: ModelT, update: dict[str, Any]) -> ModelT:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)  # type: ignore[attr-defined]
    return model.copy(update=update)


class DefaultIngestionService(IngestionService):
    """Coordinates source persistence, parsing, chunking, and candidate generation."""

    def __init__(
        self,
        source_repository: SourceRepository,
        chunk_repository: ChunkRepository,
        candidate_repository: CandidateMemoryRepository,
        job_repository: JobRepository,
        audit_repository: AuditRepository,
        parser_registry: ParserRegistry,
        chunker: Chunker,
        llm_client: LLMClient | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.source_repository = source_repository
        self.chunk_repository = chunk_repository
        self.candidate_repository = candidate_repository
        self.job_repository = job_repository
        self.audit_repository = audit_repository
        self.parser_registry = parser_registry
        self.chunker = chunker
        self.llm_client = llm_client
        self.memory_service = memory_service

    def ingest_source(self, source: SourceReference) -> IngestionJob:
        persisted_source = self.source_repository.create_source(source)
        now = datetime.utcnow()
        job = IngestionJob(
            job_id=f"job_{uuid4().hex}",
            source_id=persisted_source.source_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            error_message=None,
            metadata={"source_type": persisted_source.source_type.value},
        )
        job = self.job_repository.create_job(job)
        self._record_audit_event(
            event_type=AuditEventType.INGESTION_CREATED,
            source_id=persisted_source.source_id,
            details={"job_id": job.job_id, "source_type": persisted_source.source_type.value},
        )
        return job

    def process_job(self, job_id: str) -> IngestionJob:
        job = self.job_repository.get_job(job_id)
        if job is None:
            raise KeyError(f"Ingestion job not found: {job_id}")

        source = self.source_repository.get_source(job.source_id)
        if source is None:
            raise KeyError(f"Source not found for ingestion job: {job.source_id}")

        running_job = _model_copy(job, {"status": JobStatus.RUNNING, "updated_at": datetime.utcnow()})
        self.job_repository.update_job(running_job)

        try:
            text = self.parse_source(source.source_id)
            chunks = self.chunk_source(source.source_id, text)
            indexed_memories = self._index_source_for_retrieval(source, text, chunks)
            candidates = self.extract_candidate_memories(source.source_id, chunks)

            completed_status = JobStatus.REVIEW_REQUIRED if candidates else JobStatus.COMPLETED
            completed_job = _model_copy(
                running_job,
                {
                    "status": completed_status,
                    "updated_at": datetime.utcnow(),
                    "metadata": {
                        **running_job.metadata,
                        "chunk_count": len(chunks),
                        "indexed_memory_count": indexed_memories,
                        "candidate_count": len(candidates),
                    },
                },
            )
            self.job_repository.update_job(completed_job)
            self._record_audit_event(
                event_type=AuditEventType.INGESTION_COMPLETED,
                source_id=source.source_id,
                details={
                    "job_id": job_id,
                    "chunk_count": len(chunks),
                    "indexed_memory_count": indexed_memories,
                    "candidate_count": len(candidates),
                    "status": completed_status.value,
                },
            )
            return completed_job
        except Exception as exc:
            failed_job = _model_copy(
                running_job,
                {
                    "status": JobStatus.FAILED,
                    "updated_at": datetime.utcnow(),
                    "error_message": str(exc),
                },
            )
            self.job_repository.update_job(failed_job)
            self._record_audit_event(
                event_type=AuditEventType.INGESTION_FAILED,
                source_id=source.source_id,
                details={"job_id": job_id, "error": str(exc)},
            )
            raise

    def parse_source(self, source_id: str) -> str:
        source = self.source_repository.get_source(source_id)
        if source is None:
            raise KeyError(f"Source not found: {source_id}")
        text = self.parser_registry.parse(source).strip()
        if not text:
            raise ValueError(f"Parsed source is empty: {source_id}")
        return text

    def chunk_source(self, source_id: str, text: str) -> list[ChunkReference]:
        chunks = self.chunker.chunk(document_id=source_id, text=text, metadata={"source_id": source_id})
        return self.chunk_repository.upsert_chunks(chunks)

    def extract_candidate_memories(self, source_id: str, chunks: list[ChunkReference]) -> list[CandidateMemory]:
        if not chunks:
            return []

        source = self.source_repository.get_source(source_id)
        if source is None:
            raise KeyError(f"Source not found: {source_id}")

        if self.llm_client is None:
            candidates = [self._build_baseline_candidate(source, chunks)]
        else:
            candidates = self._extract_with_llm(source, chunks)
            if not candidates:
                candidates = [self._build_baseline_candidate(source, chunks)]

        return self.candidate_repository.save_candidates(candidates)

    def _index_source_for_retrieval(
        self,
        source: SourceReference,
        text: str,
        chunks: list[ChunkReference],
    ) -> int:
        if self.memory_service is None:
            return 0

        indexed_count = 0
        now = datetime.utcnow()
        document_memory_id = f"docmem::{source.source_id}"
        document_summary = self._build_document_summary(text)
        document_memory = MemoryRecord(
            memory_id=document_memory_id,
            memory_type=MemoryType.DOCUMENT,
            status=MemoryStatus.ACTIVE,
            content=text,
            summary=document_summary,
            source_id=source.source_id,
            session_id=None,
            conversation_id=None,
            confidence=0.85,
            importance=0.75,
            tags=["document", source.source_type.value],
            version=1,
            parent_memory_id=None,
            supersedes_memory_id=None,
            embedding_ref=None,
            graph_node_ref=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
            metadata={
                "auto_indexed": True,
                "source_type": source.source_type.value,
                "source_chunk_ids": [chunk.chunk_id for chunk in chunks],
            },
        )
        self._upsert_active_memory(document_memory)
        indexed_count += 1

        for chunk in chunks:
            chunk_memory = MemoryRecord(
                memory_id=f"chunkmem::{chunk.chunk_id}",
                memory_type=MemoryType.DOCUMENT_CHUNK,
                status=MemoryStatus.ACTIVE,
                content=chunk.text,
                summary=(chunk.text[:280] + "...") if len(chunk.text) > 280 else chunk.text,
                source_id=source.source_id,
                session_id=None,
                conversation_id=None,
                confidence=0.80,
                importance=0.60,
                tags=["document_chunk", source.source_type.value],
                version=1,
                parent_memory_id=document_memory_id,
                supersedes_memory_id=None,
                embedding_ref=None,
                graph_node_ref=None,
                created_at=now,
                updated_at=now,
                deleted_at=None,
                metadata={
                    "auto_indexed": True,
                    "source_type": source.source_type.value,
                    "chunk_id": chunk.chunk_id,
                    "source_chunk_ids": [chunk.chunk_id],
                    "document_id": chunk.document_id,
                    "sequence_index": chunk.sequence_index,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                },
            )
            self._upsert_active_memory(chunk_memory)
            indexed_count += 1

        return indexed_count

    def _extract_with_llm(self, source: SourceReference, chunks: list[ChunkReference]) -> list[CandidateMemory]:
        now = datetime.utcnow()
        candidates: list[CandidateMemory] = []

        for chunk in chunks:
            extracted_items = self.llm_client.extract_structured_memory(chunk.text, source.source_type)
            for item in extracted_items:
                memory_type = self._coerce_memory_type(item.get("memory_type"))
                confidence = self._coerce_score(item.get("confidence"), default=0.5)
                importance = self._coerce_score(item.get("importance"), default=confidence)
                content = str(item.get("content", "")).strip()
                if not content:
                    continue

                metadata = self._coerce_metadata(item.get("metadata"))
                metadata.update(
                    {
                        "auto_extracted": True,
                        "source_type": source.source_type.value,
                        "chunk_id": chunk.chunk_id,
                    }
                )

                proposed_memory = MemoryRecord(
                    memory_id=f"mem_{uuid4().hex}",
                    memory_type=memory_type,
                    status=MemoryStatus.CANDIDATE,
                    content=content,
                    summary=self._optional_text(item.get("summary")),
                    source_id=source.source_id,
                    session_id=None,
                    conversation_id=None,
                    confidence=confidence,
                    importance=importance,
                    tags=self._coerce_tags(item.get("tags")),
                    version=1,
                    parent_memory_id=None,
                    supersedes_memory_id=item.get("existing_memory_id"),
                    embedding_ref=None,
                    graph_node_ref=None,
                    created_at=now,
                    updated_at=now,
                    deleted_at=None,
                    metadata=metadata,
                )
                candidates.append(
                    CandidateMemory(
                        candidate_id=f"cand_{uuid4().hex}",
                        proposed_memory=proposed_memory,
                        extraction_reason=self._optional_text(item.get("extraction_reason"))
                        or "LLM extracted a candidate memory from an ingested chunk.",
                        source_chunk_ids=[chunk.chunk_id],
                        confidence=confidence,
                        suggested_action=self._coerce_suggested_action(item.get("suggested_action")),
                        existing_memory_id=item.get("existing_memory_id"),
                        created_at=now,
                    )
                )

        return candidates

    def _build_baseline_candidate(self, source: SourceReference, chunks: list[ChunkReference]) -> CandidateMemory:
        now = datetime.utcnow()
        combined_text = "\n\n".join(chunk.text for chunk in chunks).strip()
        preview = combined_text[:400]
        summary = preview if len(combined_text) <= 400 else f"{preview}..."
        proposed_memory = MemoryRecord(
            memory_id=f"mem_{uuid4().hex}",
            memory_type=MemoryType.DOCUMENT,
            status=MemoryStatus.CANDIDATE,
            content=combined_text,
            summary=summary,
            source_id=source.source_id,
            session_id=None,
            conversation_id=None,
            confidence=0.45,
            importance=0.50,
            tags=["document_ingest", source.source_type.value],
            version=1,
            parent_memory_id=None,
            supersedes_memory_id=None,
            embedding_ref=None,
            graph_node_ref=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
            metadata={
                "auto_extracted": True,
                "baseline_candidate": True,
                "source_type": source.source_type.value,
                "chunk_count": len(chunks),
            },
        )
        return CandidateMemory(
            candidate_id=f"cand_{uuid4().hex}",
            proposed_memory=proposed_memory,
            extraction_reason="Created a baseline document candidate because no LLM extractor was configured.",
            source_chunk_ids=[chunk.chunk_id for chunk in chunks],
            confidence=0.45,
            suggested_action="create",
            existing_memory_id=None,
            created_at=now,
        )

    def _record_audit_event(
        self,
        event_type: AuditEventType,
        source_id: str,
        details: dict[str, object],
    ) -> None:
        self.audit_repository.record_event(
            AuditEvent(
                event_id=f"audit_{uuid4().hex}",
                event_type=event_type,
                actor_id=None,
                memory_id=None,
                source_id=source_id,
                session_id=None,
                timestamp=datetime.utcnow(),
                details=details,
            )
        )

    @staticmethod
    def _coerce_memory_type(value: object) -> MemoryType:
        if isinstance(value, MemoryType):
            return value
        if isinstance(value, str):
            try:
                return MemoryType(value)
            except ValueError:
                pass
        return MemoryType.FACT

    @staticmethod
    def _coerce_tags(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_suggested_action(value: object) -> str:
        if isinstance(value, str) and value in {"create", "merge", "update"}:
            return value
        return "create"

    @staticmethod
    def _coerce_metadata(value: object) -> dict[str, object]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_score(value: object, *, default: float) -> float:
        if value is None:
            return default

        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric > 1.0:
                numeric = numeric / 100.0 if numeric <= 100.0 else 1.0
            return max(0.0, min(1.0, numeric))

        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return default

            scale_map = {
                "very low": 0.10,
                "low": 0.25,
                "medium": 0.50,
                "med": 0.50,
                "moderate": 0.50,
                "high": 0.80,
                "very high": 0.95,
            }
            if normalized in scale_map:
                return scale_map[normalized]

            if normalized.endswith("%"):
                normalized = normalized[:-1].strip()

            try:
                numeric = float(normalized)
            except ValueError:
                return default

            if numeric > 1.0:
                numeric = numeric / 100.0 if numeric <= 100.0 else 1.0
            return max(0.0, min(1.0, numeric))

        return default

    def _build_document_summary(self, text: str) -> str:
        if self.llm_client is not None:
            try:
                return self.llm_client.summarize(
                    text[:6000],
                    context="Summarize this uploaded document for future retrieval.",
                )
            except Exception:
                pass

        preview = text[:400].strip()
        return preview if len(text) <= 400 else f"{preview}..."

    def _upsert_active_memory(self, memory: MemoryRecord) -> MemoryRecord:
        existing = self.memory_service.get_memory(memory.memory_id) if self.memory_service is not None else None
        if existing is None:
            return self.memory_service.store_memory(memory)  # type: ignore[union-attr]

        updates = {
            "content": memory.content,
            "summary": memory.summary,
            "status": MemoryStatus.ACTIVE,
            "source_id": memory.source_id,
            "confidence": memory.confidence,
            "importance": memory.importance,
            "tags": memory.tags,
            "parent_memory_id": memory.parent_memory_id,
            "metadata": memory.metadata,
        }
        return self.memory_service.update_memory(memory.memory_id, updates)  # type: ignore[union-attr]
