"""Canonical memory lifecycle service implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from memory_core.domain.enums import AuditEventType, MemoryStatus
from memory_core.domain.models import AuditEvent, MemoryRecord
from memory_core.interfaces.embeddings import EmbeddingProvider
from memory_core.interfaces.graph import GraphStore, VectorIndex
from memory_core.interfaces.services import MemoryService
from memory_core.interfaces.storage import AuditRepository, MemoryRepository

from .merge import build_merged_memory_updates

ModelT = TypeVar("ModelT", bound=BaseModel)


def _model_copy(model: ModelT, update: dict[str, Any]) -> ModelT:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)  # type: ignore[attr-defined]
    return model.copy(update=update)


class DefaultMemoryService(MemoryService):
    """Coordinates canonical memory persistence with indexing and graph updates."""

    def __init__(
        self,
        memory_repository: MemoryRepository,
        audit_repository: AuditRepository,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_index: VectorIndex | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.memory_repository = memory_repository
        self.audit_repository = audit_repository
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.graph_store = graph_store

    def store_memory(self, memory: MemoryRecord) -> MemoryRecord:
        prepared_memory = self._prepare_memory_for_storage(memory, is_new=True)
        stored_memory = self.memory_repository.create_memory(prepared_memory)
        self._sync_indexes(stored_memory)
        self._record_audit_event(
            AuditEventType.MEMORY_CREATED,
            memory_id=stored_memory.memory_id,
            source_id=stored_memory.source_id,
            details={"version": stored_memory.version, "status": stored_memory.status.value},
        )
        return stored_memory

    def update_memory(self, memory_id: str, updates: dict[str, Any]) -> MemoryRecord:
        existing = self.memory_repository.get_memory(memory_id)
        if existing is None:
            raise KeyError(f"Memory not found: {memory_id}")

        merged_updates = self._normalize_updates(existing, updates)
        updated_memory = _model_copy(
            existing,
            {
                **merged_updates,
                "version": existing.version + 1,
                "updated_at": datetime.utcnow(),
            },
        )
        prepared_memory = self._prepare_memory_for_storage(updated_memory, is_new=False)
        stored_memory = self.memory_repository.update_memory(prepared_memory)
        self._sync_indexes(stored_memory)
        self._record_audit_event(
            AuditEventType.MEMORY_UPDATED,
            memory_id=stored_memory.memory_id,
            source_id=stored_memory.source_id,
            details={"version": stored_memory.version, "updated_fields": sorted(merged_updates.keys())},
        )
        return stored_memory

    def merge_memory(self, source_memory_id: str, target_memory_id: str) -> MemoryRecord:
        source_memory = self.memory_repository.get_memory(source_memory_id)
        if source_memory is None:
            raise KeyError(f"Source memory not found: {source_memory_id}")
        target_memory = self.memory_repository.get_memory(target_memory_id)
        if target_memory is None:
            raise KeyError(f"Target memory not found: {target_memory_id}")

        merged_updates = build_merged_memory_updates(target_memory, source_memory)
        merged_target = self.update_memory(target_memory_id, merged_updates)

        superseded_source = _model_copy(
            source_memory,
            {
                "status": MemoryStatus.SUPERSEDED,
                "supersedes_memory_id": target_memory_id,
                "version": source_memory.version + 1,
                "updated_at": datetime.utcnow(),
            },
        )
        superseded_source = self._prepare_memory_for_storage(superseded_source, is_new=False)
        self.memory_repository.update_memory(superseded_source)
        self._remove_from_vector_index(source_memory_id)
        if self.graph_store is not None:
            self.graph_store.upsert_memory_node(superseded_source)
            self.graph_store.create_relationship(source_memory_id, "SUPERSEDES", target_memory_id)

        self._record_audit_event(
            AuditEventType.MEMORY_MERGED,
            memory_id=target_memory_id,
            source_id=merged_target.source_id,
            details={"source_memory_id": source_memory_id, "target_memory_id": target_memory_id},
        )
        return merged_target

    def delete_memory(self, memory_id: str, reason: str) -> None:
        deleted_at = datetime.utcnow()
        self.memory_repository.soft_delete_memory(memory_id, deleted_at)
        self._remove_from_vector_index(memory_id)
        if self.graph_store is not None:
            self.graph_store.mark_deleted(memory_id, deleted_at)

        deleted_memory = self.memory_repository.get_memory(memory_id)
        self._record_audit_event(
            AuditEventType.MEMORY_SOFT_DELETED,
            memory_id=memory_id,
            source_id=deleted_memory.source_id if deleted_memory else None,
            details={"reason": reason},
        )

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self.memory_repository.get_memory(memory_id)

    def get_memory_history(self, memory_id: str) -> list[MemoryRecord]:
        return self.memory_repository.get_memory_versions(memory_id)

    def reindex_memory(self, memory_id: str) -> MemoryRecord:
        memory = self.memory_repository.get_memory(memory_id)
        if memory is None:
            raise KeyError(f"Memory not found: {memory_id}")
        self._sync_indexes(memory)
        return memory

    def reindex_all_memories(self) -> dict[str, Any]:
        memories = self.memory_repository.list_memories()
        repaired = 0
        failures: list[dict[str, str]] = []

        for memory in memories:
            try:
                if memory.status == MemoryStatus.ACTIVE:
                    self._sync_indexes(memory)
                else:
                    self._remove_from_vector_index(memory.memory_id)
                    if self.graph_store is not None and memory.status == MemoryStatus.DELETED:
                        self.graph_store.mark_deleted(memory.memory_id, memory.deleted_at or memory.updated_at)
                repaired += 1
            except Exception as exc:
                failures.append({"memory_id": memory.memory_id, "error": str(exc)})

        return {
            "total": len(memories),
            "repaired": repaired,
            "failures": failures,
        }

    def _prepare_memory_for_storage(self, memory: MemoryRecord, *, is_new: bool) -> MemoryRecord:
        update: dict[str, Any] = {
            "updated_at": memory.updated_at or datetime.utcnow(),
        }
        if is_new and not memory.created_at:
            update["created_at"] = datetime.utcnow()

        if self.vector_index is not None and self.embedding_provider is not None and memory.status == MemoryStatus.ACTIVE:
            update["embedding_ref"] = self._embedding_ref(memory.memory_id)

        if self.graph_store is not None:
            update["graph_node_ref"] = memory.memory_id

        return _model_copy(memory, update)

    def _sync_indexes(self, memory: MemoryRecord) -> None:
        if memory.status == MemoryStatus.DELETED:
            self._remove_from_vector_index(memory.memory_id)
            if self.graph_store is not None:
                self.graph_store.mark_deleted(memory.memory_id, memory.deleted_at or memory.updated_at)
            return

        if self.vector_index is not None and self.embedding_provider is not None and memory.status == MemoryStatus.ACTIVE:
            embedding = self.embedding_provider.embed_text(self._embedding_text(memory))
            self.vector_index.upsert(
                self._embedding_ref(memory.memory_id),
                embedding,
                {
                    "memory_id": memory.memory_id,
                    "memory_type": memory.memory_type.value,
                    "source_id": memory.source_id,
                    "chunk_id": memory.metadata.get("chunk_id"),
                    "source_chunk_ids": memory.metadata.get("source_chunk_ids", []),
                },
            )

        if self.graph_store is not None:
            self.graph_store.upsert_memory_node(memory)
            if memory.parent_memory_id:
                self.graph_store.create_relationship(memory.memory_id, "DERIVED_FROM", memory.parent_memory_id)
            if memory.supersedes_memory_id:
                self.graph_store.create_relationship(memory.memory_id, "SUPERSEDES", memory.supersedes_memory_id)

    def _remove_from_vector_index(self, memory_id: str) -> None:
        if self.vector_index is not None:
            self.vector_index.delete(self._embedding_ref(memory_id))

    def _normalize_updates(self, existing: MemoryRecord, updates: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(updates)
        if "tags" in normalized:
            raw_tags = normalized["tags"] or []
            normalized["tags"] = [str(tag) for tag in raw_tags if str(tag).strip()]
        if "metadata" in normalized:
            merged_metadata = dict(existing.metadata)
            incoming_metadata = normalized["metadata"]
            if isinstance(incoming_metadata, dict):
                merged_metadata.update(incoming_metadata)
            normalized["metadata"] = merged_metadata
        return normalized

    def _record_audit_event(
        self,
        event_type: AuditEventType,
        *,
        memory_id: str | None,
        source_id: str | None,
        details: dict[str, Any],
    ) -> None:
        self.audit_repository.record_event(
            AuditEvent(
                event_id=f"audit_{uuid4().hex}",
                event_type=event_type,
                actor_id=None,
                memory_id=memory_id,
                source_id=source_id,
                session_id=None,
                timestamp=datetime.utcnow(),
                details=details,
            )
        )

    @staticmethod
    def _embedding_ref(memory_id: str) -> str:
        return f"mem::{memory_id}"

    @staticmethod
    def _embedding_text(memory: MemoryRecord) -> str:
        summary = (memory.summary or "").strip()
        content = memory.content.strip()
        if summary and content:
            return f"{summary}\n\n{content[:4000]}"
        if summary:
            return summary
        return content[:4000]
