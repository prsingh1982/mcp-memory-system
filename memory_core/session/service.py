"""Session service for rolling summaries and durable cross-session memory promotion."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from memory_core.domain.enums import AuditEventType, MemoryStatus, MemoryType, SourceType
from memory_core.domain.models import AuditEvent, MemoryRecord, RetrievalQuery, SessionMessage, SessionSnapshot
from memory_core.interfaces.llm import LLMClient
from memory_core.interfaces.retrieval import RetrievalService
from memory_core.interfaces.services import MemoryService, SessionService
from memory_core.interfaces.storage import AuditRepository, SessionRepository
from memory_core.services.merge import build_merged_memory_updates

ModelT = TypeVar("ModelT", bound=BaseModel)


def _model_copy(model: ModelT, update: dict[str, Any]) -> ModelT:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)  # type: ignore[attr-defined]
    return model.copy(update=update)


class DefaultSessionService(SessionService):
    """Maintains session continuity and promotes durable chat-derived memory."""

    def __init__(
        self,
        session_repository: SessionRepository,
        memory_service: MemoryService,
        retrieval_service: RetrievalService,
        llm_client: LLMClient,
        audit_repository: AuditRepository,
        *,
        summary_window_messages: int = 8,
        promotion_window_messages: int = 12,
        promotion_confidence_threshold: float = 0.65,
        merge_score_threshold: float = 0.82,
    ) -> None:
        self.session_repository = session_repository
        self.memory_service = memory_service
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client
        self.audit_repository = audit_repository
        self.summary_window_messages = max(summary_window_messages, 2)
        self.promotion_window_messages = max(promotion_window_messages, self.summary_window_messages)
        self.promotion_confidence_threshold = promotion_confidence_threshold
        self.merge_score_threshold = merge_score_threshold

    def append_message(self, session_id: str, message: SessionMessage) -> None:
        self.session_repository.append_message(session_id, message)

    def get_context_snapshot(self, session_id: str) -> SessionSnapshot | None:
        return self.session_repository.get_session(session_id)

    def summarize_session(self, session_id: str) -> MemoryRecord | None:
        snapshot = self.session_repository.get_session(session_id)
        if snapshot is None or not snapshot.recent_messages:
            return None

        summary_messages = snapshot.recent_messages[-self.summary_window_messages :]
        summary_transcript = self._render_transcript(summary_messages)
        summary = self.llm_client.summarize(
            summary_transcript,
            context=(
                "Summarize only the durable points from this conversation window. "
                "Focus on facts, figures, repeated references, user characteristics, stable preferences, "
                "and useful inferences. Ignore casual filler."
            ),
        )
        self.session_repository.save_summary(session_id, summary)
        self._record_audit_event(
            AuditEventType.SESSION_SUMMARIZED,
            session_id=session_id,
            details={"summary_length": len(summary)},
        )

        promotion_messages = snapshot.recent_messages[-self.promotion_window_messages :]
        promotion_transcript = self._render_transcript(promotion_messages)
        promoted_memories = self._promote_durable_memories(snapshot, promotion_transcript)
        heuristic_memories = self._promote_heuristic_memories(snapshot, promotion_messages)
        all_promoted = promoted_memories + heuristic_memories
        return all_promoted[0] if all_promoted else None

    def _promote_durable_memories(
        self,
        snapshot: SessionSnapshot,
        transcript: str,
    ) -> list[MemoryRecord]:
        extracted_items = self.llm_client.extract_structured_memory(transcript, SourceType.CHAT)
        promoted: list[MemoryRecord] = []

        for item in extracted_items:
            memory_type = self._coerce_memory_type(item.get("memory_type"))
            if memory_type not in {MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.WORKFLOW_RULE}:
                continue

            confidence = self._coerce_score(item.get("confidence"), default=0.5)
            if confidence < self.promotion_confidence_threshold:
                continue

            content = str(item.get("content", "")).strip()
            if not content or len(content) < 12:
                continue

            tags = self._coerce_tags(item.get("tags"))
            tags.extend(["chat_memory", memory_type.value])
            tags = self._dedupe_strings(tags)

            metadata = self._coerce_metadata(item.get("metadata"))
            metadata.update(
                {
                    "persistent_across_sessions": True,
                    "source_kind": "chat",
                    "source_session_ids": self._dedupe_strings(
                        list(metadata.get("source_session_ids", [])) + [snapshot.session_id]
                    ),
                    "source_conversation_ids": self._dedupe_strings(
                        list(metadata.get("source_conversation_ids", [])) + [snapshot.conversation_id]
                    ),
                    "extraction_reason": self._optional_text(item.get("extraction_reason")),
                }
            )

            candidate_memory = MemoryRecord(
                memory_id=f"chatmem_{uuid4().hex}",
                memory_type=memory_type,
                status=MemoryStatus.ACTIVE,
                content=content,
                summary=self._optional_text(item.get("summary")) or content[:220],
                source_id=None,
                session_id=None,
                conversation_id=None,
                confidence=confidence,
                importance=self._coerce_score(item.get("importance"), default=confidence),
                tags=tags,
                version=1,
                parent_memory_id=None,
                supersedes_memory_id=None,
                embedding_ref=None,
                graph_node_ref=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                deleted_at=None,
                metadata=metadata,
            )

            existing = self._find_existing_memory(candidate_memory)
            if existing is None:
                stored_memory = self.memory_service.store_memory(candidate_memory)
            else:
                merged_updates = build_merged_memory_updates(existing, candidate_memory)
                merged_updates["metadata"] = self._merge_metadata(existing.metadata, candidate_memory.metadata)
                stored_memory = self.memory_service.update_memory(existing.memory_id, merged_updates)
            promoted.append(stored_memory)

        return promoted

    def _promote_heuristic_memories(
        self,
        snapshot: SessionSnapshot,
        messages: list[SessionMessage],
    ) -> list[MemoryRecord]:
        promoted: list[MemoryRecord] = []
        for message in messages:
            if message.role != "user":
                continue

            name = self._extract_name_fact(message.content)
            if name:
                promoted.append(
                    self._store_or_merge_identity_memory(
                        snapshot,
                        identity_key="user_name",
                        content=f"User's name is {name}.",
                        summary=f"User name: {name}",
                        tags=["chat_memory", "identity", "name"],
                        confidence=0.98,
                        importance=0.95,
                    )
                )

        return promoted

    def _find_existing_memory(self, candidate_memory: MemoryRecord) -> MemoryRecord | None:
        identity_key = candidate_memory.metadata.get("identity_key")
        if identity_key:
            for memory in self.retrieval_service.memory_repository.list_memories([candidate_memory.memory_type]):  # type: ignore[attr-defined]
                if memory.status != MemoryStatus.ACTIVE:
                    continue
                if memory.metadata.get("identity_key") == identity_key:
                    return memory

        retrieval_result = self.retrieval_service.retrieve(
            RetrievalQuery(
                query=candidate_memory.content,
                memory_types=[candidate_memory.memory_type],
                top_k=3,
                include_deleted=False,
                require_citations=False,
            )
        )
        if not retrieval_result.items:
            return None

        best = retrieval_result.items[0]
        if best.score.final_score < self.merge_score_threshold:
            return None
        return best.memory

    def _store_or_merge_identity_memory(
        self,
        snapshot: SessionSnapshot,
        *,
        identity_key: str,
        content: str,
        summary: str,
        tags: list[str],
        confidence: float,
        importance: float,
    ) -> MemoryRecord:
        metadata = {
            "persistent_across_sessions": True,
            "source_kind": "chat",
            "identity_key": identity_key,
            "source_session_ids": [snapshot.session_id],
            "source_conversation_ids": [snapshot.conversation_id],
            "heuristic": True,
        }
        memory = MemoryRecord(
            memory_id=f"chatmem_{uuid4().hex}",
            memory_type=MemoryType.FACT,
            status=MemoryStatus.ACTIVE,
            content=content,
            summary=summary,
            source_id=None,
            session_id=None,
            conversation_id=None,
            confidence=confidence,
            importance=importance,
            tags=self._dedupe_strings(tags),
            version=1,
            parent_memory_id=None,
            supersedes_memory_id=None,
            embedding_ref=None,
            graph_node_ref=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            deleted_at=None,
            metadata=metadata,
        )
        existing = self._find_existing_memory(memory)
        if existing is None:
            return self.memory_service.store_memory(memory)

        merged_updates = build_merged_memory_updates(existing, memory)
        merged_updates["metadata"] = self._merge_metadata(existing.metadata, memory.metadata)
        return self.memory_service.update_memory(existing.memory_id, merged_updates)

    @staticmethod
    def _render_transcript(messages: list[SessionMessage]) -> str:
        return "\n\n".join(f"{message.role}: {message.content}" for message in messages)

    @staticmethod
    def _merge_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(existing)
        merged.update(incoming)
        for list_key in ("source_session_ids", "source_conversation_ids"):
            combined = list(existing.get(list_key, [])) + list(incoming.get(list_key, []))
            merged[list_key] = DefaultSessionService._dedupe_strings(combined)
        return merged

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

    @staticmethod
    def _coerce_tags(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _coerce_metadata(value: object) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _dedupe_strings(values: list[Any]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in deduped:
                deduped.append(text)
        return deduped

    @staticmethod
    def _extract_name_fact(text: str) -> str | None:
        patterns = (
            r"\bmy name is ([A-Za-z][A-Za-z' -]{1,80})",
            r"\bi am called ([A-Za-z][A-Za-z' -]{1,80})",
            r"\bcall me ([A-Za-z][A-Za-z' -]{1,80})",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" .,!?:;")
            if candidate:
                return " ".join(part.capitalize() for part in candidate.split())
        return None

    def _record_audit_event(
        self,
        event_type: AuditEventType,
        *,
        session_id: str,
        details: dict[str, Any],
    ) -> None:
        self.audit_repository.record_event(
            AuditEvent(
                event_id=f"audit_{uuid4().hex}",
                event_type=event_type,
                actor_id=None,
                memory_id=None,
                source_id=None,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                details=details,
            )
        )
