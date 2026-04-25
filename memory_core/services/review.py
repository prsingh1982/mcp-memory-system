"""Review workflow service implementation for candidate memories."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from memory_core.domain.enums import AuditEventType, MemoryStatus, ReviewDecision
from memory_core.domain.models import AuditEvent, CandidateMemory, MemoryRecord
from memory_core.interfaces.services import MemoryService, ReviewService
from memory_core.interfaces.storage import AuditRepository, CandidateMemoryRepository

from .merge import build_merged_memory_updates

ModelT = TypeVar("ModelT", bound=BaseModel)


def _model_copy(model: ModelT, update: dict[str, Any]) -> ModelT:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)  # type: ignore[attr-defined]
    return model.copy(update=update)


class DefaultReviewService(ReviewService):
    """Applies review decisions to candidate memories from ingestion."""

    def __init__(
        self,
        candidate_repository: CandidateMemoryRepository,
        memory_service: MemoryService,
        audit_repository: AuditRepository,
    ) -> None:
        self.candidate_repository = candidate_repository
        self.memory_service = memory_service
        self.audit_repository = audit_repository

    def list_candidates(self, source_id: str | None = None) -> list[CandidateMemory]:
        return self.candidate_repository.list_candidates(source_id)

    def apply_decision(
        self,
        candidate_id: str,
        decision: ReviewDecision,
        target_memory_id: str | None = None,
    ) -> MemoryRecord | None:
        candidate = self.candidate_repository.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate memory not found: {candidate_id}")

        if decision == ReviewDecision.ACCEPT:
            approved_memory = self._activate_candidate(candidate)
            stored_memory = self.memory_service.store_memory(approved_memory)
            self.candidate_repository.delete_candidate(candidate_id)
            self._record_audit_event(
                AuditEventType.REVIEW_ACCEPTED,
                candidate,
                {"candidate_id": candidate_id, "memory_id": stored_memory.memory_id},
            )
            return stored_memory

        if decision == ReviewDecision.MERGE:
            merge_target_id = target_memory_id or candidate.existing_memory_id
            if not merge_target_id:
                raise ValueError("MERGE decisions require a target_memory_id or existing_memory_id")

            target_memory = self.memory_service.get_memory(merge_target_id)
            if target_memory is None:
                raise KeyError(f"Target memory not found: {merge_target_id}")

            merged_updates = build_merged_memory_updates(target_memory, candidate.proposed_memory)
            merged_memory = self.memory_service.update_memory(merge_target_id, merged_updates)
            self.candidate_repository.delete_candidate(candidate_id)
            self._record_audit_event(
                AuditEventType.REVIEW_ACCEPTED,
                candidate,
                {
                    "candidate_id": candidate_id,
                    "memory_id": merged_memory.memory_id,
                    "decision": "merge",
                    "target_memory_id": merge_target_id,
                },
            )
            return merged_memory

        if decision == ReviewDecision.REJECT:
            self.candidate_repository.delete_candidate(candidate_id)
            self._record_audit_event(
                AuditEventType.REVIEW_REJECTED,
                candidate,
                {"candidate_id": candidate_id},
            )
            return None

        if decision == ReviewDecision.DEFER:
            self._record_audit_event(
                AuditEventType.REVIEW_REJECTED,
                candidate,
                {"candidate_id": candidate_id, "decision": "defer"},
            )
            return None

        raise ValueError(f"Unsupported review decision: {decision}")

    def _activate_candidate(self, candidate: CandidateMemory) -> MemoryRecord:
        proposed = candidate.proposed_memory
        metadata = dict(proposed.metadata)
        if candidate.source_chunk_ids:
            metadata["source_chunk_ids"] = list(candidate.source_chunk_ids)
        metadata["reviewed"] = True
        metadata["review_decision"] = "accept"

        return _model_copy(
            proposed,
            {
                "status": MemoryStatus.ACTIVE,
                "updated_at": datetime.utcnow(),
                "metadata": metadata,
            },
        )

    def _record_audit_event(
        self,
        event_type: AuditEventType,
        candidate: CandidateMemory,
        details: dict[str, Any],
    ) -> None:
        self.audit_repository.record_event(
            AuditEvent(
                event_id=f"audit_{uuid4().hex}",
                event_type=event_type,
                actor_id=None,
                memory_id=candidate.proposed_memory.memory_id,
                source_id=candidate.proposed_memory.source_id,
                session_id=None,
                timestamp=datetime.utcnow(),
                details=details,
            )
        )
