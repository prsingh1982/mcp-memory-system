"""Two-stage retrieval implementation with vector shortlist and graph expansion."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from memory_core.domain.enums import AuditEventType, MemoryStatus, MemoryType
from memory_core.domain.models import AuditEvent, MemoryRecord, RetrievalQuery, RetrievalResult, RetrievedMemory, ScoreBreakdown, SessionSnapshot
from memory_core.interfaces.citations import CitationService
from memory_core.interfaces.embeddings import EmbeddingProvider
from memory_core.interfaces.graph import GraphStore, VectorIndex
from memory_core.interfaces.retrieval import RankingService, RetrievalService
from memory_core.interfaces.storage import AuditRepository, MemoryRepository, SessionRepository

ModelT = TypeVar("ModelT", bound=BaseModel)


def _model_copy(model: ModelT, update: dict[str, Any]) -> ModelT:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)  # type: ignore[attr-defined]
    return model.copy(update=update)


class DefaultRetrievalService(RetrievalService):
    """Coordinates semantic retrieval, graph expansion, reranking, and citations."""

    def __init__(
        self,
        memory_repository: MemoryRepository,
        vector_index: VectorIndex,
        embedding_provider: EmbeddingProvider,
        ranking_service: RankingService,
        citation_service: CitationService,
        *,
        session_repository: SessionRepository | None = None,
        graph_store: GraphStore | None = None,
        audit_repository: AuditRepository | None = None,
        shortlist_multiplier: int = 3,
        graph_depth: int = 1,
    ) -> None:
        self.memory_repository = memory_repository
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.ranking_service = ranking_service
        self.citation_service = citation_service
        self.session_repository = session_repository
        self.graph_store = graph_store
        self.audit_repository = audit_repository
        self.shortlist_multiplier = max(shortlist_multiplier, 1)
        self.graph_depth = max(graph_depth, 1)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        session = self._get_session_snapshot(query.session_id)
        shortlist = self.semantic_shortlist(query)
        expanded = self.expand_with_graph_context(shortlist)
        reranked = self.rerank(query, expanded)

        final_items = reranked[: query.top_k]
        if query.require_citations:
            final_items = [self._attach_citations(item) for item in final_items]

        audit_event_id = self._record_audit_event(
            event_type=AuditEventType.RETRIEVAL_EXECUTED,
            query=query,
            details={"result_count": len(final_items)},
        )
        self._record_audit_event(
            event_type=AuditEventType.RANKING_APPLIED,
            query=query,
            details={"reranked_count": len(reranked)},
        )

        applied_preferences = [
            item.memory
            for item in final_items
            if item.memory.memory_type in {MemoryType.PREFERENCE, MemoryType.WORKFLOW_RULE}
        ]

        return RetrievalResult(
            query=query.query,
            items=final_items,
            applied_preferences=applied_preferences,
            session_summary=session.rolling_summary if session else None,
            audit_event_id=audit_event_id,
        )

    def semantic_shortlist(self, query: RetrievalQuery) -> list[RetrievedMemory]:
        query_embedding = self.embedding_provider.embed_text(query.query)
        vector_hits = self.vector_index.search(query_embedding, max(query.top_k * self.shortlist_multiplier, query.top_k))

        best_by_memory_id: dict[str, RetrievedMemory] = {}
        for hit in vector_hits:
            vector_id = str(hit.get("vector_id", ""))
            metadata = hit.get("metadata") or {}
            memory_id = str(metadata.get("memory_id") or vector_id)
            memory = self.memory_repository.get_memory(memory_id)
            if memory is None:
                continue
            if not query.include_deleted and memory.status == MemoryStatus.DELETED:
                continue
            if query.memory_types and memory.memory_type not in query.memory_types:
                continue
            if query.tags and not set(query.tags).intersection(memory.tags):
                continue

            semantic_score = self._clamp(float(hit.get("score", 0.0)))
            matched_chunk_ids = self._resolve_matched_chunk_ids(memory, metadata)
            memory_with_signals = self._with_signals(
                memory,
                semantic_score=semantic_score,
                graph_score=float(memory.metadata.get("graph_score", 0.0)) if memory.metadata else 0.0,
                source_chunk_ids=matched_chunk_ids or None,
            )
            candidate = RetrievedMemory(
                memory=memory_with_signals,
                score=ScoreBreakdown(semantic_score=semantic_score, final_score=semantic_score),
                citations=[],
                matched_chunk_ids=matched_chunk_ids,
                reasoning="Matched via vector similarity search.",
            )

            existing = best_by_memory_id.get(memory.memory_id)
            if existing is None or candidate.score.semantic_score > existing.score.semantic_score:
                best_by_memory_id[memory.memory_id] = candidate

        return list(best_by_memory_id.values())

    def expand_with_graph_context(self, items: list[RetrievedMemory]) -> list[RetrievedMemory]:
        if self.graph_store is None or not items:
            return items

        expanded: dict[str, RetrievedMemory] = {item.memory.memory_id: item for item in items}
        for item in items:
            related_nodes = self.graph_store.get_related_nodes(item.memory.memory_id, depth=self.graph_depth)
            for related in related_nodes:
                related_id = str(related.get("node_id", ""))
                if not related_id:
                    continue
                memory = self.memory_repository.get_memory(related_id)
                if memory is None or memory.status == MemoryStatus.DELETED:
                    continue

                base_graph_score = 0.65 if related_id not in expanded else max(
                    expanded[related_id].score.graph_score,
                    0.65,
                )
                memory_with_signals = self._with_signals(
                    memory,
                    semantic_score=float(memory.metadata.get("semantic_score", 0.0)) if memory.metadata else 0.0,
                    graph_score=base_graph_score,
                )

                if related_id in expanded:
                    existing = expanded[related_id]
                    expanded[related_id] = _model_copy(
                        existing,
                        {
                            "memory": memory_with_signals,
                            "score": _model_copy(
                                existing.score,
                                {
                                    "graph_score": max(existing.score.graph_score, base_graph_score),
                                    "final_score": max(existing.score.final_score, base_graph_score),
                                },
                            ),
                        },
                    )
                else:
                    expanded[related_id] = RetrievedMemory(
                        memory=memory_with_signals,
                        score=ScoreBreakdown(graph_score=base_graph_score, final_score=base_graph_score),
                        citations=[],
                        matched_chunk_ids=[],
                        reasoning=f"Expanded from graph neighborhood of {item.memory.memory_id}.",
                    )

        return list(expanded.values())

    def rerank(self, query: RetrievalQuery, items: list[RetrievedMemory]) -> list[RetrievedMemory]:
        session = self._get_session_snapshot(query.session_id)
        reranked: list[RetrievedMemory] = []
        for item in items:
            score = self.ranking_service.score(query, item.memory, session=session)
            reranked.append(_model_copy(item, {"score": score}))

        reranked.sort(key=lambda item: item.score.final_score, reverse=True)
        return reranked

    def _attach_citations(self, item: RetrievedMemory) -> RetrievedMemory:
        citations = self.citation_service.build_citations(item.memory)
        return _model_copy(item, {"citations": citations})

    def _resolve_matched_chunk_ids(self, memory: MemoryRecord, metadata: dict[str, Any]) -> list[str]:
        chunk_ids: list[str] = []
        raw_chunk_id = metadata.get("chunk_id")
        if isinstance(raw_chunk_id, str) and raw_chunk_id.strip():
            chunk_ids.append(raw_chunk_id.strip())

        raw_chunk_ids = metadata.get("source_chunk_ids")
        if isinstance(raw_chunk_ids, list):
            chunk_ids.extend(str(value) for value in raw_chunk_ids if str(value).strip())

        source_chunk_ids = memory.metadata.get("source_chunk_ids") if memory.metadata else None
        if isinstance(source_chunk_ids, list):
            chunk_ids.extend(str(value) for value in source_chunk_ids if str(value).strip())

        if memory.memory_type == MemoryType.DOCUMENT_CHUNK and memory.memory_id not in chunk_ids:
            chunk_ids.append(memory.memory_id)

        seen: set[str] = set()
        unique_chunk_ids: list[str] = []
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunk_ids.append(chunk_id)
        return unique_chunk_ids

    def _with_signals(
        self,
        memory: MemoryRecord,
        *,
        semantic_score: float | None = None,
        graph_score: float | None = None,
        source_chunk_ids: list[str] | None = None,
    ) -> MemoryRecord:
        metadata = dict(memory.metadata)
        if semantic_score is not None:
            metadata["semantic_score"] = self._clamp(semantic_score)
        if graph_score is not None:
            metadata["graph_score"] = self._clamp(graph_score)
        if source_chunk_ids:
            metadata["source_chunk_ids"] = source_chunk_ids
        return _model_copy(memory, {"metadata": metadata})

    def _get_session_snapshot(self, session_id: str | None) -> SessionSnapshot | None:
        if session_id is None or self.session_repository is None:
            return None
        return self.session_repository.get_session(session_id)

    def _record_audit_event(
        self,
        *,
        event_type: AuditEventType,
        query: RetrievalQuery,
        details: dict[str, Any],
    ) -> str | None:
        if self.audit_repository is None:
            return None

        event = AuditEvent(
            event_id=f"audit_{uuid4().hex}",
            event_type=event_type,
            actor_id=None,
            memory_id=None,
            source_id=None,
            session_id=query.session_id,
            timestamp=datetime.utcnow(),
            details={"query": query.query, **details},
        )
        self.audit_repository.record_event(event)
        return event.event_id

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
