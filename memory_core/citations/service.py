"""Citation assembly for retrieved memories."""

from __future__ import annotations

from uuid import uuid4

from memory_core.domain.enums import MemoryType
from memory_core.domain.models import CitationReference, MemoryRecord
from memory_core.interfaces.citations import CitationService
from memory_core.interfaces.storage import ChunkRepository, SourceRepository


class DefaultCitationService(CitationService):
    """Builds source- and chunk-aware citations for a memory record."""

    def __init__(self, source_repository: SourceRepository, chunk_repository: ChunkRepository) -> None:
        self.source_repository = source_repository
        self.chunk_repository = chunk_repository

    def build_citations(self, memory: MemoryRecord) -> list[CitationReference]:
        source_id = memory.source_id
        if not source_id:
            return []

        chunk_ids = self._resolve_chunk_ids(memory)
        citations: list[CitationReference] = []
        for chunk_id in chunk_ids:
            chunk = self.chunk_repository.get_chunk(chunk_id)
            if chunk is None:
                continue
            citations.append(
                CitationReference(
                    citation_id=f"cit_{uuid4().hex}",
                    source_id=source_id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    quote=(chunk.text[:240] + "...") if len(chunk.text) > 240 else chunk.text,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    provenance_chain=[source_id, chunk.document_id, chunk.chunk_id],
                )
            )

        if citations:
            return citations

        source = self.source_repository.get_source(source_id)
        title = source.title if source else None
        return [
            CitationReference(
                citation_id=f"cit_{uuid4().hex}",
                source_id=source_id,
                document_id=source_id,
                chunk_id=None,
                quote=title,
                section_title=None,
                page_number=None,
                provenance_chain=[source_id],
            )
        ]

    @staticmethod
    def _resolve_chunk_ids(memory: MemoryRecord) -> list[str]:
        metadata = memory.metadata or {}
        chunk_ids: list[str] = []

        raw_chunk_ids = metadata.get("source_chunk_ids")
        if isinstance(raw_chunk_ids, list):
            chunk_ids.extend(str(item) for item in raw_chunk_ids if str(item).strip())

        raw_chunk_id = metadata.get("chunk_id")
        if isinstance(raw_chunk_id, str) and raw_chunk_id.strip():
            chunk_ids.append(raw_chunk_id.strip())

        if memory.memory_type == MemoryType.DOCUMENT_CHUNK and memory.memory_id not in chunk_ids:
            chunk_ids.append(memory.memory_id)

        seen: set[str] = set()
        deduped: list[str] = []
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(chunk_id)
        return deduped
