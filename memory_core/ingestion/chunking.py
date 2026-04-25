"""Chunking helpers for parsed document text."""

from __future__ import annotations

from uuid import uuid4

from memory_core.domain.models import ChunkReference
from memory_core.interfaces.parsing import Chunker


class TextChunker(Chunker):
    """Character-window chunker with overlap for retrieval-friendly segments."""

    def __init__(self, max_chars: int = 1200, overlap_chars: int = 200, min_chunk_chars: int = 250) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        if min_chunk_chars <= 0:
            raise ValueError("min_chunk_chars must be greater than zero")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, object] | None = None,
    ) -> list[ChunkReference]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        chunks: list[ChunkReference] = []
        start = 0
        sequence_index = 0
        text_length = len(normalized)

        while start < text_length:
            hard_end = min(start + self.max_chars, text_length)
            end = self._find_chunk_end(normalized, start, hard_end)
            chunk_text = normalized[start:end].strip()

            if chunk_text:
                chunks.append(
                    ChunkReference(
                        chunk_id=f"chunk_{uuid4().hex}",
                        document_id=document_id,
                        sequence_index=sequence_index,
                        text=chunk_text,
                        token_count=len(chunk_text.split()),
                        char_start=start,
                        char_end=end,
                        section_title=None,
                        page_number=None,
                    )
                )
                sequence_index += 1

            if end >= text_length:
                break

            start = max(end - self.overlap_chars, start + 1)

        return chunks

    def _find_chunk_end(self, text: str, start: int, hard_end: int) -> int:
        if hard_end == len(text):
            return hard_end

        search_start = min(start + self.min_chunk_chars, hard_end)
        candidate_positions = [
            text.rfind("\n\n", search_start, hard_end),
            text.rfind("\n", search_start, hard_end),
            text.rfind(". ", search_start, hard_end),
            text.rfind(" ", search_start, hard_end),
        ]
        end = max(candidate_positions)
        if end <= start:
            return hard_end
        return end + 1

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
