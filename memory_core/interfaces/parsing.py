"""Document parsing and chunking contracts."""

from __future__ import annotations

from typing import Any, Protocol

from memory_core.domain.enums import SourceType
from memory_core.domain.models import ChunkReference, SourceReference


class DocumentParser(Protocol):
    """Converts a source into normalized text for downstream ingestion."""

    def supports(self, source_type: SourceType) -> bool:
        """Return whether the parser can handle the source type."""

    def parse(self, source: SourceReference) -> str:
        """Parse the source and return normalized text."""


class Chunker(Protocol):
    """Splits normalized text into retrieval-friendly chunks."""

    def chunk(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkReference]:
        """Return ordered chunk references for the parsed document."""
