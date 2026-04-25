"""Citation assembly contracts."""

from __future__ import annotations

from typing import Protocol

from memory_core.domain.models import CitationReference, MemoryRecord


class CitationService(Protocol):
    """Builds citations and provenance chains for retrieved memories."""

    def build_citations(self, memory: MemoryRecord) -> list[CitationReference]:
        """Return citation references for the provided memory."""
