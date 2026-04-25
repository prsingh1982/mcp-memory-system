"""LLM-facing contracts used by orchestration services."""

from __future__ import annotations

from typing import Any, Protocol

from memory_core.domain.enums import SourceType


class LLMClient(Protocol):
    """Abstraction for text generation, summarization, and extraction."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return a generated text response for the given prompt."""

    def summarize(self, text: str, context: str | None = None) -> str:
        """Return a concise summary of the provided text."""

    def extract_structured_memory(self, text: str, source_type: SourceType) -> list[dict[str, Any]]:
        """Extract structured memory candidates from raw text."""
