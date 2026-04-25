"""Embedding provider contracts."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Abstraction over local embedding model backends."""

    def embed_text(self, text: str) -> list[float]:
        """Return an embedding vector for a single text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""

    @property
    def model_name(self) -> str:
        """Return the provider's underlying embedding model name."""
