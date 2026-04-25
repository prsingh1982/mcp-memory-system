"""Sentence-transformers embedding provider for local semantic indexing."""

from __future__ import annotations

from typing import Any

from memory_core.interfaces.embeddings import EmbeddingProvider

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled at runtime
    SentenceTransformer = None


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider backed by a sentence-transformers model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._require_dependency()

        if not model_name:
            raise ValueError("model_name must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._model_name = model_name
        self._device = device
        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size
        self._model_kwargs = dict(model_kwargs or {})
        self._model = self._load_model()

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text input."""
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        vectors = self._encode([text])
        return vectors[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text inputs in a single batch call."""
        if not texts:
            return []
        if any(not text or not text.strip() for text in texts):
            raise ValueError("all texts must be non-empty")

        return self._encode(texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _load_model(self) -> SentenceTransformer:
        load_kwargs = dict(self._model_kwargs)
        if self._device is not None:
            load_kwargs["device"] = self._device
        return SentenceTransformer(self._model_name, **load_kwargs)

    @staticmethod
    def _require_dependency() -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbeddingProvider. "
                "Install sentence-transformers to enable local embeddings."
            )
