"""FAISS vector index adapter with local persistence and metadata mapping."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from memory_core.interfaces.graph import VectorIndex

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled at runtime
    faiss = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None


class FAISSVectorIndex(VectorIndex):
    """Persistent FAISS index using cosine-style similarity via normalized inner product."""

    def __init__(self, storage_dir: str | Path, index_name: str = "memory_index") -> None:
        self._require_dependencies()

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_name = index_name
        self.index_path = self.storage_dir / f"{index_name}.faiss"
        self.metadata_path = self.storage_dir / f"{index_name}.meta.json"

        self._lock = threading.RLock()
        self._dimension: int | None = None
        self._next_faiss_id = 1
        self._faiss_id_by_vector_id: dict[str, int] = {}
        self._vector_id_by_faiss_id: dict[int, str] = {}
        self._metadata_by_vector_id: dict[str, dict[str, Any]] = {}
        self._index = None

        self._load()

    def upsert(self, vector_id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        if not vector_id:
            raise ValueError("vector_id must be non-empty")

        normalized_embedding = self._prepare_embedding(embedding)

        with self._lock:
            self._ensure_index_locked(normalized_embedding.shape[1])

            faiss_id = self._faiss_id_by_vector_id.get(vector_id)
            if faiss_id is None:
                faiss_id = self._next_faiss_id
                self._next_faiss_id += 1
            else:
                self._index.remove_ids(np.asarray([faiss_id], dtype="int64"))

            self._index.add_with_ids(
                normalized_embedding,
                np.asarray([faiss_id], dtype="int64"),
            )

            self._faiss_id_by_vector_id[vector_id] = faiss_id
            self._vector_id_by_faiss_id[faiss_id] = vector_id
            self._metadata_by_vector_id[vector_id] = dict(metadata)
            self._persist_locked()

    def delete(self, vector_id: str) -> None:
        with self._lock:
            faiss_id = self._faiss_id_by_vector_id.pop(vector_id, None)
            if faiss_id is None:
                return

            self._index.remove_ids(np.asarray([faiss_id], dtype="int64"))
            self._vector_id_by_faiss_id.pop(faiss_id, None)
            self._metadata_by_vector_id.pop(vector_id, None)
            self._persist_locked()

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        prepared_query = self._prepare_embedding(query_embedding)

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            if self._dimension != prepared_query.shape[1]:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self._dimension}, "
                    f"received {prepared_query.shape[1]}"
                )

            search_k = min(top_k, self._index.ntotal)
            distances, ids = self._index.search(prepared_query, search_k)

            results: list[dict[str, Any]] = []
            for score, raw_id in zip(distances[0], ids[0], strict=False):
                faiss_id = int(raw_id)
                if faiss_id < 0:
                    continue

                vector_id = self._vector_id_by_faiss_id.get(faiss_id)
                if vector_id is None:
                    continue

                results.append(
                    {
                        "vector_id": vector_id,
                        "score": float(score),
                        "metadata": dict(self._metadata_by_vector_id.get(vector_id, {})),
                    }
                )

        return results

    def _load(self) -> None:
        index_exists = self.index_path.exists()
        metadata_exists = self.metadata_path.exists()

        if not index_exists and not metadata_exists:
            return

        if index_exists != metadata_exists:
            raise ValueError(
                "FAISS index and metadata files are out of sync. "
                f"Expected both or neither: {self.index_path}, {self.metadata_path}"
            )

        with self._lock:
            self._index = faiss.read_index(str(self.index_path))
            manifest = json.loads(self.metadata_path.read_text(encoding="utf-8"))

            self._dimension = manifest.get("dimension")
            self._next_faiss_id = manifest.get("next_faiss_id", 1)
            self._faiss_id_by_vector_id = {}
            self._vector_id_by_faiss_id = {}
            self._metadata_by_vector_id = {}

            for entry in manifest.get("entries", []):
                vector_id = entry["vector_id"]
                faiss_id = int(entry["faiss_id"])
                self._faiss_id_by_vector_id[vector_id] = faiss_id
                self._vector_id_by_faiss_id[faiss_id] = vector_id
                self._metadata_by_vector_id[vector_id] = dict(entry.get("metadata", {}))

            if self._dimension is not None and getattr(self._index, "d", None) != self._dimension:
                raise ValueError(
                    f"Loaded FAISS index dimension {getattr(self._index, 'd', None)} "
                    f"does not match metadata dimension {self._dimension}"
                )

    def _persist_locked(self) -> None:
        if self._index is None:
            return

        faiss.write_index(self._index, str(self.index_path))
        manifest = {
            "dimension": self._dimension,
            "next_faiss_id": self._next_faiss_id,
            "entries": [
                {
                    "vector_id": vector_id,
                    "faiss_id": faiss_id,
                    "metadata": self._metadata_by_vector_id.get(vector_id, {}),
                }
                for vector_id, faiss_id in sorted(self._faiss_id_by_vector_id.items(), key=lambda item: item[1])
            ],
        }
        self._atomic_write_json(self.metadata_path, manifest)

    def _ensure_index_locked(self, dimension: int) -> None:
        if self._index is None:
            self._dimension = dimension
            base_index = faiss.IndexFlatIP(dimension)
            self._index = faiss.IndexIDMap2(base_index)
            return

        if self._dimension != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, received {dimension}"
            )

    def _prepare_embedding(self, embedding: list[float]) -> Any:
        if not embedding:
            raise ValueError("embedding must contain at least one value")

        vector = np.asarray(embedding, dtype="float32")
        if vector.ndim != 1:
            raise ValueError("embedding must be a one-dimensional vector")

        reshaped = vector.reshape(1, -1)
        faiss.normalize_L2(reshaped)
        return reshaped

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, path)

    @staticmethod
    def _require_dependencies() -> None:
        if faiss is None:
            raise ImportError(
                "faiss is required for FAISSVectorIndex. Install faiss-cpu or a compatible FAISS package."
            )
        if np is None:
            raise ImportError("numpy is required for FAISSVectorIndex.")
