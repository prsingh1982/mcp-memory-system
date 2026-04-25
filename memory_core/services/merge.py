"""Helpers for deterministic memory merge behavior."""

from __future__ import annotations

from typing import Any

from memory_core.domain.models import MemoryRecord


def build_merged_memory_updates(target: MemoryRecord, incoming: MemoryRecord) -> dict[str, Any]:
    """Create deterministic merged updates for two related memory records."""
    merged_content = _merge_content(target.content, incoming.content)
    merged_summary = target.summary or incoming.summary
    merged_tags = sorted(set(target.tags).union(incoming.tags))

    merged_metadata = dict(target.metadata)
    merged_metadata.update(incoming.metadata)

    merged_source_chunk_ids = []
    for chunk_id in _extract_chunk_ids(target) + _extract_chunk_ids(incoming):
        if chunk_id not in merged_source_chunk_ids:
            merged_source_chunk_ids.append(chunk_id)
    if merged_source_chunk_ids:
        merged_metadata["source_chunk_ids"] = merged_source_chunk_ids

    merged_memory_ids = list(merged_metadata.get("merged_memory_ids", []))
    for memory_id in [target.memory_id, incoming.memory_id]:
        if memory_id not in merged_memory_ids:
            merged_memory_ids.append(memory_id)
    merged_metadata["merged_memory_ids"] = merged_memory_ids

    return {
        "content": merged_content,
        "summary": merged_summary,
        "confidence": max(target.confidence, incoming.confidence),
        "importance": max(target.importance, incoming.importance),
        "tags": merged_tags,
        "metadata": merged_metadata,
    }


def _merge_content(target_content: str, incoming_content: str) -> str:
    target_text = target_content.strip()
    incoming_text = incoming_content.strip()
    if not incoming_text:
        return target_text
    if not target_text:
        return incoming_text
    if incoming_text in target_text:
        return target_text
    if target_text in incoming_text:
        return incoming_text
    return f"{target_text}\n\n{incoming_text}"


def _extract_chunk_ids(memory: MemoryRecord) -> list[str]:
    metadata = memory.metadata or {}
    chunk_ids: list[str] = []
    raw_chunk_ids = metadata.get("source_chunk_ids")
    if isinstance(raw_chunk_ids, list):
        chunk_ids.extend(str(item) for item in raw_chunk_ids if str(item).strip())
    raw_chunk_id = metadata.get("chunk_id")
    if isinstance(raw_chunk_id, str) and raw_chunk_id.strip():
        chunk_ids.append(raw_chunk_id.strip())
    deduped: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id not in deduped:
            deduped.append(chunk_id)
    return deduped
