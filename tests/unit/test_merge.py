from datetime import datetime

from memory_core.domain.enums import MemoryStatus, MemoryType
from memory_core.domain.models import MemoryRecord
from memory_core.services.merge import build_merged_memory_updates


def _memory(memory_id: str, content: str, tags: list[str], metadata: dict):
    now = datetime.utcnow()
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=MemoryType.FACT,
        status=MemoryStatus.ACTIVE,
        content=content,
        summary=None,
        source_id="src-1",
        session_id=None,
        conversation_id=None,
        confidence=0.6,
        importance=0.7,
        tags=tags,
        version=1,
        parent_memory_id=None,
        supersedes_memory_id=None,
        embedding_ref=None,
        graph_node_ref=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
        metadata=metadata,
    )


def test_build_merged_memory_updates_combines_tags_and_chunk_ids():
    target = _memory("mem-target", "Alpha", ["a"], {"source_chunk_ids": ["c1"]})
    incoming = _memory("mem-incoming", "Beta", ["b"], {"chunk_id": "c2"})

    merged = build_merged_memory_updates(target, incoming)

    assert merged["content"] == "Alpha\n\nBeta"
    assert merged["tags"] == ["a", "b"]
    assert merged["metadata"]["source_chunk_ids"] == ["c1", "c2"]
    assert merged["metadata"]["merged_memory_ids"] == ["mem-target", "mem-incoming"]
