from datetime import datetime, timedelta

from memory_core.domain.enums import MemoryStatus, MemoryType
from memory_core.domain.models import MemoryRecord, RetrievalQuery
from memory_core.ranking import DefaultRankingService


def _memory(memory_id: str, *, semantic_score: float, importance: float, updated_at: datetime) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        memory_type=MemoryType.FACT,
        status=MemoryStatus.ACTIVE,
        content="Example memory",
        summary=None,
        source_id="src-1",
        session_id=None,
        conversation_id=None,
        confidence=0.7,
        importance=importance,
        tags=[],
        version=1,
        parent_memory_id=None,
        supersedes_memory_id=None,
        embedding_ref=None,
        graph_node_ref=None,
        created_at=updated_at,
        updated_at=updated_at,
        deleted_at=None,
        metadata={"semantic_score": semantic_score},
    )


def test_ranking_prefers_more_recent_memory_when_other_signals_equal():
    service = DefaultRankingService()
    query = RetrievalQuery(query="example")
    recent = _memory(
        "mem-recent",
        semantic_score=0.8,
        importance=0.7,
        updated_at=datetime.utcnow(),
    )
    older = _memory(
        "mem-old",
        semantic_score=0.8,
        importance=0.7,
        updated_at=datetime.utcnow() - timedelta(days=180),
    )

    recent_score = service.score(query, recent)
    older_score = service.score(query, older)

    assert recent_score.final_score > older_score.final_score
