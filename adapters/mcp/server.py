"""MCP server wiring for the personal context memory system."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from memory_core.citations import DefaultCitationService
from memory_core.domain.enums import AuditEventType, MemoryStatus, MemoryType, ReviewDecision, SourceType
from memory_core.domain.models import MemoryRecord, RetrievalQuery, SourceReference
from memory_core.embeddings import SentenceTransformerEmbeddingProvider
from memory_core.ingestion import (
    DefaultIngestionService,
    DocxParser,
    EmailParser,
    MarkdownParser,
    ParserRegistry,
    PdfParser,
    PlainTextParser,
    TextChunker,
    WebPageParser,
)
from memory_core.llm import OllamaLLMClient
from memory_core.ranking import DefaultRankingService
from memory_core.retrieval import DefaultRetrievalService
from memory_core.session import DefaultSessionService
from memory_core.services import DefaultMemoryService, DefaultReviewService
from memory_core.storage import (
    FAISSVectorIndex,
    Neo4jGraphStore,
    SQLiteAuditRepository,
    SQLiteCandidateMemoryRepository,
    SQLiteChunkRepository,
    SQLiteDatabase,
    SQLiteJobRepository,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteSourceRepository,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - runtime dependency
    FastMCP = None


def _model_dump(model: Any) -> Any:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[call-arg]
    if hasattr(model, "json"):
        return json.loads(model.json())
    return model


@dataclass
class AppServices:
    """Concrete service bundle used by the MCP adapter."""

    ingestion_service: DefaultIngestionService
    retrieval_service: DefaultRetrievalService
    memory_service: DefaultMemoryService
    review_service: DefaultReviewService
    session_service: DefaultSessionService
    llm_client: OllamaLLMClient
    audit_repository: SQLiteAuditRepository
    job_repository: SQLiteJobRepository
    chunk_repository: SQLiteChunkRepository
    memory_repository: SQLiteMemoryRepository
    session_repository: SQLiteSessionRepository
    source_repository: SQLiteSourceRepository
    candidate_repository: SQLiteCandidateMemoryRepository
    graph_store: Neo4jGraphStore | None


def load_services_from_env(base_dir: str | Path | None = None) -> AppServices:
    """Bootstrap repositories, providers, and services from environment variables."""
    root_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    sqlite_path = Path(os.getenv("MEMORY_SQLITE_PATH", root_dir / "data" / "sqlite" / "memory.db"))
    faiss_dir = Path(os.getenv("MEMORY_FAISS_DIR", root_dir / "data" / "faiss"))

    sqlite_db = SQLiteDatabase(sqlite_path)
    source_repository = SQLiteSourceRepository(sqlite_db)
    chunk_repository = SQLiteChunkRepository(sqlite_db)
    candidate_repository = SQLiteCandidateMemoryRepository(sqlite_db)
    job_repository = SQLiteJobRepository(sqlite_db)
    audit_repository = SQLiteAuditRepository(sqlite_db)
    memory_repository = SQLiteMemoryRepository(sqlite_db)
    session_repository = SQLiteSessionRepository(sqlite_db)

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=os.getenv("MEMORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        device=os.getenv("MEMORY_EMBEDDING_DEVICE") or None,
        normalize_embeddings=True,
        batch_size=int(os.getenv("MEMORY_EMBEDDING_BATCH_SIZE", "32")),
    )
    vector_index = FAISSVectorIndex(storage_dir=faiss_dir)

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    graph_store = None
    if neo4j_uri and neo4j_user and neo4j_password:
        graph_store = Neo4jGraphStore(
            uri=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )

    llm_client = OllamaLLMClient(
        model_name=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api"),
        timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "1000")),
        keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
    )

    parser_registry = ParserRegistry(
        [
            PlainTextParser(),
            MarkdownParser(),
            WebPageParser(),
            EmailParser(),
            PdfParser(),
            DocxParser(),
        ]
    )
    chunker = TextChunker(
        max_chars=int(os.getenv("MEMORY_CHUNK_MAX_CHARS", "1200")),
        overlap_chars=int(os.getenv("MEMORY_CHUNK_OVERLAP_CHARS", "200")),
        min_chunk_chars=int(os.getenv("MEMORY_CHUNK_MIN_CHARS", "250")),
    )

    citation_service = DefaultCitationService(source_repository, chunk_repository)
    ranking_service = DefaultRankingService()
    memory_service = DefaultMemoryService(
        memory_repository,
        audit_repository,
        embedding_provider=embedding_provider,
        vector_index=vector_index,
        graph_store=graph_store,
    )
    review_service = DefaultReviewService(candidate_repository, memory_service, audit_repository)
    retrieval_service = DefaultRetrievalService(
        memory_repository,
        vector_index,
        embedding_provider,
        ranking_service,
        citation_service,
        session_repository=session_repository,
        graph_store=graph_store,
        audit_repository=audit_repository,
    )
    session_service = DefaultSessionService(
        session_repository,
        memory_service,
        retrieval_service,
        llm_client,
        audit_repository,
    )
    ingestion_service = DefaultIngestionService(
        source_repository,
        chunk_repository,
        candidate_repository,
        job_repository,
        audit_repository,
        parser_registry,
        chunker,
        llm_client=llm_client,
        memory_service=memory_service,
    )

    return AppServices(
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
        memory_service=memory_service,
        review_service=review_service,
        session_service=session_service,
        llm_client=llm_client,
        audit_repository=audit_repository,
        job_repository=job_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        session_repository=session_repository,
        source_repository=source_repository,
        candidate_repository=candidate_repository,
        graph_store=graph_store,
    )


def create_mcp_server(services: AppServices) -> Any:
    """Create a FastMCP server exposing memory tools and resources."""
    if FastMCP is None:
        raise ImportError('The "mcp" package is required to run the MCP server. Install "mcp[cli]".')

    mcp = FastMCP("Personal Context Memory", json_response=True)

    @mcp.tool()
    def ingest_source(
        source_type: str,
        title: str | None = None,
        file_path: str | None = None,
        raw_text: str | None = None,
        original_filename: str | None = None,
        mime_type: str | None = None,
        external_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an ingestion job for a new source document or text payload."""
        source_enum = SourceType(source_type)
        now = datetime.utcnow()
        source_metadata = dict(metadata or {})
        if raw_text is not None:
            source_metadata["raw_text"] = raw_text
        if external_uri is not None:
            source_metadata["external_uri"] = external_uri

        source = SourceReference(
            source_id=f"src_{uuid4().hex}",
            source_type=source_enum,
            title=title,
            file_path=file_path,
            original_filename=original_filename,
            mime_type=mime_type,
            checksum=None,
            external_uri=external_uri,
            created_at=now,
            metadata=source_metadata,
        )
        job = services.ingestion_service.ingest_source(source)
        return _model_dump(job)

    @mcp.tool()
    def process_ingestion_job(job_id: str) -> dict[str, Any]:
        """Run a previously created ingestion job through parsing and extraction."""
        return _model_dump(services.ingestion_service.process_job(job_id))

    @mcp.tool()
    def search_memory(
        query: str,
        top_k: int = 10,
        memory_types: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Search active memory using hybrid semantic and graph-aware retrieval."""
        retrieval_query = RetrievalQuery(
            query=query,
            session_id=session_id,
            conversation_id=conversation_id,
            memory_types=[MemoryType(value) for value in (memory_types or [])],
            tags=tags or [],
            top_k=top_k,
            include_deleted=include_deleted,
            require_citations=True,
        )
        return _model_dump(services.retrieval_service.retrieve(retrieval_query))

    @mcp.tool()
    def store_memory(
        memory_type: str,
        content: str,
        summary: str | None = None,
        source_id: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create and index an active canonical memory record."""
        now = datetime.utcnow()
        memory = MemoryRecord(
            memory_id=f"mem_{uuid4().hex}",
            memory_type=MemoryType(memory_type),
            status=MemoryStatus.ACTIVE,
            content=content,
            summary=summary,
            source_id=source_id,
            session_id=session_id,
            conversation_id=conversation_id,
            confidence=confidence,
            importance=importance,
            tags=tags or [],
            version=1,
            parent_memory_id=None,
            supersedes_memory_id=None,
            embedding_ref=None,
            graph_node_ref=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
            metadata=dict(metadata or {}),
        )
        return _model_dump(services.memory_service.store_memory(memory))

    @mcp.tool()
    def update_memory(memory_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update an existing canonical memory and resync indexes."""
        return _model_dump(services.memory_service.update_memory(memory_id, updates))

    @mcp.tool()
    def get_memory(memory_id: str) -> dict[str, Any] | None:
        """Fetch a canonical memory record by id."""
        memory = services.memory_service.get_memory(memory_id)
        return _model_dump(memory) if memory is not None else None

    @mcp.tool()
    def delete_memory(memory_id: str, reason: str) -> dict[str, Any]:
        """Soft-delete a memory while preserving audit and version history."""
        services.memory_service.delete_memory(memory_id, reason)
        return {"success": True, "memory_id": memory_id}

    @mcp.tool()
    def merge_memory(source_memory_id: str, target_memory_id: str) -> dict[str, Any]:
        """Merge one canonical memory into another and supersede the source."""
        return _model_dump(services.memory_service.merge_memory(source_memory_id, target_memory_id))

    @mcp.tool()
    def list_candidates(source_id: str | None = None) -> list[dict[str, Any]]:
        """List reviewable candidate memories extracted during ingestion."""
        return [_model_dump(candidate) for candidate in services.review_service.list_candidates(source_id)]

    @mcp.tool()
    def review_candidate(
        candidate_id: str,
        decision: str,
        target_memory_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Apply a review decision to a candidate memory."""
        result = services.review_service.apply_decision(
            candidate_id,
            ReviewDecision(decision),
            target_memory_id=target_memory_id,
        )
        return _model_dump(result) if result is not None else None

    @mcp.tool()
    def get_memory_history(memory_id: str) -> list[dict[str, Any]]:
        """Return all known versions of a memory lineage."""
        return [_model_dump(memory) for memory in services.memory_service.get_memory_history(memory_id)]

    @mcp.tool()
    def summarize_session(session_id: str) -> dict[str, Any] | None:
        """Refresh the rolling session summary and promote durable cross-session memories."""
        result = services.session_service.summarize_session(session_id)
        return _model_dump(result) if result is not None else None

    @mcp.tool()
    def list_audit_events(
        event_type: str | None = None,
        memory_id: str | None = None,
        source_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List audit events for ingestion, retrieval, review, and lifecycle actions."""
        audit_event_type = AuditEventType(event_type) if event_type else None
        events = services.audit_repository.list_events(
            event_type=audit_event_type,
            memory_id=memory_id,
            source_id=source_id,
            limit=limit,
        )
        return [_model_dump(event) for event in events]

    @mcp.resource("memory://{memory_id}")
    def memory_resource(memory_id: str) -> str:
        """Return a serialized memory record by id."""
        memory = services.memory_service.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory not found: {memory_id}")
        return json.dumps(_model_dump(memory), ensure_ascii=True, indent=2, sort_keys=True)

    @mcp.resource("audit://recent")
    def recent_audit_resource() -> str:
        """Return the most recent audit events."""
        events = services.audit_repository.list_events(limit=25)
        return json.dumps([_model_dump(event) for event in events], ensure_ascii=True, indent=2, sort_keys=True)

    return mcp
