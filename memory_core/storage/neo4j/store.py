"""Neo4j graph store adapter for memory relationships and provenance."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from memory_core.domain.models import MemoryRecord
from memory_core.interfaces.graph import GraphStore

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - handled at runtime
    GraphDatabase = None


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed graph adapter for canonical memory nodes and relationships."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        database: str = "neo4j",
        initialize: bool = True,
        driver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._require_dependency()

        if not uri:
            raise ValueError("uri must be non-empty")
        if not username:
            raise ValueError("username must be non-empty")
        if not password:
            raise ValueError("password must be non-empty")

        self._database = database
        self._driver_kwargs = dict(driver_kwargs or {})
        self._driver = GraphDatabase.driver(uri, auth=(username, password), **self._driver_kwargs)

        if initialize:
            self.initialize()

    def initialize(self) -> None:
        """Create constraints used by the memory graph."""
        query = """
        CREATE CONSTRAINT memory_memory_id_unique IF NOT EXISTS
        FOR (node:Memory)
        REQUIRE node.memory_id IS UNIQUE
        """
        self._run_write(query, {})

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def upsert_memory_node(self, memory: MemoryRecord) -> None:
        query = """
        MERGE (memory:Memory {memory_id: $memory_id})
        SET
            memory.memory_type = $memory_type,
            memory.status = $status,
            memory.content = $content,
            memory.summary = $summary,
            memory.source_id = $source_id,
            memory.session_id = $session_id,
            memory.conversation_id = $conversation_id,
            memory.confidence = $confidence,
            memory.importance = $importance,
            memory.tags = $tags,
            memory.version = $version,
            memory.parent_memory_id = $parent_memory_id,
            memory.supersedes_memory_id = $supersedes_memory_id,
            memory.embedding_ref = $embedding_ref,
            memory.graph_node_ref = $graph_node_ref,
            memory.created_at = $created_at,
            memory.updated_at = $updated_at,
            memory.deleted_at = $deleted_at,
            memory.metadata_json = $metadata_json
        """
        self._run_write(query, self._memory_params(memory))

    def create_relationship(
        self,
        from_node_id: str,
        relation_type: str,
        to_node_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if not from_node_id:
            raise ValueError("from_node_id must be non-empty")
        if not to_node_id:
            raise ValueError("to_node_id must be non-empty")

        safe_relation_type = self._sanitize_relation_type(relation_type)
        query = f"""
        MATCH (source:Memory {{memory_id: $from_node_id}})
        MATCH (target:Memory {{memory_id: $to_node_id}})
        MERGE (source)-[rel:{safe_relation_type}]->(target)
        SET rel += $properties
        """
        self._run_write(
            query,
            {
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "properties": self._serialize_properties(properties or {}),
            },
        )

    def get_related_nodes(self, node_id: str, depth: int = 1) -> list[dict[str, Any]]:
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if depth <= 0:
            raise ValueError("depth must be greater than zero")

        query = f"""
        MATCH path = (source:Memory {{memory_id: $node_id}})-[*1..{depth}]-(related)
        WHERE related.memory_id <> source.memory_id
        RETURN DISTINCT
            related.memory_id AS node_id,
            labels(related) AS labels,
            properties(related) AS properties
        """
        records = self._run_read(query, {"node_id": node_id})

        return [
            {
                "node_id": record["node_id"],
                "labels": list(record["labels"]),
                "properties": self._deserialize_properties(dict(record["properties"])),
            }
            for record in records
        ]

    def mark_deleted(self, node_id: str, deleted_at: datetime) -> None:
        if not node_id:
            raise ValueError("node_id must be non-empty")

        query = """
        MATCH (memory:Memory {memory_id: $node_id})
        SET
            memory.status = $status,
            memory.deleted_at = $deleted_at,
            memory.updated_at = $updated_at
        """
        timestamp = self._serialize_datetime(deleted_at)
        self._run_write(
            query,
            {
                "node_id": node_id,
                "status": "deleted",
                "deleted_at": timestamp,
                "updated_at": timestamp,
            },
        )

    def _run_write(self, query: str, parameters: dict[str, Any]) -> None:
        with self._driver.session(database=self._database) as session:
            session.execute_write(lambda tx: tx.run(query, **parameters).consume())

    def _run_read(self, query: str, parameters: dict[str, Any]) -> list[Any]:
        with self._driver.session(database=self._database) as session:
            return session.execute_read(lambda tx: list(tx.run(query, **parameters)))

    def _memory_params(self, memory: MemoryRecord) -> dict[str, Any]:
        return {
            "memory_id": memory.memory_id,
            "memory_type": memory.memory_type.value,
            "status": memory.status.value,
            "content": memory.content,
            "summary": memory.summary,
            "source_id": memory.source_id,
            "session_id": memory.session_id,
            "conversation_id": memory.conversation_id,
            "confidence": memory.confidence,
            "importance": memory.importance,
            "tags": list(memory.tags),
            "version": memory.version,
            "parent_memory_id": memory.parent_memory_id,
            "supersedes_memory_id": memory.supersedes_memory_id,
            "embedding_ref": memory.embedding_ref,
            "graph_node_ref": memory.graph_node_ref,
            "created_at": self._serialize_datetime(memory.created_at),
            "updated_at": self._serialize_datetime(memory.updated_at),
            "deleted_at": self._serialize_datetime(memory.deleted_at),
            "metadata_json": json.dumps(memory.metadata, ensure_ascii=True, sort_keys=True),
        }

    @staticmethod
    def _serialize_datetime(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    @staticmethod
    def _serialize_properties(properties: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in properties.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                serialized[key] = value
            else:
                serialized[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
        return serialized

    @classmethod
    def _deserialize_properties(cls, properties: dict[str, Any]) -> dict[str, Any]:
        deserialized = dict(properties)
        metadata_json = deserialized.get("metadata_json")
        if isinstance(metadata_json, str):
            try:
                deserialized["metadata"] = json.loads(metadata_json)
            except json.JSONDecodeError:
                pass
        return deserialized

    @staticmethod
    def _sanitize_relation_type(relation_type: str) -> str:
        safe_relation_type = re.sub(r"[^A-Z0-9_]", "_", relation_type.upper())
        safe_relation_type = re.sub(r"_+", "_", safe_relation_type).strip("_")
        if not safe_relation_type:
            raise ValueError("relation_type must contain at least one alphanumeric character")
        return safe_relation_type

    @staticmethod
    def _require_dependency() -> None:
        if GraphDatabase is None:
            raise ImportError(
                "neo4j is required for Neo4jGraphStore. Install the neo4j Python driver to enable graph storage."
            )
