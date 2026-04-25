"""SQLite schema definitions for canonical metadata storage."""

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        source_type TEXT NOT NULL,
        title TEXT,
        file_path TEXT,
        original_filename TEXT,
        mime_type TEXT,
        checksum TEXT,
        external_uri TEXT,
        created_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        memory_id TEXT PRIMARY KEY,
        memory_type TEXT NOT NULL,
        status TEXT NOT NULL,
        content TEXT NOT NULL,
        summary TEXT,
        source_id TEXT,
        session_id TEXT,
        conversation_id TEXT,
        confidence REAL NOT NULL DEFAULT 0.0,
        importance REAL NOT NULL DEFAULT 0.0,
        tags_json TEXT NOT NULL DEFAULT '[]',
        version INTEGER NOT NULL DEFAULT 1,
        parent_memory_id TEXT,
        supersedes_memory_id TEXT,
        embedding_ref TEXT,
        graph_node_ref TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (source_id) REFERENCES sources(source_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        sequence_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER,
        char_start INTEGER,
        char_end INTEGER,
        section_title TEXT,
        page_number INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_versions (
        version_id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        snapshot_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        UNIQUE(memory_id, version),
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        job_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error_message TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (source_id) REFERENCES sources(source_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_memories (
        candidate_id TEXT PRIMARY KEY,
        source_id TEXT,
        proposed_memory_json TEXT NOT NULL,
        extraction_reason TEXT NOT NULL,
        source_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
        confidence REAL NOT NULL,
        suggested_action TEXT NOT NULL,
        existing_memory_id TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (source_id) REFERENCES sources(source_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        event_id TEXT PRIMARY KEY,
        event_type TEXT NOT NULL,
        actor_id TEXT,
        memory_id TEXT,
        source_id TEXT,
        session_id TEXT,
        timestamp TEXT NOT NULL,
        details_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id),
        FOREIGN KEY (source_id) REFERENCES sources(source_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_state (
        session_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        rolling_summary TEXT,
        last_active_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_messages (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (session_id) REFERENCES session_state(session_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type)",
    "CREATE INDEX IF NOT EXISTS idx_memories_type_status ON memories(memory_type, status)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_document_sequence ON document_chunks(document_id, sequence_index)",
    "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_versions_memory ON memory_versions(memory_id, version)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_source ON ingestion_jobs(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_candidate_memories_source_created ON candidate_memories(source_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_audit_memory ON audit_events(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_source ON audit_events(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_session_messages_session_created ON session_messages(session_id, created_at)",
]
