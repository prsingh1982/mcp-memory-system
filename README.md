# Personal Context MCP Memory System

Offline, local-first document memory system built around:

- Python 3.14
- Ollama for generation and extraction
- sentence-transformers for local embeddings
- FAISS for vector retrieval
- Neo4j for graph relationships
- SQLite for canonical metadata, jobs, sessions, and audit history
- Streamlit for the user interface
- MCP for tool-style access

## Status

The core architecture is implemented:

- domain models and contracts
- SQLite repositories
- FAISS vector adapter
- Neo4j graph adapter
- ingestion pipeline
- retrieval, ranking, and citations
- Ollama client
- review and lifecycle services
- MCP server
- Streamlit UI

## Python Version

As of April 25, 2026, the latest stable Python release is `3.14.4`, released on April 7, 2026:

- [Python.org latest release page](https://www.python.org/downloads/latest/)
- [Python source releases](https://test.python.org/downloads/source/)

This project targets the Python `3.14` series.

## Project Layout

```text
memory_core/
  domain/        Core models and enums
  interfaces/    Contracts for services and adapters
  storage/       SQLite, FAISS, and Neo4j adapters
  embeddings/    Local embedding providers
  llm/           Ollama client
  ingestion/     Parsing, chunking, extraction flow
  ranking/       Fixed hybrid scoring
  retrieval/     Vector shortlist + graph expansion + reranking
  citations/     Citation assembly
  services/      Lifecycle and review orchestration
adapters/
  mcp/           FastMCP server wiring
  streamlit_ui/  Streamlit app
tests/
  unit/          Fast unit tests
```

## Setup

1. Install Python `3.14.x`.
2. Create and activate a virtual environment.
3. Install dependencies:

```powershell
python -m pip install -e .[dev]
```

4. Copy the env template if needed:

```powershell
Copy-Item .env.example .env
```

5. Start Ollama and make sure the model is available:

```powershell
ollama pull llama3.1
```

6. If using Neo4j, start it locally and update the credentials in `.env`.

## Run The Streamlit UI

```powershell
streamlit run app.py
```

## Run The MCP Server

```powershell
python mcp_server.py
```

Or after editable install:

```powershell
personal-context-mcp-server
```

## Run Tests

```powershell
pytest
```

## Notes

- The system is fully local-first by design, but you may use online LLMs hosted by OpenAI or OpenRouter or any other platform by modifying the .env file, as long as the LLM conforms to OpenAI API.
- FAISS is currently kept as the vector backend because `faiss-cpu` publishes Windows wheels for CPython 3.14.
- The Streamlit chat uses retrieval-grounded local generation through Ollama.
- Candidate memories remain reviewable before becoming canonical active memory.


## Technical Architecture

```mermaid
flowchart TB
    User["User"]
    Streamlit["Streamlit UI
    app.py
    adapters/streamlit_ui/app.py
    Purpose: chat, upload, memory browser, admin/audit"]
    MCP["MCP Server
    mcp_server.py
    adapters/mcp/server.py
    Purpose: expose memory operations as MCP tools/resources"]

    User --> Streamlit
    ExternalClient["MCP Client / Agent"] --> MCP

    subgraph Core["Application Core"]
        Ingestion["Ingestion Service
        memory_core/ingestion/service.py
        Purpose: ingest source, parse, chunk, auto-index, extract candidates"]
        Retrieval["Retrieval Service
        memory_core/retrieval/service.py
        Purpose: FAISS search, keyword fallback, profile-memory inclusion, graph expansion, reranking, citations"]
        Memory["Memory Service
        memory_core/services/lifecycle.py
        Purpose: canonical memory CRUD, indexing sync, merge, delete, reindex"]
        Review["Review Service
        memory_core/services/review.py
        Purpose: accept/merge/reject candidate memories"]
        Session["Session Service
        memory_core/session/service.py
        Purpose: session history, rolling summaries, cross-session durable chat memory promotion"]
        Ranking["Ranking Service
        memory_core/ranking/service.py
        Purpose: weighted scoring for semantic, recency, importance, continuity, graph, type"]
        Citations["Citation Service
        memory_core/citations/service.py
        Purpose: build chunk/source citations"]
        LLM["Ollama LLM Client
        memory_core/llm/ollama.py
        Purpose: generation, summarization, structured extraction"]
        Embeddings["Sentence-Transformers Embeddings
        memory_core/embeddings/sentence_transformers.py
        Purpose: local embeddings for indexing and query search"]
    end

    Streamlit --> Retrieval
    Streamlit --> Ingestion
    Streamlit --> Review
    Streamlit --> Memory
    Streamlit --> Session
    Streamlit --> LLM

    MCP --> Retrieval
    MCP --> Ingestion
    MCP --> Review
    MCP --> Memory
    MCP --> Session

    Ingestion --> LLM
    Ingestion --> Memory
    Ingestion --> Parsers
    Ingestion --> Chunker

    Session --> LLM
    Session --> Memory
    Session --> Retrieval

    Retrieval --> Embeddings
    Retrieval --> Ranking
    Retrieval --> Citations
    Retrieval --> Vector
    Retrieval --> Graph
    Retrieval --> Repos

    Memory --> Embeddings
    Memory --> Vector
    Memory --> Graph
    Memory --> Repos

    Review --> Memory
    Review --> Repos

    Citations --> Repos

    subgraph Parse["Parsing / Chunking"]
        Parsers["Parsers
        memory_core/ingestion/parsers.py
        Purpose: parse txt, md, html, email, pdf, docx"]
        Chunker["Chunker
        memory_core/ingestion/chunking.py
        Purpose: split normalized text into overlapping chunks"]
    end

    subgraph Storage["Storage Layer"]
        Repos["SQLite Repositories
        memory_core/storage/sqlite/repositories.py
        Purpose: sources, memories, versions, chunks, candidates, jobs, sessions, audit"]
        SQLite["SQLite DB
        data/sqlite/memory.db
        Purpose: system of record"]
        Vector["FAISS Vector Index
        memory_core/storage/faiss/index.py
        data/faiss/
        Purpose: semantic retrieval index"]
        Graph["Neo4j Graph Store
        memory_core/storage/neo4j/store.py
        Purpose: memory relationships, provenance, graph expansion"]
        Imports["Imported Files
        data/imports/
        Purpose: uploaded source files"]
    end

    Repos --> SQLite
    Ingestion --> Imports

    subgraph Domain["Domain / Contracts"]
        Models["Pydantic Models
        memory_core/domain/models.py
        Purpose: memory, source, chunk, retrieval, session, audit schemas"]
        Enums["Enums
        memory_core/domain/enums.py
        Purpose: canonical types and statuses"]
        Interfaces["Protocols / Interfaces
        memory_core/interfaces/
        Purpose: abstraction boundaries for services, storage, retrieval, LLM, graph"]
    end

    Streamlit -. uses .-> Models
    MCP -. uses .-> Models
    Core -. uses .-> Models
    Core -. uses .-> Enums
    Core -. follows .-> Interfaces

```


## How It Works

* `Streamlit UI` is the human-facing app for chat, upload, review, browsing, and audit.
* `MCP Server` exposes the same core capabilities as tool-style operations for external agents/clients.
* `Ingestion Service` parses uploaded content, chunks it, auto-indexes document and chunk memories, and creates reviewable extracted candidates.
* `Memory Service` is the canonical lifecycle layer. It writes memory to SQLite, syncs FAISS embeddings, syncs Neo4j graph nodes/links, and handles merge/delete/versioning.
* `Retrieval Service` is the main answer-context builder. It combines vector search, keyword fallback, profile-memory inclusion, optional graph expansion, reranking, and citations.
* `Session Service` keeps recent conversation state and promotes durable facts/preferences/workflow rules into persistent memory across sessions.
* `Review Service` lets extracted candidates become active memory only after accept/merge decisions.
* `Ollama` handles generation, summarization, and memory extraction.
* `Sentence-Transformers` handles local embeddings for both indexed memory and live queries.
* `SQLite` is the source of truth; `FAISS` is the semantic retrieval index; `Neo4j` adds relationship/provenance context.

**Key Data Flows**

1. **Document Ingestion**
* User uploads file in Streamlit
* `Ingestion Service` parses and chunks it
* `Memory Service` stores active document/chunk memories
* embeddings are generated and written to `FAISS`
* graph nodes/relationships are written to `Neo4j`
extracted higher-level candidates go to review storage

2. **Chat / Answer Generation**
* User asks question in Streamlit
* `Retrieval Service` searches `FAISS`, applies keyword fallback, optionally expands via `Neo4j`, reranks, and attaches citations
* recent session context and durable chat memories are included
* `Ollama` generates the final answer from retrieved context

3. **Cross-Session Memory**
* chat messages are stored in SQLite session tables
* `Session Service` summarizes recent conversation and promotes durable facts/preferences/workflow rules into persistent memory
* those promoted memories become searchable in future sessions

4. Audit / Admin
* ingestion, retrieval, review, and memory lifecycle actions are logged in SQLite audit tables
* Streamlit admin views expose candidate review and audit inspection

```mermaid
sequenceDiagram
    autonumber

    actor User
    participant UI as Streamlit UI
    participant Ingest as Ingestion Service
    participant Parse as Parsers + Chunker
    participant LLM as Ollama LLM Client
    participant Memory as Memory Service
    participant SQL as SQLite Repositories / DB
    participant Vec as FAISS Vector Index
    participant Graph as Neo4j Graph Store
    participant Retrieve as Retrieval Service
    participant Rank as Ranking Service
    participant Cite as Citation Service
    participant Session as Session Service

    User->>UI: Upload document
    UI->>Ingest: ingest_source(source)
    Ingest->>SQL: create source
    Ingest->>SQL: create ingestion job
    Ingest-->>UI: job created

    UI->>Ingest: process_job(job_id)
    Ingest->>SQL: update job -> RUNNING
    Ingest->>Parse: parse source text
    Parse-->>Ingest: normalized text
    Ingest->>Parse: chunk text
    Parse-->>Ingest: chunks
    Ingest->>SQL: store chunks

    rect rgb(235, 245, 255)
        Note over Ingest,Graph: Immediate retrieval indexing path
        Ingest->>LLM: summarize document (best effort)
        LLM-->>Ingest: document summary
        loop document memory + chunk memories
            Ingest->>Memory: store active memory
            Memory->>SQL: create/update canonical memory
            Memory->>LLM: none
            Memory->>Vec: upsert embedding
            Memory->>Graph: upsert node / relationships
            Memory->>SQL: record audit event
        end
    end

    rect rgb(245, 255, 235)
        Note over Ingest,SQL: Reviewable extraction path
        loop each chunk
            Ingest->>LLM: extract structured memory
            LLM-->>Ingest: candidate items
        end
        Ingest->>SQL: save candidate memories
        Ingest->>SQL: update job -> REVIEW_REQUIRED / COMPLETED
        Ingest->>SQL: record ingestion completed audit
    end

    User->>UI: Ask question about uploaded doc
    UI->>Retrieve: retrieve(query, session_id, conversation_id)

    rect rgb(255, 248, 235)
        Note over Retrieve,Graph: Retrieval pipeline
        Retrieve->>Vec: semantic search
        Vec-->>Retrieve: vector hits
        Retrieve->>SQL: load matching memories
        Retrieve->>SQL: keyword fallback scan
        Retrieve->>SQL: load session snapshot
        opt graph enabled
            Retrieve->>Graph: expand related nodes
            Graph-->>Retrieve: related memory ids
            Retrieve->>SQL: load related memories
        end
        Retrieve->>Rank: score and rerank
        Rank-->>Retrieve: ranked memories
        Retrieve->>Cite: build citations
        Cite->>SQL: load chunk/source references
        Cite-->>Retrieve: citations
        Retrieve->>SQL: record retrieval/ranking audit
    end

    Retrieve-->>UI: RetrievalResult

    UI->>LLM: generate answer with retrieved memory + session summary + recent chat
    LLM-->>UI: final answer
    UI-->>User: Answer + citations + retrieval debug

    rect rgb(245, 235, 255)
        Note over UI,Session: Post-answer session memory update
        UI->>Session: append user/assistant messages
        Session->>SQL: persist session timeline
        UI->>Session: summarize_session(session_id)
        Session->>LLM: summarize recent conversation
        LLM-->>Session: rolling summary
        Session->>SQL: save session summary
        Session->>LLM: extract durable chat memories
        LLM-->>Session: facts/preferences/workflow rules
        opt heuristic fact capture
            Session->>Session: detect explicit identity facts
        end
        loop promoted durable memories
            Session->>Memory: store or merge memory
            Memory->>SQL: create/update canonical memory
            Memory->>Vec: upsert embedding
            Memory->>Graph: upsert node / relationships
        end
        Session->>SQL: record session summarized audit
    end

```