"""Streamlit application for chat, ingestion, browsing, and admin workflows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from adapters.mcp import AppServices, load_services_from_env
from memory_core.domain.enums import AuditEventType, MemoryStatus, MemoryType, ReviewDecision, SourceType
from memory_core.domain.models import RetrievalQuery, SessionMessage, SourceReference


@st.cache_resource(show_spinner=False)
def _get_services(root_dir: str) -> AppServices:
    return load_services_from_env(root_dir)


def run_app() -> None:
    """Render the Streamlit application shell."""
    st.set_page_config(
        page_title="Personal Context MCP Memory",
        page_icon=":material/psychology:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    root_dir = Path(__file__).resolve().parents[2]
    services = _get_services(str(root_dir))

    _initialize_session_state()

    st.title("Personal Context MCP Memory")
    st.caption("Offline document memory with hybrid retrieval, reviewable extraction, and audit visibility.")

    with st.sidebar:
        st.subheader("Session")
        st.write(f"Session ID: `{st.session_state.session_id}`")
        st.write(f"Conversation ID: `{st.session_state.conversation_id}`")
        if st.button("Start New Session", use_container_width=True):
            _reset_chat_session()
            st.rerun()

        st.subheader("Retrieval")
        st.session_state.chat_top_k = st.slider(
            "Top memories",
            min_value=3,
            max_value=15,
            value=st.session_state.chat_top_k,
            step=1,
        )

    chat_tab, upload_tab, browser_tab, admin_tab = st.tabs(
        ["Chat", "Upload", "Memory Browser", "Admin & Audit"]
    )

    with chat_tab:
        _render_chat_tab(services)
    with upload_tab:
        _render_upload_tab(services, root_dir)
    with browser_tab:
        _render_memory_browser_tab(services)
    with admin_tab:
        _render_admin_tab(services)


def _initialize_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid4().hex}"
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = f"conversation_{uuid4().hex}"
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_top_k" not in st.session_state:
        st.session_state.chat_top_k = 6
    if "session_memory_warning" not in st.session_state:
        st.session_state.session_memory_warning = None


def _reset_chat_session() -> None:
    st.session_state.session_id = f"session_{uuid4().hex}"
    st.session_state.conversation_id = f"conversation_{uuid4().hex}"
    st.session_state.chat_messages = []


def _render_chat_tab(services: AppServices) -> None:
    st.subheader("Chat")
    st.write("Ask questions against stored memory. Answers are generated locally with Ollama and grounded in retrieved memory.")
    if st.session_state.session_memory_warning:
        st.warning(
            "The last cross-session memory update did not complete cleanly: "
            f"{st.session_state.session_memory_warning}"
        )
        st.session_state.session_memory_warning = None

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("Citations", expanded=False):
                    for citation in message["citations"]:
                        st.json(citation)

    prompt = st.chat_input("Ask about your stored documents, tasks, facts, or preferences...")
    if not prompt:
        return

    _append_session_message(
        services,
        role="user",
        content=prompt,
    )
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            retrieval_result = services.retrieval_service.retrieve(
                RetrievalQuery(
                    query=prompt,
                    session_id=st.session_state.session_id,
                    conversation_id=st.session_state.conversation_id,
                    top_k=st.session_state.chat_top_k,
                    require_citations=True,
                )
            )
            answer = _generate_chat_answer(services, prompt, retrieval_result)
            citations = [
                citation
                for item in retrieval_result.items
                for citation in item.citations
            ]
            st.markdown(answer)
            with st.expander("Retrieved Memory Debug", expanded=False):
                st.write(f"Session summary: {retrieval_result.session_summary or 'None'}")
                if not retrieval_result.items:
                    st.info("No memories were retrieved for this answer.")
                for item in retrieval_result.items:
                    st.json(
                        {
                            "memory_id": item.memory.memory_id,
                            "memory_type": item.memory.memory_type.value,
                            "source_id": item.memory.source_id,
                            "summary": item.memory.summary,
                            "score": _model_dump(item.score),
                            "matched_chunk_ids": item.matched_chunk_ids,
                            "reasoning": item.reasoning,
                            "tags": item.memory.tags,
                            "metadata": item.memory.metadata,
                        }
                    )
            if citations:
                with st.expander("Citations", expanded=False):
                    for citation in citations:
                        st.json(_model_dump(citation))

    _append_session_message(
        services,
        role="assistant",
        content=answer,
        metadata={"citations": [_model_dump(citation) for citation in citations]},
    )
    _update_session_summary(services)
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": [_model_dump(citation) for citation in citations],
        }
    )
    st.rerun()


def _render_upload_tab(services: AppServices, root_dir: Path) -> None:
    st.subheader("Upload & Ingest")
    st.write("Upload a file or paste raw text, then create and process an ingestion job.")

    source_type = st.selectbox("Source Type", [source_type.value for source_type in SourceType])
    title = st.text_input("Title")

    upload_col, text_col = st.columns(2)
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "txt", "md", "docx", "html", "htm", "eml"],
        )
    with text_col:
        raw_text = st.text_area("Or paste raw text", height=220)

    auto_process = st.checkbox("Process immediately after creating the ingestion job", value=True)

    if st.button("Create Ingestion Job", type="primary"):
        source_enum = SourceType(source_type)
        source_metadata: dict[str, Any] = {}
        file_path: str | None = None
        original_filename: str | None = None
        mime_type: str | None = None

        if uploaded_file is not None:
            file_path = str(_save_uploaded_file(root_dir, uploaded_file))
            original_filename = uploaded_file.name
            mime_type = uploaded_file.type or None
        elif raw_text.strip():
            source_metadata["raw_text"] = raw_text.strip()
        else:
            st.error("Provide either an uploaded file or pasted raw text.")
            return

        source = SourceReference(
            source_id=f"src_{uuid4().hex}",
            source_type=source_enum,
            title=title or None,
            file_path=file_path,
            original_filename=original_filename,
            mime_type=mime_type,
            checksum=None,
            external_uri=None,
            created_at=datetime.utcnow(),
            metadata=source_metadata,
        )
        job = services.ingestion_service.ingest_source(source)
        st.success(f"Created ingestion job `{job.job_id}` for source `{job.source_id}`.")

        if auto_process:
            processed_job = services.ingestion_service.process_job(job.job_id)
            st.info(
                f"Processed job `{processed_job.job_id}` with status `{processed_job.status.value}`."
            )
            if processed_job.metadata:
                st.json(processed_job.metadata)


def _render_memory_browser_tab(services: AppServices) -> None:
    st.subheader("Memory Browser")
    memory_type_filter = st.multiselect(
        "Filter by memory type",
        [memory_type.value for memory_type in MemoryType],
        default=[],
    )
    include_deleted = st.checkbox("Include deleted memories", value=False)

    selected_types = [MemoryType(value) for value in memory_type_filter] if memory_type_filter else None
    memories = services.memory_repository.list_memories(selected_types)
    filtered_memories = [
        memory
        for memory in memories
        if include_deleted or memory.status != MemoryStatus.DELETED
    ]

    st.write(f"{len(filtered_memories)} memories")
    for memory in filtered_memories:
        label = f"{memory.memory_type.value} | {memory.memory_id} | {memory.status.value}"
        with st.expander(label, expanded=False):
            st.write(memory.summary or memory.content[:400])
            st.json(_model_dump(memory))
            history = services.memory_service.get_memory_history(memory.memory_id)
            st.caption(f"Version history entries: {len(history)}")
            if memory.status != MemoryStatus.DELETED:
                delete_key = f"delete_{memory.memory_id}"
                if st.button("Soft Delete", key=delete_key):
                    services.memory_service.delete_memory(memory.memory_id, "Deleted from Streamlit memory browser.")
                    st.rerun()


def _render_admin_tab(services: AppServices) -> None:
    st.subheader("Admin & Audit")

    candidate_col, audit_col = st.columns([1.1, 1.4])

    with candidate_col:
        st.markdown("#### Candidate Review")
        candidates = services.review_service.list_candidates()
        if not candidates:
            st.info("No candidate memories are waiting for review.")
        else:
            all_memories = services.memory_repository.list_memories()
            memory_options = {memory.memory_id: memory for memory in all_memories}
            for candidate in candidates:
                with st.expander(
                    f"{candidate.proposed_memory.memory_type.value} | {candidate.candidate_id}",
                    expanded=False,
                ):
                    st.write(candidate.extraction_reason)
                    st.json(_model_dump(candidate))
                    merge_target = st.selectbox(
                        "Merge target (optional)",
                        options=[""] + list(memory_options.keys()),
                        key=f"merge_target_{candidate.candidate_id}",
                    )
                    action_cols = st.columns(4)
                    if action_cols[0].button("Accept", key=f"accept_{candidate.candidate_id}"):
                        services.review_service.apply_decision(candidate.candidate_id, ReviewDecision.ACCEPT)
                        st.rerun()
                    if action_cols[1].button("Reject", key=f"reject_{candidate.candidate_id}"):
                        services.review_service.apply_decision(candidate.candidate_id, ReviewDecision.REJECT)
                        st.rerun()
                    if action_cols[2].button("Defer", key=f"defer_{candidate.candidate_id}"):
                        services.review_service.apply_decision(candidate.candidate_id, ReviewDecision.DEFER)
                        st.rerun()
                    if action_cols[3].button("Merge", key=f"merge_{candidate.candidate_id}"):
                        target_memory_id = merge_target or None
                        services.review_service.apply_decision(
                            candidate.candidate_id,
                            ReviewDecision.MERGE,
                            target_memory_id=target_memory_id,
                        )
                        st.rerun()

    with audit_col:
        st.markdown("#### Audit Trail")
        event_type = st.selectbox(
            "Filter event type",
            options=[""] + [event_type.value for event_type in AuditEventType],
        )
        limit = st.slider("Audit rows", min_value=10, max_value=200, value=50, step=10)
        events = services.audit_repository.list_events(
            event_type=AuditEventType(event_type) if event_type else None,
            limit=limit,
        )
        if not events:
            st.info("No audit events found.")
        else:
            for event in events:
                with st.expander(f"{event.event_type.value} | {event.timestamp.isoformat()}", expanded=False):
                    st.json(_model_dump(event))


def _append_session_message(
    services: AppServices,
    *,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    services.session_service.append_message(
        st.session_state.session_id,
        SessionMessage(
            message_id=f"msg_{uuid4().hex}",
            role=role,
            content=content,
            created_at=datetime.utcnow(),
            metadata={
                "conversation_id": st.session_state.conversation_id,
                **dict(metadata or {}),
            },
        ),
    )


def _update_session_summary(services: AppServices) -> None:
    try:
        services.session_service.summarize_session(st.session_state.session_id)
    except Exception as exc:
        st.session_state.session_memory_warning = str(exc)


def _generate_chat_answer(services: AppServices, prompt: str, retrieval_result: Any) -> str:
    context_lines: list[str] = []
    citation_lines: list[str] = []
    recent_chat_lines: list[str] = []

    for message in st.session_state.chat_messages[-6:]:
        recent_chat_lines.append(f"{message['role']}: {message['content']}")

    for index, item in enumerate(retrieval_result.items, start=1):
        memory = item.memory
        context_lines.append(
            f"[Memory {index}] "
            f"type={memory.memory_type.value} score={item.score.final_score:.3f} "
            f"source={memory.source_id or 'n/a'}\n"
            f"{memory.summary or memory.content[:1200]}"
        )
        for citation in item.citations:
            citation_lines.append(
                f"- source_id={citation.source_id}, chunk_id={citation.chunk_id}, "
                f"document_id={citation.document_id}, quote={citation.quote}"
            )

    system_prompt = (
        "You are a generally capable assistant with access to remembered personal context. "
        "Use retrieved memory and recent conversation state when they are relevant and trustworthy. "
        "If memory is weak or absent, still answer helpfully from your own knowledge. "
        "When memory materially informs the answer, prioritize it over generic knowledge. "
        "Never claim to remember something that was not actually retrieved or present in the recent conversation. "
        "Do not fabricate citations."
    )
    user_prompt = (
        f"User question:\n{prompt}\n\n"
        f"Recent conversation:\n{chr(10).join(recent_chat_lines) if recent_chat_lines else 'No prior conversation turns.'}\n\n"
        f"Session summary:\n{retrieval_result.session_summary or 'No session summary available.'}\n\n"
        f"Retrieved memory context:\n{chr(10).join(context_lines) if context_lines else 'No relevant memory found.'}\n\n"
        f"Citation inventory:\n{chr(10).join(citation_lines) if citation_lines else 'No citations available.'}\n\n"
        "Write a helpful answer. "
        "Use remembered context where it helps. "
        "If remembered context is weak, answer normally from general knowledge and say that the answer is not based on stored memory."
    )
    return services.llm_client.generate(user_prompt, system_prompt=system_prompt)


def _save_uploaded_file(root_dir: Path, uploaded_file: Any) -> Path:
    import_dir = root_dir / "data" / "imports"
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / f"{uuid4().hex}_{uploaded_file.name}"
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _model_dump(model: Any) -> Any:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[call-arg]
    if hasattr(model, "json"):
        return json.loads(model.json())
    return model
