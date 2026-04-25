"""Domain enums shared across memory workflows."""

from enum import Enum


class MemoryType(str, Enum):
    DOCUMENT = "document"
    DOCUMENT_CHUNK = "document_chunk"
    FACT = "fact"
    EPISODE = "episode"
    PREFERENCE = "preference"
    TASK = "task"
    WORKFLOW_RULE = "workflow_rule"
    SUMMARY = "summary"


class SourceType(str, Enum):
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    EMAIL = "email"
    WEB_PAGE = "web_page"
    CHAT = "chat"
    MANUAL = "manual"


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class AuditEventType(str, Enum):
    INGESTION_CREATED = "ingestion_created"
    INGESTION_COMPLETED = "ingestion_completed"
    INGESTION_FAILED = "ingestion_failed"
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_MERGED = "memory_merged"
    MEMORY_SOFT_DELETED = "memory_soft_deleted"
    MEMORY_RESTORED = "memory_restored"
    RETRIEVAL_EXECUTED = "retrieval_executed"
    RANKING_APPLIED = "ranking_applied"
    REVIEW_ACCEPTED = "review_accepted"
    REVIEW_REJECTED = "review_rejected"
    SESSION_SUMMARIZED = "session_summarized"


class ReviewDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    MERGE = "merge"
    DEFER = "defer"
