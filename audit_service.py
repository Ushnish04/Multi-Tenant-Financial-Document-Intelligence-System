"""
app/services/audit/audit_service.py
Immutable audit logging service for all LLM calls and system events.
GDPR-safe: stores hashes of sensitive content, not raw PII.
"""
import hashlib
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.models import LLMCall

log = structlog.get_logger(__name__)


@dataclass
class AuditRecord:
    """Structured audit record for an LLM extraction call."""
    tenant_id: UUID
    user_id: UUID
    document_id: UUID | None

    # LLM metadata
    model_name: str
    model_version: str
    prompt_version: str
    operation_type: str

    # Content hashes (GDPR-safe — no raw PII)
    prompt_text: str            # Hashed before storage
    rag_context: str | None     # Hashed before storage
    raw_llm_response: str | None  # Hashed before storage
    final_output: dict | None   # Validated extraction (PII-safe subset)

    # Chunk references
    rag_chunk_ids: list[UUID] | None

    # Performance
    duration_ms: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    # Quality
    confidence_score: float | None
    validation_passed: bool
    fallback_used: bool
    retry_count: int

    # Error tracking
    error_code: str | None = None
    error_message: str | None = None

    # Token count
    prompt_token_count: int | None = None


class AuditService:
    """
    Writes immutable audit log entries to the llm_calls table.
    
    Design principles:
    - Hashes are stored for prompt/response content (not raw text) for GDPR
    - final_output stores the validated extraction (no raw PII borrower names)
    - Records are INSERT-only; no UPDATE or DELETE paths exist in application code
    - Failures to write audit logs are logged but do not break the main pipeline
    """

    @staticmethod
    def _hash(text: str | None) -> str | None:
        """SHA-256 hash of text for GDPR-safe storage."""
        if text is None:
            return None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _gdpr_safe_output(output: dict | None) -> dict | None:
        """
        Remove PII fields from final output before audit log storage.
        Retains financial metrics but removes identifiable borrower info.
        """
        if output is None:
            return None

        PII_FIELDS = {"borrower_name", "property_address", "employer_name"}
        return {k: v for k, v in output.items() if k not in PII_FIELDS}

    async def write(
        self,
        session: AsyncSession,
        record: AuditRecord,
    ) -> UUID | None:
        """
        Write an audit record to llm_calls table.
        Never raises — failures are logged but swallowed to protect main flow.
        Returns audit record ID if successful, else None.
        """
        try:
            llm_call = LLMCall(
                tenant_id=record.tenant_id,
                user_id=record.user_id,
                document_id=record.document_id,
                model_name=record.model_name,
                model_version=record.model_version,
                prompt_version=record.prompt_version,
                operation_type=record.operation_type,
                prompt_hash=self._hash(record.prompt_text) or "",
                prompt_token_count=record.prompt_token_count,
                rag_chunk_ids=record.rag_chunk_ids,
                rag_context_hash=self._hash(record.rag_context),
                raw_response_hash=self._hash(record.raw_llm_response),
                final_output=self._gdpr_safe_output(record.final_output),
                duration_ms=record.duration_ms,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                total_tokens=record.total_tokens,
                confidence_score=record.confidence_score,
                validation_passed=record.validation_passed,
                fallback_used=record.fallback_used,
                retry_count=record.retry_count,
                error_code=record.error_code,
                error_message=record.error_message,
            )
            session.add(llm_call)
            await session.flush()

            log.info(
                "audit_record_written",
                audit_id=str(llm_call.id),
                tenant_id=str(record.tenant_id),
                document_id=str(record.document_id) if record.document_id else None,
                operation=record.operation_type,
                duration_ms=record.duration_ms,
                confidence=record.confidence_score,
                validation_passed=record.validation_passed,
            )
            return llm_call.id

        except Exception as exc:
            # CRITICAL: Audit failures must not break main pipeline
            # But they must be logged at ERROR level for ops alerting
            log.error(
                "audit_write_failed",
                error=str(exc),
                tenant_id=str(record.tenant_id),
                document_id=str(record.document_id) if record.document_id else None,
            )
            return None

    async def query_audit_logs(
        self,
        session: AsyncSession,
        tenant_id: UUID,
        user_id: UUID | None = None,
        document_id: UUID | None = None,
        operation_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        days_back: int = 30,
    ) -> list[dict]:
        """
        Query audit logs with filtering. RLS ensures tenant isolation.
        """
        from sqlalchemy import select, text as sa_text, and_
        from app.db.models.models import LLMCall

        query = select(LLMCall).where(
            LLMCall.tenant_id == tenant_id,
            LLMCall.called_at >= sa_text(f"NOW() - INTERVAL '{days_back} days'"),
        )

        if user_id:
            query = query.where(LLMCall.user_id == user_id)
        if document_id:
            query = query.where(LLMCall.document_id == document_id)
        if operation_type:
            query = query.where(LLMCall.operation_type == operation_type)

        query = query.order_by(LLMCall.called_at.desc()).limit(limit).offset(offset)

        result = await session.execute(query)
        calls = result.scalars().all()

        return [
            {
                "id": str(c.id),
                "document_id": str(c.document_id) if c.document_id else None,
                "model_name": c.model_name,
                "operation_type": c.operation_type,
                "duration_ms": c.duration_ms,
                "confidence_score": float(c.confidence_score) if c.confidence_score else None,
                "validation_passed": c.validation_passed,
                "fallback_used": c.fallback_used,
                "retry_count": c.retry_count,
                "total_tokens": c.total_tokens,
                "error_code": c.error_code,
                "called_at": c.called_at.isoformat(),
            }
            for c in calls
        ]
