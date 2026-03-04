"""
app/api/v1/endpoints/documents.py
Document upload, retrieval, and extraction endpoints.
Enforces RBAC, tenant isolation, file validation, and idempotent processing.
"""
import io
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import get_settings
from app.core.exceptions import DuplicateDocumentError, ExtractionError
from app.db.base import get_tenant_session
from app.db.models.models import Document, ExtractedMetrics, RiskScore
from app.middleware.tenant import CurrentUser, require_permission
from app.services.audit.audit_service import AuditRecord, AuditService
from app.services.rag.rag_service import RAGService
from app.utils.hashing import IdempotencyCache, compute_sha256, get_redis

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])

_rag_service = RAGService()
_audit_service = AuditService()

# ── RESPONSE MODELS ────────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    sha256: str
    status: str
    is_duplicate: bool
    confidence_score: float | None
    requires_review: bool
    risk_tier: str | None
    risk_score: float | None
    risk_flags: list[str]
    extracted_metrics: dict
    duration_ms: int
    message: str


class DocumentSummary(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int | None
    confidence_score: float | None
    risk_tier: str | None
    requires_review: bool | None
    created_at: str


# ── UPLOAD ENDPOINT ────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    user: Annotated[CurrentUser, Depends(require_permission("document:write"))],
    file: UploadFile = File(..., description="PDF loan document"),
):
    """
    Upload and process a PDF loan document.
    
    - Validates file type and size
    - SHA-256 deduplication (idempotent)
    - Full RAG pipeline: parse → chunk → embed → retrieve → LLM extract
    - Returns structured financial metrics + risk assessment
    """
    # File validation
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted",
        )

    pdf_bytes = await file.read()

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    if len(pdf_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File appears to be empty or corrupted",
        )

    # Idempotency cache
    redis_client = await get_redis()
    cache = IdempotencyCache(redis_client)

    # Quick duplicate check before DB transaction
    sha256 = compute_sha256(pdf_bytes)
    existing_doc_id = await cache.check_duplicate(user.tenant_id, sha256)
    if existing_doc_id:
        cached_result = await cache.get_cached_result(user.tenant_id, sha256)
        if cached_result:
            return DocumentUploadResponse(
                **cached_result,
                message="Duplicate document — returning cached extraction",
            )

    async with get_tenant_session(user.tenant_id) as session:
        async with session.begin():
            # Create document record
            storage_path = f"tenants/{user.tenant_id}/documents/{sha256}/{file.filename}"
            document = Document(
                tenant_id=user.tenant_id,
                uploaded_by=user.user_id,
                filename=file.filename or "document.pdf",
                storage_path=storage_path,
                file_size_bytes=len(pdf_bytes),
                status="pending",
            )
            session.add(document)
            await session.flush()

            document_id = document.id

            try:
                result = await _rag_service.process_document(
                    session=session,
                    tenant_id=user.tenant_id,
                    user_id=user.user_id,
                    document_id=document_id,
                    pdf_bytes=pdf_bytes,
                    idempotency_cache=cache,
                    filename=file.filename or "document.pdf",
                )

                # Write audit log (non-blocking)
                from app.core.config import get_settings
                cfg = get_settings()
                audit_record = AuditRecord(
                    tenant_id=user.tenant_id,
                    user_id=user.user_id,
                    document_id=document_id,
                    model_name=cfg.LLM_MODEL,
                    model_version="1",
                    prompt_version=cfg.PROMPT_VERSION,
                    operation_type="extraction",
                    prompt_text="[extraction_prompt]",  # Hashed in audit service
                    rag_context=None,
                    raw_llm_response=None,
                    final_output=result.get("extracted_metrics"),
                    rag_chunk_ids=None,
                    duration_ms=result["duration_ms"],
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    confidence_score=result["confidence_score"],
                    validation_passed=not result["requires_review"],
                    fallback_used=result["fallback_used"],
                    retry_count=0,
                )
                await _audit_service.write(session, audit_record)

            except ExtractionError as exc:
                document.status = "failed"
                document.error_message = str(exc)
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Extraction failed: {exc.message}",
                ) from exc

    return DocumentUploadResponse(
        **result,
        message="Document processed successfully",
    )


# ── RETRIEVAL ENDPOINTS ────────────────────────────────────────────────────────

@router.get("/{document_id}", response_model=dict)
async def get_document(
    document_id: UUID,
    user: Annotated[CurrentUser, Depends(require_permission("document:read"))],
):
    """Retrieve document with extracted metrics and risk scores."""
    async with get_tenant_session(user.tenant_id) as session:
        doc = await session.get(Document, document_id)
        if not doc or doc.tenant_id != user.tenant_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Document not found")

        metrics = doc.extracted_metrics
        risk = doc.risk_score

        return {
            "id": str(doc.id),
            "filename": doc.filename,
            "status": doc.status,
            "page_count": doc.page_count,
            "file_size_bytes": doc.file_size_bytes,
            "created_at": doc.created_at.isoformat(),
            "extracted_metrics": metrics.validated_extraction if metrics else None,
            "confidence_score": float(metrics.confidence_score) if metrics else None,
            "requires_review": metrics.requires_review if metrics else None,
            "risk_tier": risk.risk_tier if risk else None,
            "risk_score": float(risk.overall_score) if risk else None,
            "risk_flags": risk.flags if risk else [],
        }


@router.get("/", response_model=list[DocumentSummary])
async def list_documents(
    user: Annotated[CurrentUser, Depends(require_permission("document:read"))],
    limit: int = 50,
    offset: int = 0,
):
    """List tenant documents with pagination. RLS enforces tenant isolation."""
    async with get_tenant_session(user.tenant_id) as session:
        from sqlalchemy.orm import joinedload
        result = await session.execute(
            select(Document)
            .where(Document.tenant_id == user.tenant_id)
            .options(
                joinedload(Document.extracted_metrics),
                joinedload(Document.risk_score),
            )
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        docs = result.unique().scalars().all()

        return [
            DocumentSummary(
                id=str(d.id),
                filename=d.filename,
                status=d.status,
                page_count=d.page_count,
                confidence_score=float(d.extracted_metrics.confidence_score)
                                 if d.extracted_metrics else None,
                risk_tier=d.risk_score.risk_tier if d.risk_score else None,
                requires_review=d.extracted_metrics.requires_review
                                if d.extracted_metrics else None,
                created_at=d.created_at.isoformat(),
            )
            for d in docs
        ]
