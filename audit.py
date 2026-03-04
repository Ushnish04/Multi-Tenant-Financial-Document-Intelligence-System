"""
app/api/v1/endpoints/audit.py
Audit log retrieval endpoints — queryable by tenant, user, document, operation.
Enforces RBAC: only admin and analyst roles can access audit logs.
"""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.db.base import get_tenant_session
from app.middleware.tenant import CurrentUser, require_permission
from app.services.audit.audit_service import AuditService

router = APIRouter(prefix="/audit", tags=["Audit"])
_audit_service = AuditService()


class AuditLogEntry(BaseModel):
    id: str
    document_id: str | None
    model_name: str
    operation_type: str
    duration_ms: int
    confidence_score: float | None
    validation_passed: bool
    fallback_used: bool
    retry_count: int
    total_tokens: int | None
    error_code: str | None
    called_at: str


@router.get("/logs", response_model=list[AuditLogEntry])
async def get_audit_logs(
    user: Annotated[CurrentUser, Depends(require_permission("audit:read"))],
    user_id: UUID | None = Query(None, description="Filter by user ID"),
    document_id: UUID | None = Query(None, description="Filter by document ID"),
    operation_type: str | None = Query(None, description="Filter by operation type"),
    days_back: int = Query(30, ge=1, le=365, description="Lookback window in days"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Query audit logs for the current tenant.
    
    Access control:
    - Analysts can see all tenant logs
    - (Optionally restrict users to their own logs only via user_id enforcement)
    
    GDPR compliance:
    - No raw PII exposed (prompt hashes only, no raw prompt text)
    - Borrower names excluded from final_output in audit records
    """
    # Non-admins can only query their own logs
    if not user.has_role("admin") and user_id and user_id != user.user_id:
        user_id = user.user_id  # Silently restrict to own logs

    async with get_tenant_session(user.tenant_id) as session:
        logs = await _audit_service.query_audit_logs(
            session=session,
            tenant_id=user.tenant_id,
            user_id=user_id,
            document_id=document_id,
            operation_type=operation_type,
            limit=limit,
            offset=offset,
            days_back=days_back,
        )

    return [AuditLogEntry(**entry) for entry in logs]


@router.get("/logs/summary")
async def get_audit_summary(
    user: Annotated[CurrentUser, Depends(require_permission("audit:read"))],
    days_back: int = Query(7, ge=1, le=90),
):
    """Aggregate audit metrics for the tenant — for dashboards and compliance."""
    async with get_tenant_session(user.tenant_id) as session:
        from sqlalchemy import func, select, text
        from app.db.models.models import LLMCall

        result = await session.execute(
            select(
                func.count(LLMCall.id).label("total_calls"),
                func.avg(LLMCall.duration_ms).label("avg_duration_ms"),
                func.sum(LLMCall.total_tokens).label("total_tokens"),
                func.avg(LLMCall.confidence_score).label("avg_confidence"),
                func.sum(
                    func.cast(LLMCall.validation_passed == False, func.Integer)  # noqa: E712
                ).label("validation_failures"),
                func.sum(
                    func.cast(LLMCall.fallback_used == True, func.Integer)  # noqa: E712
                ).label("fallback_count"),
            ).where(
                LLMCall.tenant_id == user.tenant_id,
                LLMCall.called_at >= text(f"NOW() - INTERVAL '{days_back} days'"),
            )
        )
        row = result.one()

    return {
        "period_days": days_back,
        "total_calls": row.total_calls or 0,
        "avg_duration_ms": round(float(row.avg_duration_ms or 0), 1),
        "total_tokens_consumed": row.total_tokens or 0,
        "avg_confidence_score": round(float(row.avg_confidence or 0), 3),
        "validation_failure_count": row.validation_failures or 0,
        "fallback_extraction_count": row.fallback_count or 0,
    }
