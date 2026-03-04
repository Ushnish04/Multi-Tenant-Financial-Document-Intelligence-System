"""
app/db/models/models.py
SQLAlchemy 2.0 ORM models mirroring the PostgreSQL schema.
All models are async-compatible with mapped columns.
"""
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Date, DateTime,
    ForeignKey, Index, Integer, Numeric, String, Text,
    UniqueConstraint, func, text,
)
from sqlalchemy.dialects.postgresql import ARRAY, INET, JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Tenant(Base):
    __tablename__ = "tenants"
    __table_args__ = {"schema": "finrag"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    plan: Mapped[str] = mapped_column(String(50), nullable=False, default="standard")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                  onupdate=func.now())

    users: Mapped[list["User"]] = relationship("User", back_populates="tenant")
    documents: Mapped[list["Document"]] = relationship("Document", back_populates="tenant")


class Role(Base):
    __tablename__ = "roles"
    __table_args__ = {"schema": "finrag"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    permissions: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="users_tenant_email_unique"),
        {"schema": "finrag"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"),
                                             nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="users")
    user_roles: Mapped[list["UserRole"]] = relationship("UserRole", back_populates="user")
    documents: Mapped[list["Document"]] = relationship("Document", back_populates="uploader")


class UserRole(Base):
    __tablename__ = "user_roles"
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", "tenant_id", name="user_roles_unique"),
        {"schema": "finrag"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                           ForeignKey("finrag.users.id", ondelete="CASCADE"))
    role_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                           ForeignKey("finrag.roles.id", ondelete="CASCADE"))
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"))
    granted_by: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True),
                                                        ForeignKey("finrag.users.id", ondelete="SET NULL"))
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="user_roles", foreign_keys=[user_id])
    role: Mapped["Role"] = relationship("Role")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    __table_args__ = {"schema": "finrag"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                           ForeignKey("finrag.users.id", ondelete="CASCADE"))
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(INET)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "finrag"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"))
    uploaded_by: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                               ForeignKey("finrag.users.id", ondelete="RESTRICT"))
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False, default="application/pdf")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="documents")
    uploader: Mapped["User"] = relationship("User", back_populates="documents")
    document_hash: Mapped[Optional["DocumentHash"]] = relationship("DocumentHash",
                                                                     back_populates="document",
                                                                     uselist=False)
    embeddings: Mapped[list["Embedding"]] = relationship("Embedding", back_populates="document")
    extracted_metrics: Mapped[Optional["ExtractedMetrics"]] = relationship(
        "ExtractedMetrics", back_populates="document", uselist=False)
    risk_score: Mapped[Optional["RiskScore"]] = relationship("RiskScore", back_populates="document",
                                                              uselist=False)


class DocumentHash(Base):
    __tablename__ = "document_hashes"
    __table_args__ = (
        UniqueConstraint("tenant_id", "sha256_hash", name="document_hashes_tenant_hash_unique"),
        {"schema": "finrag"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"))
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                               ForeignKey("finrag.documents.id", ondelete="CASCADE"))
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship("Document", back_populates="document_hash")


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="embeddings_doc_chunk_unique"),
        {"schema": "finrag"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"))
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                               ForeignKey("finrag.documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship("Document", back_populates="embeddings")


class ExtractedMetrics(Base):
    __tablename__ = "extracted_metrics"
    __table_args__ = {"schema": "finrag"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"))
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                               ForeignKey("finrag.documents.id", ondelete="CASCADE"),
                                               unique=True)
    loan_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2))
    loan_term_months: Mapped[Optional[int]] = mapped_column(Integer)
    interest_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 5))
    annual_percentage_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 5))
    monthly_payment: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2))
    origination_fee: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2))
    borrower_income_annual: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2))
    debt_to_income_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    credit_score: Mapped[Optional[int]] = mapped_column(Integer)
    employment_status: Mapped[Optional[str]] = mapped_column(String(100))
    employer_name: Mapped[Optional[str]] = mapped_column(String(255))
    property_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 2))
    loan_to_value_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    property_type: Mapped[Optional[str]] = mapped_column(String(100))
    property_address: Mapped[Optional[str]] = mapped_column(Text)
    document_date: Mapped[Optional[date]] = mapped_column(Date)
    lender_name: Mapped[Optional[str]] = mapped_column(String(255))
    borrower_name: Mapped[Optional[str]] = mapped_column(String(255))
    loan_purpose: Mapped[Optional[str]] = mapped_column(String(255))
    loan_type: Mapped[Optional[str]] = mapped_column(String(100))
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    extraction_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    requires_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_extraction: Mapped[dict] = mapped_column(JSONB, nullable=False)
    validated_extraction: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship("Document", back_populates="extracted_metrics")
    risk_score: Mapped[Optional["RiskScore"]] = relationship("RiskScore", back_populates="metrics")


class RiskScore(Base):
    __tablename__ = "risk_scores"
    __table_args__ = {"schema": "finrag"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="CASCADE"))
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                               ForeignKey("finrag.documents.id", ondelete="CASCADE"),
                                               unique=True)
    metrics_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                              ForeignKey("finrag.extracted_metrics.id",
                                                         ondelete="CASCADE"))
    overall_score: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(50), nullable=False)
    dti_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 3))
    ltv_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 3))
    credit_score_n: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 3))
    income_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 3))
    flags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    scoring_version: Mapped[str] = mapped_column(String(50), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship("Document", back_populates="risk_score")
    metrics: Mapped["ExtractedMetrics"] = relationship("ExtractedMetrics", back_populates="risk_score")


class LLMCall(Base):
    __tablename__ = "llm_calls"
    __table_args__ = {"schema": "finrag"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                             ForeignKey("finrag.tenants.id", ondelete="RESTRICT"))
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True),
                                           ForeignKey("finrag.users.id", ondelete="RESTRICT"))
    document_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True),
                                                         ForeignKey("finrag.documents.id",
                                                                    ondelete="SET NULL"))
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    operation_type: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_token_count: Mapped[Optional[int]] = mapped_column(Integer)
    rag_chunk_ids: Mapped[Optional[list]] = mapped_column(ARRAY(PGUUID(as_uuid=True)))
    rag_context_hash: Mapped[Optional[str]] = mapped_column(String(64))
    raw_response_hash: Mapped[Optional[str]] = mapped_column(String(64))
    final_output: Mapped[Optional[dict]] = mapped_column(JSONB)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    confidence_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 3))
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[Optional[str]] = mapped_column(String(100))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    called_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
