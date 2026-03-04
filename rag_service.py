"""
app/services/rag/rag_service.py
RAG Orchestration Service.
Coordinates: PDF parse → chunk → embed → store → retrieve → assemble → LLM → validate → persist.
"""
import io
import time
from uuid import UUID

import pdfplumber
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import ExtractionError
from app.db.models.models import Document, DocumentHash, ExtractedMetrics, RiskScore
from app.services.llm.wrapper import DeterministicLLMWrapper
from app.services.rag.chunker import SemanticChunker
from app.services.rag.embeddings import EmbeddingService
from app.utils.hashing import IdempotencyCache, compute_sha256

settings = get_settings()
log = structlog.get_logger(__name__)

EXTRACTION_SCHEMA_VERSION = settings.EXTRACTION_SCHEMA_VERSION
PROMPT_VERSION = settings.PROMPT_VERSION


class PDFParser:
    """Robust PDF text extraction using pdfplumber."""

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> tuple[str, int, int]:
        """
        Extract full text from PDF bytes.
        Returns (full_text, page_count, word_count).
        """
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages: list[str] = []
            for page in pdf.pages:
                text = page.extract_text(
                    x_tolerance=3,
                    y_tolerance=3,
                    layout=True,
                    x_density=7.25,
                    y_density=13,
                ) or ""
                pages.append(text)

            full_text = "\n\n".join(pages)
            word_count = len(full_text.split())
            page_count = len(pdf.pages)

        return full_text, page_count, word_count


def _assemble_context(similar_chunks: list[dict]) -> str:
    """
    Assemble retrieved chunks into ordered context string.
    Deduplicates by chunk text and sorts by similarity score descending.
    """
    seen: set[str] = set()
    ordered: list[dict] = sorted(similar_chunks, key=lambda c: c["similarity_score"], reverse=True)

    context_parts: list[str] = []
    for chunk in ordered:
        text = chunk["chunk_text"].strip()
        if text not in seen and len(text) > 20:
            seen.add(text)
            context_parts.append(
                f"[Chunk {chunk['chunk_index']} | Score: {chunk['similarity_score']:.3f}]\n{text}"
            )

    return "\n\n---\n\n".join(context_parts)


def _compute_risk_scores(metrics: ExtractedMetrics) -> dict:
    """
    Compute normalized risk sub-scores from extracted financial metrics.
    Each sub-score is 0.0 (lowest risk) to 1.0 (highest risk).
    """
    scores: dict[str, float | None] = {}
    flags: list[str] = []

    # DTI score: >0.43 is high risk (CFPB QM threshold)
    if metrics.debt_to_income_ratio is not None:
        dti = float(metrics.debt_to_income_ratio)
        scores["dti_score"] = min(dti / 0.50, 1.0)
        if dti > 0.43:
            flags.append(f"HIGH_DTI:{dti:.1%}")
    else:
        scores["dti_score"] = None

    # LTV score: >0.80 typically requires PMI; >0.97 is high risk
    if metrics.loan_to_value_ratio is not None:
        ltv = float(metrics.loan_to_value_ratio)
        scores["ltv_score"] = min(ltv / 1.0, 1.0)
        if ltv > 0.97:
            flags.append(f"HIGH_LTV:{ltv:.1%}")
    else:
        scores["ltv_score"] = None

    # Credit score (inverted: lower credit = higher risk)
    if metrics.credit_score is not None:
        cs = metrics.credit_score
        # Normalize: 300=worst(1.0), 850=best(0.0)
        scores["credit_score_n"] = 1.0 - (cs - 300) / 550.0
        if cs < 620:
            flags.append(f"LOW_CREDIT_SCORE:{cs}")
    else:
        scores["credit_score_n"] = None

    # Income score: cross-check against loan amount
    if metrics.borrower_income_annual is not None and metrics.loan_amount is not None:
        income = float(metrics.borrower_income_annual)
        loan = float(metrics.loan_amount)
        ratio = loan / max(income, 1)
        scores["income_score"] = min(ratio / 10.0, 1.0)  # >10x income = max risk
        if ratio > 5.0:
            flags.append(f"HIGH_LOAN_TO_INCOME:{ratio:.1f}x")
    else:
        scores["income_score"] = None

    # Overall score: weighted average of available sub-scores
    weights = {"dti_score": 0.30, "ltv_score": 0.25, "credit_score_n": 0.35, "income_score": 0.10}
    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        if scores.get(key) is not None:
            weighted_sum += scores[key] * weight  # type: ignore[operator]
            total_weight += weight

    overall = weighted_sum / total_weight if total_weight > 0 else 0.5

    # Risk tier classification
    if overall < 0.25:
        tier = "low"
    elif overall < 0.50:
        tier = "medium"
    elif overall < 0.75:
        tier = "high"
    else:
        tier = "critical"

    if tier in ("high", "critical"):
        flags.append(f"RISK_TIER:{tier.upper()}")

    return {
        "overall_score": round(overall, 3),
        "risk_tier": tier,
        "flags": flags,
        **{k: round(v, 3) if v is not None else None for k, v in scores.items()},
    }


class RAGService:
    """
    Main orchestrator for the RAG + LLM extraction pipeline.
    
    Usage:
        result = await rag_service.process_document(
            session=session,
            tenant_id=tenant_id,
            user_id=user_id,
            document_id=document_id,
            pdf_bytes=pdf_bytes,
            idempotency_cache=cache,
        )
    """

    def __init__(self) -> None:
        self._chunker = SemanticChunker()
        self._embedding_service = EmbeddingService()
        self._llm = DeterministicLLMWrapper()

    async def process_document(
        self,
        session: AsyncSession,
        tenant_id: UUID,
        user_id: UUID,
        document_id: UUID,
        pdf_bytes: bytes,
        idempotency_cache: IdempotencyCache,
        filename: str = "document.pdf",
    ) -> dict:
        """
        Full RAG pipeline for a single PDF document.
        
        Pipeline:
        1. SHA-256 hash + duplicate check
        2. PDF parse
        3. Semantic chunking
        4. Embedding generation (batched async)
        5. pgvector storage
        6. Query embedding for retrieval
        7. Similarity search (top-k)
        8. Context assembly
        9. Deterministic LLM extraction
        10. Risk scoring
        11. DB persistence
        12. Cache update
        13. Return structured result
        """
        start_time = time.monotonic()
        doc_log = log.bind(
            document_id=str(document_id),
            tenant_id=str(tenant_id),
            filename=filename,
        )

        # ── Step 1: Duplicate detection ──────────────────────────────────────
        sha256 = compute_sha256(pdf_bytes)
        cached_doc_id = await idempotency_cache.check_duplicate(tenant_id, sha256)

        if cached_doc_id:
            cached_result = await idempotency_cache.get_cached_result(tenant_id, sha256)
            if cached_result:
                doc_log.info("duplicate_document_cache_hit", sha256=sha256[:12])
                return {**cached_result, "is_duplicate": True, "original_document_id": cached_doc_id}

        # ── Step 2: PDF parsing ───────────────────────────────────────────────
        doc_log.info("pdf_parsing_start")
        full_text, page_count, word_count = PDFParser.extract_text(pdf_bytes)
        doc_log.info("pdf_parsing_complete", pages=page_count, words=word_count)

        if len(full_text.strip()) < 50:
            raise ExtractionError("PDF appears to be empty or unreadable")

        # ── Step 3: Update document metadata ─────────────────────────────────
        doc = await session.get(Document, document_id)
        if not doc:
            raise ExtractionError(f"Document record {document_id} not found")
        doc.status = "processing"
        doc.page_count = page_count
        doc.word_count = word_count
        await session.flush()

        # Store document hash (idempotency record)
        doc_hash = DocumentHash(
            tenant_id=tenant_id,
            document_id=document_id,
            sha256_hash=sha256,
        )
        session.add(doc_hash)
        await session.flush()

        # ── Step 4: Semantic chunking ─────────────────────────────────────────
        doc_log.info("chunking_start")
        chunks = self._chunker.chunk(full_text)
        doc_log.info("chunking_complete", chunk_count=len(chunks))

        # ── Step 5: Embedding generation ─────────────────────────────────────
        doc_log.info("embedding_generation_start", chunk_count=len(chunks))
        chunk_texts = [c.text for c in chunks]
        vectors = await self._embedding_service.embed_texts(chunk_texts)

        # ── Step 6: Store embeddings ──────────────────────────────────────────
        embedding_ids = await self._embedding_service.store_embeddings(
            session=session,
            tenant_id=tenant_id,
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
        )

        # ── Step 7: Retrieval query ───────────────────────────────────────────
        # Use document-specific retrieval query for financial extraction
        retrieval_query = (
            "loan amount interest rate APR monthly payment borrower income "
            "debt-to-income ratio credit score property value LTV loan term "
            "origination fee lender borrower employment"
        )
        query_vector = await self._embedding_service.embed_single(retrieval_query)

        # ── Step 8: Similarity search ─────────────────────────────────────────
        doc_log.info("similarity_search_start", top_k=settings.RAG_TOP_K)
        similar_chunks = await self._embedding_service.similarity_search(
            session=session,
            tenant_id=tenant_id,
            query_vector=query_vector,
            top_k=settings.RAG_TOP_K,
            document_ids=[document_id],
        )
        doc_log.info("similarity_search_complete", retrieved=len(similar_chunks))

        # ── Step 9: Context assembly ──────────────────────────────────────────
        context = _assemble_context(similar_chunks)

        # ── Step 10: Deterministic LLM extraction ─────────────────────────────
        doc_log.info("llm_extraction_start")
        extraction_result = await self._llm.extract_financial_metrics(context=context)
        doc_log.info("llm_extraction_complete",
                     confidence=extraction_result.confidence_score,
                     fallback_used=extraction_result.fallback_used)

        # ── Step 11: Persist extracted metrics ───────────────────────────────
        ext = extraction_result.extraction
        metrics = ExtractedMetrics(
            tenant_id=tenant_id,
            document_id=document_id,
            loan_amount=ext.loan_amount,
            loan_term_months=ext.loan_term_months,
            interest_rate=ext.interest_rate,
            annual_percentage_rate=ext.annual_percentage_rate,
            monthly_payment=ext.monthly_payment,
            origination_fee=ext.origination_fee,
            borrower_income_annual=ext.borrower_income_annual,
            debt_to_income_ratio=ext.debt_to_income_ratio,
            credit_score=ext.credit_score,
            employment_status=ext.employment_status,
            employer_name=ext.employer_name,
            property_value=ext.property_value,
            loan_to_value_ratio=ext.loan_to_value_ratio,
            property_type=ext.property_type,
            property_address=ext.property_address,
            lender_name=ext.lender_name,
            borrower_name=ext.borrower_name,
            loan_purpose=ext.loan_purpose,
            loan_type=ext.loan_type,
            confidence_score=extraction_result.confidence_score,
            extraction_version=EXTRACTION_SCHEMA_VERSION,
            prompt_version=PROMPT_VERSION,
            requires_review=extraction_result.requires_review,
            raw_extraction=extraction_result.model_dump()["extraction"],
            validated_extraction=ext.model_dump(exclude_none=True),
        )
        session.add(metrics)
        await session.flush()

        # ── Step 12: Risk scoring ─────────────────────────────────────────────
        risk_data = _compute_risk_scores(metrics)
        risk = RiskScore(
            tenant_id=tenant_id,
            document_id=document_id,
            metrics_id=metrics.id,
            overall_score=risk_data["overall_score"],
            risk_tier=risk_data["risk_tier"],
            dti_score=risk_data.get("dti_score"),
            ltv_score=risk_data.get("ltv_score"),
            credit_score_n=risk_data.get("credit_score_n"),
            income_score=risk_data.get("income_score"),
            flags=risk_data["flags"],
            scoring_version="v1.0",
        )
        session.add(risk)

        # ── Step 13: Mark document complete ──────────────────────────────────
        doc.status = "completed"
        await session.flush()

        duration_ms = int((time.monotonic() - start_time) * 1000)

        result = {
            "document_id": str(document_id),
            "sha256": sha256,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "embedding_count": len(embedding_ids),
            "confidence_score": extraction_result.confidence_score,
            "requires_review": extraction_result.requires_review,
            "fallback_used": extraction_result.fallback_used,
            "risk_tier": risk_data["risk_tier"],
            "risk_score": risk_data["overall_score"],
            "risk_flags": risk_data["flags"],
            "extracted_metrics": ext.model_dump(exclude_none=True),
            "duration_ms": duration_ms,
            "is_duplicate": False,
        }

        # ── Step 14: Update idempotency cache ─────────────────────────────────
        await idempotency_cache.set_processed(tenant_id, sha256, document_id)
        await idempotency_cache.cache_extraction_result(tenant_id, sha256, result)

        doc_log.info("rag_pipeline_complete",
                     duration_ms=duration_ms,
                     confidence=extraction_result.confidence_score,
                     risk_tier=risk_data["risk_tier"])

        return result
