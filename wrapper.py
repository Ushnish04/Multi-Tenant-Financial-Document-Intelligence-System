"""
app/services/llm/wrapper.py
Deterministic LLM wrapper with:
- Forced temperature=0
- Frozen prompt versioning
- Strict Pydantic JSON schema validation
- Numeric range validation
- Confidence scoring
- Regex fallback extraction
- Exponential retry
- Structured audit logging
"""
import hashlib
import json
import re
import time
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog
from openai import AsyncOpenAI, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import get_settings
from app.core.exceptions import LLMTimeoutError, LLMValidationError, ExtractionError
from app.utils.retry import async_retry_with_backoff

settings = get_settings()
log = structlog.get_logger(__name__)


# =============================================================================
# EXTRACTION SCHEMA — STRICT PYDANTIC MODEL (v1.0)
# Schema is versioned and frozen. Changes increment version → new model class.
# =============================================================================

class FinancialExtractionV1(BaseModel):
    """
    Strict extraction schema for loan documents.
    Version: v1.0 — DO NOT MODIFY. Create FinancialExtractionV2 for changes.
    All fields optional to handle partial documents; confidence reflects completeness.
    """
    model_config = {"extra": "forbid", "strict": True}

    # Loan terms
    loan_amount: float | None = Field(None, ge=1_000, le=100_000_000,
                                       description="Principal loan amount in USD")
    loan_term_months: int | None = Field(None, ge=1, le=600,
                                          description="Loan term in months")
    interest_rate: float | None = Field(None, ge=0.001, le=0.50,
                                         description="Annual interest rate as decimal (0.07 = 7%)")
    annual_percentage_rate: float | None = Field(None, ge=0.001, le=0.60,
                                                  description="APR as decimal")
    monthly_payment: float | None = Field(None, ge=0, le=1_000_000)
    origination_fee: float | None = Field(None, ge=0, le=100_000)

    # Borrower financials
    borrower_income_annual: float | None = Field(None, ge=0, le=100_000_000)
    debt_to_income_ratio: float | None = Field(None, ge=0.0, le=1.0,
                                                description="DTI as decimal (0.43 = 43%)")
    credit_score: int | None = Field(None, ge=300, le=850)
    employment_status: str | None = Field(None, max_length=100)
    employer_name: str | None = Field(None, max_length=255)

    # Property / collateral
    property_value: float | None = Field(None, ge=0, le=1_000_000_000)
    loan_to_value_ratio: float | None = Field(None, ge=0.0, le=2.0,
                                               description="LTV as decimal (0.80 = 80%)")
    property_type: str | None = Field(None, max_length=100)
    property_address: str | None = Field(None, max_length=500)

    # Document metadata
    document_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$",
                                       description="ISO 8601 date: YYYY-MM-DD")
    lender_name: str | None = Field(None, max_length=255)
    borrower_name: str | None = Field(None, max_length=255)
    loan_purpose: str | None = Field(None, max_length=255)
    loan_type: str | None = Field(None, max_length=100)

    @field_validator("interest_rate", "annual_percentage_rate")
    @classmethod
    def rate_cannot_be_zero(cls, v: float | None) -> float | None:
        if v is not None and v == 0.0:
            raise ValueError("Interest rate cannot be exactly 0")
        return v

    @model_validator(mode="after")
    def apr_must_exceed_interest_rate(self) -> "FinancialExtractionV1":
        if (self.annual_percentage_rate is not None and
                self.interest_rate is not None and
                self.annual_percentage_rate < self.interest_rate):
            raise ValueError("APR cannot be less than interest rate")
        return self


class ExtractionResult(BaseModel):
    """Wrapper including confidence and validation metadata."""
    extraction: FinancialExtractionV1
    confidence_score: float = Field(ge=0.0, le=1.0)
    requires_review: bool
    fallback_used: bool
    validation_errors: list[str]
    field_coverage: float   # Fraction of non-None fields
    raw_response: str
    prompt_hash: str
    context_hash: str


# =============================================================================
# FROZEN PROMPT TEMPLATES
# Versioned and immutable. Hash is stored in audit log.
# =============================================================================

EXTRACTION_PROMPT_V1 = """You are a precise financial document analyst. Extract structured data from the provided loan document context.

CRITICAL RULES:
1. Return ONLY valid JSON matching the exact schema below. No markdown, no explanation.
2. Use null for any field not clearly stated in the document.
3. Express rates as decimals: 7.5% → 0.075
4. Express DTI and LTV as decimals: 43% → 0.43
5. All monetary amounts in USD without formatting.
6. Dates in YYYY-MM-DD format only.
7. Do not infer or estimate values not explicitly present.

REQUIRED JSON SCHEMA:
{{
  "loan_amount": <number|null>,
  "loan_term_months": <integer|null>,
  "interest_rate": <number|null>,
  "annual_percentage_rate": <number|null>,
  "monthly_payment": <number|null>,
  "origination_fee": <number|null>,
  "borrower_income_annual": <number|null>,
  "debt_to_income_ratio": <number|null>,
  "credit_score": <integer|null>,
  "employment_status": <string|null>,
  "employer_name": <string|null>,
  "property_value": <number|null>,
  "loan_to_value_ratio": <number|null>,
  "property_type": <string|null>,
  "property_address": <string|null>,
  "document_date": <"YYYY-MM-DD"|null>,
  "lender_name": <string|null>,
  "borrower_name": <string|null>,
  "loan_purpose": <string|null>,
  "loan_type": <string|null>
}}

DOCUMENT CONTEXT:
{context}

Respond with the JSON object only."""

PROMPT_HASH_V1 = hashlib.sha256(EXTRACTION_PROMPT_V1.encode()).hexdigest()[:16]


# =============================================================================
# REGEX FALLBACK PATTERNS
# Used when LLM output fails JSON parsing
# =============================================================================

FALLBACK_PATTERNS: dict[str, re.Pattern] = {
    "loan_amount": re.compile(
        r"(?:loan\s+amount|principal)[:\s]+\$?([\d,]+(?:\.\d{2})?)", re.IGNORECASE),
    "interest_rate": re.compile(
        r"(?:interest\s+rate|rate)[:\s]+([\d.]+)\s*%", re.IGNORECASE),
    "credit_score": re.compile(
        r"(?:credit\s+score|fico)[:\s]+(\d{3})", re.IGNORECASE),
    "loan_term_months": re.compile(
        r"(\d+)[- ](?:month|mo)(?:s)?\s+(?:term|loan)", re.IGNORECASE),
    "debt_to_income_ratio": re.compile(
        r"(?:dti|debt[- ]to[- ]income)[:\s]+([\d.]+)\s*%", re.IGNORECASE),
    "loan_to_value_ratio": re.compile(
        r"(?:ltv|loan[- ]to[- ]value)[:\s]+([\d.]+)\s*%", re.IGNORECASE),
    "annual_percentage_rate": re.compile(
        r"(?:apr|annual\s+percentage\s+rate)[:\s]+([\d.]+)\s*%", re.IGNORECASE),
    "monthly_payment": re.compile(
        r"(?:monthly\s+payment|payment)[:\s]+\$?([\d,]+(?:\.\d{2})?)", re.IGNORECASE),
    "property_value": re.compile(
        r"(?:property\s+value|appraised\s+value)[:\s]+\$?([\d,]+(?:\.\d{2})?)", re.IGNORECASE),
}


def _fallback_extract(text: str) -> dict[str, Any]:
    """Regex-based extraction as fallback when LLM JSON fails."""
    result: dict[str, Any] = {}

    for field, pattern in FALLBACK_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        try:
            if field in ("credit_score", "loan_term_months"):
                result[field] = int(raw)
            elif field in ("interest_rate", "annual_percentage_rate",
                           "debt_to_income_ratio", "loan_to_value_ratio"):
                # Convert percentage to decimal
                result[field] = float(raw) / 100.0
            else:
                result[field] = float(raw)
        except ValueError:
            pass

    return result


def _compute_confidence(extraction: FinancialExtractionV1,
                        fallback_used: bool,
                        validation_errors: list[str]) -> tuple[float, float]:
    """
    Compute confidence score and field coverage fraction.
    
    Scoring:
    - Base: field coverage (non-null fields / total fields)
    - Penalty: -0.15 for fallback usage
    - Penalty: -0.05 per validation error (capped at -0.30)
    - Critical fields bonus: loan_amount, credit_score, interest_rate present
    """
    all_fields = extraction.model_fields.keys()
    non_null = sum(1 for f in all_fields if getattr(extraction, f) is not None)
    coverage = non_null / len(list(all_fields))

    score = coverage
    if fallback_used:
        score -= 0.15
    score -= min(len(validation_errors) * 0.05, 0.30)

    # Bonus for critical financial fields
    critical = [extraction.loan_amount, extraction.credit_score, extraction.interest_rate]
    if all(v is not None for v in critical):
        score = min(score + 0.10, 1.0)

    return max(0.0, min(1.0, score)), coverage


# =============================================================================
# DETERMINISTIC LLM WRAPPER
# =============================================================================

class DeterministicLLMWrapper:
    """
    Production LLM wrapper enforcing determinism, validation, and audit trails.
    temperature=0 is enforced — any deviation raises a hard error.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
        # Validate temperature lock
        assert settings.LLM_TEMPERATURE == 0.0, \
            "CRITICAL: LLM_TEMPERATURE must be 0.0. Check configuration."

    @async_retry_with_backoff(
        max_retries=settings.LLM_MAX_RETRIES,
        base_delay=1.0,
        exceptions=(APITimeoutError, RateLimitError),
    )
    async def _call_llm(self, prompt: str) -> tuple[str, dict[str, int]]:
        """
        Raw LLM API call with deterministic settings.
        Returns (raw_text_response, token_usage).
        """
        response = await self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial document parsing system. "
                               "Return only valid JSON. No prose, no markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,            # HARD LOCK — never change
            max_tokens=settings.LLM_MAX_TOKENS,
            response_format={"type": "json_object"},  # Force JSON mode
            seed=42,                    # Additional determinism signal (where supported)
        )

        content = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        return content, usage

    async def extract_financial_metrics(
        self,
        context: str,
    ) -> ExtractionResult:
        """
        Main extraction entry point.
        
        1. Build frozen prompt
        2. Call LLM (temperature=0, JSON mode)
        3. Validate Pydantic schema
        4. On failure → fallback regex extraction
        5. Compute confidence score
        6. Return structured ExtractionResult
        """
        start_time = time.monotonic()
        fallback_used = False
        validation_errors: list[str] = []

        # Build prompt with frozen template
        prompt = EXTRACTION_PROMPT_V1.format(context=context)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        context_hash = hashlib.sha256(context.encode()).hexdigest()

        log.info("llm_extraction_start",
                 prompt_hash=prompt_hash[:12],
                 context_length=len(context))

        # LLM call
        raw_response, token_usage = await self._call_llm(prompt)

        # Primary validation: parse and validate JSON
        extraction_data: dict[str, Any] | None = None
        try:
            extraction_data = json.loads(raw_response)
            validated = FinancialExtractionV1.model_validate(extraction_data)
        except (json.JSONDecodeError, Exception) as exc:
            validation_errors.append(str(exc))
            log.warning("llm_primary_validation_failed",
                        error=str(exc),
                        prompt_hash=prompt_hash[:12])

            # Fallback: regex extraction from raw response + original context
            log.info("llm_fallback_extraction_triggered")
            fallback_data = _fallback_extract(raw_response + "\n" + context)
            fallback_used = True

            try:
                validated = FinancialExtractionV1.model_validate(fallback_data)
            except Exception as fallback_exc:
                validation_errors.append(f"fallback_failed: {fallback_exc}")
                # Return minimal valid extraction rather than raising
                validated = FinancialExtractionV1.model_validate({})

        confidence, coverage = _compute_confidence(validated, fallback_used, validation_errors)
        requires_review = confidence < settings.MIN_CONFIDENCE_SCORE

        duration_ms = int((time.monotonic() - start_time) * 1000)

        log.info("llm_extraction_complete",
                 confidence=confidence,
                 coverage=coverage,
                 fallback_used=fallback_used,
                 requires_review=requires_review,
                 duration_ms=duration_ms,
                 **token_usage)

        return ExtractionResult(
            extraction=validated,
            confidence_score=confidence,
            requires_review=requires_review,
            fallback_used=fallback_used,
            validation_errors=validation_errors,
            field_coverage=coverage,
            raw_response=raw_response,
            prompt_hash=prompt_hash,
            context_hash=context_hash,
        )
