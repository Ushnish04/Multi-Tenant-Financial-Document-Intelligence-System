"""
tests/unit/test_chunker.py
Unit tests for the semantic chunker.
"""
import pytest
from app.services.rag.chunker import SemanticChunker, TextChunk


class TestSemanticChunker:

    def setup_method(self):
        self.chunker = SemanticChunker(chunk_size=100, overlap=20)

    def test_empty_text_returns_no_chunks(self):
        chunks = self.chunker.chunk("")
        assert chunks == []

    def test_short_text_single_chunk(self):
        text = "This is a short loan document with basic information."
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].token_count > 0

    def test_chunks_are_indexed_sequentially(self):
        text = " ".join(["This is sentence number {}.".format(i) for i in range(50)])
        chunks = self.chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_text_not_empty(self):
        text = "Loan amount is $500,000. Interest rate is 6.5%. " * 20
        chunks = self.chunker.chunk(text)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0
            assert chunk.token_count > 0

    def test_overlap_is_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="Overlap must be smaller"):
            SemanticChunker(chunk_size=50, overlap=60)

    def test_long_document_produces_multiple_chunks(self):
        # Generate text that will exceed chunk_size=100 tokens
        text = "The borrower has excellent credit. " * 100
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1


"""
tests/unit/test_hashing.py
Unit tests for document hashing utilities.
"""
from app.utils.hashing import compute_sha256, compute_sha256_str


class TestHashing:

    def test_sha256_deterministic(self):
        content = b"test pdf content"
        assert compute_sha256(content) == compute_sha256(content)

    def test_sha256_different_content(self):
        assert compute_sha256(b"content_a") != compute_sha256(b"content_b")

    def test_sha256_returns_64_char_hex(self):
        result = compute_sha256(b"test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_sha256_str_matches_bytes(self):
        text = "hello world"
        assert compute_sha256_str(text) == compute_sha256(text.encode("utf-8"))


"""
tests/unit/test_llm_wrapper.py
Unit tests for the deterministic LLM wrapper validation logic.
"""
import pytest
from app.services.llm.wrapper import (
    FinancialExtractionV1,
    _fallback_extract,
    _compute_confidence,
)


class TestFinancialExtractionSchema:

    def test_valid_extraction(self):
        data = {
            "loan_amount": 500000.0,
            "interest_rate": 0.065,
            "credit_score": 750,
        }
        ext = FinancialExtractionV1.model_validate(data)
        assert ext.loan_amount == 500000.0
        assert ext.interest_rate == 0.065

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            FinancialExtractionV1.model_validate({"unknown_field": "value"})

    def test_credit_score_out_of_range(self):
        with pytest.raises(Exception):
            FinancialExtractionV1.model_validate({"credit_score": 900})

    def test_apr_below_interest_rate_rejected(self):
        with pytest.raises(Exception):
            FinancialExtractionV1.model_validate({
                "interest_rate": 0.08,
                "annual_percentage_rate": 0.07,  # APR < rate → invalid
            })

    def test_nulls_allowed(self):
        ext = FinancialExtractionV1.model_validate({})
        assert ext.loan_amount is None
        assert ext.credit_score is None

    def test_dti_must_be_decimal(self):
        with pytest.raises(Exception):
            FinancialExtractionV1.model_validate({"debt_to_income_ratio": 1.5})


class TestFallbackExtraction:

    def test_extracts_loan_amount(self):
        text = "Loan Amount: $450,000.00"
        result = _fallback_extract(text)
        assert result.get("loan_amount") == 450000.0

    def test_extracts_interest_rate_as_decimal(self):
        text = "Interest Rate: 6.75%"
        result = _fallback_extract(text)
        assert abs(result.get("interest_rate", 0) - 0.0675) < 0.0001

    def test_extracts_credit_score(self):
        text = "FICO Score: 742"
        result = _fallback_extract(text)
        assert result.get("credit_score") == 742

    def test_extracts_dti_as_decimal(self):
        text = "DTI: 38.5%"
        result = _fallback_extract(text)
        assert abs(result.get("debt_to_income_ratio", 0) - 0.385) < 0.001

    def test_returns_empty_dict_for_no_matches(self):
        result = _fallback_extract("This text has no financial data")
        assert result == {}


class TestConfidenceScoring:

    def test_full_extraction_high_confidence(self):
        ext = FinancialExtractionV1.model_validate({
            "loan_amount": 500000.0,
            "interest_rate": 0.065,
            "credit_score": 750,
            "debt_to_income_ratio": 0.38,
            "loan_to_value_ratio": 0.80,
            "annual_percentage_rate": 0.068,
            "monthly_payment": 3200.0,
            "loan_term_months": 360,
            "borrower_income_annual": 120000.0,
            "property_value": 625000.0,
        })
        score, coverage = _compute_confidence(ext, fallback_used=False, validation_errors=[])
        assert score > 0.5

    def test_empty_extraction_low_confidence(self):
        ext = FinancialExtractionV1.model_validate({})
        score, coverage = _compute_confidence(ext, fallback_used=False, validation_errors=[])
        assert score < 0.3

    def test_fallback_penalizes_confidence(self):
        ext = FinancialExtractionV1.model_validate({
            "loan_amount": 500000.0,
            "credit_score": 750,
            "interest_rate": 0.065,
        })
        score_no_fallback, _ = _compute_confidence(ext, False, [])
        score_fallback, _ = _compute_confidence(ext, True, [])
        assert score_fallback < score_no_fallback
