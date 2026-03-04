"""
app/core/exceptions.py
Domain-specific exceptions with structured error context.
"""
from typing import Any


class FinRAGException(Exception):
    """Base exception for all FinRAG errors."""
    def __init__(self, message: str, code: str, context: dict[str, Any] | None = None):
        self.message = message
        self.code = code
        self.context = context or {}
        super().__init__(message)


class TenantIsolationError(FinRAGException):
    """Raised when a cross-tenant access attempt is detected."""
    def __init__(self, detail: str = "Cross-tenant access denied"):
        super().__init__(detail, "TENANT_ISOLATION_VIOLATION")


class DocumentNotFoundError(FinRAGException):
    def __init__(self, document_id: str):
        super().__init__(f"Document {document_id} not found", "DOCUMENT_NOT_FOUND",
                         {"document_id": document_id})


class DuplicateDocumentError(FinRAGException):
    def __init__(self, sha256: str, existing_doc_id: str):
        super().__init__("Document already processed", "DUPLICATE_DOCUMENT",
                         {"sha256": sha256, "existing_document_id": existing_doc_id})


class LLMTimeoutError(FinRAGException):
    def __init__(self, model: str, timeout: int):
        super().__init__(f"LLM call timed out after {timeout}s", "LLM_TIMEOUT",
                         {"model": model, "timeout_seconds": timeout})


class LLMValidationError(FinRAGException):
    def __init__(self, errors: list[dict]):
        super().__init__("LLM response failed schema validation", "LLM_VALIDATION_FAILED",
                         {"validation_errors": errors})


class ExtractionError(FinRAGException):
    def __init__(self, detail: str):
        super().__init__(detail, "EXTRACTION_FAILED")


class EmbeddingError(FinRAGException):
    def __init__(self, detail: str):
        super().__init__(detail, "EMBEDDING_FAILED")


class AuthenticationError(FinRAGException):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(detail, "AUTH_FAILED")


class AuthorizationError(FinRAGException):
    def __init__(self, required_permission: str):
        super().__init__(f"Permission denied: {required_permission}", "PERMISSION_DENIED",
                         {"required_permission": required_permission})


class TokenRevocationError(FinRAGException):
    def __init__(self):
        super().__init__("Token has been revoked", "TOKEN_REVOKED")
