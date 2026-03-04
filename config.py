"""
app/core/config.py
Production configuration via pydantic-settings.
All secrets sourced from environment variables / secrets manager.
"""
from functools import lru_cache
from typing import Literal
from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    APP_NAME: str = "FinRAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "production"
    DEBUG: bool = False

    # --- API ---
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: list[str] = ["https://app.yourcompany.com"]
    MAX_UPLOAD_SIZE_MB: int = 50

    # --- Database ---
    DATABASE_URL: PostgresDsn
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False

    # --- Redis ---
    REDIS_URL: RedisDsn
    CACHE_TTL_SECONDS: int = 86400          # 24h idempotency cache
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # --- JWT ---
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- LLM ---
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4o"
    LLM_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_EMBEDDING_DIMENSIONS: int = 1536
    LLM_TEMPERATURE: float = 0.0            # NEVER change — determinism
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT_SECONDS: int = 30
    LLM_MAX_RETRIES: int = 3

    # --- RAG ---
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    RAG_TOP_K: int = 5
    MIN_CONFIDENCE_SCORE: float = 0.75      # Below this → requires_review=True

    # --- Extraction ---
    EXTRACTION_SCHEMA_VERSION: str = "v1.0"
    PROMPT_VERSION: str = "v1.0"

    # --- Storage ---
    STORAGE_BACKEND: Literal["local", "s3", "gcs"] = "s3"
    S3_BUCKET: str = ""
    S3_REGION: str = "us-east-1"
    LOCAL_STORAGE_PATH: str = "/data/uploads"

    # --- Audit ---
    AUDIT_LOG_RETENTION_DAYS: int = 2555    # 7 years (financial compliance)
    GDPR_SAFE_LOGGING: bool = True

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def temperature_must_be_zero(cls, v: float) -> float:
        if v != 0.0:
            raise ValueError("LLM_TEMPERATURE must be 0.0 for deterministic outputs")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
