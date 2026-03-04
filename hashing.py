"""
app/utils/hashing.py
SHA-256 document hashing for idempotent deduplication.
Includes Redis-based idempotency cache for fast duplicate detection.
"""
import hashlib
import json
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis

from app.core.config import get_settings

settings = get_settings()


def compute_sha256(content: bytes) -> str:
    """Compute SHA-256 hash of binary content. Returns hex string."""
    return hashlib.sha256(content).hexdigest()


def compute_sha256_str(text: str) -> str:
    """Compute SHA-256 hash of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IdempotencyCache:
    """
    Redis-backed idempotency cache for duplicate document detection.
    Stores sha256_hash → document_id mappings with TTL.
    
    Cache key format: finrag:idem:{tenant_id}:{sha256_hash}
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._ttl = settings.CACHE_TTL_SECONDS

    def _cache_key(self, tenant_id: UUID, sha256_hash: str) -> str:
        return f"finrag:idem:{tenant_id}:{sha256_hash}"

    def _result_key(self, tenant_id: UUID, sha256_hash: str) -> str:
        return f"finrag:result:{tenant_id}:{sha256_hash}"

    async def check_duplicate(
        self, tenant_id: UUID, sha256_hash: str
    ) -> str | None:
        """
        Check if document hash exists in cache.
        Returns document_id string if duplicate, else None.
        """
        key = self._cache_key(tenant_id, sha256_hash)
        result = await self._redis.get(key)
        return result.decode() if result else None

    async def set_processed(
        self, tenant_id: UUID, sha256_hash: str, document_id: UUID
    ) -> None:
        """Mark document as processed in cache."""
        key = self._cache_key(tenant_id, sha256_hash)
        await self._redis.setex(key, self._ttl, str(document_id))

    async def cache_extraction_result(
        self, tenant_id: UUID, sha256_hash: str, result: dict[str, Any]
    ) -> None:
        """Cache full extraction result for idempotent re-retrieval."""
        key = self._result_key(tenant_id, sha256_hash)
        await self._redis.setex(key, self._ttl, json.dumps(result))

    async def get_cached_result(
        self, tenant_id: UUID, sha256_hash: str
    ) -> dict[str, Any] | None:
        """Retrieve cached extraction result."""
        key = self._result_key(tenant_id, sha256_hash)
        result = await self._redis.get(key)
        return json.loads(result) if result else None

    async def invalidate(self, tenant_id: UUID, sha256_hash: str) -> None:
        """Remove cache entries for a document (e.g., on reprocessing)."""
        await self._redis.delete(
            self._cache_key(tenant_id, sha256_hash),
            self._result_key(tenant_id, sha256_hash),
        )


# Redis client singleton factory
_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Get or create Redis async client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            str(settings.REDIS_URL),
            encoding="utf-8",
            decode_responses=False,
            max_connections=50,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
        )
    return _redis_client
