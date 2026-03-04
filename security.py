"""
app/core/security.py
JWT access/refresh token generation, bcrypt password hashing,
and token validation utilities.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, TokenRevocationError

settings = get_settings()

# Bcrypt with cost factor 12 (OWASP recommended minimum)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

# JWT claims structure
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def hash_password(plain_password: str) -> str:
    """Hash password with bcrypt."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time bcrypt verification."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    user_id: UUID,
    tenant_id: UUID,
    email: str,
    roles: list[str],
    permissions: list[str],
) -> str:
    """
    Create a short-lived JWT access token.
    Embeds tenant_id as a claim for middleware enforcement.
    """
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "email": email,
        "roles": roles,
        "permissions": permissions,
        "type": TOKEN_TYPE_ACCESS,
        "iat": now,
        "exp": expire,
        "jti": secrets.token_hex(16),   # Unique token ID for revocation
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(
    user_id: UUID,
    tenant_id: UUID,
) -> tuple[str, str]:
    """
    Create a long-lived refresh token.
    Returns (raw_token, sha256_hash) — only hash is stored in DB.
    """
    now = datetime.now(tz=timezone.utc)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    raw_token = secrets.token_urlsafe(64)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "type": TOKEN_TYPE_REFRESH,
        "iat": now,
        "exp": expire,
        "jti": secrets.token_hex(16),
    }
    signed_token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    # Hash the raw token for storage (never store raw refresh tokens)
    token_hash = hashlib.sha256(signed_token.encode()).hexdigest()
    return signed_token, token_hash


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode and validate JWT access token.
    Raises AuthenticationError on any failure.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != TOKEN_TYPE_ACCESS:
            raise AuthenticationError("Invalid token type")
        return payload
    except JWTError as exc:
        raise AuthenticationError(f"Token validation failed: {exc}") from exc


def decode_refresh_token(token: str) -> dict[str, Any]:
    """Decode and validate JWT refresh token."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != TOKEN_TYPE_REFRESH:
            raise AuthenticationError("Invalid token type")
        return payload
    except JWTError as exc:
        raise AuthenticationError(f"Refresh token validation failed: {exc}") from exc


def hash_token(token: str) -> str:
    """SHA-256 hash a token string for DB storage."""
    return hashlib.sha256(token.encode()).hexdigest()
