"""
app/services/auth/auth_service.py
JWT authentication service with RBAC, refresh tokens, and revocation.
"""
from datetime import datetime, timezone
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.exceptions import AuthenticationError, AuthorizationError, TokenRevocationError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from app.db.models.models import RefreshToken, Role, User, UserRole

log = structlog.get_logger(__name__)


class AuthService:
    """Authentication service handling login, token lifecycle, and RBAC."""

    async def authenticate_user(
        self, session: AsyncSession, email: str, password: str
    ) -> User:
        """Verify credentials. Returns User or raises AuthenticationError."""
        result = await session.execute(
            select(User)
            .where(User.email == email, User.is_active == True)  # noqa: E712
        )
        user = result.scalar_one_or_none()

        if not user or not verify_password(password, user.hashed_password):
            # Constant-time comparison prevents timing attacks
            log.warning("auth_failed", email=email)
            raise AuthenticationError("Invalid email or password")

        if not user.is_verified:
            raise AuthenticationError("Email not verified")

        log.info("auth_success", user_id=str(user.id), tenant_id=str(user.tenant_id))
        return user

    async def get_user_roles_and_permissions(
        self, session: AsyncSession, user_id: UUID, tenant_id: UUID
    ) -> tuple[list[str], list[str]]:
        """Fetch user's roles and flattened permissions for JWT claims."""
        result = await session.execute(
            select(UserRole)
            .options(joinedload(UserRole.role))
            .where(UserRole.user_id == user_id, UserRole.tenant_id == tenant_id)
        )
        user_roles = result.scalars().all()

        roles: list[str] = []
        permissions: set[str] = set()
        for ur in user_roles:
            roles.append(ur.role.name)
            permissions.update(ur.role.permissions or [])

        return roles, list(permissions)

    async def login(
        self,
        session: AsyncSession,
        email: str,
        password: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> dict:
        """
        Authenticate and issue access + refresh tokens.
        Returns token pair with expiry metadata.
        """
        user = await self.authenticate_user(session, email, password)
        roles, permissions = await self.get_user_roles_and_permissions(
            session, user.id, user.tenant_id
        )

        access_token = create_access_token(
            user_id=user.id,
            tenant_id=user.tenant_id,
            email=user.email,
            roles=roles,
            permissions=permissions,
        )
        refresh_token_raw, refresh_token_hash = create_refresh_token(
            user_id=user.id,
            tenant_id=user.tenant_id,
        )

        # Store hashed refresh token
        from app.core.config import get_settings
        settings = get_settings()
        from datetime import timedelta
        rt = RefreshToken(
            user_id=user.id,
            tenant_id=user.tenant_id,
            token_hash=refresh_token_hash,
            expires_at=datetime.now(tz=timezone.utc) + timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            user_agent=user_agent,
            ip_address=ip_address,
        )
        session.add(rt)

        # Update last login
        user.last_login_at = datetime.now(tz=timezone.utc)
        await session.flush()

        return {
            "access_token": access_token,
            "refresh_token": refresh_token_raw,
            "token_type": "bearer",
            "user_id": str(user.id),
            "tenant_id": str(user.tenant_id),
            "roles": roles,
        }

    async def refresh_tokens(
        self, session: AsyncSession, refresh_token: str
    ) -> dict:
        """
        Issue new access token from valid refresh token.
        Implements refresh token rotation (old token revoked, new one issued).
        """
        payload = decode_refresh_token(refresh_token)
        token_hash = hash_token(refresh_token)

        # Verify token exists and is not revoked
        result = await session.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked_at == None,  # noqa: E711
            )
        )
        stored_token = result.scalar_one_or_none()

        if not stored_token:
            log.warning("refresh_token_invalid_or_revoked", token_hash=token_hash[:12])
            raise TokenRevocationError()

        if stored_token.expires_at < datetime.now(tz=timezone.utc):
            raise AuthenticationError("Refresh token expired")

        # Revoke current token (rotation)
        stored_token.revoked_at = datetime.now(tz=timezone.utc)

        user_id = UUID(payload["sub"])
        tenant_id = UUID(payload["tenant_id"])

        # Load user
        user = await session.get(User, user_id)
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")

        roles, permissions = await self.get_user_roles_and_permissions(
            session, user_id, tenant_id
        )

        # Issue new access token
        new_access_token = create_access_token(
            user_id=user_id,
            tenant_id=tenant_id,
            email=user.email,
            roles=roles,
            permissions=permissions,
        )

        # Issue new refresh token (rotation)
        new_refresh_raw, new_refresh_hash = create_refresh_token(user_id, tenant_id)
        from app.core.config import get_settings
        settings = get_settings()
        from datetime import timedelta
        new_rt = RefreshToken(
            user_id=user_id,
            tenant_id=tenant_id,
            token_hash=new_refresh_hash,
            expires_at=datetime.now(tz=timezone.utc) + timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        session.add(new_rt)
        await session.flush()

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_raw,
            "token_type": "bearer",
        }

    async def revoke_all_user_tokens(
        self, session: AsyncSession, user_id: UUID, tenant_id: UUID
    ) -> int:
        """Revoke all active refresh tokens for a user (logout from all devices)."""
        from sqlalchemy import update
        result = await session.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.tenant_id == tenant_id,
                RefreshToken.revoked_at == None,  # noqa: E711
            )
        )
        tokens = result.scalars().all()
        now = datetime.now(tz=timezone.utc)
        count = 0
        for token in tokens:
            token.revoked_at = now
            count += 1
        await session.flush()
        log.info("tokens_revoked", user_id=str(user_id), count=count)
        return count
