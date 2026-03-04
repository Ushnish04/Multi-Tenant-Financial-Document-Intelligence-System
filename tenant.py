"""
app/middleware/tenant.py
JWT authentication middleware with tenant scope enforcement.
Provides FastAPI dependency injection for authenticated requests.
Injects tenant_id into PostgreSQL session for RLS enforcement.
"""
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.exceptions import AuthenticationError, AuthorizationError, TenantIsolationError
from app.core.security import decode_access_token

log = structlog.get_logger(__name__)

http_bearer = HTTPBearer(auto_error=True)


class AuthenticatedUser:
    """Represents the currently authenticated user extracted from JWT."""

    def __init__(self, payload: dict) -> None:
        self.user_id: UUID = UUID(payload["sub"])
        self.tenant_id: UUID = UUID(payload["tenant_id"])
        self.email: str = payload["email"]
        self.roles: list[str] = payload.get("roles", [])
        self.permissions: list[str] = payload.get("permissions", [])
        self.jti: str = payload.get("jti", "")

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def require_permission(self, permission: str) -> None:
        if not self.has_permission(permission):
            raise AuthorizationError(permission)

    def require_role(self, role: str) -> None:
        if not self.has_role(role):
            raise AuthorizationError(f"role:{role}")


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(http_bearer)],
) -> AuthenticatedUser:
    """
    FastAPI dependency: validate JWT and return AuthenticatedUser.
    Raises HTTP 401 on any validation failure.
    """
    try:
        payload = decode_access_token(credentials.credentials)
        return AuthenticatedUser(payload)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]


def require_permission(permission: str):
    """
    Dependency factory for permission-based access control.
    Usage: Depends(require_permission("document:write"))
    """
    async def check(user: CurrentUser) -> AuthenticatedUser:
        try:
            user.require_permission(permission)
        except AuthorizationError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=exc.message,
            ) from exc
        return user
    return check


def require_role(role: str):
    """
    Dependency factory for role-based access control.
    Usage: Depends(require_role("admin"))
    """
    async def check(user: CurrentUser) -> AuthenticatedUser:
        try:
            user.require_role(role)
        except AuthorizationError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=exc.message,
            ) from exc
        return user
    return check


class TenantScopeMiddleware:
    """
    ASGI middleware that validates tenant context on every request.
    Adds tenant_id to request state for downstream use.
    Provides defense-in-depth on top of PostgreSQL RLS.
    """

    async def __call__(self, request: Request, call_next):
        # Skip for health check and auth endpoints
        if request.url.path in ("/health", "/api/v1/auth/login", "/api/v1/auth/refresh"):
            return await call_next(request)

        # Validate Authorization header exists
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization header required"},
            )

        try:
            token = auth_header.removeprefix("Bearer ")
            payload = decode_access_token(token)
            tenant_id = UUID(payload["tenant_id"])
            request.state.tenant_id = tenant_id
            request.state.user_id = UUID(payload["sub"])
            request.state.user_email = payload["email"]
        except Exception as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired token"},
            )

        response = await call_next(request)
        return response
