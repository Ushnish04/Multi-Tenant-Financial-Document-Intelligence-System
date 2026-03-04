"""
app/api/v1/endpoints/auth.py
Authentication endpoints: login and token refresh.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError, TokenRevocationError
from app.db.base import get_tenant_session
from app.middleware.tenant import CurrentUser
from app.services.auth.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])
_auth_service = AuthService()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    tenant_slug: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user_id: str
    tenant_id: str
    roles: list[str]


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login", response_model=TokenResponse)
async def login(request: Request, body: LoginRequest):
    """
    Authenticate with email + password + tenant slug.
    Returns JWT access token (15min) + refresh token (7days).
    """
    from sqlalchemy import select, text
    from app.db.models.models import Tenant
    from app.db.base import AsyncSessionFactory

    # First: resolve tenant without RLS (login doesn't have a JWT yet)
    async with AsyncSessionFactory() as session:
        result = await session.execute(
            select(Tenant).where(Tenant.slug == body.tenant_slug, Tenant.is_active == True)
        )
        tenant = result.scalar_one_or_none()
        if not tenant:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Invalid tenant or credentials")

        # Now set RLS context and authenticate
        await session.execute(
            text("SELECT set_config('app.current_tenant_id', :tid, TRUE)"),
            {"tid": str(tenant.id)},
        )
        try:
            async with session.begin():
                tokens = await _auth_service.login(
                    session=session,
                    email=body.email,
                    password=body.password,
                    user_agent=request.headers.get("User-Agent"),
                    ip_address=request.client.host if request.client else None,
                )
        except AuthenticationError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail=exc.message) from exc

    return TokenResponse(**tokens)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest):
    """Rotate refresh token and issue new access token."""
    from app.db.base import AsyncSessionFactory
    from sqlalchemy import text
    from app.core.security import decode_refresh_token
    from uuid import UUID

    try:
        payload = decode_refresh_token(body.refresh_token)
        tenant_id = UUID(payload["tenant_id"])
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message)

    async with AsyncSessionFactory() as session:
        await session.execute(
            text("SELECT set_config('app.current_tenant_id', :tid, TRUE)"),
            {"tid": str(tenant_id)},
        )
        try:
            async with session.begin():
                tokens = await _auth_service.refresh_tokens(session, body.refresh_token)
        except (AuthenticationError, TokenRevocationError) as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message)

    return TokenResponse(**tokens, user_id="", tenant_id=str(tenant_id), roles=[])


@router.post("/logout")
async def logout(user: CurrentUser):
    """Revoke all user refresh tokens (logout from all devices)."""
    from app.db.base import get_tenant_session
    async with get_tenant_session(user.tenant_id) as session:
        async with session.begin():
            count = await _auth_service.revoke_all_user_tokens(
                session, user.user_id, user.tenant_id
            )
    return {"revoked_tokens": count, "message": "Logged out from all devices"}
