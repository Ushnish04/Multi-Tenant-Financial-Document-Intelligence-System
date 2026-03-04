"""
app/main.py
FastAPI application entrypoint.
Configures middleware, exception handlers, routers, and startup events.
"""
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import auth, documents, audit
from app.core.config import get_settings
from app.core.exceptions import (
    AuthenticationError, AuthorizationError, FinRAGException,
    TenantIsolationError,
)
from app.middleware.tenant import TenantScopeMiddleware

settings = get_settings()
log = structlog.get_logger(__name__)


# =============================================================================
# STRUCTURED LOGGING CONFIGURATION
# =============================================================================

def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            20 if settings.ENVIRONMENT == "production" else 10
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# =============================================================================
# LIFESPAN (startup / shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    configure_logging()
    log.info("finrag_starting",
             version=settings.APP_VERSION,
             environment=settings.ENVIRONMENT)

    # Validate DB connectivity on startup
    from app.db.base import engine
    from sqlalchemy import text
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        log.info("database_connected")
    except Exception as exc:
        log.error("database_connection_failed", error=str(exc))
        raise

    # Validate Redis connectivity
    from app.utils.hashing import get_redis
    try:
        redis = await get_redis()
        await redis.ping()
        log.info("redis_connected")
    except Exception as exc:
        log.error("redis_connection_failed", error=str(exc))
        raise

    log.info("finrag_ready")
    yield

    # Shutdown
    await engine.dispose()
    log.info("finrag_shutdown")


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_application() -> FastAPI:
    app = FastAPI(
        title="FinRAG — Financial Document Intelligence API",
        version=settings.APP_VERSION,
        docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/api/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ── Middleware stack (applied in reverse order) ───────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # ── Request ID + timing middleware ────────────────────────────────────────
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)

        log.info("request_complete",
                 status_code=response.status_code,
                 duration_ms=duration_ms)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Duration-MS"] = str(duration_ms)
        return response

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(TenantIsolationError)
    async def tenant_isolation_handler(request: Request, exc: TenantIsolationError):
        log.error("tenant_isolation_violation",
                  path=request.url.path,
                  code=exc.code)
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": exc.code, "detail": "Access denied"},
        )

    @app.exception_handler(AuthenticationError)
    async def auth_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": exc.code, "detail": exc.message},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(AuthorizationError)
    async def authz_handler(request: Request, exc: AuthorizationError):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": exc.code, "detail": exc.message},
        )

    @app.exception_handler(FinRAGException)
    async def finrag_handler(request: Request, exc: FinRAGException):
        log.error("domain_exception",
                  code=exc.code,
                  message=exc.message,
                  context=exc.context)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": exc.code, "detail": exc.message, "context": exc.context},
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "INTERNAL_ERROR", "detail": "An unexpected error occurred"},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
    app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
    app.include_router(audit.router, prefix=settings.API_V1_PREFIX)

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy", "version": settings.APP_VERSION}

    @app.get("/health/ready", tags=["Health"])
    async def readiness_check():
        """Deep health check validating DB + Redis connectivity."""
        checks: dict[str, str] = {}
        from app.db.base import engine
        from sqlalchemy import text

        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["database"] = "healthy"
        except Exception as exc:
            checks["database"] = f"unhealthy: {exc}"

        try:
            from app.utils.hashing import get_redis
            r = await get_redis()
            await r.ping()
            checks["redis"] = "healthy"
        except Exception as exc:
            checks["redis"] = f"unhealthy: {exc}"

        is_ready = all(v == "healthy" for v in checks.values())
        return JSONResponse(
            status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "ready" if is_ready else "not_ready", "checks": checks},
        )

    return app


app = create_application()
