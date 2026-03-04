"""
app/db/base.py
Async SQLAlchemy engine, session factory, and RLS session context manager.
Enforces per-request tenant isolation via PostgreSQL session variables.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import UUID

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text, event

from app.core.config import get_settings

settings = get_settings()

# Async engine with production-grade pooling
engine = create_async_engine(
    str(settings.DATABASE_URL).replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,         # Validate connections before use
    pool_recycle=3600,          # Recycle connections hourly
    echo=settings.DB_ECHO,
    connect_args={
        "server_settings": {
            "search_path": "finrag,public",
            "application_name": "finrag_api",
        }
    },
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


@asynccontextmanager
async def get_tenant_session(
    tenant_id: UUID,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager that provides a session with RLS tenant scope set.
    Sets app.current_tenant_id PostgreSQL session variable before every query,
    enabling Row-Level Security policies to enforce tenant isolation.
    """
    async with AsyncSessionFactory() as session:
        async with session.begin():
            # Set RLS session variable — this is the core of tenant isolation
            await session.execute(
                text("SELECT set_config('app.current_tenant_id', :tid, TRUE)"),
                {"tid": str(tenant_id)},
            )
            try:
                yield session
            except Exception:
                await session.rollback()
                raise


async def get_db(tenant_id: UUID) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions with RLS enforcement.
    Usage: db: AsyncSession = Depends(get_db_dep)
    """
    async with get_tenant_session(tenant_id) as session:
        yield session
