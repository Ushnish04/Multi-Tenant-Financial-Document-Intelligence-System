# =============================================================================
# FinRAG Production Dockerfile
# Multi-stage build: builder → runtime
# Python 3.12 / Alpine-based for minimal attack surface
# =============================================================================

# ── Stage 1: Dependency builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies (not in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r <(pip install --dry-run . 2>&1 | grep "Would install" | sed 's/Would install //' | tr ' ' '\n')

# Install all deps into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 10001 finrag && \
    useradd --uid 10001 --gid finrag --shell /bin/false --no-create-home finrag

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libpoppler-cpp0v5 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Security: no write permissions on app code
RUN chown -R finrag:finrag /app && \
    chmod -R 555 /app/app && \
    mkdir -p /data/uploads && chown finrag:finrag /data/uploads

USER finrag

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Production server: uvicorn with multiple workers
# Use gunicorn + uvicorn workers for process management in production
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
