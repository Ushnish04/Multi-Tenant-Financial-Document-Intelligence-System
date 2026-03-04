-- =============================================================================
-- SECTION 2: PRODUCTION DATABASE SCHEMA
-- Multi-Tenant Financial Document Intelligence System
-- PostgreSQL 16 + pgvector 0.7
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- =============================================================================
-- SCHEMA SETUP
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS finrag;
SET search_path TO finrag, public;

-- =============================================================================
-- TABLE: tenants
-- Root of multi-tenancy hierarchy
-- =============================================================================
CREATE TABLE tenants (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(255) NOT NULL,
    slug            VARCHAR(100) NOT NULL,
    plan            VARCHAR(50) NOT NULL DEFAULT 'standard'
                        CHECK (plan IN ('trial', 'standard', 'enterprise')),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    settings        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT tenants_slug_unique UNIQUE (slug)
);

CREATE INDEX idx_tenants_slug ON tenants (slug);
CREATE INDEX idx_tenants_is_active ON tenants (is_active);

-- =============================================================================
-- TABLE: roles
-- RBAC role definitions (system-wide)
-- =============================================================================
CREATE TABLE roles (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(100) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT roles_name_unique UNIQUE (name)
);

-- Seed default roles
INSERT INTO roles (name, permissions) VALUES
    ('admin',    '["document:read","document:write","document:delete","audit:read","user:manage"]'),
    ('analyst',  '["document:read","document:write","audit:read"]'),
    ('viewer',   '["document:read"]');

-- =============================================================================
-- TABLE: users
-- Per-tenant users
-- =============================================================================
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email           VARCHAR(320) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name       VARCHAR(255),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified     BOOLEAN NOT NULL DEFAULT FALSE,
    last_login_at   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT users_tenant_email_unique UNIQUE (tenant_id, email)
);

CREATE INDEX idx_users_tenant_id ON users (tenant_id);
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_is_active ON users (tenant_id, is_active);

-- =============================================================================
-- TABLE: user_roles
-- Many-to-many: users <-> roles (scoped to tenant)
-- =============================================================================
CREATE TABLE user_roles (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id     UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    granted_by  UUID REFERENCES users(id) ON DELETE SET NULL,
    granted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT user_roles_unique UNIQUE (user_id, role_id, tenant_id)
);

CREATE INDEX idx_user_roles_user_id ON user_roles (user_id);
CREATE INDEX idx_user_roles_tenant_id ON user_roles (tenant_id);

-- =============================================================================
-- TABLE: refresh_tokens
-- JWT refresh token store with revocation support
-- =============================================================================
CREATE TABLE refresh_tokens (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    token_hash  VARCHAR(64) NOT NULL,          -- SHA-256 of raw token
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked_at  TIMESTAMPTZ,
    user_agent  TEXT,
    ip_address  INET,
    CONSTRAINT refresh_tokens_hash_unique UNIQUE (token_hash)
);

CREATE INDEX idx_refresh_tokens_user_id ON refresh_tokens (user_id);
CREATE INDEX idx_refresh_tokens_hash ON refresh_tokens (token_hash);
CREATE INDEX idx_refresh_tokens_expires ON refresh_tokens (expires_at)
    WHERE revoked_at IS NULL;

-- =============================================================================
-- TABLE: documents
-- Uploaded financial PDF documents
-- =============================================================================
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    uploaded_by     UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    filename        VARCHAR(512) NOT NULL,
    storage_path    TEXT NOT NULL,             -- S3/GCS/local path
    file_size_bytes BIGINT NOT NULL,
    mime_type       VARCHAR(100) NOT NULL DEFAULT 'application/pdf',
    status          VARCHAR(50) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','processing','completed','failed','duplicate')),
    page_count      INT,
    word_count      INT,
    error_message   TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_tenant_id ON documents (tenant_id);
CREATE INDEX idx_documents_status ON documents (tenant_id, status);
CREATE INDEX idx_documents_created_at ON documents (tenant_id, created_at DESC);
CREATE INDEX idx_documents_uploaded_by ON documents (uploaded_by);

-- =============================================================================
-- TABLE: document_hashes
-- SHA-256 hash for idempotent deduplication
-- =============================================================================
CREATE TABLE document_hashes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sha256_hash     VARCHAR(64) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT document_hashes_tenant_hash_unique UNIQUE (tenant_id, sha256_hash)
);

CREATE INDEX idx_document_hashes_hash ON document_hashes (tenant_id, sha256_hash);

-- =============================================================================
-- TABLE: embeddings
-- pgvector semantic embeddings per chunk
-- =============================================================================
CREATE TABLE embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    chunk_text      TEXT NOT NULL,
    token_count     INT NOT NULL,
    embedding       vector(1536) NOT NULL,      -- text-embedding-3-small dims
    model_name      VARCHAR(100) NOT NULL,
    model_version   VARCHAR(50) NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT embeddings_doc_chunk_unique UNIQUE (document_id, chunk_index)
);

-- HNSW index for approximate nearest neighbor search
-- ef_construction=128 and m=16 are production-grade defaults
CREATE INDEX idx_embeddings_hnsw ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- IVFFlat as alternative for very large datasets (>1M rows)
-- CREATE INDEX idx_embeddings_ivfflat ON embeddings
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 1000);

CREATE INDEX idx_embeddings_tenant_doc ON embeddings (tenant_id, document_id);
CREATE INDEX idx_embeddings_tenant_id ON embeddings (tenant_id);

-- =============================================================================
-- TABLE: extracted_metrics
-- Structured financial metrics from LLM extraction
-- =============================================================================
CREATE TABLE extracted_metrics (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id               UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id             UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Core loan metrics
    loan_amount             NUMERIC(20, 2),
    loan_term_months        INT,
    interest_rate           NUMERIC(8, 5),       -- e.g. 0.07250 = 7.25%
    annual_percentage_rate  NUMERIC(8, 5),
    monthly_payment         NUMERIC(20, 2),
    origination_fee         NUMERIC(20, 2),

    -- Borrower financials
    borrower_income_annual  NUMERIC(20, 2),
    debt_to_income_ratio    NUMERIC(6, 4),
    credit_score            INT CHECK (credit_score BETWEEN 300 AND 850),
    employment_status       VARCHAR(100),
    employer_name           VARCHAR(255),

    -- Property / collateral
    property_value          NUMERIC(20, 2),
    loan_to_value_ratio     NUMERIC(6, 4),
    property_type           VARCHAR(100),
    property_address        TEXT,

    -- Document metadata
    document_date           DATE,
    lender_name             VARCHAR(255),
    borrower_name           VARCHAR(255),       -- PII — stored encrypted in prod
    loan_purpose            VARCHAR(255),
    loan_type               VARCHAR(100),

    -- Extraction quality
    confidence_score        NUMERIC(4, 3) NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    extraction_version      VARCHAR(50) NOT NULL,
    prompt_version          VARCHAR(50) NOT NULL,
    requires_review         BOOLEAN NOT NULL DEFAULT FALSE,
    raw_extraction          JSONB NOT NULL,     -- Full LLM JSON response
    validated_extraction    JSONB NOT NULL,     -- Post-validation JSON

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT extracted_metrics_document_unique UNIQUE (document_id)
);

CREATE INDEX idx_extracted_metrics_tenant_id ON extracted_metrics (tenant_id);
CREATE INDEX idx_extracted_metrics_doc_id ON extracted_metrics (document_id);
CREATE INDEX idx_extracted_metrics_confidence ON extracted_metrics (tenant_id, confidence_score);
CREATE INDEX idx_extracted_metrics_review ON extracted_metrics (tenant_id, requires_review)
    WHERE requires_review = TRUE;
CREATE INDEX idx_extracted_metrics_raw ON extracted_metrics USING gin (raw_extraction);

-- =============================================================================
-- TABLE: risk_scores
-- Computed risk signals per document
-- =============================================================================
CREATE TABLE risk_scores (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    metrics_id      UUID NOT NULL REFERENCES extracted_metrics(id) ON DELETE CASCADE,

    overall_score   NUMERIC(4, 3) NOT NULL CHECK (overall_score BETWEEN 0 AND 1),
    risk_tier       VARCHAR(50) NOT NULL CHECK (risk_tier IN ('low','medium','high','critical')),
    dti_score       NUMERIC(4, 3),
    ltv_score       NUMERIC(4, 3),
    credit_score_n  NUMERIC(4, 3),
    income_score    NUMERIC(4, 3),

    flags           JSONB NOT NULL DEFAULT '[]',     -- Array of risk flag strings
    scoring_version VARCHAR(50) NOT NULL,
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT risk_scores_document_unique UNIQUE (document_id)
);

CREATE INDEX idx_risk_scores_tenant_id ON risk_scores (tenant_id);
CREATE INDEX idx_risk_scores_tier ON risk_scores (tenant_id, risk_tier);

-- =============================================================================
-- TABLE: llm_calls (PARTITIONED AUDIT LOG)
-- Immutable audit trail of every LLM interaction
-- Partitioned by month for scalability
-- =============================================================================
CREATE TABLE llm_calls (
    id                  UUID NOT NULL DEFAULT uuid_generate_v4(),
    tenant_id           UUID NOT NULL REFERENCES tenants(id) ON DELETE RESTRICT,
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    document_id         UUID REFERENCES documents(id) ON DELETE SET NULL,

    -- LLM call metadata
    model_name          VARCHAR(100) NOT NULL,
    model_version       VARCHAR(50) NOT NULL,
    prompt_version      VARCHAR(50) NOT NULL,
    operation_type      VARCHAR(100) NOT NULL,  -- 'extraction', 'embedding', etc.

    -- Input/output (GDPR: no raw PII in prompts stored here)
    prompt_hash         VARCHAR(64) NOT NULL,   -- SHA-256 of prompt (not raw)
    prompt_token_count  INT,
    rag_chunk_ids       UUID[],                 -- References to embeddings used
    rag_context_hash    VARCHAR(64),            -- SHA-256 of assembled context
    raw_response_hash   VARCHAR(64),            -- SHA-256 of raw LLM response
    final_output        JSONB,                  -- Validated extraction result

    -- Performance & quality
    duration_ms         INT NOT NULL,
    input_tokens        INT,
    output_tokens       INT,
    total_tokens        INT,
    confidence_score    NUMERIC(4, 3),
    validation_passed   BOOLEAN NOT NULL,
    fallback_used       BOOLEAN NOT NULL DEFAULT FALSE,
    retry_count         INT NOT NULL DEFAULT 0,
    error_code          VARCHAR(100),
    error_message       TEXT,

    -- Timestamps
    called_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, called_at)
) PARTITION BY RANGE (called_at);

-- Create monthly partitions (generate programmatically in production)
CREATE TABLE llm_calls_2024_01 PARTITION OF llm_calls
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE llm_calls_2024_02 PARTITION OF llm_calls
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ... (alembic generates remaining partitions dynamically)
CREATE TABLE llm_calls_default PARTITION OF llm_calls DEFAULT;

-- Indexes on partitioned table
CREATE INDEX idx_llm_calls_tenant_called ON llm_calls (tenant_id, called_at DESC);
CREATE INDEX idx_llm_calls_user_called ON llm_calls (user_id, called_at DESC);
CREATE INDEX idx_llm_calls_document ON llm_calls (document_id, called_at DESC);
CREATE INDEX idx_llm_calls_validation ON llm_calls (tenant_id, validation_passed, called_at DESC);

-- =============================================================================
-- ROW-LEVEL SECURITY (RLS)
-- Enforces tenant isolation at the database layer
-- JWT tenant_id claim maps to app.current_tenant_id session variable
-- =============================================================================

-- Function to get current tenant from session variable
CREATE OR REPLACE FUNCTION current_tenant_id() RETURNS UUID AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_tenant_id', TRUE), '')::UUID;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Enable RLS on all tenant-scoped tables
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_hashes ENABLE ROW LEVEL SECURITY;
ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE extracted_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE refresh_tokens ENABLE ROW LEVEL SECURITY;

-- RLS POLICIES: documents
CREATE POLICY tenant_isolation_documents ON documents
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- RLS POLICIES: document_hashes
CREATE POLICY tenant_isolation_document_hashes ON document_hashes
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- RLS POLICIES: embeddings
CREATE POLICY tenant_isolation_embeddings ON embeddings
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- RLS POLICIES: extracted_metrics
CREATE POLICY tenant_isolation_extracted_metrics ON extracted_metrics
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- RLS POLICIES: risk_scores
CREATE POLICY tenant_isolation_risk_scores ON risk_scores
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- RLS POLICIES: llm_calls (audit — read own tenant, immutable)
CREATE POLICY tenant_isolation_llm_calls_select ON llm_calls
    AS PERMISSIVE FOR SELECT
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

CREATE POLICY tenant_isolation_llm_calls_insert ON llm_calls
    AS PERMISSIVE FOR INSERT
    TO finrag_app_role
    WITH CHECK (tenant_id = current_tenant_id());

-- No UPDATE or DELETE on llm_calls (immutable audit log)

-- RLS POLICIES: users
CREATE POLICY tenant_isolation_users ON users
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- RLS POLICIES: user_roles
CREATE POLICY tenant_isolation_user_roles ON user_roles
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- RLS POLICIES: refresh_tokens
CREATE POLICY tenant_isolation_refresh_tokens ON refresh_tokens
    AS PERMISSIVE FOR ALL
    TO finrag_app_role
    USING (tenant_id = current_tenant_id());

-- Create application database role
CREATE ROLE finrag_app_role LOGIN PASSWORD 'change_in_production';
GRANT USAGE ON SCHEMA finrag TO finrag_app_role;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA finrag TO finrag_app_role;
REVOKE UPDATE, DELETE ON llm_calls FROM finrag_app_role; -- Audit immutability

-- =============================================================================
-- EXAMPLE QUERIES
-- =============================================================================

-- [1] Retrieve all documents for the current tenant (RLS enforces isolation)
-- SET LOCAL app.current_tenant_id = '<tenant_uuid>';
SELECT
    d.id,
    d.filename,
    d.status,
    d.created_at,
    em.confidence_score,
    rs.risk_tier
FROM documents d
LEFT JOIN extracted_metrics em ON em.document_id = d.id
LEFT JOIN risk_scores rs ON rs.document_id = d.id
WHERE d.tenant_id = current_tenant_id()
ORDER BY d.created_at DESC
LIMIT 50;

-- [2] Similarity search: retrieve top-5 semantically similar chunks
-- for a given query embedding (tenant-scoped via RLS)
-- :query_embedding is the vectorized user query
SELECT
    e.id,
    e.document_id,
    e.chunk_index,
    e.chunk_text,
    1 - (e.embedding <=> :query_embedding::vector) AS similarity_score
FROM embeddings e
WHERE e.tenant_id = current_tenant_id()
ORDER BY e.embedding <=> :query_embedding::vector
LIMIT 5;

-- [3] Audit logs per user with pagination
SELECT
    lc.id,
    lc.document_id,
    lc.model_name,
    lc.operation_type,
    lc.duration_ms,
    lc.confidence_score,
    lc.validation_passed,
    lc.fallback_used,
    lc.called_at
FROM llm_calls lc
WHERE lc.tenant_id = current_tenant_id()
  AND lc.user_id = :user_id
  AND lc.called_at >= NOW() - INTERVAL '30 days'
ORDER BY lc.called_at DESC
LIMIT 100 OFFSET :offset;

-- [4] Documents requiring human review
SELECT d.filename, em.confidence_score, em.requires_review, rs.risk_tier
FROM documents d
JOIN extracted_metrics em ON em.document_id = d.id
JOIN risk_scores rs ON rs.document_id = d.id
WHERE em.requires_review = TRUE
  AND d.tenant_id = current_tenant_id()
ORDER BY em.confidence_score ASC;

-- =============================================================================
-- ARCHIVAL STRATEGY
-- =============================================================================
-- Partition detachment for archival (run monthly via pg_cron):
-- ALTER TABLE llm_calls DETACH PARTITION llm_calls_2023_01;
-- COPY llm_calls_2023_01 TO '/archive/llm_calls_2023_01.csv' CSV HEADER;
-- DROP TABLE llm_calls_2023_01;

-- =============================================================================
-- SCALING STRATEGY FOR MILLIONS OF EMBEDDINGS
-- =============================================================================
-- 1. HNSW index with m=16 handles ~10M rows efficiently
-- 2. Partition embeddings table by tenant_id hash for horizontal scale:
--    CREATE TABLE embeddings_0 PARTITION OF embeddings FOR VALUES WITH (MODULUS 8, REMAINDER 0);
-- 3. Use pgBouncer connection pooling in transaction mode
-- 4. Set maintenance_work_mem = 4GB for index builds
-- 5. Enable parallel index builds: SET max_parallel_maintenance_workers = 4;
-- 6. For >50M embeddings: consider pgvector + Citus or migrate to dedicated vector DB
