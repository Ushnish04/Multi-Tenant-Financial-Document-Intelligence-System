# Multi-Tenant Financial Document Intelligence System
## Production-Grade RAG + LLM Extraction Platform

---

## SECTION 1 — SYSTEM ARCHITECTURE

### Overview

The system is a production-grade, multi-tenant financial document intelligence platform. It ingests PDF loan documents, extracts structured financial metrics via a deterministic LLM pipeline, stores embeddings in PostgreSQL + pgvector, and exposes all capabilities through a secure, auditable FastAPI REST API.

---

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLIENT LAYER                                      │
│              (Web App / Partner API / Internal Tools)                        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │ HTTPS / REST
┌─────────────────────────────▼───────────────────────────────────────────────┐
│                         API GATEWAY LAYER                                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │  Rate Limiter│  │  JWT Middleware   │  │  Tenant Scope Middleware     │  │
│  │  (Redis)     │  │  (Access+Refresh) │  │  (RLS Enforcement)          │  │
│  └──────────────┘  └──────────────────┘  └──────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     FastAPI Application                              │    │
│  │  /auth  /documents  /extractions  /audit  /health                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────────┐
│                       PROCESSING LAYER                                       │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │  PDF Parser      │  │  Chunking Engine  │  │  Hash / Dedup Service   │   │
│  │  (pdfplumber)    │  │  (token-based)    │  │  (SHA-256 idempotency)  │   │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────┬─────────────┘   │
│           │                     │                          │                 │
│  ┌────────▼─────────────────────▼──────────────────────────▼─────────────┐ │
│  │                    RAG Orchestration Service                            │ │
│  │   chunk → embed → store → retrieve → assemble context → LLM call      │ │
│  └─────────────────────────────┬──────────────────────────────────────────┘ │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                           LLM LAYER                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Deterministic LLM Wrapper                            │ │
│  │   temperature=0 │ frozen prompt │ schema lock │ JSON validation        │ │
│  │   retry (exp backoff) │ timeout │ fallback regex │ confidence score    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │  OpenAI / Anthropic  │    │  Pydantic Validation + Range Checker     │   │
│  │  (via LiteLLM)       │    │  + Fallback Extraction                   │   │
│  └──────────────────────┘    └──────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                          STORAGE LAYER                                       │
│  ┌──────────────────────────────────────┐  ┌─────────────────────────────┐  │
│  │         PostgreSQL 16                │  │         Redis 7             │  │
│  │  ┌───────────────────────────────┐   │  │  ┌───────────────────────┐  │  │
│  │  │  tenants / users / roles      │   │  │  │  Rate limiting        │  │  │
│  │  │  documents / doc_hashes       │   │  │  │  Token blacklist      │  │  │
│  │  │  embeddings (pgvector HNSW)   │   │  │  │  Idempotency cache    │  │  │
│  │  │  extracted_metrics            │   │  │  │  Session store        │  │  │
│  │  │  risk_scores                  │   │  │  └───────────────────────┘  │  │
│  │  │  llm_calls (partitioned)      │   │  └─────────────────────────────┘  │
│  │  │  RLS policies per tenant      │   │                                    │
│  │  └───────────────────────────────┘   │                                    │
│  └──────────────────────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Detailed Data Flow

```
PDF Upload
    │
    ▼
[1] SHA-256 Hash Computation
    │
    ▼
[2] Duplicate Detection (document_hashes table + Redis cache)
    │   ├─ DUPLICATE → return cached extraction (idempotent)
    │   └─ NEW → continue
    ▼
[3] PDF Text Extraction (pdfplumber)
    │
    ▼
[4] Token-Based Semantic Chunking (512 tokens, 64 overlap)
    │
    ▼
[5] Embedding Generation (text-embedding-3-small, async batched)
    │
    ▼
[6] pgvector Storage (with tenant_id, doc_id, chunk_index)
    │
    ▼
[7] Similarity Search (HNSW ANN, top-k=5, cosine distance)
    │
    ▼
[8] Context Assembly (ranked chunks → prompt template)
    │
    ▼
[9] Deterministic LLM Call (temperature=0, frozen prompt v{N})
    │
    ▼
[10] JSON Schema Validation (Pydantic + numeric range checks)
    │   ├─ PASS → continue
    │   └─ FAIL → Fallback Regex Extraction → re-validate
    ▼
[11] Confidence Scoring (field coverage + range compliance)
    │
    ▼
[12] PostgreSQL Persistence (extracted_metrics + risk_scores)
    │
    ▼
[13] Audit Log Written (llm_calls table: prompt, context, response, duration)
    │
    ▼
[14] API Response Returned
```

---

### Determinism Strategy

| Mechanism | Implementation |
|-----------|----------------|
| Fixed temperature | `temperature=0` on every LLM call, enforced in wrapper |
| Prompt freezing | Prompt templates versioned and immutable per schema version |
| Schema locking | Pydantic model with strict mode, no extra fields |
| Output validation | Pydantic + numeric range + required field enforcement |
| Hash-based caching | SHA-256 of PDF content → Redis cache → idempotent response |
| Retry policy | Exponential backoff: 1s, 2s, 4s, max 3 retries |

---

### Failure Handling

| Failure | Response |
|---------|----------|
| LLM timeout (>30s) | Retry with exponential backoff; mark partial on exhaustion |
| Partial extraction | Confidence score < threshold → flag for human review |
| DB write failure | Transaction rollback; document marked as `failed` |
| Embedding failure | Retry batch; fall back to sequential single-item embedding |
| Duplicate submission | Return cached result immediately (idempotent) |

---

### Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.12 |
| API Framework | FastAPI | 0.115 |
| ORM | SQLAlchemy | 2.0 (async) |
| Migrations | Alembic | 1.13 |
| Database | PostgreSQL | 16 |
| Vector Store | pgvector | 0.7 |
| Cache / Rate Limit | Redis | 7.2 |
| LLM Client | LiteLLM | 1.40 |
| PDF Parsing | pdfplumber | 0.11 |
| Tokenizer | tiktoken | 0.7 |
| Auth | python-jose + passlib | 3.3 / 1.7 |
| Validation | Pydantic | 2.7 |
| Containerization | Docker + Compose | 27 / 2.27 |
| Logging | structlog | 24.1 |

---

### Folder Structure

```
finrag/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── auth.py
│   │           ├── documents.py
│   │           ├── extractions.py
│   │           └── audit.py
│   ├── core/
│   │   ├── config.py          # Settings (pydantic-settings)
│   │   ├── security.py        # JWT + bcrypt utilities
│   │   └── exceptions.py      # Domain exceptions
│   ├── db/
│   │   ├── base.py            # SQLAlchemy engine + session
│   │   ├── models/            # ORM models
│   │   └── repositories/      # Repository pattern DAOs
│   ├── services/
│   │   ├── rag/
│   │   │   ├── chunker.py
│   │   │   ├── embeddings.py
│   │   │   └── rag_service.py
│   │   ├── llm/
│   │   │   ├── wrapper.py
│   │   │   └── validator.py
│   │   ├── auth/
│   │   │   └── auth_service.py
│   │   └── audit/
│   │       └── audit_service.py
│   ├── schemas/               # Pydantic request/response models
│   ├── middleware/
│   │   ├── tenant.py
│   │   └── audit.py
│   └── utils/
│       ├── hashing.py
│       └── retry.py
├── alembic/
│   ├── versions/
│   └── env.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/
│   └── init_db.sql
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── pyproject.toml
└── README.md
```
