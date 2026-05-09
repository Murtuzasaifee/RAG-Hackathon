# LineageRAG

**Trace every answer back to the source.**

LineageRAG is a production-grade Retrieval-Augmented Generation backend built with Python 3.13 and FastAPI. It combines hybrid retrieval, grounded generation, document versioning, safety checks, caching, observability, and citation-level source lineage with PDF bounding-box highlighting.

**One command to run the full stack:** `docker compose up`

## Architecture

```mermaid
graph TB
    Client["Client (curl / SDK / Demo UI)"]
    App["FastAPI App :8000"]
    Logfire["Logfire / Langfuse (traces + logs)"]

    subgraph Ingestion
        ADI["Azure Document Intelligence"]
        Chunker["Document-Aware Chunker"]
        DenseEmb["Dense Embeddings (OpenAI)"]
        SparseEmb["SPLADE v3 Sparse Embeddings (local)"]
        Qdrant["Qdrant (dense + sparse vectors)"]
    end

    subgraph Retrieval & Generation
        Hybrid["Hybrid RRF Fusion"]
        Reranker["Cohere Rerank (via Bifrost)"]
        Generator["LLM Generation (MeshAPI via Bifrost)"]
    end

    subgraph Security
        Auth["API Key Auth + RBAC (reader / editor / admin)"]
        ACL["Document ACL (owner_id filter)"]
        GuardIn["LLM Guard — Input Scan"]
        GuardOut["LLM Guard — Output Scan"]
    end

    subgraph Caching
        Redis["Redis Cache (3 layers)"]
    end

    subgraph Bifrost["Bifrost AI Gateway (all provider traffic)"]
        BifrostOpenAI["openai/ → OpenAI API"]
        BifrostMeshAPI["meshapi/ → MeshAPI (custom provider)"]
        BifrostCohere["cohere/ → Cohere API"]
    end

    Client --> App
    App --> Auth
    Auth --> GuardIn
    Auth --> ACL
    GuardIn --> Redis
    Redis --> Hybrid
    ADI --> Chunker --> DenseEmb --> Qdrant
    Chunker --> SparseEmb --> Qdrant
    Hybrid --> Reranker --> Generator --> GuardOut
    GuardOut --> Redis
    DenseEmb --> BifrostOpenAI
    Reranker --> BifrostCohere
    Generator --> BifrostMeshAPI
    ACL --> Qdrant
    App --> Logfire```

### Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API server | FastAPI (async) | REST endpoints |
| Document parsing | Azure Document Intelligence | Structure + bbox extraction |
| Chunking | Document-Aware Chunker | Section-bounded, atomic elements, cross-page context |
| Dense embeddings | OpenAI `text-embedding-3-small` via Bifrost | 1536-dim cosine vectors |
| Sparse embeddings | SPLADE v3 (local CPU) | Symmetric sparse model for hybrid retrieval |
| Vector store | Qdrant | Named vectors + RRF fusion |
| Reranking | Cohere `rerank-english-v3.0` via Bifrost | Cross-encoder re-scoring |
| Generation | MeshAPI `gpt-5.4` via Bifrost custom provider | Grounded answer synthesis |
| Auth & RBAC | API key auth, 3 roles (reader/editor/admin) | Role-based endpoint access + document-level ACL |
| Content safety | LLM Guard sidecar | Input scanning (prompt injection, PII), output groundedness (NLI) |
| Gateway | Bifrost AI Gateway | Unified gateway for all provider traffic (OpenAI, MeshAPI, Cohere) |
| Caching | Redis | 3-tier TTL cache |
| Observability | Logfire or Langfuse + structlog | Pluggable tracing backend, JSON logs + distributed traces |

## Prerequisites

- Docker + Docker Compose
- API keys:
  - [Azure Document Intelligence](https://portal.azure.com/#create/Microsoft.CognitiveServicesFormRecognizer)
  - [OpenAI](https://platform.openai.com/api-keys) — embeddings via Bifrost
  - [MeshAPI](https://meshapi.ai) — LLM generation via Bifrost custom provider
  - [Cohere](https://dashboard.cohere.com/api-keys) — reranking via Bifrost
   - [Logfire](https://logfire.pydantic.dev/) (free tier works) **or** [Langfuse](https://cloud.langfuse.com/) (free tier works)

## Quickstart

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your API keys

# 2. Start the stack
docker compose up

# 3. Verify health
curl http://localhost:8000/health

# 4. Open the demo UI
http://localhost:8000/demo
```

## Demo UI

A lightweight, unauthenticated demo UI is served by the FastAPI app at:

```text
http://localhost:8000/demo
```

The UI is intentionally simple and backend-focused:

- Upload a PDF and watch ingestion progress across parsing, chunking, embedding, and indexing.
- Ask questions against indexed documents.
- Inspect returned citations with `doc_id`, `version_id`, page, section path, score, bbox, and chunk text.
- Render the uploaded PDF in the browser and highlight citation bounding boxes when bbox data is available.
- Show whether a query response was served from Redis answer cache.
- Link directly to Bifrost and the active observability backend (Logfire or Langfuse) from the top bar.

PDF preview uses the PDF file selected in the current browser session. If you refresh the page, query/citation metadata still works, but you need to re-select or re-upload the PDF to render the source preview again.

For citation highlights, re-ingest documents after bbox/chunking changes so Qdrant contains overlay-ready citation metadata.

## Docker Commands & Debugging

### Running the Stack
- **Start all containers in the background (detached mode):**
  ```bash
  docker compose up -d
  ```
- **Stop all containers gracefully:**
  ```bash
  docker compose down
  ```
- **Stop containers and completely wipe all persistent data (Qdrant & Redis volumes):**
  ```bash
  docker compose down -v
  ```
- **Rebuild the FastAPI app image (run this if you modify code or install new packages):**
  ```bash
  docker compose up -d --build app
  ```
  > **Note on Caching:** `docker compose down` followed by `up` will reuse cached image layers — it does **NOT** re-run `uv sync` inside the container unless you force a rebuild. If you change dependencies and things aren't updating, do a clean build:
  ```bash
  docker compose build --no-cache app
  docker compose up
  ```

### Running App Locally (Sidecars in Docker)
To iterate faster without building the Docker image every time, start only sidecars via Compose and run the FastAPI app directly:

```bash
docker compose up qdrant redis bifrost llm-guard -d
uv run uvicorn app.api.app:app --reload
```
> **Note:** The app reads `.env` for sidecar URLs. When running locally, swap your `_URL` variables to their `localhost` variants (provided as comments in `.env.example`).

### Checking Logs
- **View logs for all containers (live tail):**
  ```bash
  docker compose logs -f
  ```
- **View logs for the main FastAPI backend:**
  ```bash
  docker compose logs -f app
  ```
- **View logs for Bifrost AI Gateway:**
  ```bash
  docker compose logs -f bifrost
  ```
- **View logs for LLM Guard (security sidecar):**
  ```bash
  docker compose logs -f llm-guard
  ```

### Executing & Inspecting
- **Check the status and health of all running containers:**
  ```bash
  docker compose ps
  ```
- **Open a bash shell directly inside the running app container:**
  ```bash
  docker compose exec app bash
  ```

The stack takes ~60s to become healthy. All five services (app, Qdrant, Redis, Bifrost, LLM Guard) must pass health checks before the app accepts traffic.

## API Reference

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

### `POST /ingest`

Upload a PDF for async ingestion. Returns a `job_id` for polling.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: editor-test-key-def456" \
  -H "X-Request-Id: $(uuidgen)" \
  -F "file=@document.pdf" \
  -F "doc_id=my-doc-1"
```

Response:
```json
{
  "job_id": "01J...",
  "doc_id": "my-doc-1",
  "version_id": "01J..."
}
```

### `GET /jobs/{job_id}`

Poll ingestion job status.

```bash
curl http://localhost:8000/jobs/01J... \
  -H "X-API-Key: reader-test-key-abc123" \
  -H "X-Request-Id: $(uuidgen)"
```

Response:
```json
{
  "job_id": "01J...",
  "doc_id": "my-doc-1",
  "version_id": "01J...",
  "state": "done",
  "stage": "done",
  "progress": 100,
  "error": null
}
```

States: `pending` → `running` (stages: `parsing` → `chunking` → `embedding_dense` → `embedding_sparse` → `indexing`) → `done` | `failed`

### `POST /api/v1/query`

Query the RAG pipeline with hybrid retrieval + reranking + grounded generation.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: reader-test-key-abc123" \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: $(uuidgen)" \
  -d '{
    "query": "What are the caching layers?",
    "doc_ids": ["my-doc-1"],
    "top_k": 20,
    "top_n": 5
  }'
```

Response:
```json
{
  "answer": "The system uses three Redis caching layers...",
  "citations": [
    {
      "doc_id": "my-doc-1",
      "version_id": "01J...",
      "page": 3,
      "section_path": ["Introduction", "Architecture"],
      "bbox": [0.1, 0.2, 0.8, 0.4],
      "page_bboxes": [
        {
          "page": 3,
          "bbox": [0.1, 0.2, 0.8, 0.4]
        }
      ],
      "chunk_text": "The caching architecture consists of...",
      "chunk_type": "text",
      "score": 0.95
    }
  ],
  "request_id": "abc-123",
  "timings_ms": {
    "guard_input_ms": 45,
    "retrieve_ms": 120,
    "rerank_ms": 85,
    "generate_ms": 340,
    "guard_output_ms": 1800
  },
  "warnings": [],
  "cache_hit": false
}
```

**Optional filters:**
- `doc_ids` — restrict retrieval to specific documents
- `version_ids` — query specific versions (bypasses active filter)
- `top_k` — candidates from hybrid retrieval (default: 20)
- `top_n` — final results after reranking (default: 5)

### `PUT /documents/{doc_id}`

Re-ingest a document with a new version. The new version becomes active automatically.

```bash
curl -X PUT http://localhost:8000/documents/my-doc-1 \
  -H "X-API-Key: editor-test-key-def456" \
  -H "X-Request-Id: $(uuidgen)" \
  -F "file=@document-v2.pdf"
```

### `DELETE /documents/{doc_id}`

Delete document points from Qdrant.

```bash
# Soft delete — editor or admin
curl -X DELETE "http://localhost:8000/documents/my-doc-1?mode=soft" \
  -H "X-API-Key: editor-test-key-def456"

# Hard delete — admin only
curl -X DELETE "http://localhost:8000/documents/my-doc-1?mode=hard" \
  -H "X-API-Key: admin-test-key-ghi789"

# Delete specific version only
curl -X DELETE "http://localhost:8000/documents/my-doc-1?mode=soft&version_id=01J..." \
  -H "X-API-Key: editor-test-key-def456"
```

### `GET /health`

Health check endpoint. Returns `{"status": "ok"}`.

## Document Versioning

All Qdrant points carry `(doc_id, version_id, active)` payload. Default retrieval filters `active == True`.

- **Re-ingest** creates a new version and flips `active` atomically
- **Soft delete** marks points `active=False` (queryable via explicit `version_ids`)
- **Hard delete** removes points and bumps a cache epoch to invalidate cached answers
- Old versions remain queryable by passing `version_ids` in the query request

## Security

### API Key Authentication & RBAC

All endpoints (except `GET /health`, `GET /demo`, `GET /demo/config`) require an `X-API-Key` header. Three roles are supported, hierarchical (admin ⊃ editor ⊃ reader):

| Role | Permitted endpoints |
|------|-------------------|
| `reader` | `POST /api/v1/query`, `GET /jobs/{job_id}` |
| `editor` | reader + `POST /ingest`, `PUT /documents/{doc_id}`, `DELETE /documents/{doc_id}?mode=soft` |
| `admin` | editor + `DELETE /documents/{doc_id}?mode=hard`, `POST /eval/run` |

Missing key → `401`. Valid key, insufficient role → `403`.

**Seeding keys for local development:**

```bash
# Start Redis first (or docker compose up redis)
uv run python scripts/seed_keys.py
```

This seeds three test keys (`reader-test-key-abc123`, `editor-test-key-def456`, `admin-test-key-ghi789`) into Redis. Edit `scripts/seed_keys.py` to change them.

**Bootstrap an admin key on startup** — set `ADMIN_API_KEY` in `.env`:

```bash
# Generate a secure key
python -c "import secrets; print(secrets.token_hex(32))"

# Add to .env
ADMIN_API_KEY=<generated-key>
```

The key is seeded into Redis idempotently on every startup.

**Bypass auth for local development** (never in production):

```bash
AUTH_ENABLED=false
```

### Document-Level ACL

Each ingested document is tagged with the `owner_id` of the API key that uploaded it (stored in Qdrant payload). At query time:

- `reader` / `editor` — Qdrant filter `owner_id == principal.key_id` applied automatically. Users only see documents they ingested.
- `admin` — no `owner_id` filter. Sees all documents across all owners.

No extra parameters needed — ACL is enforced transparently based on the API key role.

### LLM Guard Content Scanning

Two-layer scanning via LLM Guard sidecar (CPU inference, lazy model load):

- **Input scanning** (toggle: `LLM_GUARD_INPUT_ENABLED`): Prompt injection detection, token limit enforcement
- **Output scanning** (toggle: `LLM_GUARD_OUTPUT_ENABLED`): Factual consistency check via NLI (FactualConsistency scanner)

Input block → `400` with error details. Output concern → `warnings` array in response (does not block). Both layers fail-open on timeout or sidecar unavailability.

> LLM Guard NLI inference is slow on CPU (~30–120s). Disable output scanning with `LLM_GUARD_OUTPUT_ENABLED=false` for faster iteration.

## Observability

The tracing backend is **pluggable** — switch between Logfire and Langfuse via a single env var `OTEL_BACKEND`. All pipeline stages emit spans with the same granularity regardless of backend.

### Backends

| Backend | Config | Notes |
|---------|--------|-------|
| **Logfire** (default) | `OTEL_BACKEND=logfire` + `LOGFIRE_TOKEN` | Auto-instruments FastAPI, httpx, Pydantic |
| **Langfuse Cloud** | `OTEL_BACKEND=langfuse` + `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` | Free tier (5k traces/mo); set `LANGFUSE_BASE_URL` for self-hosted |

### Span coverage

Per-stage spans: `parse`, `chunk`, `embed.dense`, `embed.sparse`, `index`, `retrieve.hybrid`, `rerank`, `generate`, `guard.input`, `guard.output`, `gateway.embed`, `gateway.chat`, `cache.lookup`, `cache.store`, `version.flip_active`, `version.soft_delete`, `version.hard_delete`

- **structlog** — JSON-structured logs with request ID correlation, per-stage timing, hit counts, and score breakdowns
- Every response includes `timings_ms` for full latency breakdown across all pipeline stages
- The demo UI links to the active observability backend from `/demo/config`

## Configuration

All settings are env-var driven. See `.env.example` for the full list.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_DI_ENDPOINT` | Yes | — | Azure DI resource endpoint |
| `AZURE_DI_KEY` | Yes | — | Azure DI API key |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (used by Bifrost for embeddings) |
| `MESH_API_KEY` | Yes | — | MeshAPI key — read by Bifrost for LLM generation (custom provider) |
| `COHERE_API_KEY` | Yes | — | Cohere API key — read by Bifrost for reranking |
| `LOGFIRE_TOKEN` | No* | — | Logfire token (*required when `OTEL_BACKEND=logfire`) |
| `HUGGINGFACE_TOKEN` | No | — | HuggingFace token (required for gated SPLADE v3 model) |
| `OTEL_BACKEND` | No | `logfire` | Tracing backend: `logfire` or `langfuse` |
| `LANGFUSE_SECRET_KEY` | No* | — | Langfuse secret key (*required when `OTEL_BACKEND=langfuse`) |
| `LANGFUSE_PUBLIC_KEY` | No* | — | Langfuse public key (*required when `OTEL_BACKEND=langfuse`) |
| `LANGFUSE_BASE_URL` | No | `https://cloud.langfuse.com` | Langfuse host URL (change for self-hosted) |
| `BIFROST_URL` | No | `http://localhost:8080` | Bifrost gateway URL |
| `BIFROST_PUBLIC_URL` | No | — | Browser-visible Bifrost URL shown in the demo UI; falls back to `BIFROST_URL` |
| `LOGFIRE_URL` | No | `https://logfire-us.pydantic.dev/msaifee/rag-hackathon` | Logfire project link shown in the demo UI |
| `LLM_GUARD_URL` | No | `http://localhost:8001` | LLM Guard sidecar URL |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant URL |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis URL |
| `LLM_MODEL` | No | `gpt-5.4` | LLM model for generation |
| `LLM_PROVIDER` | No | `meshapi` | LLM provider (`meshapi` or `openai`) |
| `AUTH_ENABLED` | No | `true` | Enable/disable API key auth (set `false` for local dev only) |
| `ADMIN_API_KEY` | No | — | Raw admin key seeded into Redis idempotently on startup |
| `SPARSE_ENABLED` | No | `true` | Enable/disable sparse channel |
| `LLM_GUARD_INPUT_ENABLED` | No | `true` | Enable/disable input prompt scanning |
| `LLM_GUARD_OUTPUT_ENABLED` | No | `true` | Enable/disable output groundedness scanning |
| `CHUNK_MAX_TOKENS` | No | `512` | Max tokens per chunk |
| `CACHE_TTL_ANSWER` | No | `3600` | Answer cache TTL (seconds) |

## Development

### Tests

```bash
uv sync
uv run pytest tests/ -v
```

### Linting & Formatting (Ruff)

Ruff replaces both flake8 (linting) and black (formatting). Config is in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`.

```bash
# Lint check (no changes)
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/

# Format check (no changes)
uv run ruff format --check src/ tests/

# Auto-format
uv run ruff format src/ tests/

# Full lint + format pass
uv run ruff check --fix src/ tests/ && uv run ruff format src/ tests/
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| App crashes on start | Missing required env vars | Ensure all required keys are in `.env` for your chosen `OTEL_BACKEND` |
| Bifrost call fails "no keys found" | API key env var not set, or model not in explicit allowlist | Check `docker/bifrost/config.json` model lists and that all three keys (`OPENAI_API_KEY`, `MESH_API_KEY`, `COHERE_API_KEY`) are in `.env` |
| SPLADE 401 on load | Gated HuggingFace repo | Set `HUGGINGFACE_TOKEN=hf_...` in `.env` |
| SPLADE OOM on low-RAM machines | SPLADE model loads ~1 GB into memory | Set `SPARSE_ENABLED=false` in `.env` |
| LLM Guard container killed | OOM — too many NLP models loaded | Memory capped at 4 GB in compose; reduce further if needed |
| LLM Guard timeout (ReadTimeout) | NLI inference slow on CPU | Disable output scan: `LLM_GUARD_OUTPUT_ENABLED=false` |
| Ingest stuck at `embedding_sparse` | First SPLADE run downloads model (~500 MB) | Wait; subsequent runs use cached model |
| Cache misses after version flip | Expected — version hash changes | Normal behavior; cache rebuilds on next query |

## Low-Memory Mode

For machines with <16 GB RAM, set in `.env`:

```bash
SPARSE_ENABLED=false
LLM_GUARD_OUTPUT_ENABLED=false
```

This skips sparse embedding and output scanning entirely. Dense-only retrieval with input-only guard scanning.

## Project Structure

```
src/app/
├── api/                    # FastAPI routes, schemas, middleware
│   ├── routers/            # ingest, query, documents, health, demo
│   ├── services/           # QueryService orchestration
│   └── schemas.py          # Pydantic request/response models
├── cache/                  # Redis cache (3-tier TTL)
├── core/                   # Settings, types, errors
├── generation/             # Grounded LLM generation
├── frontend/               # Minimal static demo UI served at /demo
├── gateway/                # Bifrost client (embeddings + chat + rerank — all providers)
├── ingestion/              # Parser, chunker, embedders, indexer, jobs
├── observability/          # Pluggable tracing (Logfire / Langfuse), structlog config
├── retrieval/              # Hybrid Qdrant retriever, Cohere reranker
├── security/               # LLM Guard client (input/output)
└── versioning/             # Document version manager
```
