# Testing Guide — RAG Hackathon Backend

## Prerequisites

```bash
# 1. Ensure .env is configured with real API keys
cp .env.example .env
# Edit .env with your actual keys

# 2. Install dependencies
uv sync

# 3. Verify Python environment
uv run python -c "from rag_hackathon.api.app import app; print('OK')"
```

---

## Phase 1 — Foundations (U1–U4)

### Unit Tests (no sidecars needed)

```bash
uv run pytest tests/core/ tests/observability/ tests/api/test_health.py tests/api/test_middleware.py -v
```

Expected: all pass.

### Manual Verification

```bash
# App boots with real env vars (sidecars NOT required for boot)
uv run uvicorn rag_hackathon.api.app:app --port 8000 &

# Health check (deps will show as unhealthy without sidecars — that's OK)
curl -s http://localhost:8000/health | python -m json.tool

# Request-ID middleware
curl -s -H "X-Request-Id: test-123" http://localhost:8000/health -D - | grep -i x-request-id

# Kill server
kill %1
```

### Docker Compose Smoke Test

```bash
docker compose up -d
# Wait ~60s for all services to be healthy
docker compose ps  # all should show "healthy"

curl -s http://localhost:8000/health | python -m json.tool
# Should return 200 with deps showing qdrant, redis status

docker compose down
```

---

## Phase 2 — Ingestion Pipeline (U5–U10)

### Unit Tests (no sidecars needed)

```bash
uv run pytest tests/gateway/ tests/ingestion/ tests/api/test_health.py -v
```

Expected: 55 tests pass.

### Integration Test (requires all sidecars + Azure DI credentials)

```bash
# Start sidecars
docker compose up -d qdrant redis bifrost
# Wait until healthy
docker compose ps

# Start app locally (easier to debug than in-container)
BIFROST_URL=http://localhost:8080 \
QDRANT_URL=http://localhost:6333 \
REDIS_URL=redis://localhost:6379/0 \
uv run uvicorn rag_hackathon.api.app:app --port 8000 --reload &

# Ingest a PDF
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@samples/example.pdf" \
  | python -m json.tool
# Returns: {"job_id": "...", "doc_id": "...", "version_id": "..."}

# Poll job status (replace JOB_ID from above)
JOB_ID="<from-above>"
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | python -m json.tool
# Repeat until state == "done"

# Verify in Qdrant
curl -s "http://localhost:6333/collections/documents" | python -m json.tool
# Should show points_count > 0

curl -s -X POST "http://localhost:6333/collections/documents/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 3, "with_payload": true}' | python -m json.tool
# Verify payload fields: doc_id, version_id, active, page, section_path, bbox, chunk_text, chunk_type

kill %1
docker compose down
```

---

## Phase 3 — Retrieval & Generation (U11–U13)

### Unit Tests (no sidecars needed)

```bash
uv run pytest tests/retrieval/ tests/generation/ tests/api/test_query*.py -v
```

Expected: 22 tests pass (8 retrieval + 7 generation + 7 query).

### Integration Test (requires all sidecars + provider credentials)

```bash
# Full stack up
docker compose up -d

# Wait for all healthy
docker compose ps

# Or run app locally with sidecar networking:
BIFROST_URL=http://localhost:8080 \
QDRANT_URL=http://localhost:6333 \
REDIS_URL=redis://localhost:6379/0 \
uv run uvicorn rag_hackathon.api.app:app --port 8000 --reload &

# Step 1: Ingest a PDF (if not already done)
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@samples/example.pdf" | python -m json.tool

# Poll until done
JOB_ID="<from-above>"
while true; do
  STATE=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | python -c "import sys,json; print(json.load(sys.stdin)['state'])")
  echo "State: $STATE"
  [ "$STATE" = "done" ] && break
  sleep 2
done

# Step 2: Query the ingested document
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: test-$(uuidgen)" \
  -d '{
    "query": "What is the main topic of this document?",
    "top_k": 10,
    "top_n": 3
  }' | python -m json.tool

# Expected response shape:
# {
#   "answer": "...",
#   "citations": [
#     {
#       "doc_id": "...",
#       "version_id": "...",
#       "page": 1,
#       "section_path": ["..."],
#       "bbox": [...],
#       "chunk_text": "...",
#       "score": 0.95
#     }
#   ],
#   "request_id": "...",
#   "timings_ms": {"retrieve_ms": ..., "rerank_ms": ..., "generate_ms": ...},
#   "warnings": []
# }

# Step 3: Query with doc_id filter
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "summary",
    "doc_ids": ["<DOC_ID from ingest>"],
    "top_n": 2
  }' | python -m json.tool

# Step 4: Query with specific version_ids (older versions)
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "summary",
    "version_ids": ["<VERSION_ID>"]
  }' | python -m json.tool

# Step 5: Empty result test (nonexistent doc)
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is quantum computing",
    "doc_ids": ["nonexistent-doc-id"]
  }' | python -m json.tool
# Should return: "I don't know based on the provided documents."

# Step 6: Validation test
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": ""}' | python -m json.tool
# Should return 422

kill %1
docker compose down
```

---

## Phase 4 — Security (U14) *(upcoming)*

> LLM Guard input/output scanning. Tests will be added with the implementation.

---

## Phase 5 — Caching & Document Lifecycle (U15–U16) *(upcoming)*

> Redis cache verification + version flip + delete. Tests will be added with the implementation.

---

## Phase 6 — Eval (U17) *(upcoming)*

> RAGAS eval endpoint. Tests will be added with the implementation.

---

## Phase 7 — Polish & Demo (U18) *(upcoming)*

> End-to-end demo script. Will be added with the implementation.

---

## Full Test Suite (All Phases)

```bash
# Run everything
uv run pytest tests/ -v --tb=short

# Run with coverage
uv run pytest tests/ -v --cov=rag_hackathon --cov-report=term-missing

# Lint check
uv run ruff check src/ tests/
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ValidationError` on boot | Check `.env` has all 5 required secrets |
| Qdrant connection refused | `docker compose up -d qdrant` or set `QDRANT_URL` |
| Redis connection refused | `docker compose up -d redis` or set `REDIS_URL` |
| Splade OOM on low-RAM machine | Set `SPARSE_ENABLED=false` in `.env` |
| Bifrost 503 | Ensure Bifrost image is pulled and healthy: `docker compose logs bifrost` |
| `ImportError` after adding deps | Run `uv sync` |
| Tests fail after edit | Run `uv run ruff check src/ tests/ --fix` then re-run tests |
