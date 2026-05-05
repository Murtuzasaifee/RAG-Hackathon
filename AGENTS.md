# AGENTS.md

Project instructions for Codex when working in this repository.

## Project Overview

**Production-grade RAG backend (hackathon entry).** Python 3.13 + FastAPI async. Backend-only — no frontend in scope. Demo target: `docker compose up` brings the entire stack to a healthy state on a single laptop.

The stack is intentionally production-shaped: pluggable strategy interfaces at every layer, sidecar architecture, full source lineage on citations, observability, eval, and document lifecycle (versioning + update + delete).

## Source-of-Truth Documents

- **Requirements (WHAT):** `docs/brainstorms/2026-05-02-rag-hackathon-backend-requirements.md` — 20 numbered requirements (R1–R20) plus key decisions and scope boundaries.
- **Plan (HOW):** `docs/plans/2026-05-02-001-feat-rag-hackathon-backend-plan.md` — 18 implementation units across 7 phases. Read this before starting any unit; each unit lists files, approach, test scenarios, and verification.

When implementing, treat these documents as canonical. Do not re-invent decisions already made there.

## Architecture (One-Paragraph Map)

FastAPI service orchestrates the pipeline: Azure Document Intelligence (parse) → structure-aware chunker → OpenAI dense embeddings (via Bifrost) + local Splade sparse embeddings → Qdrant index with named vectors and version payload → Qdrant prefetch + RRF fusion → Cohere rerank (direct API call) → grounded LLM generation via MeshAPI (direct API call) → response with `citations[]` carrying full lineage (page, section_path, bbox, chunk_text, score). Embeddings route through Bifrost AI Gateway to OpenAI. Chat/LLM calls go directly to MeshAPI. Cohere rerank calls go directly to Cohere API. LLM Guard sidecar scans both input and output (groundedness via NLI, fail-open). Redis caches query embeddings, rerank results, and final answers. Logfire + structlog provide observability. RAGAS eval endpoint validates groundedness on a bundled golden set.

Sidecars (all in `docker-compose.yml`): Bifrost, LLM Guard, Qdrant, Redis.

## Toolchain

- **Package manager:** `uv` (always). Add deps with `uv add <pkg>`. Sync with `uv sync`.
- **Python:** 3.13.
- **Test runner:** `uv run pytest`.
- **Run server (local):** `uv run uvicorn app.api.app:app --reload`.
- **Run stack:** `docker compose up`.

## Code Conventions

- **Layout:** `src/app/` package with submodules `api/`, `ingestion/`, `retrieval/`, `generation/`, `security/`, `gateway/`, `cache/`, `versioning/`, `eval/`, `observability/`, `core/`. `tests/` mirrors `src/` structure. Routers: `health`, `ingest`, `query`, `documents`, `eval`.
- **Strategy boundaries:** every swappable component (chunker, embedder, retriever, reranker, generator, guard, gateway client) defines its `Protocol` in `protocols.py` inside its submodule. Default impl lives next to it. The query route pulls `QueryService` from `app.state`; ingest/documents/eval routers construct dependencies inline for simplicity.
- **Pydantic v2 at every API boundary.** Request/response models live in `api/schemas.py`. Inter-module value objects live in `core/types.py` (`ParsedDocument`, `Chunk`, `Citation`, `RetrievalHit`, `Answer`, `JobStatus`).
- **Settings:** single `Settings` class via `pydantic-settings`, accessed through `lru_cache`-wrapped `get_settings()`. Env-var driven. Secrets only in env, never in code or logs.
- **Errors:** `RAGError` hierarchy in `core/errors.py`. Mapped to HTTP responses by a global FastAPI exception handler. Worker errors land in `JobStore`, not in HTTP responses.
- **Async first.** FastAPI routes async. Use `AsyncQdrantClient`, `AsyncOpenAI`, `httpx.AsyncClient`, `redis.asyncio`. Never block the event loop.
- **Observability:** every pipeline stage opens a Logfire span (`stage_span("parse")`, `stage_span("retrieve.hybrid")`, etc.) carrying `request_id`, `job_id`, `doc_id`, `version_id`, or `query_hash` as appropriate. structlog emits JSON logs routed through Logfire.
- **Repo-relative paths only** in docs and plans. Never absolute paths.

## Key Implementation Invariants

- **Provider traffic routing:** Embeddings go through Bifrost AI Gateway to OpenAI (`openai/text-embedding-3-small`). Chat/LLM calls go directly to MeshAPI (`https://api.meshapi.ai`) with the `MESH_API_KEY`. Cohere rerank calls go directly to `https://api.cohere.com/v2/rerank` from `CohereReranker`. Bifrost config (`docker/bifrost/config.json`) defines only the OpenAI provider for embeddings.
- **Qdrant point id is deterministic:** `str(uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:{version_id}:{chunk_index}"))` where `NAMESPACE_RAG` is a fixed UUID constant in `core/types.py`. Re-upsert is idempotent.
- **Single Qdrant collection (`documents`) with two named vectors:** `dense` (1536-dim cosine) and `sparse`. Each point carries the full lineage payload — `doc_id`, `version_id`, `active`, `page`, `section_path`, `bbox`, `chunk_text`, `chunk_type`, `chunk_index`.
- **Versioning is payload-based, not collection-per-version.** Default retrieval filters `active == True`. Explicit `version_ids` parameter bypasses the `active` filter so older versions remain queryable.
- **SPLADE v3 is symmetric.** Single model `naver/splade-v3` used for both ingestion and query-side sparse embeddings. Controlled by `SPARSE_ENABLED` and `SPARSE_MODEL` settings.
- **`SPARSE_ENABLED=False` is a runtime escape hatch** — sparse channel skipped at both ingest and retrieve, dense-only fallback. No code changes required for low-memory machines.
- **LLM Guard `FactualConsistency` has a ~512-token NLI window.** Long retrieved contexts must be split into ≤512-token slices, scanned via `asyncio.gather` concurrently, then aggregated: `is_valid = all(slices)`, `score = max(slices)`.
- **Citations are returned as a separate `citations` array** on `QueryResponse` — the LLM is explicitly instructed not to emit inline `[N]` markers.
- **`Citation.score` semantics:** rerank score when reranking ran for this query, else the RRF fusion score. `RetrievalHit.score` carries the same semantic.
- **Query orchestration lives in `api/services/query_service.py`.** Both the `POST /query` route handler and the eval runner call `QueryService.run(...)` directly. The route handler is thin; the eval runner avoids re-wiring FastAPI dependency injection.
- **Cache key for the answer cache** is `sha256(query + filters + active_version_hash + doc_cache_epoch)`. The `active_version_hash` is read from a Redis-maintained index (`doc:active:{doc_id}` hash), not by scanning Qdrant. `doc_cache_epoch` (`doc:epoch:{doc_id}` integer) is bumped on every hard delete so cached answers referencing that `doc_id` become unreachable without a keyspace scan.
- **`JobRunner` ships with `BackgroundTasksRunner` (FastAPI BackgroundTasks)** as the active implementation, with a Redis-backed `JobStore` for status. Migration to arq/celery is a deferred follow-up; the protocol exists to localize that swap.

## Working Style

- **Plan-first.** Murtuza prefers thorough upfront design before implementation. Confirm approach, data flow, and component boundaries before writing code. The plan in `docs/plans/` already does this — follow it phase-by-phase.
- **Production-grade by default.** Proper error handling at system boundaries, structured logging, health checks, async I/O, type-safe contracts. No prototype shortcuts.
- **Quality over speed.** No hacks, no backwards-compatibility shims, no speculative abstractions. Clean module boundaries.
- **Cost-conscious.** When AWS / cloud architecture choices come up, surface trade-offs (always-on vs on-demand, data transfer, model inference cost). Not the primary focus this hackathon — cloud deployment is out of scope — but the discipline applies to model and embedding choices.
- **Documentation discipline.** Always read API documentation before integrating a new SDK or service. Saves more tokens than it costs.

## Validation Loop

- After any Python edit, run the affected module or its tests via Bash before moving on. Catch import errors at the edit site.
- Do not chain multiple unrelated edits without a validation checkpoint between logical units.

## Edit Discipline

- Always `Read` a file before editing. Never modify a file not read in the current session.
- Prefer `Edit` over `Write`. Reserve `Write` for new files or full rewrites the user explicitly asked for.
- Use one `Write` over many sequential `Edit` calls when changes are widespread in a single file.

## Out of Scope (Do Not Build)

These are explicit non-goals from the requirements doc — do not add them speculatively:

- Frontend / UI
- AuthN / AuthZ / multi-tenant isolation
- Prometheus / Grafana / OTel collector (Logfire is the chosen substitute)
- Kubernetes manifests, CI/CD pipelines, load testing
- Hosted / cloud Qdrant
- Token-streaming SSE responses
- Embedding / rerank model fine-tuning
- Switching `JobRunner` from `BackgroundTasks` to arq/celery (post-hackathon hardening)

## Required Environment Variables

Listed in full in U1 of the plan. Required (no defaults, app fails to start without them): `AZURE_DI_ENDPOINT`, `AZURE_DI_KEY`, `OPENAI_API_KEY`, `MESH_API_KEY`, `COHERE_API_KEY`, `LOGFIRE_TOKEN`. Sidecar URLs and tunables have sensible defaults — see `Settings` in `src/app/core/settings.py` and `.env.example`.

## Project Tracker

Not configured yet. If issue creation is requested, ask whether to use GitHub or Linear.
