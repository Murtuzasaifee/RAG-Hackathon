---
title: Production-Grade RAG Backend (Hackathon)
type: requirements
status: confirmed
date: 2026-05-02
---

# Production-Grade RAG Backend (Hackathon)

## Summary

Production-shaped RAG backend in Python/FastAPI for a hackathon, no frontend in scope. Async ingestion pipeline parses documents with Azure Document Intelligence, chunks structure-aware, embeds via OpenAI + Splade for hybrid retrieval into a local Qdrant (RRF fusion), reranks with Cohere, and generates guarded LLM answers with full page+section+bbox citations. LLM Guard scans both input and output. Bifrost AI Gateway proxies all provider calls. Redis caches embeddings, rerank results, and answers. Documents are versioned with update/delete workflows. Logfire + structlog handle observability. RAGAS eval endpoint validates groundedness on a small golden set. Whole stack ships via docker-compose.

---

## Problem Frame

A hackathon entry must demo a *production-grade* RAG pipeline — meaning the architecture, contracts, and operational posture should look real, not toy. Generic-domain so it can be re-pointed at any document corpus later without rewriting the pipeline. Backend-only this round; frontend deferred. Judging weight is on demo quality plus the credibility of the production-shaped components (security, observability, eval, lineage), so each piece must be demonstrable end-to-end on a single laptop via `docker compose up`.

---

## Requirements

- R1. Backend in Python 3.13 + FastAPI (async). No frontend.
- R2. Document ingestion via Azure Document Intelligence; preserve hierarchical structure (sections, tables, headings, bbox).
- R3. Chunking is structure-aware (section-bounded + token cap + tables-as-chunks) and exposed behind a pluggable strategy interface.
- R4. Hybrid retrieval over Qdrant: dense embeddings (OpenAI) + sparse Splade vectors fused with RRF.
- R5. Reranking with Cohere on the fused candidate set.
- R6. Generation through an LLM, with retrieved chunks provided as grounding context.
- R7. Response payload returns full source lineage per citation: `doc_id`, `page`, `section_path` (heading hierarchy), `bbox`, `chunk_text`, `score`. Citations are returned as a separate `citations[]` array (no inline `[1]` markers in answer text).
- R8. Async ingestion: `POST /ingest` returns a `job_id`; `GET /jobs/{job_id}` returns parse → chunk → embed → index status.
- R9. LLM Guard scans on both input (prompt injection, PII, toxicity) and output (groundedness vs cited chunks, sensitive data leak).
- R10. Bifrost AI Gateway proxies all provider traffic — OpenAI embeddings, OpenAI chat, Cohere rerank — over its OpenAI-compatible REST surface.
- R11. Redis caching layer covering: query embedding cache, Cohere rerank cache, final answer cache keyed by `hash(query + filters + active_doc_versions)`. TTL configurable per layer.
- R12. Document versioning: each ingestion of an existing `doc_id` creates a new `version_id`. Qdrant points carry `(doc_id, version_id)` payload. Default query filter targets latest active versions; older versions are still queryable via explicit filter.
- R13. Document update endpoint = re-ingest under same `doc_id` → new version + invalidation of dependent Redis keys.
- R14. Document delete endpoint = soft delete (mark version inactive, exclude from query); hard delete purges Qdrant points and dependent caches.
- R15. RAGAS eval endpoint (`POST /eval/run`) executes faithfulness, context_precision, answer_relevancy on a curated 5–10 Q&A golden set bundled with the repo.
- R16. Observability: Logfire instruments FastAPI + httpx + Qdrant client; explicit spans per pipeline stage (parse, chunk, embed, retrieve, rerank, generate, guard).
- R17. Logging: structlog producing structured JSON logs; correlation/request id propagated through every stage.
- R18. Whole stack runs locally via `docker-compose up` — FastAPI service, Qdrant, Redis, Bifrost gateway, LLM Guard sidecar.
- R19. Pluggable component boundaries (chunker, embedder, retriever, reranker, generator, guard, gateway client) so any single piece can be swapped without touching the rest.
- R20. Pydantic v2 models at every API boundary; provider-call boundary inputs/outputs typed.

---

## Scope Boundaries

- Frontend / UI work
- AuthN, AuthZ, multi-tenant isolation
- Prometheus / Grafana / OTel collector (Logfire covers observability for this scope)
- Kubernetes manifests, CI/CD pipelines, load testing
- Hosted / cloud Qdrant
- Token-streaming SSE responses for MVP
- Embedding or rerank model fine-tuning
- Domain-specific extractors (table understanding, OCR fallback) — pluggable interface only

---

## Key Decisions

- **Bifrost in front of every provider.** Single egress point gives uniform observability, retries, virtual keys, and cost tracking. Not optional even for hackathon scope — it is one of the headline production-grade signals.
- **LLM Guard as a sidecar service**, not in-process. Models are heavy (~16 GB RAM recommended); keeping them in the FastAPI worker would bloat startup and resource use. Sidecar scales independently and the FastAPI service calls it over HTTP.
- **Structure-aware chunking owned by Azure DI output.** Throwing away DI's hierarchical structure with fixed-window chunking would waste the tool choice. Section-bounded chunks with a token cap fallback for oversize sections; tables become their own markdown chunks.
- **Citations as separate `citations[]` array.** Avoids the prompt-engineering complexity of stable inline `[N]` markers; the array is sufficient for any frontend to render lineage and is simpler to validate against groundedness checks.
- **Hybrid via RRF (reciprocal rank fusion).** Score-free fusion sidesteps Splade-vs-dense score normalization issues; well-known default that tunes minimally.
- **FastAPI BackgroundTasks for the demo.** arq/celery would add operational overhead with no demo value at this scope; the background runner sits behind a `JobRunner` interface so it can be swapped to arq/celery later.
- **Versioning via payload filters, not separate collections.** A single Qdrant collection per namespace with `(doc_id, version_id, active)` payload keeps the index simple and reuses query-time filtering.
- **Logfire over Prometheus/OTel.** Logfire bundles traces + metrics + structured log forwarding with minimal setup; Prometheus stack would be over-scoped for hackathon. Bifrost has its own OTel exporter that can be wired later if needed.

---

## Dependencies / Assumptions

- Azure Document Intelligence endpoint + API key are available (assumed user has Azure subscription).
- OpenAI and Cohere API keys are available, both routed via Bifrost.
- Splade model (e.g., `naver/efficient-splade-VI-BT-large-doc`) runs on CPU at acceptable demo latency for small corpora; if not, fall back to a smaller variant.
- LLM Guard `FactualConsistency` scanner has a ~512-token NLI window; long retrieved contexts are chunked and the max entailment score is taken.
- Bifrost image is pinned to a post-April-2026 build for Cohere v2 rerank fixes.
- Hackathon corpus is small enough (tens of documents) that local Qdrant + in-process Splade are sufficient; horizontal scaling is out of scope.

---

## Open Questions (Deferred to Implementation)

- Exact Splade model variant — pick smallest viable on hackathon machine.
- LLM Guard sidecar GPU vs CPU — depends on demo machine; CPU acceptable with chunked groundedness.
- Final RAGAS golden-set source — to be assembled during eval-unit implementation from a representative ingest.
