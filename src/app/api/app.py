from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from app.api.error_handlers import RAGError, rag_error_handler
from app.api.middleware import RequestIdMiddleware
from app.api.routers import demo, documents, eval, health, ingest, query
from app.api.services.query_service import QueryService
from app.cache.redis_cache import RedisCache
from app.cache.semantic_cache import SemanticQueryCache
from app.core.settings import get_settings
from app.generation.generator import GroundedGenerator
from app.gateway.bifrost import BifrostClient
from app.ingestion.embedders.openai_dense import OpenAIDenseEmbedder
from app.ingestion.embedders.splade_sparse import SpladeSparseEmbedder
from app.observability.logging import configure_logging, get_logger
from app.observability.tracing import configure_tracing
from app.retrieval.hybrid_qdrant import HybridQdrantRetriever
from app.retrieval.reranker_cohere import CohereReranker
from app.security.auth import seed_api_key
from app.security.guard import LLMGuardClient

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()

    bifrost = BifrostClient(
        base_url=settings.bifrost_url,
        api_key=settings.openai_api_key,
        chat_provider=settings.llm_provider,
    )
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    redis_client = aioredis.from_url(settings.redis_url)

    dense_embedder = OpenAIDenseEmbedder(bifrost, settings.embedding_model)
    sparse_embedder = (
        SpladeSparseEmbedder(settings.splade_model, hf_token=settings.huggingface_token)
        if settings.sparse_enabled
        else None
    )

    retriever = HybridQdrantRetriever(
        client=qdrant,
        collection=settings.qdrant_collection,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        rrf_k=settings.rrf_k,
    )
    reranker = CohereReranker(settings.bifrost_url, settings.rerank_model)
    generator = GroundedGenerator(bifrost, settings.llm_model)
    cache = RedisCache(redis_client)
    guard = LLMGuardClient(settings.llm_guard_url)

    semantic_cache: SemanticQueryCache | None = None
    if settings.semantic_cache_enabled:
        semantic_cache = SemanticQueryCache(
            client=qdrant,
            embedder=dense_embedder,
            collection=settings.semantic_cache_collection,
            threshold=settings.semantic_cache_threshold,
        )
        await semantic_cache.ensure_collection()

    app.state.query_service = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        input_guard=guard if settings.llm_guard_input_enabled else None,
        output_guard=guard if settings.llm_guard_output_enabled else None,
        cache=cache,
        cache_ttl_answer=settings.cache_ttl_answer,
        semantic_cache=semantic_cache,
    )
    app.state.redis = redis_client
    app.state.qdrant = qdrant
    app.state.bifrost = bifrost
    app.state.semantic_cache = semantic_cache

    if not settings.auth_enabled:
        logger.warning(
            "auth.disabled",
            message="AUTH_ENABLED=false — all requests treated as admin. Never use in production.",
        )
    if settings.admin_api_key:
        await seed_api_key(redis_client, settings.admin_api_key, "admin", "admin-bootstrap")
        logger.info("auth.bootstrap.complete", message="Admin API key seeded from ADMIN_API_KEY")
    else:
        logger.info("auth.bootstrap.skipped", message="ADMIN_API_KEY not set; skipping bootstrap")

    # Warm up LLM Guard if either guard is enabled
    if settings.llm_guard_input_enabled or settings.llm_guard_output_enabled:
        logger.info(
            "llm_guard.warmup.starting",
            llm_guard_url=settings.llm_guard_url,
            message="Waiting for LLM Guard to load ML scanner models — startup will resume once warmup completes",
        )
        await guard.warmup()
        logger.info("llm_guard.warmup.complete", message="LLM Guard warmup finished; all scanners ready")
    else:
        logger.info("llm_guard.warmup.skipped", message="Skipping LLM Guard warmup because both input and output guards are disabled")

    yield

    await guard.close()
    await reranker.close()
    await bifrost.close()
    await qdrant.close()
    await redis_client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Hackathon Backend",
        version="0.1.0",
        lifespan=lifespan,
    )

    configure_tracing(app)
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(RAGError, rag_error_handler)
    app.mount("/static/demo", demo.static_files, name="demo-static")

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(demo.router)
    # app.include_router(eval.router)

    return app


app = create_app()
