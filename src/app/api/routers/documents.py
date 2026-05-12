from __future__ import annotations

from typing import Annotated

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from qdrant_client import AsyncQdrantClient, models

from app.api.schemas import DocumentInfo, IngestResponse
from app.cache.redis_cache import RedisCache
from app.core.errors import ForbiddenError
from app.core.settings import get_settings
from app.core.types import JobStage, JobState, JobStatus
from app.ingestion.chunkers import get_chunker
from app.ingestion.embedders.protocols import SparseVector
from app.ingestion.indexer import QdrantIndexer
from app.ingestion.jobs import (
    RedisJobStore,
    make_job_id,
    make_version_id,
    now_utc,
)
from app.ingestion.parser import AzureDIParser
from app.security.auth import (
    ROLE_ORDER,
    Principal,
    require_role,
    verify_document_ownership,
)
from app.versioning.manager import VersionManager

logger = structlog.get_logger("app.api.documents")

router = APIRouter(prefix="/documents", tags=["documents"])


async def _run_reingest_pipeline(
    job_id: str,
    doc_id: str,
    version_id: str,
    file_bytes: bytes,
    store: RedisJobStore,
    owner_id: str | None = None,
) -> None:
    settings = get_settings()

    async def _update(
        state: JobState,
        stage: JobStage,
        progress: int,
        error: str | None = None,
    ) -> None:
        await store.set_status(
            JobStatus(
                job_id=job_id,
                doc_id=doc_id,
                version_id=version_id,
                state=state,
                stage=stage,
                progress=progress,
                error=error,
                owner_id=owner_id,
                created_at=now_utc(),
                updated_at=now_utc(),
            )
        )

    try:
        await _update("running", "parsing", 10)

        parser = AzureDIParser(
            endpoint=settings.azure_di_endpoint,
            key=settings.azure_di_key,
        )
        parsed = await parser.parse(file_bytes, doc_id)

        await _update("running", "chunking", 30)
        chunker = get_chunker(settings.chunker_strategy)
        chunks = chunker.chunk(parsed, max_tokens=settings.chunk_max_tokens)

        for i, c in enumerate(chunks):
            chunks[i] = c.model_copy(update={"version_id": version_id})

        await _update("running", "embedding_dense", 50)
        from app.gateway.bifrost import BifrostClient

        bifrost = BifrostClient(
            base_url=settings.bifrost_url,
            api_key=settings.openai_api_key,
        )
        try:
            texts = [c.text for c in chunks]
            dense_vectors = await bifrost.embed(texts, settings.embedding_model)

            sparse_vectors: list[SparseVector] | None = None
            if settings.sparse_enabled:
                await _update("running", "embedding_sparse", 65)
                from app.ingestion.embedders.splade_sparse import (
                    SpladeSparseEmbedder,
                )

                sparse_embedder = SpladeSparseEmbedder(
                    settings.splade_model, hf_token=settings.huggingface_token
                )
                sparse_vectors = await sparse_embedder.embed(texts)

            await _update("running", "indexing", 80)
            qdrant = AsyncQdrantClient(url=settings.qdrant_url)
            try:
                indexer = QdrantIndexer(qdrant, settings.qdrant_collection)
                await indexer.ensure_collection()
                await indexer.upsert(
                    chunks,
                    dense_vectors,
                    sparse_vectors,
                    doc_id,
                    version_id,
                    owner_id=owner_id,
                )
            finally:
                await qdrant.close()

            redis_client = aioredis.from_url(settings.redis_url)
            cache = RedisCache(redis_client)
            qdrant2 = AsyncQdrantClient(url=settings.qdrant_url)
            try:
                vm = VersionManager(qdrant2, settings.qdrant_collection, cache)
                await vm.flip_active(doc_id, version_id)
            finally:
                await qdrant2.close()

            if settings.semantic_cache_enabled:
                from app.cache.semantic_cache import SemanticQueryCache
                from app.ingestion.embedders.openai_dense import OpenAIDenseEmbedder
                from app.gateway.bifrost import BifrostClient as _Bifrost

                _bifrost = _Bifrost(base_url=settings.bifrost_url, api_key=settings.openai_api_key)
                _embedder = OpenAIDenseEmbedder(_bifrost, settings.embedding_model)
                _qdrant_inv = AsyncQdrantClient(url=settings.qdrant_url)
                try:
                    sc = SemanticQueryCache(
                        client=_qdrant_inv,
                        embedder=_embedder,
                        collection=settings.semantic_cache_collection,
                        threshold=settings.semantic_cache_threshold,
                    )
                    await sc.invalidate_doc(doc_id)
                finally:
                    await _qdrant_inv.close()
                    await _bifrost.close()

            await _update("done", "done", 100)
        finally:
            await bifrost.close()

    except Exception as exc:
        logger.error("reingest_failed", job_id=job_id, error=str(exc))
        await _update("failed", "failed", 0, error=str(exc))


@router.get("", response_model=list[DocumentInfo])
async def list_documents(
    principal: Annotated[Principal, Depends(require_role("reader"))],
):
    settings = get_settings()
    is_admin = ROLE_ORDER.get(principal.role, -1) >= ROLE_ORDER["admin"]
    owner_id = None if is_admin else principal.key_id

    conditions: list[models.Condition] = [
        models.FieldCondition(key="active", match=models.MatchValue(value=True)),
    ]
    if owner_id is not None:
        conditions.append(
            models.FieldCondition(
                key="owner_id", match=models.MatchValue(value=owner_id)
            )
        )

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        points, _ = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=models.Filter(must=conditions),
            limit=1000,
            with_payload=["doc_id", "version_id", "owner_id"],
        )
    finally:
        await qdrant.close()

    docs: dict[str, dict] = {}
    for point in points:
        p = point.payload or {}
        did = p.get("doc_id", "")
        if did not in docs:
            docs[did] = {
                "doc_id": did,
                "active_version_id": p.get("version_id"),
                "total_chunks": 1,
                "owner_id": p.get("owner_id"),
            }
        else:
            docs[did]["total_chunks"] += 1

    return [DocumentInfo(**d) for d in docs.values()]


@router.put("/{doc_id}", response_model=IngestResponse)  # noqa: B008
async def update_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    principal: Annotated[Principal, Depends(require_role("editor"))],
    file: UploadFile = File(...),  # noqa: B008
):
    settings = get_settings()

    qdrant_check = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        await verify_document_ownership(
            qdrant_check, settings.qdrant_collection, doc_id, principal
        )
    finally:
        await qdrant_check.close()

    file_bytes = await file.read()
    version_id = make_version_id()
    job_id = make_job_id()

    redis_client = aioredis.from_url(settings.redis_url)
    store = RedisJobStore(redis_client)

    now = now_utc()
    await store.set_status(
        JobStatus(
            job_id=job_id,
            doc_id=doc_id,
            version_id=version_id,
            state="pending",
            stage="queued",
            progress=0,
            owner_id=principal.key_id,
            created_at=now,
            updated_at=now,
        )
    )

    background_tasks.add_task(
        _run_reingest_pipeline,
        job_id,
        doc_id,
        version_id,
        file_bytes,
        store,
        principal.key_id,
    )

    return IngestResponse(
        job_id=job_id,
        doc_id=doc_id,
        version_id=version_id,
    )


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_role("editor"))],
    mode: str = "soft",
    version_id: str | None = None,
):
    if mode not in ("soft", "hard"):
        raise HTTPException(
            status_code=400,
            detail="mode must be 'soft' or 'hard'",
        )

    if mode == "hard" and ROLE_ORDER.get(principal.role, -1) < ROLE_ORDER["admin"]:
        raise ForbiddenError("hard delete requires admin role")

    settings = get_settings()

    qdrant_check = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        await verify_document_ownership(
            qdrant_check, settings.qdrant_collection, doc_id, principal
        )
    finally:
        await qdrant_check.close()

    redis_client = aioredis.from_url(settings.redis_url)
    cache = RedisCache(redis_client)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        vm = VersionManager(qdrant, settings.qdrant_collection, cache)

        if mode == "soft":
            n = await vm.soft_delete(doc_id, version_id)
        else:
            n = await vm.hard_delete(doc_id, version_id)

        if n == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No points found for doc_id={doc_id}"
                + (f" version_id={version_id}" if version_id else ""),
            )

        semantic_cache = getattr(request.app.state, "semantic_cache", None)
        if semantic_cache is not None:
            await semantic_cache.invalidate_doc(doc_id)

        return {
            "doc_id": doc_id,
            "mode": mode,
            "version_id": version_id,
            "points_affected": n,
        }
    finally:
        await qdrant.close()
