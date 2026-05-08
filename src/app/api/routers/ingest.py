from __future__ import annotations

import json
import pathlib
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile

from app.api.schemas import IngestResponse, JobStatusResponse
from app.cache.redis_cache import RedisCache
from app.security.auth import Principal, require_role
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
from app.versioning.manager import VersionManager

logger = structlog.get_logger("app.api.ingest")

_OUTPUT_DIR = pathlib.Path("output")

router = APIRouter()


def _write_debug(doc_id: str, version_id: str, label: str, data: object) -> None:
    try:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / f"{doc_id}__{version_id}__{label}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("debug_output_written", path=str(path))
    except Exception as exc:
        logger.warning("debug_output_failed", error=str(exc))


async def _run_ingest_pipeline(
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
                created_at=now_utc(),
                updated_at=now_utc(),
            )
        )

    try:
        logger.info("ingest_started", job_id=job_id, doc_id=doc_id, version_id=version_id, file_size_bytes=len(file_bytes))
        await _update("running", "parsing", 10)

        parser = AzureDIParser(
            endpoint=settings.azure_di_endpoint,
            key=settings.azure_di_key,
        )
        parsed = await parser.parse(file_bytes, doc_id)
        logger.info(
            "parse_done",
            job_id=job_id,
            doc_id=doc_id,
            total_elements=len(parsed.elements),
            pages=len({e.page for e in parsed.elements}),
            labels={
                lbl: sum(1 for e in parsed.elements if e.label == lbl)
                for lbl in {e.label for e in parsed.elements}
            },
        )
        _write_debug(doc_id, version_id, "parsed", parsed.model_dump())

        await _update("running", "chunking", 30)
        chunker = get_chunker(settings.chunker_strategy)
        chunks = chunker.chunk(parsed, max_tokens=settings.chunk_max_tokens)
        chunk_type_counts = {}
        for c in chunks:
            chunk_type_counts[c.chunk_type] = chunk_type_counts.get(c.chunk_type, 0) + 1
        logger.info(
            "chunk_done",
            job_id=job_id,
            doc_id=doc_id,
            total_chunks=len(chunks),
            by_type=chunk_type_counts,
            strategy=settings.chunker_strategy,
            max_tokens=settings.chunk_max_tokens,
        )
        _write_debug(doc_id, version_id, "chunks", [c.model_dump() for c in chunks])

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
            logger.info("embedding_dense_started", job_id=job_id, doc_id=doc_id, n_texts=len(texts))
            dense_vectors = await bifrost.embed(texts, settings.embedding_model)
            logger.info("embedding_dense_done", job_id=job_id, doc_id=doc_id, n_vectors=len(dense_vectors))

            sparse_vectors: list[SparseVector] | None = None
            if settings.sparse_enabled:
                await _update("running", "embedding_sparse", 65)
                from app.ingestion.embedders.splade_sparse import (
                    SpladeSparseEmbedder,
                )
                logger.info("embedding_sparse_started", job_id=job_id, doc_id=doc_id, model=settings.splade_model)
                sparse_embedder = SpladeSparseEmbedder(settings.splade_model, hf_token=settings.huggingface_token)
                sparse_vectors = await sparse_embedder.embed(texts)
                logger.info("embedding_sparse_done", job_id=job_id, doc_id=doc_id, n_vectors=len(sparse_vectors))
            else:
                logger.info("embedding_sparse_skipped", job_id=job_id, doc_id=doc_id)

            await _update("running", "indexing", 80)
            from qdrant_client import AsyncQdrantClient

            qdrant = AsyncQdrantClient(url=settings.qdrant_url)
            try:
                indexer = QdrantIndexer(qdrant, settings.qdrant_collection)
                await indexer.ensure_collection()
                await indexer.upsert(
                    chunks, dense_vectors, sparse_vectors, doc_id, version_id, owner_id=owner_id
                )
                logger.info(
                    "indexing_done",
                    job_id=job_id,
                    doc_id=doc_id,
                    version_id=version_id,
                    points_upserted=len(chunks),
                    collection=settings.qdrant_collection,
                )
            finally:
                await qdrant.close()

            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(settings.redis_url)
            cache = RedisCache(redis_client)
            qdrant2 = AsyncQdrantClient(url=settings.qdrant_url)
            try:
                vm = VersionManager(qdrant2, settings.qdrant_collection, cache)
                await vm.flip_active(doc_id, version_id)
            finally:
                await qdrant2.close()

            await _update("done", "done", 100)
            logger.info("ingest_complete", job_id=job_id, doc_id=doc_id, version_id=version_id, total_chunks=len(chunks))
        finally:
            await bifrost.close()

    except Exception as exc:
        logger.error("ingest_failed", job_id=job_id, doc_id=doc_id, error=str(exc), exc_info=True)
        await _update("failed", "failed", 0, error=str(exc))


@router.post("/ingest", response_model=IngestResponse)  # noqa: B008
async def ingest(
    background_tasks: BackgroundTasks,
    principal: Annotated[Principal, Depends(require_role("editor"))],
    file: UploadFile = File(...),  # noqa: B008
    doc_id: str | None = Form(None),  # noqa: B008
):
    settings = get_settings()
    file_bytes = await file.read()

    resolved_doc_id = doc_id or file.filename or make_job_id()
    version_id = make_version_id()
    job_id = make_job_id()

    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(settings.redis_url)
    store = RedisJobStore(redis_client)

    now = now_utc()
    await store.set_status(
        JobStatus(
            job_id=job_id,
            doc_id=resolved_doc_id,
            version_id=version_id,
            state="pending",
            stage="queued",
            progress=0,
            created_at=now,
            updated_at=now,
        )
    )

    background_tasks.add_task(
        _run_ingest_pipeline,
        job_id,
        resolved_doc_id,
        version_id,
        file_bytes,
        store,
        principal.key_id,
    )

    return IngestResponse(
        job_id=job_id,
        doc_id=resolved_doc_id,
        version_id=version_id,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    principal: Annotated[Principal, Depends(require_role("reader"))],
):
    settings = get_settings()
    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(settings.redis_url)
    store = RedisJobStore(redis_client)
    status = await store.get_status(job_id)

    if status is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=status.job_id,
        doc_id=status.doc_id,
        version_id=status.version_id,
        state=status.state,
        stage=status.stage,
        progress=status.progress,
        error=status.error,
    )
