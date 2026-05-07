from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.schemas import EvalRunResponse
from app.security.auth import Principal, require_role
from app.eval.ragas_runner import EvalReport, run_eval

router = APIRouter(prefix="/eval", tags=["eval"])

_last_report: EvalReport | None = None


async def _run_eval_background() -> EvalReport:
    from qdrant_client import AsyncQdrantClient

    from app.api.services.query_service import QueryService
    from app.core.settings import get_settings
    from app.gateway.bifrost import BifrostClient
    from app.generation.generator import GroundedGenerator
    from app.ingestion.embedders.openai_dense import OpenAIDenseEmbedder
    from app.ingestion.embedders.splade_sparse import SpladeSparseEmbedder
    from app.retrieval.hybrid_qdrant import HybridQdrantRetriever
    from app.retrieval.reranker_cohere import CohereReranker

    settings = get_settings()

    bifrost = BifrostClient(
        base_url=settings.bifrost_url,
        api_key=settings.openai_api_key,
        chat_provider=settings.llm_provider,
    )
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        dense = OpenAIDenseEmbedder(bifrost, settings.embedding_model)
        sparse = (
            SpladeSparseEmbedder(settings.splade_model, hf_token=settings.huggingface_token)
            if settings.sparse_enabled
            else None
        )
        retriever = HybridQdrantRetriever(
            client=qdrant,
            collection=settings.qdrant_collection,
            dense_embedder=dense,
            sparse_embedder=sparse,
            rrf_k=settings.rrf_k,
        )
        reranker = CohereReranker(settings.bifrost_url, settings.rerank_model)
        generator = GroundedGenerator(bifrost, settings.llm_model)

        query_service = QueryService(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        return await run_eval(query_service)
    finally:
        await bifrost.close()
        await qdrant.close()


@router.post("/run", response_model=EvalRunResponse)
async def trigger_eval(
    background_tasks: BackgroundTasks,
    principal: Annotated[Principal, Depends(require_role("admin"))],
):
    global _last_report  # noqa: PLW0603

    report = await _run_eval_background()
    _last_report = report

    return EvalRunResponse(
        total_questions=report.total_questions,
        failed_questions=report.failed_questions,
        elapsed_seconds=report.elapsed_seconds,
        aggregate=report.aggregate,
        per_question=[
            {
                "question": r.question,
                "scores": r.scores,
                "error": r.error,
            }
            for r in report.results
        ],
    )
