from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from rag_hackathon.api.schemas import EvalRunResponse
from rag_hackathon.eval.ragas_runner import EvalReport, run_eval

router = APIRouter(prefix="/eval", tags=["eval"])

_last_report: EvalReport | None = None


async def _run_eval_background() -> EvalReport:
    from rag_hackathon.api.services.query_service import QueryService
    from rag_hackathon.core.settings import get_settings
    from rag_hackathon.gateway.bifrost import BifrostClient
    from rag_hackathon.generation.generator import GroundedGenerator
    from rag_hackathon.retrieval.hybrid_qdrant import HybridQdrantRetriever
    from rag_hackathon.retrieval.reranker_cohere import CohereReranker

    settings = get_settings()

    bifrost = BifrostClient(
        base_url=settings.bifrost_url,
        api_key=settings.openai_api_key,
    )
    try:
        retriever = HybridQdrantRetriever(
            qdrant_url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            rrf_k=settings.rrf_k,
            sparse_enabled=settings.sparse_enabled,
        )
        reranker = CohereReranker(bifrost, settings.rerank_model)
        generator = GroundedGenerator(bifrost, settings.llm_model)

        query_service = QueryService(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        return await run_eval(query_service)
    finally:
        await bifrost.close()


@router.post("/run", response_model=EvalRunResponse)
async def trigger_eval(background_tasks: BackgroundTasks):
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
