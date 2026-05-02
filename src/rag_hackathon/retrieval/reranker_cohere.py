from __future__ import annotations

import structlog

from rag_hackathon.core.types import RetrievalHit
from rag_hackathon.gateway.protocols import GatewayClient
from rag_hackathon.observability.tracing import stage_span

logger = structlog.get_logger("rag_hackathon.retrieval.reranker")


class CohereReranker:
    def __init__(self, client: GatewayClient, model: str) -> None:
        self._client = client
        self._model = model

    async def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_n: int = 5,
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        with stage_span("rerank", model=self._model, n_hits=len(hits), top_n=top_n):
            effective_top_n = min(top_n, len(hits))
            documents = [h.chunk_text for h in hits]
            reranked = await self._client.rerank(
                query=query,
                documents=documents,
                model=self._model,
                top_n=effective_top_n,
            )
            results: list[RetrievalHit] = []
            for r in reranked:
                original = hits[r.index]
                results.append(
                    RetrievalHit(
                        doc_id=original.doc_id,
                        version_id=original.version_id,
                        chunk_index=original.chunk_index,
                        chunk_text=original.chunk_text,
                        page=original.page,
                        section_path=original.section_path,
                        bbox=original.bbox,
                        chunk_type=original.chunk_type,
                        score=r.relevance_score,
                    )
                )
            logger.debug(
                "rerank complete",
                n_results=len(results),
            )
            return results
