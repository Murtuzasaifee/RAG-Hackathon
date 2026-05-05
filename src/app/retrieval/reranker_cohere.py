from __future__ import annotations

import httpx
import structlog

from app.core.types import RetrievalHit
from app.observability.tracing import stage_span

logger = structlog.get_logger("app.retrieval.reranker")


class CohereReranker:
    def __init__(self, bifrost_url: str, model: str) -> None:
        self._model = model
        self._http = httpx.AsyncClient(
            base_url=bifrost_url.rstrip("/"),
            timeout=httpx.Timeout(30.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

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
            resp = await self._http.post(
                "/cohere/v2/rerank",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "top_n": effective_top_n,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results: list[RetrievalHit] = []
            for r in data.get("results", []):
                idx = r["index"]
                original = hits[idx]
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
                        score=r["relevance_score"],
                    )
                )
            logger.debug("rerank complete", n_results=len(results))
            return results
