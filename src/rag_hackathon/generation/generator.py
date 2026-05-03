from __future__ import annotations

import structlog

from rag_hackathon.core.errors import GenerationError
from rag_hackathon.core.types import Answer, Citation, RetrievalHit
from rag_hackathon.gateway.protocols import GatewayClient
from rag_hackathon.generation.prompt import build_messages, hits_to_dicts
from rag_hackathon.observability.tracing import stage_span

logger = structlog.get_logger("rag_hackathon.generation")

_NO_CONTEXT_ANSWER = "I don't know based on the provided documents."


class GroundedGenerator:
    def __init__(self, client: GatewayClient, model: str) -> None:
        self._client = client
        self._model = model

    async def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> Answer:
        if not hits:
            return Answer(
                text=_NO_CONTEXT_ANSWER,
                citations=[],
                model=self._model,
                usage={},
            )

        with stage_span("generate", model=self._model, n_hits=len(hits)):
            messages = build_messages(query, hits_to_dicts(hits))
            logger.info("generate", query=query, hits=hits_to_dicts(hits))
            try:
                text = await self._client.chat(messages, self._model)
            except Exception as exc:
                raise GenerationError(f"LLM generation failed: {exc}") from exc

            citations = [
                Citation(
                    doc_id=h.doc_id,
                    version_id=h.version_id,
                    page=h.page,
                    section_path=list(h.section_path),
                    bbox=list(h.bbox),
                    chunk_text=h.chunk_text,
                    chunk_type=h.chunk_type,
                    score=h.score,
                )
                for h in hits
            ]

            return Answer(
                text=text,
                citations=citations,
                model=self._model,
                usage={},
            )
