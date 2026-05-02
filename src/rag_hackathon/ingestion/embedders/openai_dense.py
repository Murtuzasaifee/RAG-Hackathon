from __future__ import annotations

import structlog

from rag_hackathon.gateway.bifrost import BifrostClient
from rag_hackathon.observability.tracing import stage_span

logger = structlog.get_logger("rag_hackathon.ingestion.embedders.dense")


class OpenAIDenseEmbedder:
    def __init__(self, client: BifrostClient, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with stage_span("embed.dense", model=self._model, n=len(texts)):
            return await self._client.embed(texts, self._model)
