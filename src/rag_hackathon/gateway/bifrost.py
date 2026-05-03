from __future__ import annotations

import httpx
import structlog
from openai import AsyncOpenAI

from rag_hackathon.core.errors import GatewayError
from rag_hackathon.gateway.protocols import RerankHit
from rag_hackathon.observability.tracing import stage_span

logger = structlog.get_logger("rag_hackathon.gateway")


class BifrostClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "unused",
        chat_provider: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_provider = chat_provider
        self._openai = AsyncOpenAI(
            base_url=f"{self._base_url}/openai/v1",
            api_key=api_key,
        )
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60.0),
        )

    async def close(self) -> None:
        await self._openai.close()
        await self._http.aclose()

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        routed_model = f"openai/{model}" if not model.startswith("openai/") else model
        with stage_span("gateway.embed", model=routed_model, n_texts=len(texts)):
            try:
                resp = await self._openai.embeddings.create(
                    input=texts, model=routed_model
                )
                return [d.embedding for d in resp.data]
            except Exception as exc:
                raise GatewayError(f"Embedding call failed: {exc}") from exc

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        **opts: object,
    ) -> str:
        routed_model = f"{self._chat_provider}/{model}" if self._chat_provider else model
        with stage_span("gateway.chat", model=routed_model):
            try:
                resp = await self._openai.chat.completions.create(
                    model=routed_model,
                    messages=messages,
                )
                content = resp.choices[0].message.content
                return content or ""
            except Exception as exc:
                raise GatewayError(f"Chat call failed: {exc}") from exc

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int = 5,
    ) -> list[RerankHit]:
        if not documents:
            return []
        with stage_span("gateway.rerank", model=model, n_docs=len(documents)):
            try:
                resp = await self._http.post(
                    "/v1/rerank",
                    json={
                        "model": model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    RerankHit(
                        index=r["index"],
                        document=r["document"]["text"],
                        relevance_score=r["relevance_score"],
                    )
                    for r in data.get("results", [])
                ]
            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"Rerank call failed: {exc.response.status_code}"
                ) from exc
            except Exception as exc:
                raise GatewayError(f"Rerank call failed: {exc}") from exc
