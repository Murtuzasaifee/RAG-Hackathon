from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from rag_hackathon.core.errors import GatewayError
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

        # Single OpenAI-compat client for both embeddings and chat via /openai/v1.
        # Custom providers referenced as meshapi/openai/gpt-5.4 — Bifrost routes by model prefix.
        self._openai = AsyncOpenAI(
            base_url=f"{self._base_url}/openai/v1",
            api_key=api_key,
        )

    async def close(self) -> None:
        await self._openai.close()

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
        if self._chat_provider and self._chat_provider != "openai":
            openai_model = f"openai/{model}" if not model.startswith("openai/") else model
            routed_model = f"{self._chat_provider}/{openai_model}"
        else:
            routed_model = f"openai/{model}" if not model.startswith("openai/") else model

        provider_label = self._chat_provider or "openai"
        with stage_span("gateway.chat", model=routed_model, provider=provider_label):
            try:
                resp = await self._openai.chat.completions.create(
                    model=routed_model,
                    messages=messages,
                )
                content = resp.choices[0].message.content
                return content or ""
            except Exception as exc:
                raise GatewayError(f"Chat call failed: {exc}") from exc
