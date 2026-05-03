from __future__ import annotations

import httpx
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
        chat_api_key: str | None = None,
        chat_base_url: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._chat_provider = chat_provider
        self._chat_api_key = chat_api_key or api_key
        self._chat_base_url = chat_base_url

        self._openai = AsyncOpenAI(
            base_url=f"{self._base_url}/openai/v1",
            api_key=api_key,
        )
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60.0),
        )
        self._chat_http: httpx.AsyncClient | None = None
        if chat_base_url:
            self._chat_http = httpx.AsyncClient(
                base_url=chat_base_url.rstrip("/"),
                timeout=httpx.Timeout(60.0),
            )

    async def close(self) -> None:
        await self._openai.close()
        await self._http.aclose()
        if self._chat_http:
            await self._chat_http.aclose()

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
            return await self._chat_direct(messages, model)

        routed_model = f"openai/{model}" if not model.startswith("openai/") else model
        with stage_span("gateway.chat", model=routed_model, provider="openai"):
            try:
                resp = await self._openai.chat.completions.create(
                    model=routed_model,
                    messages=messages,
                )
                content = resp.choices[0].message.content
                return content or ""
            except Exception as exc:
                raise GatewayError(f"Chat call failed: {exc}") from exc

    async def _chat_direct(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> str:
        routed_model = f"openai/{model}" if not model.startswith("openai/") else model
        with stage_span("gateway.chat", model=routed_model, provider=self._chat_provider):
            client = self._chat_http or self._http
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._chat_api_key}",
            }
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": routed_model,
                        "messages": messages,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content or ""
            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"Chat call failed: {exc.response.status_code} - {exc.response.text}"
                ) from exc
            except Exception as exc:
                raise GatewayError(f"Chat call failed: {exc}") from exc
