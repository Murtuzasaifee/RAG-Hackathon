from __future__ import annotations

from typing import Protocol


class GatewayClient(Protocol):
    async def embed(
        self, texts: list[str], model: str
    ) -> list[list[float]]: ...

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        **opts: object,
    ) -> str: ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int,
    ) -> list[RerankHit]: ...


class RerankHit:
    __slots__ = ("index", "document", "relevance_score")

    def __init__(
        self, index: int, document: str, relevance_score: float
    ) -> None:
        self.index = index
        self.document = document
        self.relevance_score = relevance_score
