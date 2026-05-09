from __future__ import annotations

from typing import Protocol

from app.core.types import RetrievalHit


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        version_ids: list[str] | None = None,
        top_k: int = 20,
        owner_id: str | None = None,
    ) -> list[RetrievalHit]: ...


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_n: int = 5,
    ) -> list[RetrievalHit]: ...
