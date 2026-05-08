from __future__ import annotations

from typing import Protocol

from app.core.types import Answer, RetrievalHit


class Generator(Protocol):
    async def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> Answer: ...
