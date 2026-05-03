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
