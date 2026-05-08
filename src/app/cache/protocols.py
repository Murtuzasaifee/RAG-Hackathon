from __future__ import annotations

from typing import Any, Protocol


class CacheStore(Protocol):
    async def get(self, prefix: str, key: str) -> bytes | None: ...

    async def set(
        self, prefix: str, key: str, value: bytes, ttl: int
    ) -> None: ...

    async def delete(self, prefix: str, key: str) -> None: ...

    async def get_json(self, prefix: str, key: str) -> dict[str, Any] | None: ...

    async def set_json(
        self, prefix: str, key: str, value: dict[str, Any], ttl: int
    ) -> None: ...

    async def get_msgpack(self, prefix: str, key: str) -> list[Any] | None: ...

    async def set_msgpack(
        self, prefix: str, key: str, value: list[Any], ttl: int
    ) -> None: ...
