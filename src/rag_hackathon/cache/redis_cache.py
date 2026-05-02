from __future__ import annotations

import hashlib
import json
from typing import Any

import msgpack
import redis.asyncio as aioredis
import structlog

from rag_hackathon.observability.tracing import stage_span

logger = structlog.get_logger("rag_hackathon.cache")


def cache_key(prefix: str, key: str) -> str:
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"cache:{prefix}:{h}"


class RedisCache:
    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def get(self, prefix: str, key: str) -> bytes | None:
        with stage_span("cache.lookup", prefix=prefix):
            try:
                val = await self._client.get(cache_key(prefix, key))
                if val is not None:
                    logger.debug("cache.hit", prefix=prefix)
                else:
                    logger.debug("cache.miss", prefix=prefix)
                return val
            except Exception as exc:
                logger.warning("cache.get.failed", prefix=prefix, error=str(exc))
                return None

    async def set(
        self, prefix: str, key: str, value: bytes, ttl: int
    ) -> None:
        with stage_span("cache.store", prefix=prefix, ttl=ttl):
            try:
                await self._client.set(cache_key(prefix, key), value, ex=ttl)
                logger.debug("cache.stored", prefix=prefix, ttl=ttl)
            except Exception as exc:
                logger.warning("cache.set.failed", prefix=prefix, error=str(exc))

    async def delete(self, prefix: str, key: str) -> None:
        try:
            await self._client.delete(cache_key(prefix, key))
        except Exception as exc:
            logger.warning("cache.delete.failed", prefix=prefix, error=str(exc))

    async def get_json(self, prefix: str, key: str) -> dict[str, Any] | None:
        raw = await self.get(prefix, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("cache.json_decode.failed", prefix=prefix)
            return None

    async def set_json(
        self, prefix: str, key: str, value: dict[str, Any], ttl: int
    ) -> None:
        await self.set(prefix, key, json.dumps(value).encode(), ttl)

    async def get_msgpack(self, prefix: str, key: str) -> list[Any] | None:
        raw = await self.get(prefix, key)
        if raw is None:
            return None
        try:
            decoded = msgpack.unpackb(raw, raw=False)
            if isinstance(decoded, list):
                return decoded
            logger.warning("cache.msgpack.unexpected_type", prefix=prefix)
            return None
        except Exception as exc:
            logger.warning("cache.msgpack_decode.failed", prefix=prefix, error=str(exc))
            return None

    async def set_msgpack(
        self, prefix: str, key: str, value: list[Any], ttl: int
    ) -> None:
        await self.set(prefix, key, msgpack.packb(value, use_bin_type=True), ttl)

    async def incr(self, key: str) -> int:
        try:
            return await self._client.incr(key)
        except Exception as exc:
            logger.warning("cache.incr.failed", key=key, error=str(exc))
            return 0

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        try:
            return await self._client.hgetall(key)
        except Exception as exc:
            logger.warning("cache.hgetall.failed", key=key, error=str(exc))
            return {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        try:
            await self._client.hset(key, mapping=mapping)
        except Exception as exc:
            logger.warning("cache.hset.failed", key=key, error=str(exc))
