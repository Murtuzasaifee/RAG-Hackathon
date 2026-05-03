from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_hackathon.cache.redis_cache import RedisCache, cache_key


@pytest.fixture
def redis_mock() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def cache(redis_mock: AsyncMock) -> RedisCache:
    return RedisCache(client=redis_mock)


class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key("answer", "hello")
        k2 = cache_key("answer", "hello")
        assert k1 == k2

    def test_different_prefix(self):
        k1 = cache_key("answer", "hello")
        k2 = cache_key("rerank", "hello")
        assert k1 != k2

    def test_different_key(self):
        k1 = cache_key("answer", "hello")
        k2 = cache_key("answer", "world")
        assert k1 != k2

    def test_format(self):
        k = cache_key("answer", "hello")
        assert k.startswith("cache:answer:")


class TestGet:
    @pytest.mark.asyncio
    async def test_get_hit(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.get.return_value = b'{"a": 1}'
        result = await cache.get("answer", "key1")
        assert result == b'{"a": 1}'

    @pytest.mark.asyncio
    async def test_get_miss(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.get.return_value = None
        result = await cache.get("answer", "key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_redis_down_graceful(
        self, cache: RedisCache, redis_mock: AsyncMock
    ):
        redis_mock.get.side_effect = ConnectionError("refused")
        result = await cache.get("answer", "key1")
        assert result is None


class TestSet:
    @pytest.mark.asyncio
    async def test_set_ok(self, cache: RedisCache, redis_mock: AsyncMock):
        await cache.set("answer", "key1", b"value", ttl=3600)
        redis_mock.set.assert_awaited_once()
        args = redis_mock.set.call_args
        assert args[0][1] == b"value"
        assert args[1]["ex"] == 3600

    @pytest.mark.asyncio
    async def test_set_redis_down_graceful(
        self, cache: RedisCache, redis_mock: AsyncMock
    ):
        redis_mock.set.side_effect = ConnectionError("refused")
        await cache.set("answer", "key1", b"value", ttl=3600)


class TestGetJson:
    @pytest.mark.asyncio
    async def test_json_hit(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.get.return_value = b'{"answer": "hello"}'
        result = await cache.get_json("answer", "k")
        assert result == {"answer": "hello"}

    @pytest.mark.asyncio
    async def test_json_miss(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.get.return_value = None
        result = await cache.get_json("answer", "k")
        assert result is None

    @pytest.mark.asyncio
    async def test_json_corrupt(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.get.return_value = b"not json"
        result = await cache.get_json("answer", "k")
        assert result is None


class TestSetJson:
    @pytest.mark.asyncio
    async def test_set_json(self, cache: RedisCache, redis_mock: AsyncMock):
        await cache.set_json("answer", "k", {"a": 1}, ttl=60)
        redis_mock.set.assert_awaited_once()
        stored_key, stored_val, *_ = redis_mock.set.call_args[0]
        assert b'"a"' in stored_val


class TestMsgpack:
    @pytest.mark.asyncio
    async def test_msgpack_roundtrip(
        self, cache: RedisCache, redis_mock: AsyncMock
    ):
        stored_val = None

        async def fake_set(key, val, **kw):
            nonlocal stored_val
            stored_val = val

        async def fake_get(key, **kw):
            return stored_val

        redis_mock.set.side_effect = fake_set
        redis_mock.get.side_effect = fake_get

        data = [{"doc_id": "d1", "score": 0.95}]
        await cache.set_msgpack("rerank", "k", data, ttl=60)
        result = await cache.get_msgpack("rerank", "k")
        assert result == data

    @pytest.mark.asyncio
    async def test_msgpack_miss(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.get.return_value = None
        result = await cache.get_msgpack("rerank", "k")
        assert result is None

    @pytest.mark.asyncio
    async def test_msgpack_corrupt(
        self, cache: RedisCache, redis_mock: AsyncMock
    ):
        redis_mock.get.return_value = b"\xff\xff"
        result = await cache.get_msgpack("rerank", "k")
        assert result is None


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_ok(self, cache: RedisCache, redis_mock: AsyncMock):
        await cache.delete("answer", "k")
        redis_mock.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_redis_down(
        self, cache: RedisCache, redis_mock: AsyncMock
    ):
        redis_mock.delete.side_effect = ConnectionError("refused")
        await cache.delete("answer", "k")


class TestIncr:
    @pytest.mark.asyncio
    async def test_incr_ok(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.incr.return_value = 3
        result = await cache.incr("doc:epoch:d1")
        assert result == 3

    @pytest.mark.asyncio
    async def test_incr_error(self, cache: RedisCache, redis_mock: AsyncMock):
        redis_mock.incr.side_effect = ConnectionError("refused")
        result = await cache.incr("doc:epoch:d1")
        assert result == 0


class TestHsetHgetall:
    @pytest.mark.asyncio
    async def test_hset_hgetall(
        self, cache: RedisCache, redis_mock: AsyncMock
    ):
        await cache.hset("doc:active:d1", mapping={"version_id": "v2"})
        redis_mock.hset.assert_awaited_once()

        redis_mock.hgetall.return_value = {b"version_id": b"v2"}
        result = await cache.hgetall("doc:active:d1")
        assert result[b"version_id"] == b"v2"
