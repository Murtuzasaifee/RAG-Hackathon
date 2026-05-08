from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.api.schemas import QueryRequest
from app.api.services.query_service import QueryService
from app.cache.redis_cache import RedisCache
from app.core.types import Answer, Citation, RetrievalHit


def _make_hit(text: str = "chunk text", score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        doc_id="d1",
        version_id="v1",
        chunk_index=0,
        chunk_text=text,
        page=1,
        section_path=["H1"],
        bbox=[0.0, 0.0, 1.0, 1.0],
        chunk_type="text",
        score=score,
    )


def _make_answer() -> Answer:
    return Answer(
        text="The answer is 42.",
        citations=[
            Citation(
                doc_id="d1",
                version_id="v1",
                page=1,
                section_path=["H1"],
                bbox=[0.0, 0.0, 1.0, 1.0],
                chunk_text="chunk text",
                score=0.95,
            )
        ],
        model="gpt-5.4",
        usage={},
    )


class FakeRedis:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    async def get(self, key: str, **_kw):
        return self._store.get(key)

    async def set(self, key: str, value: bytes, **_kw):
        self._store[key] = value

    async def delete(self, key: str, **_kw):
        self._store.pop(key, None)

    async def incr(self, key: str):
        cur = int(self._store.get(key, b"0"))
        cur += 1
        self._store[key] = str(cur).encode()
        return cur

    async def hgetall(self, key: str):
        return {}

    async def hset(self, key: str, **_kw):
        pass


@pytest.fixture
def retriever() -> AsyncMock:
    r = AsyncMock()
    r.retrieve.return_value = [_make_hit()]
    return r


@pytest.fixture
def reranker() -> AsyncMock:
    r = AsyncMock()
    r.rerank.return_value = [_make_hit(score=0.95)]
    return r


@pytest.fixture
def generator() -> AsyncMock:
    g = AsyncMock()
    g.generate.return_value = _make_answer()
    return g


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def cache(fake_redis: FakeRedis) -> RedisCache:
    return RedisCache(client=fake_redis)


@pytest.mark.asyncio
async def test_query_cached_answer_hit(
    retriever, reranker, generator, cache
):
    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        cache=cache,
        cache_ttl_answer=3600,
    )
    req = QueryRequest(query="What is the answer?")
    resp1 = await svc.run(req, request_id="r1")
    assert resp1.answer == "The answer is 42."
    assert generator.generate.await_count == 1

    resp2 = await svc.run(req, request_id="r2")
    assert resp2.answer == "The answer is 42."
    assert "cache_hit" in resp2.timings_ms
    assert generator.generate.await_count == 1


@pytest.mark.asyncio
async def test_query_no_cache_works(retriever, reranker, generator):
    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )
    req = QueryRequest(query="What is X?")
    resp = await svc.run(req)
    assert resp.answer == "The answer is 42."
    retriever.retrieve.assert_awaited_once()
    reranker.rerank.assert_awaited_once()
    generator.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_cache_different_query_miss(
    retriever, reranker, generator, cache
):
    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        cache=cache,
        cache_ttl_answer=3600,
    )

    await svc.run(QueryRequest(query="query A"), request_id="r1")
    await svc.run(QueryRequest(query="query B"), request_id="r2")

    assert retriever.retrieve.await_count == 2
    assert generator.generate.await_count == 2


@pytest.mark.asyncio
async def test_query_cache_epoch_change_invalidates(
    retriever, reranker, generator, cache, fake_redis
):
    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        cache=cache,
        cache_ttl_answer=3600,
    )

    req = QueryRequest(query="same query", doc_ids=["d1"])

    resp1 = await svc.run(req, request_id="r1")
    assert resp1.answer == "The answer is 42."
    assert generator.generate.await_count == 1

    from app.cache.redis_cache import cache_key

    epoch_key = cache_key("doc_epoch", "d1")
    fake_redis._store[epoch_key] = b"1"

    resp2 = await svc.run(req, request_id="r2")
    assert resp2.answer == "The answer is 42."
    assert generator.generate.await_count == 2


@pytest.mark.asyncio
async def test_query_cache_redis_down_graceful(
    retriever, reranker, generator, cache
):
    redis_mock = AsyncMock()
    redis_mock.get.side_effect = ConnectionError("refused")
    redis_mock.set.side_effect = ConnectionError("refused")
    cache._client = redis_mock

    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        cache=cache,
        cache_ttl_answer=3600,
    )

    req = QueryRequest(query="What is X?")
    resp = await svc.run(req)
    assert resp.answer == "The answer is 42."
    retriever.retrieve.assert_awaited_once()
    generator.generate.assert_awaited_once()
