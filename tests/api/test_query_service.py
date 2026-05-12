from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.schemas import CitationResponse, QueryRequest, QueryResponse
from app.api.services.query_service import QueryService
from app.core.types import Answer, Citation, RetrievalHit


def _make_hit(text: str = "some text", score: float = 0.9) -> RetrievalHit:
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


@pytest.fixture
def retriever() -> AsyncMock:
    r = AsyncMock()
    r.retrieve.return_value = [_make_hit("relevant chunk")]
    return r


@pytest.fixture
def reranker() -> AsyncMock:
    r = AsyncMock()
    reranked = [_make_hit("relevant chunk", score=0.95)]
    r.rerank.return_value = reranked
    return r


@pytest.fixture
def generator() -> AsyncMock:
    g = AsyncMock()
    g.generate.return_value = Answer(
        text="The answer is X.",
        citations=[
            Citation(
                doc_id="d1",
                version_id="v1",
                page=1,
                section_path=["H1"],
                bbox=[0.0, 0.0, 1.0, 1.0],
                chunk_text="relevant chunk",
                score=0.95,
            )
        ],
        model="gpt-5.4",
        usage={},
    )
    return g


@pytest.mark.asyncio
async def test_query_service_full_pipeline(retriever, reranker, generator):
    svc = QueryService(retriever=retriever, reranker=reranker, generator=generator)
    req = QueryRequest(query="What is X?")
    resp = await svc.run(req, request_id="test-123")

    assert resp.answer == "The answer is X."
    assert len(resp.citations) == 1
    assert resp.citations[0].doc_id == "d1"
    assert resp.citations[0].score == 0.95
    assert resp.request_id == "test-123"
    assert "retrieve_ms" in resp.timings_ms
    assert "rerank_ms" in resp.timings_ms
    assert "generate_ms" in resp.timings_ms

    retriever.retrieve.assert_awaited_once()
    reranker.rerank.assert_awaited_once()
    generator.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_service_no_hits(retriever, reranker):
    retriever.retrieve.return_value = []
    reranker.rerank.return_value = []
    generator = AsyncMock()
    generator.generate.return_value = Answer(
        text="I don't know based on the provided documents.",
        citations=[],
        model="gpt-5.4",
        usage={},
    )

    svc = QueryService(retriever=retriever, reranker=reranker, generator=generator)
    req = QueryRequest(query="Unknown topic")
    resp = await svc.run(req)

    assert "I don't know" in resp.answer
    assert resp.citations == []


@pytest.fixture
def semantic_cache_hit() -> AsyncMock:
    m = AsyncMock()
    cached = QueryResponse(
        answer="cached answer",
        citations=[],
        request_id="",
        timings_ms={"semantic_cache_hit": 1},
        cache_hit=True,
    )
    m.lookup.return_value = (cached, [0.1] * 1536)
    m.store = AsyncMock()
    return m


@pytest.fixture
def semantic_cache_miss() -> AsyncMock:
    m = AsyncMock()
    m.lookup.return_value = (None, [0.1] * 1536)
    m.store = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_semantic_cache_hit_skips_pipeline(retriever, reranker, generator, semantic_cache_hit):
    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        semantic_cache=semantic_cache_hit,
    )
    req = QueryRequest(query="What is X?")
    resp = await svc.run(req, request_id="test-sem")

    assert resp.cache_hit is True
    assert resp.answer == "cached answer"
    retriever.retrieve.assert_not_awaited()
    reranker.rerank.assert_not_awaited()
    generator.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_cache_miss_runs_full_pipeline(retriever, reranker, generator, semantic_cache_miss):
    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        semantic_cache=semantic_cache_miss,
    )
    req = QueryRequest(query="What is X?")
    resp = await svc.run(req, request_id="test-miss")

    assert resp.cache_hit is False
    retriever.retrieve.assert_awaited_once()
    reranker.rerank.assert_awaited_once()
    generator.generate.assert_awaited_once()
    # Store called with reused vec from lookup
    semantic_cache_miss.store.assert_awaited_once()
    store_kwargs = semantic_cache_miss.store.call_args.kwargs
    assert store_kwargs["query_vec"] == [0.1] * 1536


@pytest.mark.asyncio
async def test_no_semantic_cache_behavior_unchanged(retriever, reranker, generator):
    svc = QueryService(retriever=retriever, reranker=reranker, generator=generator)
    req = QueryRequest(query="What is X?")
    resp = await svc.run(req)

    assert resp.cache_hit is False
    retriever.retrieve.assert_awaited_once()
    reranker.rerank.assert_awaited_once()
    generator.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_cache_lookup_error_falls_through(retriever, reranker, generator):
    sem = AsyncMock()
    sem.lookup.side_effect = Exception("Qdrant down")
    sem.store = AsyncMock()

    svc = QueryService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        semantic_cache=sem,
    )
    req = QueryRequest(query="What is X?")
    # Should not raise — fail open, run full pipeline
    resp = await svc.run(req)
    retriever.retrieve.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_service_passes_filters(retriever, reranker, generator):
    svc = QueryService(retriever=retriever, reranker=reranker, generator=generator)
    req = QueryRequest(
        query="What is X?",
        doc_ids=["d1", "d2"],
        version_ids=["v1"],
        top_k=10,
        top_n=3,
    )
    await svc.run(req)

    retriever.retrieve.assert_awaited_once()
    call_kwargs = retriever.retrieve.call_args
    assert call_kwargs.kwargs.get("doc_ids") == ["d1", "d2"]
    assert call_kwargs.kwargs.get("version_ids") == ["v1"]
    assert call_kwargs.kwargs.get("top_k") == 10

    reranker.rerank.assert_awaited_once()
    rerank_kwargs = reranker.rerank.call_args
    assert rerank_kwargs.kwargs.get("top_n") == 3
