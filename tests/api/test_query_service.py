from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_hackathon.api.schemas import QueryRequest
from rag_hackathon.api.services.query_service import QueryService
from rag_hackathon.core.types import Answer, Citation, RetrievalHit


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
