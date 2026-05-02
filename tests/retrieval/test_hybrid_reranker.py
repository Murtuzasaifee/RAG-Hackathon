from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_hackathon.core.types import RetrievalHit
from rag_hackathon.gateway.protocols import RerankHit
from rag_hackathon.retrieval.hybrid_qdrant import HybridQdrantRetriever
from rag_hackathon.retrieval.reranker_cohere import CohereReranker


def _make_hit(
    doc_id: str = "d1",
    version_id: str = "v1",
    score: float = 0.9,
    chunk_index: int = 0,
    chunk_text: str = "sample text",
) -> RetrievalHit:
    return RetrievalHit(
        doc_id=doc_id,
        version_id=version_id,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        page=1,
        section_path=["H1"],
        bbox=[0.0, 0.0, 1.0, 1.0],
        chunk_type="text",
        score=score,
    )


class TestHybridQdrantRetriever:
    @pytest.fixture
    def dense_embedder(self) -> AsyncMock:
        emb = AsyncMock()
        emb.embed.return_value = [[0.1] * 1536]
        return emb

    @pytest.fixture
    def sparse_embedder(self) -> AsyncMock:
        from rag_hackathon.ingestion.embedders.protocols import SparseVector

        emb = AsyncMock()
        emb.embed.return_value = [
            SparseVector(indices=[1, 5, 10], values=[0.3, 0.7, 0.1])
        ]
        return emb

    @pytest.fixture
    def qdrant_client(self) -> AsyncMock:

        client = AsyncMock()

        point = AsyncMock()
        point.payload = {
            "doc_id": "d1",
            "version_id": "v1",
            "chunk_index": 0,
            "chunk_text": "hello world",
            "page": 1,
            "section_path": ["Intro"],
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "chunk_type": "text",
        }
        point.score = 0.95

        resp = AsyncMock()
        resp.points = [point]
        client.query_points.return_value = resp
        return client

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_returns_hits(
        self, qdrant_client, dense_embedder, sparse_embedder
    ):
        retriever = HybridQdrantRetriever(
            client=qdrant_client,
            collection="documents",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        hits = await retriever.retrieve("hello", top_k=10)
        assert len(hits) == 1
        assert hits[0].doc_id == "d1"
        assert hits[0].chunk_text == "hello world"
        assert hits[0].score == 0.95
        qdrant_client.query_points.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_dense_only(self, qdrant_client, dense_embedder):
        retriever = HybridQdrantRetriever(
            client=qdrant_client,
            collection="documents",
            dense_embedder=dense_embedder,
            sparse_embedder=None,
        )
        hits = await retriever.retrieve("hello", top_k=10)
        assert len(hits) == 1

        call_kwargs = qdrant_client.query_points.call_args
        prefetch = call_kwargs.kwargs.get("prefetch") or call_kwargs[1].get("prefetch")
        assert len(prefetch) == 1
        assert prefetch[0].using == "dense"

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_filters_by_doc_ids(
        self, qdrant_client, dense_embedder, sparse_embedder
    ):
        retriever = HybridQdrantRetriever(
            client=qdrant_client,
            collection="documents",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        await retriever.retrieve("hello", doc_ids=["d1", "d2"], top_k=10)
        call_kwargs = qdrant_client.query_points.call_args
        prefetch = call_kwargs.kwargs.get("prefetch") or call_kwargs[1].get("prefetch")
        f = prefetch[0].filter
        assert f is not None
        must_fields = [c.key for c in f.must]
        assert "doc_id" in must_fields

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_version_ids_skips_active_filter(
        self, qdrant_client, dense_embedder, sparse_embedder
    ):
        retriever = HybridQdrantRetriever(
            client=qdrant_client,
            collection="documents",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        await retriever.retrieve("hello", version_ids=["v2"], top_k=10)
        call_kwargs = qdrant_client.query_points.call_args
        prefetch = call_kwargs.kwargs.get("prefetch") or call_kwargs[1].get("prefetch")
        f = prefetch[0].filter
        must_fields = [c.key for c in f.must]
        assert "active" not in must_fields
        assert "version_id" in must_fields

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_qdrant_error_raises(
        self, dense_embedder, sparse_embedder
    ):
        qdrant_client = AsyncMock()
        qdrant_client.query_points.side_effect = Exception("connection refused")
        from rag_hackathon.core.errors import RetrievalError

        retriever = HybridQdrantRetriever(
            client=qdrant_client,
            collection="documents",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        with pytest.raises(RetrievalError, match="Qdrant query failed"):
            await retriever.retrieve("hello")


class TestCohereReranker:
    @pytest.fixture
    def gateway(self) -> AsyncMock:
        gw = AsyncMock()
        return gw

    @pytest.mark.asyncio
    async def test_rerank_returns_reordered(self, gateway):
        gateway.rerank.return_value = [
            RerankHit(index=1, document="second", relevance_score=0.95),
            RerankHit(index=0, document="first", relevance_score=0.80),
        ]
        hits = [
            _make_hit(chunk_index=0, score=0.5),
            _make_hit(chunk_index=1, score=0.6),
        ]
        reranker = CohereReranker(client=gateway, model="rerank-english-v3.0")
        result = await reranker.rerank("query", hits, top_n=2)
        assert len(result) == 2
        assert result[0].score == 0.95
        assert result[0].chunk_index == 1
        assert result[1].score == 0.80

    @pytest.mark.asyncio
    async def test_rerank_empty_hits(self, gateway):
        reranker = CohereReranker(client=gateway, model="rerank-english-v3.0")
        result = await reranker.rerank("query", [], top_n=5)
        assert result == []
        gateway.rerank.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rerank_top_n_greater_than_hits(self, gateway):
        gateway.rerank.return_value = [
            RerankHit(index=0, document="only one", relevance_score=0.9),
        ]
        hits = [_make_hit(chunk_index=0, score=0.5)]
        reranker = CohereReranker(client=gateway, model="rerank-english-v3.0")
        result = await reranker.rerank("query", hits, top_n=10)
        assert len(result) == 1
        gateway.rerank.assert_awaited_once()
        call_kwargs = gateway.rerank.call_args
        assert call_kwargs.kwargs.get("top_n") == 1
