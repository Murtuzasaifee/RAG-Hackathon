from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.schemas import CitationResponse, QueryResponse
from app.cache.semantic_cache import SemanticQueryCache


def _make_hit(score: float = 0.92, epoch_map: dict | None = None, top_k: int = 20, top_n: int = 5) -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = {
        "answer": "The answer is 42.",
        "citations": [
            {
                "doc_id": "d1",
                "version_id": "v1",
                "page": 1,
                "section_path": ["H1"],
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "page_bboxes": [],
                "chunk_text": "relevant text",
                "chunk_type": "text",
                "score": 0.9,
            }
        ],
        "epoch_map": epoch_map if epoch_map is not None else {"d1": 0},
        "top_k": top_k,
        "top_n": top_n,
    }
    return point


@pytest.fixture
def qdrant_mock() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def embedder_mock() -> AsyncMock:
    m = AsyncMock()
    m.embed.return_value = [[0.1] * 1536]
    return m


@pytest.fixture
def sem_cache(qdrant_mock: AsyncMock, embedder_mock: AsyncMock) -> SemanticQueryCache:
    return SemanticQueryCache(
        client=qdrant_mock,
        embedder=embedder_mock,
        collection="query_cache",
        threshold=0.85,
    )


class TestEnsureCollection:
    @pytest.mark.asyncio
    async def test_creates_when_missing(self, sem_cache, qdrant_mock):
        qdrant_mock.collection_exists.return_value = False
        await sem_cache.ensure_collection()
        qdrant_mock.create_collection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_when_exists(self, sem_cache, qdrant_mock):
        qdrant_mock.collection_exists.return_value = True
        await sem_cache.ensure_collection()
        qdrant_mock.create_collection.assert_not_awaited()


class TestLookup:
    @pytest.mark.asyncio
    async def test_hit_matching_epoch(self, sem_cache, qdrant_mock, embedder_mock):
        result_mock = MagicMock()
        result_mock.points = [_make_hit(score=0.92, epoch_map={"d1": 0})]
        qdrant_mock.query_points.return_value = result_mock

        response, vec = await sem_cache.lookup(
            query="What is X?",
            doc_ids=["d1"],
            owner_id="user1",
            epoch_map={"d1": 0},
            top_k=20,
            top_n=5,
        )

        assert response is not None
        assert response.cache_hit is True
        assert response.answer == "The answer is 42."
        assert len(response.citations) == 1
        assert vec == [0.1] * 1536

    @pytest.mark.asyncio
    async def test_miss_epoch_mismatch(self, sem_cache, qdrant_mock):
        result_mock = MagicMock()
        result_mock.points = [_make_hit(score=0.92, epoch_map={"d1": 0})]
        qdrant_mock.query_points.return_value = result_mock

        response, vec = await sem_cache.lookup(
            query="What is X?",
            doc_ids=["d1"],
            owner_id="user1",
            epoch_map={"d1": 1},  # epoch bumped
            top_k=20,
            top_n=5,
        )

        assert response is None
        assert vec == [0.1] * 1536

    @pytest.mark.asyncio
    async def test_miss_no_qdrant_results(self, sem_cache, qdrant_mock):
        result_mock = MagicMock()
        result_mock.points = []
        qdrant_mock.query_points.return_value = result_mock

        response, vec = await sem_cache.lookup(
            query="Unknown query",
            doc_ids=None,
            owner_id=None,
            epoch_map={},
            top_k=20,
            top_n=5,
        )

        assert response is None

    @pytest.mark.asyncio
    async def test_admin_lookup_no_owner_filter(self, sem_cache, qdrant_mock):
        result_mock = MagicMock()
        result_mock.points = []
        qdrant_mock.query_points.return_value = result_mock

        await sem_cache.lookup(
            query="What is X?",
            doc_ids=None,
            owner_id=None,
            epoch_map={},
            top_k=20,
            top_n=5,
        )

        call_args = qdrant_mock.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        # owner_id None uses IsNullCondition, not MatchValue
        assert query_filter is not None
        condition_types = [type(c).__name__ for c in query_filter.must]
        assert "IsNullCondition" in condition_types

    @pytest.mark.asyncio
    async def test_qdrant_error_fail_open(self, sem_cache, qdrant_mock):
        qdrant_mock.query_points.side_effect = Exception("Qdrant down")

        response, vec = await sem_cache.lookup(
            query="What is X?",
            doc_ids=None,
            owner_id=None,
            epoch_map={},
            top_k=20,
            top_n=5,
        )

        assert response is None
        assert vec == []


class TestStore:
    @pytest.mark.asyncio
    async def test_store_calls_upsert_with_payload(self, sem_cache, qdrant_mock):
        citations = [
            CitationResponse(
                doc_id="d1",
                version_id="v1",
                page=1,
                section_path=["H1"],
                bbox=[0.0, 0.0, 1.0, 1.0],
                chunk_text="text",
                score=0.9,
            )
        ]

        await sem_cache.store(
            query="What is X?",
            doc_ids=["d1"],
            owner_id="user1",
            epoch_map={"d1": 0},
            top_k=20,
            top_n=5,
            query_vec=[0.1] * 1536,
            answer_text="The answer is 42.",
            citations=citations,
        )

        qdrant_mock.upsert.assert_awaited_once()
        call_args = qdrant_mock.upsert.call_args
        points = call_args.kwargs.get("points") or call_args.args[1] if call_args.args else call_args.kwargs["points"]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["answer"] == "The answer is 42."
        assert payload["epoch_map"] == {"d1": 0}
        assert payload["top_k"] == 20
        assert payload["top_n"] == 5
        assert "created_at" in payload

    @pytest.mark.asyncio
    async def test_store_error_fail_open(self, sem_cache, qdrant_mock):
        qdrant_mock.upsert.side_effect = Exception("Qdrant down")

        await sem_cache.store(
            query="What is X?",
            doc_ids=None,
            owner_id=None,
            epoch_map={},
            top_k=20,
            top_n=5,
            query_vec=[0.1] * 1536,
            answer_text="answer",
            citations=[],
        )
        # Should not raise


class TestInvalidateDoc:
    @pytest.mark.asyncio
    async def test_invalidate_calls_delete_with_filter(self, sem_cache, qdrant_mock):
        await sem_cache.invalidate_doc("d1")

        qdrant_mock.delete.assert_awaited_once()
        call_args = qdrant_mock.delete.call_args
        assert call_args.kwargs.get("collection_name") == "query_cache"
        selector = call_args.kwargs.get("points_selector")
        assert selector is not None

    @pytest.mark.asyncio
    async def test_invalidate_error_fail_open(self, sem_cache, qdrant_mock):
        qdrant_mock.delete.side_effect = Exception("Qdrant down")
        await sem_cache.invalidate_doc("d1")
        # Should not raise
