from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_hackathon.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestDeleteDocument:
    def test_delete_soft_success(self, client):
        mock_qdrant = AsyncMock()
        mock_qdrant.count.return_value = MagicMock(count=3)
        mock_qdrant.delete = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_cache = AsyncMock()
        mock_cache.incr.return_value = 1

        mock_redis = AsyncMock()

        with patch(
            "rag_hackathon.api.routers.documents.AsyncQdrantClient",
            return_value=mock_qdrant,
        ), patch(
            "rag_hackathon.api.routers.documents.RedisCache",
            return_value=mock_cache,
        ), patch(
            "rag_hackathon.api.routers.documents.aioredis.from_url",
            return_value=mock_redis,
        ):
            resp = client.delete("/documents/doc1?mode=soft")

        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "doc1"
        assert data["mode"] == "soft"
        assert data["points_affected"] == 3

    def test_delete_hard_success(self, client):
        mock_qdrant = AsyncMock()
        mock_qdrant.count.return_value = MagicMock(count=3)
        mock_qdrant.delete = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_cache = AsyncMock()
        mock_cache.incr.return_value = 1

        mock_redis = AsyncMock()

        with patch(
            "rag_hackathon.api.routers.documents.AsyncQdrantClient",
            return_value=mock_qdrant,
        ), patch(
            "rag_hackathon.api.routers.documents.RedisCache",
            return_value=mock_cache,
        ), patch(
            "rag_hackathon.api.routers.documents.aioredis.from_url",
            return_value=mock_redis,
        ):
            resp = client.delete("/documents/doc1?mode=hard")

        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "hard"
        assert data["points_affected"] == 3

    def test_delete_with_version_id(self, client):
        mock_qdrant = AsyncMock()
        mock_qdrant.count.return_value = MagicMock(count=2)
        mock_qdrant.delete = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_cache = AsyncMock()
        mock_redis = AsyncMock()

        with patch(
            "rag_hackathon.api.routers.documents.AsyncQdrantClient",
            return_value=mock_qdrant,
        ), patch(
            "rag_hackathon.api.routers.documents.RedisCache",
            return_value=mock_cache,
        ), patch(
            "rag_hackathon.api.routers.documents.aioredis.from_url",
            return_value=mock_redis,
        ):
            resp = client.delete(
                "/documents/doc1?mode=soft&version_id=v1"
            )

        assert resp.status_code == 200

    def test_delete_invalid_mode(self, client):
        mock_redis = AsyncMock()

        with patch(
            "rag_hackathon.api.routers.documents.aioredis.from_url",
            return_value=mock_redis,
        ):
            resp = client.delete("/documents/doc1?mode=invalid")

        assert resp.status_code == 400

    def test_delete_not_found(self, client):
        mock_qdrant = AsyncMock()
        mock_qdrant.count.return_value = MagicMock(count=0)
        mock_qdrant.close = AsyncMock()

        mock_cache = AsyncMock()
        mock_redis = AsyncMock()

        with patch(
            "rag_hackathon.api.routers.documents.AsyncQdrantClient",
            return_value=mock_qdrant,
        ), patch(
            "rag_hackathon.api.routers.documents.RedisCache",
            return_value=mock_cache,
        ), patch(
            "rag_hackathon.api.routers.documents.aioredis.from_url",
            return_value=mock_redis,
        ):
            resp = client.delete("/documents/nonexistent?mode=soft")

        assert resp.status_code == 404


class TestUpdateDocument:
    def test_update_returns_job(self, client):
        mock_redis = AsyncMock()
        mock_store = AsyncMock()
        mock_store.set_status = AsyncMock()
        mock_store.get_status = AsyncMock(return_value=None)

        with patch(
            "rag_hackathon.api.routers.documents.aioredis.from_url",
            return_value=mock_redis,
        ), patch(
            "rag_hackathon.api.routers.documents.RedisJobStore",
            return_value=mock_store,
        ):
            resp = client.put(
                "/documents/doc1",
                files={"file": ("test.pdf", b"fake content", "application/pdf")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "doc1"
        assert data["job_id"] is not None
        assert data["version_id"] is not None
