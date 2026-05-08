from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.services.query_service import QueryService


@pytest.fixture
def mock_query_service() -> AsyncMock:
    from app.api.schemas import CitationResponse, QueryResponse

    svc = AsyncMock(spec=QueryService)
    svc.run.return_value = QueryResponse(
        answer="X is a framework.",
        citations=[
            CitationResponse(
                doc_id="d1",
                version_id="v1",
                page=1,
                section_path=["Intro"],
                bbox=[0.0, 0.0, 1.0, 1.0],
                chunk_text="X is a framework for apps.",
                score=0.95,
            )
        ],
        request_id="test-req",
        timings_ms={"retrieve_ms": 50, "rerank_ms": 30, "generate_ms": 200},
    )
    return svc


@pytest.fixture
def client(mock_query_service: AsyncMock) -> TestClient:
    app = create_app()
    app.state.query_service = mock_query_service
    return TestClient(app)


def test_query_returns_200(client: TestClient):
    resp = client.post(
        "/api/v1/query",
        json={"query": "What is X?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "X is a framework."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["doc_id"] == "d1"
    assert data["citations"][0]["score"] == 0.95
    assert "retrieve_ms" in data["timings_ms"]


def test_query_empty_string_returns_422(client: TestClient):
    resp = client.post(
        "/api/v1/query",
        json={"query": ""},
    )
    assert resp.status_code == 422


def test_query_with_optional_filters(client: TestClient, mock_query_service: AsyncMock):
    resp = client.post(
        "/api/v1/query",
        json={
            "query": "What is X?",
            "doc_ids": ["d1"],
            "version_ids": ["v1"],
            "top_k": 10,
            "top_n": 3,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_query_service.run.call_args
    req_arg = call_kwargs.args[0]
    assert req_arg.doc_ids == ["d1"]
    assert req_arg.version_ids == ["v1"]
    assert req_arg.top_k == 10
    assert req_arg.top_n == 3


def test_query_has_request_id_header(client: TestClient):
    resp = client.post(
        "/api/v1/query",
        json={"query": "What is X?"},
        headers={"X-Request-Id": "custom-123"},
    )
    assert resp.status_code == 200
    assert "X-Request-Id" in resp.headers
