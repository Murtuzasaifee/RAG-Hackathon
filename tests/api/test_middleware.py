from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from rag_hackathon.api.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_request_id_header_set(client: AsyncClient):
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


async def test_request_id_echoed_when_provided(client: AsyncClient):
    rid = "my-custom-id-123"
    resp = await client.get("/health", headers={"X-Request-Id": rid})
    assert resp.headers["x-request-id"] == rid


async def test_error_handler_maps_ingestion_error(client: AsyncClient):
    from fastapi import FastAPI

    from rag_hackathon.api.error_handlers import RAGError, rag_error_handler
    from rag_hackathon.core.errors import IngestionError

    test_app = FastAPI()
    test_app.add_exception_handler(RAGError, rag_error_handler)

    @test_app.get("/fail")
    async def fail():
        raise IngestionError("bad pdf")

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/fail")
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"] == "ingestion_error"
        assert body["message"] == "bad pdf"
