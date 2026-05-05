from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.errors import GatewayError
from app.gateway.bifrost import BifrostClient


@pytest.fixture
def client():
    c = BifrostClient(base_url="http://fake:8080")
    yield c


async def test_embed_returns_vectors(client: BifrostClient):
    mock_response = AsyncMock()
    mock_response.data = [
        AsyncMock(embedding=[0.1] * 1536),
        AsyncMock(embedding=[0.2] * 1536),
    ]
    with patch.object(
        client._openai.embeddings, "create", new=AsyncMock(return_value=mock_response)
    ):
        result = await client.embed(["hello", "world"], "text-embedding-3-small")
        assert len(result) == 2
        assert len(result[0]) == 1536


async def test_embed_empty_input(client: BifrostClient):
    result = await client.embed([], "text-embedding-3-small")
    assert result == []


async def test_chat_returns_content(client: BifrostClient):
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="Paris"))]
    with patch.object(
        client._openai.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_response),
    ):
        result = await client.chat(
            [{"role": "user", "content": "capital of france?"}], "gpt-5.4"
        )
        assert result == "Paris"


async def test_chat_failure_raises_gateway_error(client: BifrostClient):
    with (
        patch.object(
            client._openai.chat.completions,
            "create",
            new=AsyncMock(side_effect=Exception("timeout")),
        ),
        pytest.raises(GatewayError, match="Chat call failed"),
    ):
        await client.chat([{"role": "user", "content": "hi"}], "gpt-5.4")


async def test_rerank_returns_sorted_hits(client: BifrostClient):
    mock_resp = httpx.Response(
        200,
        json={
            "results": [
                {"index": 1, "document": {"text": "doc1"}, "relevance_score": 0.9},
                {"index": 0, "document": {"text": "doc0"}, "relevance_score": 0.5},
            ]
        },
        request=httpx.Request("POST", "http://fake/v1/rerank"),
    )
    with patch.object(client._http, "post", return_value=mock_resp):
        hits = await client.rerank("query", ["a", "b"], "rerank-english-v3.0", 5)
        assert len(hits) == 2
        assert hits[0].relevance_score == 0.9
        assert hits[0].index == 1


async def test_rerank_empty_documents(client: BifrostClient):
    result = await client.rerank("q", [], "model", 5)
    assert result == []


async def test_rerank_http_error_raises_gateway_error(client: BifrostClient):
    mock_resp = httpx.Response(
        503,
        request=httpx.Request("POST", "http://fake/v1/rerank"),
    )
    with (
        patch.object(client._http, "post", return_value=mock_resp),
        pytest.raises(GatewayError, match="Rerank call failed"),
    ):
        await client.rerank("q", ["doc"], "model", 5)
