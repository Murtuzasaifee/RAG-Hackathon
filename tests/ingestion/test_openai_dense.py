from __future__ import annotations

from unittest.mock import AsyncMock

from rag_hackathon.ingestion.embedders.openai_dense import OpenAIDenseEmbedder


async def test_embed_returns_vectors():
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(
        return_value=[[0.1] * 1536, [0.2] * 1536]
    )
    embedder = OpenAIDenseEmbedder(client=mock_client, model="text-embedding-3-small")
    result = await embedder.embed(["hello", "world"])

    assert len(result) == 2
    assert len(result[0]) == 1536
    mock_client.embed.assert_called_once_with(
        ["hello", "world"], "text-embedding-3-small"
    )


async def test_embed_empty_input():
    mock_client = AsyncMock()
    embedder = OpenAIDenseEmbedder(client=mock_client, model="text-embedding-3-small")
    result = await embedder.embed([])

    assert result == []
    mock_client.embed.assert_not_called()
