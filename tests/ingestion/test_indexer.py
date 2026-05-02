from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_hackathon.core.types import Chunk
from rag_hackathon.ingestion.embedders.protocols import SparseVector
from rag_hackathon.ingestion.indexer import QdrantIndexer


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    return client


@pytest.fixture
def indexer(mock_client):
    return QdrantIndexer(client=mock_client, collection="documents")


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            doc_id="doc1",
            version_id="v1",
            chunk_index=0,
            text="chunk zero",
            page=1,
            section_path=["H1"],
            bbox=[0.1, 0.2, 0.8, 0.9],
            chunk_type="text",
        ),
        Chunk(
            doc_id="doc1",
            version_id="v1",
            chunk_index=1,
            text="chunk one",
            page=2,
            section_path=["H1", "H2"],
            bbox=[],
            chunk_type="text",
        ),
    ]


async def test_ensure_collection_creates_when_missing(
    indexer: QdrantIndexer, mock_client
):
    await indexer.ensure_collection()
    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args
    assert call_kwargs.kwargs["collection_name"] == "documents"


async def test_ensure_collection_skips_when_exists(
    indexer: QdrantIndexer, mock_client
):
    mock_client.collection_exists = AsyncMock(return_value=True)
    await indexer.ensure_collection()
    mock_client.create_collection.assert_not_called()


async def test_upsert_sends_points(
    indexer: QdrantIndexer, mock_client, sample_chunks
):
    dense = [[0.1] * 1536, [0.2] * 1536]
    sparse = [SparseVector([1, 2], [0.5, 0.3]), SparseVector([3], [0.7])]

    await indexer.upsert(
        sample_chunks, dense, sparse, "doc1", "v1"
    )

    mock_client.upsert.assert_called_once()
    points = mock_client.upsert.call_args.kwargs["points"]
    assert len(points) == 2
    assert points[0].payload["doc_id"] == "doc1"
    assert points[0].payload["version_id"] == "v1"
    assert points[0].payload["active"] is True
    assert points[0].payload["chunk_text"] == "chunk zero"
    assert "dense" in points[0].vector
    assert "sparse" in points[0].vector


async def test_upsert_without_sparse(
    indexer: QdrantIndexer, mock_client, sample_chunks
):
    dense = [[0.1] * 1536, [0.2] * 1536]

    await indexer.upsert(sample_chunks, dense, None, "doc1", "v1")

    points = mock_client.upsert.call_args.kwargs["points"]
    assert "sparse" not in points[0].vector


async def test_upsert_empty_chunks(indexer: QdrantIndexer, mock_client):
    await indexer.upsert([], [], None, "doc1", "v1")
    mock_client.upsert.assert_not_called()


async def test_upsert_failure_raises_ingestion_error(
    indexer: QdrantIndexer, mock_client, sample_chunks
):
    mock_client.upsert = AsyncMock(side_effect=Exception("connection lost"))
    dense = [[0.1] * 1536, [0.2] * 1536]

    with pytest.raises(Exception, match="Qdrant upsert failed"):
        await indexer.upsert(sample_chunks, dense, None, "doc1", "v1")
