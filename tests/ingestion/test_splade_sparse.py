from __future__ import annotations

from unittest.mock import patch

from app.ingestion.embedders.protocols import SparseVector
from app.ingestion.embedders.splade_sparse import SpladeSparseEmbedder


async def test_embed_returns_sparse_vectors():
    embedder = SpladeSparseEmbedder(model_name="fake-model")
    expected = SparseVector(indices=[10, 20, 30], values=[0.5, 0.3, 0.1])

    with (
        patch.object(embedder, "_load"),
        patch.object(embedder, "_compute_sparse", return_value=expected),
    ):
        result = await embedder.embed(["test text"])

    assert len(result) == 1
    assert result[0].indices == [10, 20, 30]
    assert result[0].values == [0.5, 0.3, 0.1]


async def test_embed_empty_input():
    embedder = SpladeSparseEmbedder(model_name="fake-model")
    result = await embedder.embed([])
    assert result == []


async def test_embed_multiple_texts():
    embedder = SpladeSparseEmbedder(model_name="fake-model")
    sv1 = SparseVector(indices=[1], values=[0.9])
    sv2 = SparseVector(indices=[2, 5], values=[0.4, 0.6])

    with (
        patch.object(embedder, "_load"),
        patch.object(embedder, "_compute_sparse", side_effect=[sv1, sv2]),
    ):
        result = await embedder.embed(["text1", "text2"])

    assert len(result) == 2
    assert result[0].indices == [1]
    assert result[1].indices == [2, 5]
