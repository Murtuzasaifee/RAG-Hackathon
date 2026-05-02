from __future__ import annotations

from typing import Protocol


class SparseVector:
    __slots__ = ("indices", "values")

    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class DenseEmbedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    async def embed(self, texts: list[str]) -> list[SparseVector]: ...
