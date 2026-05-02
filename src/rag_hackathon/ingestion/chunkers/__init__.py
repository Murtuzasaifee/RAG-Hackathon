from __future__ import annotations

from rag_hackathon.ingestion.chunkers.protocols import Chunker
from rag_hackathon.ingestion.chunkers.structure_aware import StructureAwareChunker

_REGISTRY: dict[str, type[Chunker]] = {
    "structure_aware": StructureAwareChunker,
}


def get_chunker(name: str) -> Chunker:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown chunker: {name}")
    return cls()
