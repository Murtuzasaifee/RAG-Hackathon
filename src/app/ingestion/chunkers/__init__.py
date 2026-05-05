from __future__ import annotations

from app.ingestion.chunkers.protocols import Chunker
from app.ingestion.chunkers.chunker import DocumentAwareChunker

_REGISTRY: dict[str, type[Chunker]] = {
    "document_aware": DocumentAwareChunker
}


def get_chunker(name: str) -> Chunker:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown chunker: {name}")
    return cls()
