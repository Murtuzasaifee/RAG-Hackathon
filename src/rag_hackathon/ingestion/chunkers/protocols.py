from __future__ import annotations

from typing import Protocol

from rag_hackathon.core.types import Chunk, ParsedDocument


class Chunker(Protocol):
    def chunk(
        self,
        parsed: ParsedDocument,
        *,
        max_tokens: int = 512,
        overlap: int = 50,
    ) -> list[Chunk]: ...
