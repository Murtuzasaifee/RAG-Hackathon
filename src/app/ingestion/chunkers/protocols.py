from __future__ import annotations

from typing import Protocol

from app.core.types import Chunk, ParsedDocument


class Chunker(Protocol):
    def chunk(
        self,
        parsed: ParsedDocument,
        *,
        max_tokens: int = 512,
    ) -> list[Chunk]: ...
