from __future__ import annotations

import tiktoken

from rag_hackathon.core.types import Chunk, ParsedDocument
from rag_hackathon.observability.tracing import stage_span


class StructureAwareChunker:
    def __init__(self) -> None:
        self._encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    def chunk(
        self,
        parsed: ParsedDocument,
        *,
        max_tokens: int = 512,
        overlap: int = 50,
    ) -> list[Chunk]:
        with stage_span("chunk", doc_id=parsed.doc_id):
            chunks: list[Chunk] = []
            idx = 0

            for para in parsed.paragraphs:
                text = para["content"].strip()
                if not text:
                    continue

                section_path = para.get("section_path", [])
                page = para.get("page", 1)
                bbox = para.get("bbox", [])

                token_count = len(self._encoding.encode(text))

                if token_count <= max_tokens:
                    chunks.append(
                        Chunk(
                            doc_id=parsed.doc_id,
                            version_id="",
                            chunk_index=idx,
                            text=text,
                            page=page,
                            section_path=section_path,
                            bbox=bbox,
                            chunk_type="text",
                        )
                    )
                    idx += 1
                else:
                    windows = self._split_to_windows(
                        text, max_tokens, overlap
                    )
                    for window in windows:
                        chunks.append(
                            Chunk(
                                doc_id=parsed.doc_id,
                                version_id="",
                                chunk_index=idx,
                                text=window,
                                page=page,
                                section_path=section_path,
                                bbox=bbox,
                                chunk_type="text",
                            )
                        )
                        idx += 1

            for table in parsed.tables:
                if not table.markdown.strip():
                    continue
                chunks.append(
                    Chunk(
                        doc_id=parsed.doc_id,
                        version_id="",
                        chunk_index=idx,
                        text=table.markdown,
                        page=table.page,
                        section_path=table.section_path,
                        bbox=table.bbox,
                        chunk_type="table",
                    )
                )
                idx += 1

            return chunks

    def _split_to_windows(
        self, text: str, max_tokens: int, overlap: int
    ) -> list[str]:
        tokens = self._encoding.encode(text)
        windows: list[str] = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            window_tokens = tokens[start:end]
            windows.append(self._encoding.decode(window_tokens))
            start += max_tokens - overlap
        return windows
