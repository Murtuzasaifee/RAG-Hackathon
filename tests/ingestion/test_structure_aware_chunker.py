from __future__ import annotations

import pytest

from rag_hackathon.core.types import ParsedDocument, ParsedTable
from rag_hackathon.ingestion.chunkers import get_chunker
from rag_hackathon.ingestion.chunkers.document_aware import StructureAwareChunker

_P1 = {
    "content": "Short paragraph one.",
    "page": 1,
    "bbox": [],
    "section_path": ["H1"],
    "role": None,
}
_P2 = {
    "content": "Short paragraph two.",
    "page": 1,
    "bbox": [],
    "section_path": ["H1"],
    "role": None,
}
_P3 = {
    "content": "Short paragraph three.",
    "page": 2,
    "bbox": [],
    "section_path": ["H2"],
    "role": None,
}


def _make_doc(paragraphs=None, tables=None):
    return ParsedDocument(
        doc_id="test-doc",
        paragraphs=paragraphs or [],
        tables=tables or [],
    )


def test_short_sections_produce_one_chunk_each():
    doc = _make_doc(paragraphs=[_P1, _P2, _P3])
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)

    assert len(chunks) == 3
    assert all(c.chunk_type == "text" for c in chunks)
    assert chunks[0].section_path == ["H1"]
    assert chunks[2].section_path == ["H2"]
    assert chunks[0].page == 1
    assert chunks[2].page == 2


def test_oversize_section_splits_with_overlap():
    long_text = " ".join(f"Word{i}" for i in range(200))
    doc = _make_doc(
        paragraphs=[
            {
                "content": long_text,
                "page": 1,
                "bbox": [],
                "section_path": ["Big"],
                "role": None,
            },
        ]
    )
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc, max_tokens=64, overlap=10)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.section_path == ["Big"]
        assert c.chunk_type == "text"


def test_table_becomes_standalone_chunk():
    doc = _make_doc(
        tables=[
            ParsedTable(
                markdown="| a | b |\n|---|---|\n| 1 | 2 |",
                page=1,
                bbox=[0.0, 0.0, 1.0, 1.0],
                section_path=["Data"],
            )
        ]
    )
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
    assert "| a | b |" in chunks[0].text
    assert chunks[0].section_path == ["Data"]


def test_empty_paragraph_produces_no_chunk():
    doc = _make_doc(
        paragraphs=[
            {
                "content": "   ",
                "page": 1,
                "bbox": [],
                "section_path": [],
                "role": None,
            },
        ]
    )
    chunker = StructureAwareChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 0


def test_get_chunker_returns_document_aware():
    chunker = get_chunker("document_aware")
    assert isinstance(chunker, StructureAwareChunker)


def test_get_chunker_unknown_raises():
    with pytest.raises(ValueError, match="Unknown chunker"):
        get_chunker("nonexistent")
