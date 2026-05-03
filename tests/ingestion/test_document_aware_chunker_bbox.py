from __future__ import annotations

from rag_hackathon.core.types import ParsedDocument, ParsedElement
from rag_hackathon.ingestion.chunkers.chunker import DocumentAwareChunker


def test_title_plus_single_body_chunk_preserves_body_bbox():
    parsed = ParsedDocument(
        doc_id="doc-1",
        elements=[
            ParsedElement(
                label="paragraph_title",
                text="Layout Analysis Model",
                page=1,
                reading_order=0,
            ),
            ParsedElement(
                label="text",
                text="The layout analysis model predicts bounding boxes.",
                page=1,
                reading_order=1,
                bbox=[0.1, 0.2, 0.8, 0.3],
            ),
        ],
    )

    chunks = DocumentAwareChunker().chunk(parsed)

    assert len(chunks) == 1
    assert chunks[0].bbox == [0.1, 0.2, 0.8, 0.3]


def test_multi_body_chunk_uses_primary_page_bbox_union():
    parsed = ParsedDocument(
        doc_id="doc-1",
        elements=[
            ParsedElement(
                label="text",
                text="First paragraph.",
                page=1,
                reading_order=0,
                bbox=[0.1, 0.2, 0.8, 0.3],
            ),
            ParsedElement(
                label="text",
                text="Second paragraph.",
                page=1,
                reading_order=1,
                bbox=[0.1, 0.4, 0.8, 0.5],
            ),
        ],
    )

    chunks = DocumentAwareChunker().chunk(parsed)

    assert len(chunks) == 1
    assert chunks[0].bbox == [0.1, 0.2, 0.8, 0.5]
