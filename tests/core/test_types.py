from datetime import UTC, datetime

from app.core.types import (
    Chunk,
    Citation,
    JobStatus,
    ParsedDocument,
    ParsedTable,
    RetrievalHit,
    make_point_id,
)


def test_citation_roundtrip():
    c = Citation(
        doc_id="doc1",
        version_id="v1",
        page=3,
        section_path=["H1", "H2"],
        bbox=[0.1, 0.2, 0.8, 0.9],
        chunk_text="some text",
        score=0.92,
    )
    dumped = c.model_dump_json()
    loaded = Citation.model_validate_json(dumped)
    assert loaded == c
    assert loaded.score == 0.92
    assert loaded.section_path == ["H1", "H2"]


def test_chunk_frozen():
    c = Chunk(
        doc_id="d",
        version_id="v",
        chunk_index=0,
        text="hello",
        page=1,
    )
    assert c.chunk_type == "text"


def test_parsed_table():
    t = ParsedTable(
        markdown="| a | b |\n|---|---|",
        page=2,
        bbox=[0.0, 0.0, 1.0, 1.0],
        section_path=["Intro"],
    )
    assert t.page == 2


def test_parsed_document_empty():
    d = ParsedDocument(doc_id="empty")
    assert d.sections == []
    assert d.paragraphs == []
    assert d.tables == []


def test_job_status():
    now = datetime.now(UTC)
    js = JobStatus(
        job_id="j1",
        doc_id="d1",
        version_id="v1",
        state="running",
        stage="chunking",
        progress=40,
        created_at=now,
        updated_at=now,
    )
    assert js.state == "running"
    assert js.error is None


def test_retrieval_hit():
    rh = RetrievalHit(
        doc_id="d",
        version_id="v",
        chunk_index=0,
        chunk_text="text",
        page=1,
        score=0.85,
    )
    assert rh.chunk_type == "text"
    assert rh.score == 0.85


def test_make_point_id_deterministic():
    id1 = make_point_id("doc1", "v1", 0)
    id2 = make_point_id("doc1", "v1", 0)
    assert id1 == id2


def test_make_point_id_unique_per_inputs():
    id1 = make_point_id("doc1", "v1", 0)
    id2 = make_point_id("doc1", "v1", 1)
    id3 = make_point_id("doc1", "v2", 0)
    assert id1 != id2
    assert id1 != id3
