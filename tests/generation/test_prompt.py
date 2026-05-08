from __future__ import annotations

from app.generation.prompt import build_messages, hits_to_dicts


def test_build_messages_no_hits():
    msgs = build_messages("What is X?", [])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "No relevant context" in msgs[1]["content"]


def test_build_messages_with_hits():
    hits = [
        {
            "chunk_text": "X is a framework.",
            "page": 3,
            "section_path": ["Intro", "Overview"],
        }
    ]
    msgs = build_messages("What is X?", hits)
    assert len(msgs) == 2
    user_content = msgs[1]["content"]
    assert "[1]" in user_content
    assert "Page 3" in user_content
    assert "Intro > Overview" in user_content
    assert "X is a framework." in user_content


def test_build_messages_multiple_hits():
    hits = [
        {"chunk_text": "First chunk.", "page": 1, "section_path": ["H1"]},
        {"chunk_text": "Second chunk.", "page": 2, "section_path": ["H2"]},
    ]
    msgs = build_messages("query", hits)
    user_content = msgs[1]["content"]
    assert "[1]" in user_content
    assert "[2]" in user_content
    assert "First chunk." in user_content
    assert "Second chunk." in user_content


def test_hits_to_dicts():
    from app.core.types import RetrievalHit

    hits = [
        RetrievalHit(
            doc_id="d1",
            version_id="v1",
            chunk_index=0,
            chunk_text="hello",
            page=1,
            section_path=["S1"],
            bbox=[0.0, 0.0, 1.0, 1.0],
            chunk_type="text",
            score=0.9,
        )
    ]
    result = hits_to_dicts(hits)
    assert len(result) == 1
    assert result[0]["chunk_text"] == "hello"
    assert result[0]["page"] == 1
    assert result[0]["section_path"] == ["S1"]
