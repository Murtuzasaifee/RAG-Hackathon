from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.errors import GenerationError
from app.core.types import RetrievalHit
from app.generation.generator import GroundedGenerator


def _make_hit(chunk_text: str = "some text", score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        doc_id="d1",
        version_id="v1",
        chunk_index=0,
        chunk_text=chunk_text,
        page=1,
        section_path=["H1"],
        bbox=[0.0, 0.0, 1.0, 1.0],
        chunk_type="text",
        score=score,
    )


@pytest.mark.asyncio
async def test_generate_with_hits():
    gateway = AsyncMock()
    gateway.chat.return_value = "X is a framework for building apps."
    gen = GroundedGenerator(client=gateway, model="gpt-5.4")

    hits = [_make_hit("X is a framework."), _make_hit("It supports plugins.")]
    answer = await gen.generate("What is X?", hits)

    assert answer.text == "X is a framework for building apps."
    assert len(answer.citations) == 2
    assert answer.citations[0].doc_id == "d1"
    assert answer.citations[0].chunk_text == "X is a framework."
    assert answer.citations[0].score == 0.9
    assert answer.model == "gpt-5.4"
    gateway.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_no_hits_short_circuits():
    gateway = AsyncMock()
    gen = GroundedGenerator(client=gateway, model="gpt-5.4")

    answer = await gen.generate("What is X?", [])

    assert "I don't know" in answer.text
    assert answer.citations == []
    gateway.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_llm_failure_raises():
    gateway = AsyncMock()
    gateway.chat.side_effect = Exception("timeout")
    gen = GroundedGenerator(client=gateway, model="gpt-5.4")

    with pytest.raises(GenerationError, match="LLM generation failed"):
        await gen.generate("What is X?", [_make_hit()])
