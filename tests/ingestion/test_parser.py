from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_hackathon.core.errors import IngestionError
from rag_hackathon.ingestion.parser import AzureDIParser


def _make_mock_result():
    result = MagicMock()

    para1 = MagicMock()
    para1.content = "Introduction"
    para1.role = "title"
    para1.bounding_regions = [
        MagicMock(page_number=1, polygon=[0.0, 0.0, 6.0, 0.0, 6.0, 1.0, 0.0, 1.0])
    ]

    para2 = MagicMock()
    para2.content = "This is the first paragraph."
    para2.role = None
    para2.bounding_regions = [
        MagicMock(page_number=1, polygon=[0.0, 1.0, 6.0, 1.0, 6.0, 2.0, 0.0, 2.0])
    ]

    para3 = MagicMock()
    para3.content = "Methods"
    para3.role = "sectionHeading"
    para3.bounding_regions = [
        MagicMock(page_number=2, polygon=[0.0, 0.0, 6.0, 0.0, 6.0, 1.0, 0.0, 1.0])
    ]

    para4 = MagicMock()
    para4.content = "The methodology is as follows."
    para4.role = None
    para4.bounding_regions = [
        MagicMock(page_number=2, polygon=[0.0, 1.0, 6.0, 1.0, 6.0, 2.0, 0.0, 2.0])
    ]

    result.paragraphs = [para1, para2, para3, para4]

    cell = MagicMock()
    cell.row_index = 0
    cell.column_index = 0
    cell.content = "Header"
    cell2 = MagicMock()
    cell2.row_index = 1
    cell2.column_index = 0
    cell2.content = "Value"

    table = MagicMock()
    table.row_count = 2
    table.column_count = 1
    table.cells = [cell, cell2]
    table.bounding_regions = [
        MagicMock(page_number=1, polygon=[0.0, 3.0, 6.0, 3.0, 6.0, 5.0, 0.0, 5.0])
    ]
    result.tables = [table]
    result.pages = [MagicMock(page_number=1), MagicMock(page_number=2)]

    return result


@pytest.fixture
def parser():
    return AzureDIParser(
        endpoint="https://fake.cognitiveservices.azure.com",
        key="fake-key",
    )


async def test_parse_returns_parsed_document(parser: AzureDIParser):
    mock_result = _make_mock_result()
    mock_poller = AsyncMock()
    mock_poller.result = AsyncMock(return_value=mock_result)

    mock_client = AsyncMock()
    mock_client.begin_analyze_document = AsyncMock(return_value=mock_poller)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "rag_hackathon.ingestion.parser.DocumentIntelligenceClient",
        return_value=mock_client,
    ):
        doc = await parser.parse(b"fake-bytes", "doc-1")

    assert doc.doc_id == "doc-1"
    assert len(doc.paragraphs) == 4
    assert doc.paragraphs[0]["role"] == "title"
    assert doc.paragraphs[1]["section_path"] == ["Introduction"]
    assert doc.paragraphs[3]["section_path"] == ["Introduction", "Methods"]
    assert len(doc.tables) == 1
    assert doc.tables[0].page == 1
    assert "Header" in doc.tables[0].markdown
    assert len(doc.sections) == 2


async def test_parse_empty_document(parser: AzureDIParser):
    mock_result = MagicMock()
    mock_result.paragraphs = None
    mock_result.tables = None
    mock_result.pages = []

    mock_poller = AsyncMock()
    mock_poller.result = AsyncMock(return_value=mock_result)

    mock_client = AsyncMock()
    mock_client.begin_analyze_document = AsyncMock(return_value=mock_poller)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "rag_hackathon.ingestion.parser.DocumentIntelligenceClient",
        return_value=mock_client,
    ):
        doc = await parser.parse(b"empty", "doc-empty")

    assert doc.doc_id == "doc-empty"
    assert doc.paragraphs == []
    assert doc.tables == []
    assert doc.sections == []


async def test_parse_bad_bytes_raises_ingestion_error(parser: AzureDIParser):
    mock_client = AsyncMock()
    mock_client.begin_analyze_document = AsyncMock(side_effect=Exception("bad data"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "rag_hackathon.ingestion.parser.DocumentIntelligenceClient",
            return_value=mock_client,
        ),
        pytest.raises(IngestionError, match="Document parsing failed"),
    ):
        await parser.parse(b"garbage", "doc-bad")


async def test_section_path_deep_hierarchy(parser: AzureDIParser):
    mock_result = MagicMock()
    h1 = MagicMock(content="Chapter 1", role="title", bounding_regions=[])
    h2 = MagicMock(content="Section A", role="sectionHeading", bounding_regions=[])
    h3 = MagicMock(content="Subsection i", role="sectionHeading", bounding_regions=[])
    body = MagicMock(content="Content here", role=None, bounding_regions=[])

    mock_result.paragraphs = [h1, h2, h3, body]
    mock_result.tables = None
    mock_result.pages = []

    mock_poller = AsyncMock()
    mock_poller.result = AsyncMock(return_value=mock_result)

    mock_client = AsyncMock()
    mock_client.begin_analyze_document = AsyncMock(return_value=mock_poller)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "rag_hackathon.ingestion.parser.DocumentIntelligenceClient",
        return_value=mock_client,
    ):
        doc = await parser.parse(b"bytes", "doc-hier")

    assert doc.paragraphs[3]["section_path"] == [
        "Chapter 1",
        "Section A",
        "Subsection i",
    ]
