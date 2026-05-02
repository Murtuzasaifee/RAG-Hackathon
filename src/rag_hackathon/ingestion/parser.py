from __future__ import annotations

import io
from typing import Any

import structlog
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from rag_hackathon.core.errors import IngestionError
from rag_hackathon.core.types import ParsedDocument, ParsedTable, Section
from rag_hackathon.observability.tracing import stage_span

logger = structlog.get_logger("rag_hackathon.ingestion.parser")


def _table_to_markdown(table: Any) -> str:
    rows: list[list[str]] = []
    for cell in table.cells:
        ri = cell.row_index
        while len(rows) <= ri:
            rows.append([])
        col = cell.column_index
        row = rows[ri]
        while len(row) <= col:
            row.append("")
        row[col] = cell.content

    if not rows:
        return ""

    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "|".join("---" for _ in row) + "|")
    return "\n".join(lines)


def _polygon_to_bbox(polygon: list[float]) -> list[float]:
    if len(polygon) >= 8:
        xs = polygon[0::2]
        ys = polygon[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    return []


def _build_section_path(
    paragraph: Any, para_idx: int, paragraphs: list[Any]
) -> list[str]:
    path: list[str] = []
    for i in range(para_idx, -1, -1):
        p = paragraphs[i]
        role = getattr(p, "role", None)
        if role in ("title", "sectionHeading"):
            path.insert(0, p.content.strip())
    return path


class AzureDIParser:
    def __init__(self, endpoint: str, key: str) -> None:
        self._endpoint = endpoint
        self._key = key

    async def parse(self, file_bytes: bytes, doc_id: str) -> ParsedDocument:
        with stage_span("parse", doc_id=doc_id):
            try:
                credential = AzureKeyCredential(self._key)
                async with DocumentIntelligenceClient(
                    endpoint=self._endpoint, credential=credential
                ) as client:
                    poller = await client.begin_analyze_document(
                        "prebuilt-layout",
                        io.BytesIO(file_bytes),
                        content_type="application/octet-stream",
                    )
                    result = await poller.result()
            except Exception as exc:
                raise IngestionError(f"Document parsing failed: {exc}") from exc

            paragraphs = result.paragraphs or []
            sections = self._extract_sections(paragraphs)
            tables = self._extract_tables(result)
            raw_paragraphs = self._extract_paragraphs(paragraphs)

            return ParsedDocument(
                doc_id=doc_id,
                sections=sections,
                paragraphs=raw_paragraphs,
                tables=tables,
            )

    def _extract_sections(self, paragraphs: list[Any]) -> list[Section]:
        sections: list[Section] = []
        for p in paragraphs:
            role = getattr(p, "role", None)
            if role in ("title", "sectionHeading"):
                sections.append(
                    Section(
                        heading=p.content.strip(),
                        level=1 if role == "title" else 2,
                    )
                )
        return sections

    def _extract_tables(self, result: Any) -> list[ParsedTable]:
        tables: list[ParsedTable] = []
        for table in result.tables or []:
            bbox: list[float] = []
            page = 1
            if table.bounding_regions:
                br = table.bounding_regions[0]
                page = br.page_number
                bbox = _polygon_to_bbox(br.polygon)
            tables.append(
                ParsedTable(
                    markdown=_table_to_markdown(table),
                    page=page,
                    bbox=bbox,
                )
            )
        return tables

    def _extract_paragraphs(self, paragraphs: list[Any]) -> list[dict]:
        result_list: list[dict] = []
        for idx, p in enumerate(paragraphs):
            page = 1
            bbox: list[float] = []
            if p.bounding_regions:
                br = p.bounding_regions[0]
                page = br.page_number
                bbox = _polygon_to_bbox(br.polygon)
            section_path = _build_section_path(p, idx, paragraphs)
            result_list.append(
                {
                    "content": p.content,
                    "page": page,
                    "bbox": bbox,
                    "section_path": section_path,
                    "role": getattr(p, "role", None),
                }
            )
        return result_list
