from __future__ import annotations

import io
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import structlog
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from rag_hackathon.core.errors import IngestionError
from rag_hackathon.core.types import ParsedDocument, ParsedElement
from rag_hackathon.observability.tracing import stage_span

logger = structlog.get_logger("rag_hackathon.ingestion.parser")

_OUTPUT_DIR = pathlib.Path("output")

_SKIP_ROLES: frozenset[str] = frozenset(
    {"pageHeader", "pageFooter", "pageNumber", "footnote"}
)

_ROLE_TO_LABEL: dict[str, str] = {
    "title": "document_title",
    "sectionHeading": "paragraph_title",
}


@dataclass
class _RawElement:
    page: int
    y_min: float
    label: str
    text: str
    bbox: list[float]


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _is_pdf(file_bytes: bytes) -> bool:
    return file_bytes[:4] == b"%PDF"


def _split_pdf_pages(file_bytes: bytes) -> list[bytes]:
    """Return list of single-page PDF byte blobs, one per original page."""
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    n = len(reader.pages)
    logger.debug("pdf_split_start", total_pages=n)
    pages: list[bytes] = []
    for i, page in enumerate(reader.pages):
        writer = pypdf.PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        pages.append(buf.getvalue())
        logger.debug("pdf_page_extracted", page=i + 1, total_pages=n)
    return pages


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _save_raw_json(doc_id: str, page_num: int, data: Any) -> None:
    try:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / f"{doc_id}__page_{page_num:03d}__azure_raw.json"
        if hasattr(data, "as_dict"):
            serializable = data.as_dict()
        else:
            serializable = data
        path.write_text(json.dumps(serializable, indent=2, default=str))
        logger.debug("raw_json_saved", doc_id=doc_id, page=page_num, path=str(path))
    except Exception as exc:
        logger.warning("raw_json_save_failed", doc_id=doc_id, page=page_num, error=str(exc))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polygon_to_bbox(polygon: list[float]) -> list[float]:
    if len(polygon) >= 8:
        xs = polygon[0::2]
        ys = polygon[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]
    return []


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
        row[col] = (cell.content or "").strip()

    if not rows:
        return ""

    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "|".join("---" for _ in row) + "|")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Element extraction from a single Azure DI result
# All page numbers are overridden with `actual_page`.
# ---------------------------------------------------------------------------

def _extract_paragraphs(result: Any, actual_page: int) -> list[_RawElement]:
    raw: list[_RawElement] = []
    for p in result.paragraphs or []:
        role = getattr(p, "role", None)
        if role in _SKIP_ROLES:
            continue
        text = (p.content or "").strip()
        if not text:
            continue
        label = _ROLE_TO_LABEL.get(role or "", "text")
        bbox: list[float] = []
        y_min = 0.0
        if p.bounding_regions:
            bbox = _polygon_to_bbox(p.bounding_regions[0].polygon or [])
            y_min = bbox[1] if len(bbox) >= 4 else 0.0
        raw.append(_RawElement(page=actual_page, y_min=y_min, label=label, text=text, bbox=bbox))
    return raw


def _extract_tables(result: Any, actual_page: int) -> list[_RawElement]:
    raw: list[_RawElement] = []
    for table in result.tables or []:
        markdown = _table_to_markdown(table)
        if not markdown.strip():
            continue
        bbox: list[float] = []
        y_min = 0.0
        if table.bounding_regions:
            bbox = _polygon_to_bbox(table.bounding_regions[0].polygon or [])
            y_min = bbox[1] if len(bbox) >= 4 else 0.0
        raw.append(_RawElement(page=actual_page, y_min=y_min, label="table", text=markdown, bbox=bbox))
    return raw


def _extract_figures(result: Any, actual_page: int) -> list[_RawElement]:
    raw: list[_RawElement] = []
    for figure in getattr(result, "figures", None) or []:
        caption = ""
        if hasattr(figure, "caption") and figure.caption:
            caption = (getattr(figure.caption, "content", "") or "").strip()
        bbox: list[float] = []
        y_min = 0.0
        if figure.bounding_regions:
            bbox = _polygon_to_bbox(figure.bounding_regions[0].polygon or [])
            y_min = bbox[1] if len(bbox) >= 4 else 0.0
        raw.append(_RawElement(page=actual_page, y_min=y_min, label="image", text=caption, bbox=bbox))
    return raw


def _result_to_raw_elements(result: Any, actual_page: int) -> list[_RawElement]:
    raw: list[_RawElement] = []
    raw.extend(_extract_paragraphs(result, actual_page))
    raw.extend(_extract_tables(result, actual_page))
    raw.extend(_extract_figures(result, actual_page))
    return raw


def _label_counts(raw: list[_RawElement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for el in raw:
        counts[el.label] = counts.get(el.label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class AzureDIParser:
    def __init__(self, endpoint: str, key: str) -> None:
        self._endpoint = endpoint
        self._key = key

    async def parse(self, file_bytes: bytes, doc_id: str) -> ParsedDocument:
        with stage_span("parse", doc_id=doc_id):
            is_pdf = _is_pdf(file_bytes)
            if is_pdf:
                pages_bytes = _split_pdf_pages(file_bytes)
            else:
                pages_bytes = [file_bytes]

            logger.info(
                "parse_started",
                doc_id=doc_id,
                file_type="pdf" if is_pdf else "other",
                total_pages=len(pages_bytes),
                file_size_bytes=len(file_bytes),
            )

            all_raw: list[_RawElement] = []

            credential = AzureKeyCredential(self._key)
            for page_num, page_bytes in enumerate(pages_bytes, start=1):
                logger.info(
                    "page_parse_started",
                    doc_id=doc_id,
                    page=page_num,
                    total_pages=len(pages_bytes),
                    page_size_bytes=len(page_bytes),
                )
                try:
                    async with DocumentIntelligenceClient(
                        endpoint=self._endpoint, credential=credential
                    ) as client:
                        poller = await client.begin_analyze_document(
                            "prebuilt-layout",
                            io.BytesIO(page_bytes),
                            content_type="application/octet-stream",
                        )
                        result = await poller.result()
                except Exception as exc:
                    raise IngestionError(
                        f"Document parsing failed on page {page_num}: {exc}"
                    ) from exc

                _save_raw_json(doc_id, page_num, result)

                page_raw = _result_to_raw_elements(result, actual_page=page_num)
                all_raw.extend(page_raw)

                logger.info(
                    "page_parse_done",
                    doc_id=doc_id,
                    page=page_num,
                    total_pages=len(pages_bytes),
                    elements_on_page=len(page_raw),
                    by_label=_label_counts(page_raw),
                    paragraphs=len(result.paragraphs or []),
                    tables=len(result.tables or []),
                    figures=len(getattr(result, "figures", None) or []),
                )

            # Sort by (page, y_min) and assign per-page reading_order
            all_raw.sort(key=lambda e: (e.page, e.y_min))

            elements: list[ParsedElement] = []
            page_counters: dict[int, int] = {}
            for raw_el in all_raw:
                order = page_counters.get(raw_el.page, 0)
                page_counters[raw_el.page] = order + 1
                elements.append(
                    ParsedElement(
                        label=raw_el.label,
                        text=raw_el.text,
                        bbox=raw_el.bbox,
                        reading_order=order,
                        page=raw_el.page,
                    )
                )

            logger.info(
                "parse_complete",
                doc_id=doc_id,
                total_pages=len(pages_bytes),
                total_elements=len(elements),
                by_label=_label_counts(all_raw),
            )

            return ParsedDocument(doc_id=doc_id, elements=elements)
