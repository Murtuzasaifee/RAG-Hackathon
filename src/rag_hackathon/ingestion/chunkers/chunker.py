from __future__ import annotations

from typing import Any

from rag_hackathon.core.types import Chunk, ChunkType, ParsedDocument, ParsedElement
from rag_hackathon.observability.tracing import stage_span

_TOKEN_WORD_RATIO: float = 1.3

ATOMIC_LABELS: frozenset[str] = frozenset(
    {"table", "formula", "inline_formula", "algorithm", "image", "figure"}
)

TITLE_LABELS: frozenset[str] = frozenset(
    {"document_title", "paragraph_title", "figure_title"}
)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOKEN_WORD_RATIO)


def _split_text(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    words_per_chunk = max(1, int(max_tokens / _TOKEN_WORD_RATIO))
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def _infer_chunk_type(labels: list[str]) -> ChunkType:
    types = frozenset(labels)
    if "table" in types:
        return "table"
    if types & {"image", "figure"}:
        return "image"
    if types & {"formula", "inline_formula"}:
        return "formula"
    if "algorithm" in types:
        return "algorithm"
    return "text"


def _update_section_path(section_path: list[str], label: str, text: str) -> list[str]:
    if label == "document_title":
        return [text]
    if label == "paragraph_title":
        return [section_path[0], text] if section_path else [text]
    return section_path  # figure_title doesn't change section hierarchy


def _union_bbox(bboxes: list[list[float]]) -> list[float]:
    """Compute the bounding union of multiple [x0, y0, x1, y1] boxes."""
    valid = [b for b in bboxes if len(b) == 4]
    if not valid:
        return []
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


class DocumentAwareChunker:
    def chunk(
        self,
        parsed: ParsedDocument,
        *,
        max_tokens: int = 512,
    ) -> list[Chunk]:
        with stage_span("chunk", doc_id=parsed.doc_id):
            return _chunk_document(parsed.elements, parsed.doc_id, max_tokens)


def _chunk_document(
    elements: list[ParsedElement],
    doc_id: str,
    max_tokens: int,
) -> list[Chunk]:
    sorted_els = sorted(elements, key=lambda e: (e.page, e.reading_order))
    if not sorted_els:
        return []

    chunks: list[Chunk] = []
    chunk_idx = 0
    section_path: list[str] = []

    current_texts: list[str] = []
    current_labels: list[str] = []
    current_tokens: int = 0
    current_page: int = sorted_els[0].page
    # Accumulate all per-element {page, bbox} pairs for the current chunk
    current_page_bboxes: list[dict[str, Any]] = []

    pending_title: str | None = None
    pending_title_label: str | None = None
    pending_title_page: int = current_page

    def _make_chunk(
        texts: list[str],
        labels: list[str],
        page: int,
        page_bboxes: list[dict[str, Any]],
    ) -> Chunk:
        nonlocal chunk_idx
        # Union of all bboxes on the primary page for the top-level bbox field
        primary_bboxes = [pb["bbox"] for pb in page_bboxes if pb["page"] == page]
        union = _union_bbox(primary_bboxes)
        c = Chunk(
            doc_id=doc_id,
            version_id="",
            chunk_index=chunk_idx,
            text="\n\n".join(texts),
            page=page,
            section_path=list(section_path),
            bbox=union,
            page_bboxes=list(page_bboxes),
            chunk_type=_infer_chunk_type(labels),
        )
        chunk_idx += 1
        return c

    def flush_current() -> None:
        nonlocal current_texts, current_labels, current_tokens
        nonlocal pending_title, pending_title_label, pending_title_page
        nonlocal current_page, current_page_bboxes

        if not current_texts and pending_title is None:
            return

        texts_to_flush: list[str] = []
        labels_to_flush: list[str] = []
        page_to_use = (
            pending_title_page if (pending_title and not current_texts) else current_page
        )

        if pending_title is not None:
            texts_to_flush.append(pending_title)
            labels_to_flush.append(pending_title_label or "paragraph_title")
            pending_title = None
            pending_title_label = None

        texts_to_flush.extend(current_texts)
        labels_to_flush.extend(current_labels)

        if not texts_to_flush:
            return

        chunks.append(_make_chunk(texts_to_flush, labels_to_flush, page_to_use, current_page_bboxes))
        current_texts = []
        current_labels = []
        current_tokens = 0
        current_page_bboxes = []

    for el in sorted_els:
        label = el.label
        text = el.text.strip()

        if label in ATOMIC_LABELS:
            # Figure caption: pending figure_title belongs to this atomic element
            figure_caption: str | None = None
            if pending_title is not None and pending_title_label == "figure_title":
                figure_caption = pending_title
                pending_title = None
                pending_title_label = None

            flush_current()

            if figure_caption:
                atomic_text = f"{figure_caption}\n\n{text}" if text else figure_caption
                atomic_labels = ["figure_title", label]
            else:
                atomic_text = text
                atomic_labels = [label]

            if atomic_text:
                # Atomic elements always have their own bbox — never cleared
                el_page_bboxes = [{"page": el.page, "bbox": el.bbox}] if el.bbox else []
                chunks.append(_make_chunk([atomic_text], atomic_labels, el.page, el_page_bboxes))
            continue

        if not text:
            continue

        if label in TITLE_LABELS:
            section_path = _update_section_path(section_path, label, text)
            if current_texts:
                flush_current()
            elif pending_title is not None:
                flush_current()
            pending_title = text
            pending_title_label = label
            pending_title_page = el.page
            continue

        token_estimate = _estimate_tokens(text)
        pending_tokens = _estimate_tokens(pending_title) if pending_title else 0

        if token_estimate > max_tokens:
            flush_current()
            sub_texts = _split_text(text, max_tokens)
            for sub_idx, sub_text in enumerate(sub_texts):
                # First sub-chunk inherits the source element bbox; rest get empty
                # (can't split bbox without line-level geometry)
                sub_bboxes = [{"page": el.page, "bbox": el.bbox}] if (sub_idx == 0 and el.bbox) else []
                chunks.append(_make_chunk([sub_text], [label], el.page, sub_bboxes))
            continue

        if current_texts and (current_tokens + token_estimate + pending_tokens > max_tokens):
            flush_current()

        if pending_title is not None:
            if not current_texts:
                current_page = pending_title_page
            current_texts.append(pending_title)
            current_labels.append(pending_title_label or "paragraph_title")
            current_tokens += _estimate_tokens(pending_title)
            pending_title = None
            pending_title_label = None

        if not current_texts:
            current_page = el.page

        # Always record bbox regardless of how many sources contribute
        if el.bbox:
            current_page_bboxes.append({"page": el.page, "bbox": el.bbox})

        current_texts.append(text)
        current_labels.append(label)
        current_tokens += token_estimate

        if current_tokens >= max_tokens:
            flush_current()

    flush_current()
    return chunks
