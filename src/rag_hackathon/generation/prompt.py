from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly on the "
    "provided document context. "
    "Answer ONLY using the provided context. If the answer is not in the "
    "context, say \"I don't know based on the provided documents.\" "
    "Do not include citation markers or inline references in your response — "
    "citations are returned separately."
)


def build_messages(query: str, hits: list[dict]) -> list[dict[str, str]]:
    if not hits:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nNo relevant context was found."},
        ]

    context_parts: list[str] = []
    for i, h in enumerate(hits, 1):
        page = h.get("page", "?")
        section = " > ".join(h.get("section_path", []))
        context_parts.append(
            f"[{i}] (Page {page}, Section: {section})\n{h.get('chunk_text', '')}"
        )
    context = "\n\n".join(context_parts)
    user_msg = f"Question: {query}\n\nContext:\n{context}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def hits_to_dicts(hits: list) -> list[dict]:
    return [
        {
            "chunk_text": h.chunk_text,
            "page": h.page,
            "section_path": list(h.section_path),
        }
        for h in hits
    ]
