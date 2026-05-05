from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.errors import (
    GatewayError,
    GenerationError,
    GuardError,
    IngestionError,
    RAGError,
    RetrievalError,
)


def _error_body(error: str, message: str) -> dict:
    return {"error": error, "message": message}


_ERROR_MAP: dict[type[RAGError], tuple[int, str]] = {
    IngestionError: (422, "ingestion_error"),
    RetrievalError: (502, "retrieval_error"),
    GenerationError: (502, "generation_error"),
    GuardError: (400, "guard_error"),
    GatewayError: (502, "gateway_error"),
}


async def rag_error_handler(_request: Request, exc: RAGError) -> JSONResponse:
    status_code, error_name = _ERROR_MAP.get(type(exc), (500, "internal_error"))
    return JSONResponse(
        status_code=status_code,
        content=_error_body(error_name, exc.message),
    )
