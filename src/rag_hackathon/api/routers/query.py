from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from rag_hackathon.api.schemas import QueryRequest, QueryResponse
from rag_hackathon.api.services.query_service import QueryService
from rag_hackathon.observability.logging import request_id_var

router = APIRouter(prefix="/api/v1", tags=["query"])

logger = structlog.get_logger("rag_hackathon.api.query")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, fastapi_req: Request) -> QueryResponse:
    request_id = request_id_var.get() or ""
    svc: QueryService = fastapi_req.app.state.query_service
    return await svc.run(request, request_id=request_id)
