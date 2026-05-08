from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request

from app.api.schemas import QueryRequest, QueryResponse
from app.api.services.query_service import QueryService
from app.observability.logging import request_id_var
from app.security.auth import ROLE_ORDER, Principal, require_role

router = APIRouter(prefix="/api/v1", tags=["query"])

logger = structlog.get_logger("app.api.query")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    fastapi_req: Request,
    principal: Annotated[Principal, Depends(require_role("reader"))],
) -> QueryResponse:
    request_id = request_id_var.get() or ""
    svc: QueryService = fastapi_req.app.state.query_service
    # admin sees all documents; reader/editor scoped to their own ingested docs
    owner_id = None if ROLE_ORDER.get(principal.role, 0) >= ROLE_ORDER["admin"] else principal.key_id
    return await svc.run(request, request_id=request_id, owner_id=owner_id)
