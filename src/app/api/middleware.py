from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import structlog

from app.observability.logging import request_id_var

logger = structlog.get_logger("app.middleware")

_HEADER = "x-request-id"


class RequestIdMiddleware:
    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(
        self, scope: dict[str, Any], receive: Callable, send: Callable
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(
            (k.decode().lower(), v.decode()) for k, v in scope.get("headers", [])
        )
        rid = headers.get(_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(rid)

        async def _send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append(
                    (f"{_HEADER}".encode(), rid.encode())
                )
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            request_id_var.reset(token)
