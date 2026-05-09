from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import logfire
from fastapi import FastAPI

from app.observability.logging import request_id_var


class LogfireTracingBackend:
    def __init__(self, token: str) -> None:
        self._token = token

    def configure(self, app: FastAPI) -> None:
        logfire.configure(token=self._token)
        logfire.instrument_fastapi(app)
        logfire.instrument_httpx()
        logfire.instrument_pydantic()

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        span_attrs: dict[str, Any] = dict(attrs)
        rid = request_id_var.get()
        if rid is not None:
            span_attrs["request_id"] = rid
        with logfire.span(name, **span_attrs):
            yield
