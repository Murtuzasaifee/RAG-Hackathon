from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import logfire
from fastapi import FastAPI

from app.core.settings import get_settings
from app.observability.logging import request_id_var


def configure_tracing(app: FastAPI) -> None:
    settings = get_settings()
    logfire.configure(token=settings.logfire_token)
    logfire.instrument_fastapi(app)
    logfire.instrument_httpx()
    logfire.instrument_pydantic()


@contextmanager
def stage_span(name: str, **attrs: Any) -> Iterator[None]:
    span_attrs: dict[str, Any] = dict(attrs)
    rid = request_id_var.get()
    if rid is not None:
        span_attrs["request_id"] = rid
    with logfire.span(name, **span_attrs):
        yield
