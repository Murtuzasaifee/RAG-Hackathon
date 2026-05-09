from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI

from app.core.settings import get_settings
from app.observability.langfuse_backend import LangfuseTracingBackend
from app.observability.logfire_backend import LogfireTracingBackend

_backend: LogfireTracingBackend | LangfuseTracingBackend | None = None


def configure_tracing(app: FastAPI) -> None:
    global _backend

    settings = get_settings()
    backend_name = settings.otel_backend

    if backend_name == "logfire":
        if not settings.logfire_token:
            raise ValueError(
                "OTEL_BACKEND=logfire but LOGFIRE_TOKEN is not set. "
                "Provide LOGFIRE_TOKEN or switch OTEL_BACKEND=langfuse."
            )
        _backend = LogfireTracingBackend(token=settings.logfire_token)

    elif backend_name == "langfuse":
        if not settings.langfuse_secret_key or not settings.langfuse_public_key:
            raise ValueError(
                "OTEL_BACKEND=langfuse but LANGFUSE_SECRET_KEY "
                "and/or LANGFUSE_PUBLIC_KEY are not set. "
                "Provide both keys (and optionally LANGFUSE_BASE_URL) "
                "or switch OTEL_BACKEND=logfire."
            )
        _backend = LangfuseTracingBackend(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            base_url=settings.langfuse_base_url or "https://cloud.langfuse.com",
        )

    if _backend is not None:
        _backend.configure(app)


@contextmanager
def stage_span(name: str, **attrs: Any) -> Iterator[None]:
    if _backend is not None:
        with _backend.span(name, **attrs):
            yield
    else:
        yield


def get_tracing_backend() -> LogfireTracingBackend | LangfuseTracingBackend | None:
    return _backend
