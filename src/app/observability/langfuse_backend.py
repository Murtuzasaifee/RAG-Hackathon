from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI
from langfuse import Langfuse

from app.observability.logging import request_id_var


class LangfuseTracingBackend:
    def __init__(
        self,
        secret_key: str,
        public_key: str,
        base_url: str,
    ) -> None:
        self._secret_key = secret_key
        self._public_key = public_key
        self._base_url = base_url
        self._client: Langfuse | None = None

    def configure(self, app: FastAPI) -> None:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", self._secret_key)
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self._public_key)
        os.environ.setdefault("LANGFUSE_BASE_URL", self._base_url)

        self._client = Langfuse(
            secret_key=self._secret_key,
            public_key=self._public_key,
            host=self._base_url,
        )

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        if self._client is None:
            yield
            return

        span_attrs: dict[str, Any] = dict(attrs)
        rid = request_id_var.get()
        if rid is not None:
            span_attrs["request_id"] = rid

        with self._client.start_as_current_observation(
            as_type="span",
            name=name,
            metadata=span_attrs,
        ):
            yield

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()
