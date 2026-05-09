from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI


@runtime_checkable
class TracingBackend(Protocol):
    def configure(self, app: FastAPI) -> None: ...

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]: ...
