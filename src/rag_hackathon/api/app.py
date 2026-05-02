from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_hackathon.api.error_handlers import RAGError, rag_error_handler
from rag_hackathon.api.middleware import RequestIdMiddleware
from rag_hackathon.api.routers import health
from rag_hackathon.observability.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Hackathon Backend",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(RAGError, rag_error_handler)

    app.include_router(health.router)

    return app


app = create_app()
