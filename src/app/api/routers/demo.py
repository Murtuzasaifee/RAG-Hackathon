from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.settings import get_settings

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

router = APIRouter(tags=["demo"])
static_files = StaticFiles(directory=FRONTEND_DIR)


@router.get("/demo", include_in_schema=False)
async def demo() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@router.get("/demo/config", include_in_schema=False)
async def demo_config() -> dict[str, str]:
    settings = get_settings()
    return {
        "bifrost_url": settings.bifrost_public_url or settings.bifrost_url,
        "logfire_project_url": settings.logfire_project_url,
    }
