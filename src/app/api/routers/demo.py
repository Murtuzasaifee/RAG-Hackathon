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
async def demo_config() -> dict:
    settings = get_settings()
    return {
        "bifrost_url": settings.bifrost_public_url or settings.bifrost_url,
        "observability_url": settings.otel_project_url,
        "otel_backend": settings.otel_backend,
        "role_presets": [
            {"label": "reader",  "key": settings.demo_reader_key},
            {"label": "editor1",  "key": settings.demo_editor1_key},
            {"label": "editor2", "key": settings.demo_editor2_key},
            {"label": "admin",   "key": settings.demo_admin_key},
        ],
    }
