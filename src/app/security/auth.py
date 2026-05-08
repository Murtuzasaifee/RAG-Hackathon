from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

import structlog
from fastapi import Depends, Header, Request
from pydantic import BaseModel, ConfigDict
from ulid import ULID

from app.core.errors import AuthError, ForbiddenError
from app.core.settings import get_settings

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

logger = structlog.get_logger("rag_hackathon.security.auth")

ROLE_ORDER: dict[str, int] = {"reader": 0, "editor": 1, "admin": 2}


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)
    key_id: str
    role: str


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def authenticate(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    settings = get_settings()

    if not settings.auth_enabled:
        return Principal(key_id="dev", role="admin")

    if x_api_key is None:
        logger.warning("auth.rejected", reason="missing_key", path=request.url.path)
        raise AuthError("missing X-API-Key header")

    redis_key = f"apikey:{_hash_key(x_api_key)}"
    data: dict = await request.app.state.redis.hgetall(redis_key)

    if not data:
        logger.warning("auth.rejected", reason="invalid_key", path=request.url.path)
        raise AuthError("invalid or unknown api key")

    role = data[b"role"].decode()
    key_id = data[b"key_id"].decode()
    logger.debug("auth.ok", key_id=key_id, role=role, path=request.url.path)
    return Principal(key_id=key_id, role=role)


def require_role(min_role: str) -> Callable:
    async def _dependency(
        principal: Annotated[Principal, Depends(authenticate)],
    ) -> Principal:
        if ROLE_ORDER.get(principal.role, -1) < ROLE_ORDER[min_role]:
            logger.warning(
                "auth.forbidden",
                key_id=principal.key_id,
                role=principal.role,
                required=min_role,
            )
            raise ForbiddenError(
                f"role '{principal.role}' insufficient — requires '{min_role}'"
            )
        return principal

    return _dependency


async def seed_api_key(redis, raw_key: str, role: str, label: str) -> str:
    redis_key = f"apikey:{_hash_key(raw_key)}"
    existing = await redis.exists(redis_key)
    if existing:
        key_id = (await redis.hget(redis_key, "key_id")).decode()
        logger.info("auth.seed.skipped", label=label, key_id=key_id, reason="already_exists")
        return key_id

    key_id = str(ULID())
    await redis.hset(
        redis_key,
        mapping={
            "key_id": key_id,
            "role": role,
            "created_at": datetime.now(UTC).isoformat(),
            "label": label,
        },
    )
    logger.info("auth.seed.stored", label=label, role=role, key_id=key_id)
    return key_id


async def verify_document_ownership(
    qdrant: AsyncQdrantClient,
    collection: str,
    doc_id: str,
    principal: Principal,
) -> None:
    if ROLE_ORDER.get(principal.role, -1) >= ROLE_ORDER["admin"]:
        return

    from qdrant_client import models

    points, _ = await qdrant.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                ),
            ]
        ),
        limit=1,
        with_payload=["owner_id"],
    )

    if not points:
        return

    owner_id = (points[0].payload or {}).get("owner_id")
    if owner_id is not None and owner_id != principal.key_id:
        logger.warning(
            "ownership.denied",
            key_id=principal.key_id,
            doc_id=doc_id,
            owner_id=owner_id,
        )
        raise ForbiddenError("you do not own this document")
