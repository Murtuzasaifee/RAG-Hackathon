from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

import redis.asyncio as aioredis
import structlog
from ulid import ULID

from rag_hackathon.core.types import JobStatus

logger = structlog.get_logger("rag_hackathon.ingestion.jobs")


class JobStore(Protocol):
    async def set_status(self, job: JobStatus) -> None: ...
    async def get_status(self, job_id: str) -> JobStatus | None: ...


class RedisJobStore:
    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    async def set_status(self, job: JobStatus) -> None:
        key = f"job:{job.job_id}"
        await self._client.hset(
            key,
            mapping={
                "job_id": job.job_id,
                "doc_id": job.doc_id,
                "version_id": job.version_id,
                "state": job.state,
                "stage": job.stage,
                "progress": str(job.progress),
                "error": job.error or "",
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            },
        )

    async def get_status(self, job_id: str) -> JobStatus | None:
        key = f"job:{job_id}"
        data = await self._client.hgetall(key)
        if not data:
            return None
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        return JobStatus(
            job_id=decoded["job_id"],
            doc_id=decoded["doc_id"],
            version_id=decoded["version_id"],
            state=decoded["state"],
            stage=decoded["stage"],
            progress=int(decoded["progress"]),
            error=decoded["error"] or None,
            created_at=datetime.fromisoformat(decoded["created_at"]),
            updated_at=datetime.fromisoformat(decoded["updated_at"]),
        )


class JobRunner(Protocol):
    async def submit(self, coro) -> str: ...


class BackgroundTasksRunner:
    def __init__(self) -> None:
        self._tasks: dict[str, object] = {}

    async def submit(self, coro) -> str:
        import asyncio

        job_id = str(ULID())
        asyncio.create_task(coro(job_id))
        return job_id


def make_job_id() -> str:
    return str(ULID())


def make_version_id() -> str:
    return str(ULID())


def now_utc() -> datetime:
    return datetime.now(UTC)
