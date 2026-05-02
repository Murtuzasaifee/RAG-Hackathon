from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from rag_hackathon.core.types import JobStatus
from rag_hackathon.ingestion.jobs import RedisJobStore, make_job_id, make_version_id


async def test_make_job_id_unique():
    assert make_job_id() != make_job_id()


async def test_make_version_id_unique():
    assert make_version_id() != make_version_id()


async def test_redis_job_store_roundtrip():
    mock_redis = AsyncMock()
    mock_redis.hset = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={})

    store = RedisJobStore(mock_redis)

    now = datetime.now(UTC)
    job = JobStatus(
        job_id="j1",
        doc_id="d1",
        version_id="v1",
        state="running",
        stage="chunking",
        progress=40,
        error=None,
        created_at=now,
        updated_at=now,
    )

    await store.set_status(job)
    mock_redis.hset.assert_called_once()

    key = mock_redis.hset.call_args[0][0]
    assert key == "job:j1"


async def test_redis_job_store_get_not_found():
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={})
    store = RedisJobStore(mock_redis)

    result = await store.get_status("nonexistent")
    assert result is None


async def test_redis_job_store_get_existing():
    now = datetime.now(UTC)
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(
        return_value={
            b"job_id": b"j1",
            b"doc_id": b"d1",
            b"version_id": b"v1",
            b"state": b"done",
            b"stage": b"done",
            b"progress": b"100",
            b"error": b"",
            b"created_at": now.isoformat().encode(),
            b"updated_at": now.isoformat().encode(),
        }
    )
    store = RedisJobStore(mock_redis)

    result = await store.get_status("j1")
    assert result is not None
    assert result.job_id == "j1"
    assert result.state == "done"
    assert result.progress == 100
