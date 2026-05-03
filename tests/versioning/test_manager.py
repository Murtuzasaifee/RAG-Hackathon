from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from rag_hackathon.versioning.manager import VersionManager


@pytest.fixture
def qdrant_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.count.return_value = MagicMock(count=5)
    return mock


@pytest.fixture
def cache_mock() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def vm(qdrant_mock: AsyncMock, cache_mock: AsyncMock) -> VersionManager:
    return VersionManager(
        qdrant=qdrant_mock,
        collection="documents",
        cache=cache_mock,
    )


class TestFlipActive:
    @pytest.mark.asyncio
    async def test_flip_active_sets_new_version_active(
        self, vm: VersionManager, qdrant_mock: AsyncMock
    ):
        await vm.flip_active("doc1", "v2")

        assert qdrant_mock.set_payload.await_count == 2

        first_call = qdrant_mock.set_payload.call_args_list[0]
        assert first_call.kwargs["payload"] == {"active": True}
        first_filter = first_call.kwargs["points"]
        must_conditions = first_filter.must
        assert len(must_conditions) == 2
        assert any(
            c.key == "version_id"
            and isinstance(c.match, models.MatchValue)
            and c.match.value == "v2"
            for c in must_conditions
        )

        second_call = qdrant_mock.set_payload.call_args_list[1]
        assert second_call.kwargs["payload"] == {"active": False}
        second_filter = second_call.kwargs["points"]
        assert second_filter.must_not is not None

    @pytest.mark.asyncio
    async def test_flip_active_without_cache(
        self, qdrant_mock: AsyncMock
    ):
        vm_no_cache = VersionManager(
            qdrant=qdrant_mock,
            collection="documents",
            cache=None,
        )
        await vm_no_cache.flip_active("doc1", "v2")
        assert qdrant_mock.set_payload.await_count == 2


class TestSoftDelete:
    @pytest.mark.asyncio
    async def test_soft_delete_all_versions(
        self, vm: VersionManager, qdrant_mock: AsyncMock
    ):
        n = await vm.soft_delete("doc1")
        assert n == 5
        qdrant_mock.set_payload.assert_awaited_once()
        call_kwargs = qdrant_mock.set_payload.call_args.kwargs
        assert call_kwargs["payload"] == {"active": False}

    @pytest.mark.asyncio
    async def test_soft_delete_specific_version(
        self, vm: VersionManager, qdrant_mock: AsyncMock
    ):
        n = await vm.soft_delete("doc1", version_id="v1")
        assert n == 5
        call_kwargs = qdrant_mock.set_payload.call_args.kwargs
        conditions = call_kwargs["points"].must
        assert len(conditions) == 2

    @pytest.mark.asyncio
    async def test_soft_delete_no_points(
        self, vm: VersionManager, qdrant_mock: AsyncMock
    ):
        qdrant_mock.count.return_value = MagicMock(count=0)
        n = await vm.soft_delete("nonexistent")
        assert n == 0


class TestHardDelete:
    @pytest.mark.asyncio
    async def test_hard_delete_all_versions(
        self, vm: VersionManager, qdrant_mock: AsyncMock, cache_mock: AsyncMock
    ):
        n = await vm.hard_delete("doc1")
        assert n == 5
        qdrant_mock.delete.assert_awaited_once()
        cache_mock.incr.assert_awaited_once_with("doc:epoch:doc1")

    @pytest.mark.asyncio
    async def test_hard_delete_specific_version(
        self, vm: VersionManager, qdrant_mock: AsyncMock, cache_mock: AsyncMock
    ):
        n = await vm.hard_delete("doc1", version_id="v1")
        assert n == 5
        call_kwargs = qdrant_mock.delete.call_args.kwargs
        selector = call_kwargs["points_selector"]
        conditions = selector.filter.must
        assert len(conditions) == 2

    @pytest.mark.asyncio
    async def test_hard_delete_bumps_epoch(
        self, vm: VersionManager, cache_mock: AsyncMock
    ):
        cache_mock.incr.return_value = 3
        await vm.hard_delete("doc1")
        cache_mock.incr.assert_awaited_once_with("doc:epoch:doc1")

    @pytest.mark.asyncio
    async def test_hard_delete_without_cache(
        self, qdrant_mock: AsyncMock
    ):
        vm_no_cache = VersionManager(
            qdrant=qdrant_mock,
            collection="documents",
            cache=None,
        )
        await vm_no_cache.hard_delete("doc1")
        qdrant_mock.delete.assert_awaited_once()
