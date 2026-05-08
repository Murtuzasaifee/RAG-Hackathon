from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient, models

from app.cache.protocols import CacheStore
from app.observability.tracing import stage_span

logger = structlog.get_logger("app.versioning.manager")


class VersionManager:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        collection: str,
        cache: CacheStore | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._collection = collection
        self._cache = cache

    async def flip_active(
        self, doc_id: str, new_version_id: str
    ) -> None:
        with stage_span(
            "version.flip_active",
            doc_id=doc_id,
            new_version=new_version_id,
        ):
            await self._qdrant.set_payload(
                collection_name=self._collection,
                payload={"active": True},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        ),
                        models.FieldCondition(
                            key="version_id",
                            match=models.MatchValue(value=new_version_id),
                        ),
                    ]
                ),
            )

            await self._qdrant.set_payload(
                collection_name=self._collection,
                payload={"active": False},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        ),
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="version_id",
                            match=models.MatchValue(value=new_version_id),
                        ),
                    ],
                ),
            )

            logger.info(
                "version flipped",
                doc_id=doc_id,
                new_version=new_version_id,
            )

    async def soft_delete(
        self,
        doc_id: str,
        version_id: str | None = None,
    ) -> int:
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            ),
        ]
        if version_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="version_id",
                    match=models.MatchValue(value=version_id),
                )
            )

        with stage_span(
            "version.soft_delete",
            doc_id=doc_id,
            version_id=version_id,
        ):
            count_before = await self._qdrant.count(
                collection_name=self._collection,
                count_filter=models.Filter(must=conditions),
            )
            n = count_before.count

            await self._qdrant.set_payload(
                collection_name=self._collection,
                payload={"active": False},
                points=models.Filter(must=conditions),
            )

            logger.info(
                "soft deleted",
                doc_id=doc_id,
                version_id=version_id,
                n_points=n,
            )
            return n

    async def hard_delete(
        self,
        doc_id: str,
        version_id: str | None = None,
    ) -> int:
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            ),
        ]
        if version_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="version_id",
                    match=models.MatchValue(value=version_id),
                )
            )

        with stage_span(
            "version.hard_delete",
            doc_id=doc_id,
            version_id=version_id,
        ):
            count_before = await self._qdrant.count(
                collection_name=self._collection,
                count_filter=models.Filter(must=conditions),
            )
            n = count_before.count

            await self._qdrant.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=conditions),
                ),
            )

            if self._cache is not None:
                await self._cache.incr(f"doc:epoch:{doc_id}")

            logger.info(
                "hard deleted",
                doc_id=doc_id,
                version_id=version_id,
                n_points=n,
            )
            return n
