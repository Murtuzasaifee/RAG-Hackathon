from __future__ import annotations

import hashlib
import json
import time
import uuid

import structlog
from qdrant_client import AsyncQdrantClient, models

from app.api.schemas import CitationResponse, QueryResponse
from app.ingestion.embedders.protocols import DenseEmbedder
from app.observability.tracing import stage_span

logger = structlog.get_logger("app.cache.semantic")

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(query: str, doc_ids: list[str] | None, owner_id: str | None, top_k: int, top_n: int) -> str:
    key = "|".join([
        query,
        json.dumps(sorted(doc_ids or [])),
        owner_id or "",
        str(top_k),
        str(top_n),
    ])
    return str(uuid.uuid5(_NAMESPACE, hashlib.sha256(key.encode()).hexdigest()))


class SemanticQueryCache:
    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: DenseEmbedder,
        collection: str,
        threshold: float,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._threshold = threshold

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if exists:
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
        logger.info("semantic_cache.collection_created", collection=self._collection)

    async def lookup(
        self,
        query: str,
        doc_ids: list[str] | None,
        owner_id: str | None,
        epoch_map: dict[str, int],
        top_k: int,
        top_n: int,
    ) -> tuple[QueryResponse | None, list[float]]:
        """Return (cached response, query_vec). query_vec is always returned for reuse in store()."""
        with stage_span("cache.semantic.lookup", threshold=self._threshold):
            try:
                vecs = await self._embedder.embed([query])
                query_vec = vecs[0] if vecs else []

                conditions: list[models.Condition] = [
                    models.FieldCondition(key="top_k", match=models.MatchValue(value=top_k)),
                    models.FieldCondition(key="top_n", match=models.MatchValue(value=top_n)),
                ]
                if owner_id is not None:
                    conditions.append(
                        models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id))
                    )
                else:
                    conditions.append(
                        models.IsNullCondition(is_null=models.PayloadField(key="owner_id"))
                    )

                result = await self._client.query_points(
                    collection_name=self._collection,
                    query=query_vec,
                    limit=1,
                    score_threshold=self._threshold,
                    query_filter=models.Filter(must=conditions),
                    with_payload=True,
                    with_vectors=False,
                )

                if not result.points:
                    logger.debug("cache.semantic.miss", reason="no_hit")
                    return None, query_vec

                point = result.points[0]
                payload = point.payload or {}

                stored_epoch = payload.get("epoch_map", {})
                if stored_epoch != epoch_map:
                    logger.debug("cache.semantic.miss", reason="stale_epoch", score=point.score)
                    return None, query_vec

                logger.info(
                    "cache.semantic.hit",
                    score=point.score,
                    collection=self._collection,
                )
                citations = [CitationResponse(**c) for c in payload.get("citations", [])]
                response = QueryResponse(
                    answer=payload["answer"],
                    citations=citations,
                    request_id="",
                    timings_ms={"semantic_cache_hit": 1},
                    warnings=[],
                    cache_hit=True,
                )
                return response, query_vec

            except Exception as exc:
                logger.warning("cache.semantic.lookup.failed", error=str(exc))
                return None, []

    async def store(
        self,
        query: str,
        doc_ids: list[str] | None,
        owner_id: str | None,
        epoch_map: dict[str, int],
        top_k: int,
        top_n: int,
        query_vec: list[float],
        answer_text: str,
        citations: list[CitationResponse],
    ) -> None:
        with stage_span("cache.semantic.store"):
            try:
                point_id = _point_id(query, doc_ids, owner_id, top_k, top_n)
                payload: dict = {
                    "query_text": query,
                    "answer": answer_text,
                    "citations": [c.model_dump() for c in citations],
                    "doc_ids": sorted(doc_ids) if doc_ids else [],
                    "owner_id": owner_id,
                    "epoch_map": epoch_map,
                    "top_k": top_k,
                    "top_n": top_n,
                    "created_at": time.time(),
                }
                await self._client.upsert(
                    collection_name=self._collection,
                    points=[models.PointStruct(id=point_id, vector=query_vec, payload=payload)],
                )
                logger.debug("cache.semantic.stored", collection=self._collection)
            except Exception as exc:
                logger.warning("cache.semantic.store.failed", error=str(exc))

    async def invalidate_doc(self, doc_id: str) -> None:
        with stage_span("cache.semantic.invalidate", doc_id=doc_id):
            try:
                await self._client.delete(
                    collection_name=self._collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="doc_ids",
                                    match=models.MatchValue(value=doc_id),
                                )
                            ]
                        )
                    ),
                )
                logger.info("cache.semantic.invalidated", doc_id=doc_id)
            except Exception as exc:
                logger.warning("cache.semantic.invalidate.failed", doc_id=doc_id, error=str(exc))
