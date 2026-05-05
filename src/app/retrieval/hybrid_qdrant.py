from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient, models

from app.core.errors import RetrievalError
from app.core.types import RetrievalHit
from app.ingestion.embedders.protocols import DenseEmbedder, SparseEmbedder
from app.observability.tracing import stage_span

logger = structlog.get_logger("app.retrieval.hybrid")


class HybridQdrantRetriever:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._client = client
        self._collection = collection
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        *,
        doc_ids: list[str] | None = None,
        version_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[RetrievalHit]:
        with stage_span("retrieve.hybrid", query_len=len(query), top_k=top_k):
            dense_vec = await self._dense.embed([query])
            if not dense_vec:
                raise RetrievalError("Dense embedding returned empty result")
            query_dense = dense_vec[0]

            conditions: list[models.Condition] = []
            if version_ids is not None:
                conditions.append(
                    models.FieldCondition(
                        key="version_id",
                        match=models.MatchAny(any=version_ids),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="active", match=models.MatchValue(value=True)
                    )
                )
            if doc_ids is not None:
                conditions.append(
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=doc_ids),
                    )
                )
            query_filter = models.Filter(must=conditions) if conditions else None

            prefetch: list[models.Prefetch] = [
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=top_k,
                    filter=query_filter,
                ),
            ]

            if self._sparse is not None:
                sparse_vecs = await self._sparse.embed([query])
                if sparse_vecs:
                    query_sparse = sparse_vecs[0]
                    prefetch.append(
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=query_sparse.indices,
                                values=query_sparse.values,
                            ),
                            using="sparse",
                            limit=top_k,
                            filter=query_filter,
                        )
                    )

            try:
                resp = await self._client.query_points(
                    collection_name=self._collection,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
            except Exception as exc:
                raise RetrievalError(f"Qdrant query failed: {exc}") from exc

            hits: list[RetrievalHit] = []
            for point in resp.points:
                p = point.payload or {}
                hits.append(
                    RetrievalHit(
                        doc_id=p.get("doc_id", ""),
                        version_id=p.get("version_id", ""),
                        chunk_index=p.get("chunk_index", 0),
                        chunk_text=p.get("chunk_text", ""),
                        page=p.get("page", 0),
                        section_path=p.get("section_path", []),
                        bbox=p.get("bbox", []),
                        page_bboxes=p.get("page_bboxes", []),
                        chunk_type=p.get("chunk_type", "text"),
                        score=point.score or 0.0,
                    )
                )
            logger.debug(
                "hybrid retrieval complete",
                n_hits=len(hits),
            )
            return hits
