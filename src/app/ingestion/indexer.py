from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient, models

from app.core.errors import IngestionError
from app.core.types import Chunk, make_point_id
from app.ingestion.embedders.protocols import SparseVector
from app.observability.tracing import stage_span

logger = structlog.get_logger("app.ingestion.indexer")


class QdrantIndexer:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if exists:
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )

    async def upsert(
        self,
        chunks: list[Chunk],
        vectors_dense: list[list[float]],
        vectors_sparse: list[SparseVector] | None,
        doc_id: str,
        version_id: str,
    ) -> None:
        if not chunks:
            return
        with stage_span(
            "index",
            doc_id=doc_id,
            version_id=version_id,
            n_chunks=len(chunks),
        ):
            points: list[models.PointStruct] = []
            for i, chunk in enumerate(chunks):
                point_id = make_point_id(doc_id, version_id, chunk.chunk_index)
                vectors: dict[str, models.Vector] = {
                    "dense": vectors_dense[i],
                }
                if vectors_sparse is not None:
                    vectors["sparse"] = models.SparseVector(
                        indices=vectors_sparse[i].indices,
                        values=vectors_sparse[i].values,
                    )
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vectors,
                        payload={
                            "doc_id": doc_id,
                            "version_id": version_id,
                            "active": True,
                            "page": chunk.page,
                            "section_path": chunk.section_path,
                            "bbox": chunk.bbox,
                            "page_bboxes": chunk.page_bboxes,
                            "chunk_text": chunk.text,
                            "chunk_type": chunk.chunk_type,
                            "chunk_index": chunk.chunk_index,
                        },
                    )
                )

            batch_size = 64
            n_batches = (len(points) + batch_size - 1) // batch_size
            logger.info("upsert_started", doc_id=doc_id, version_id=version_id, total_points=len(points), n_batches=n_batches)
            for offset in range(0, len(points), batch_size):
                batch = points[offset : offset + batch_size]
                batch_num = offset // batch_size + 1
                try:
                    await self._client.upsert(
                        collection_name=self._collection,
                        points=batch,
                    )
                    logger.debug("upsert_batch_done", doc_id=doc_id, batch=batch_num, n_batches=n_batches, size=len(batch))
                except Exception as exc:
                    raise IngestionError(
                        f"Qdrant upsert failed at batch {batch_num}: {exc}"
                    ) from exc
            logger.info("upsert_complete", doc_id=doc_id, version_id=version_id, total_points=len(points))
