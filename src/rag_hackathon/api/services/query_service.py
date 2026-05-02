from __future__ import annotations

import hashlib
import json
import time

import structlog

from rag_hackathon.api.schemas import CitationResponse, QueryRequest, QueryResponse
from rag_hackathon.cache.protocols import CacheStore
from rag_hackathon.core.errors import GuardError
from rag_hackathon.generation.protocols import Generator
from rag_hackathon.retrieval.protocols import Reranker, Retriever
from rag_hackathon.security.protocols import InputGuard, OutputGuard

logger = structlog.get_logger("rag_hackathon.query_service")


def _answer_cache_key(
    query: str,
    doc_ids: list[str] | None,
    version_ids: list[str] | None,
    top_k: int,
    top_n: int,
    version_hash: str,
    epoch_map: dict[str, int],
) -> str:
    parts = [
        query,
        json.dumps(sorted(doc_ids or [])),
        json.dumps(sorted(version_ids or [])),
        str(top_k),
        str(top_n),
        version_hash,
        json.dumps(sorted(epoch_map.items())),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class QueryService:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
        cache: CacheStore | None = None,
        cache_ttl_answer: int = 3600,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._cache = cache
        self._cache_ttl_answer = cache_ttl_answer

    async def _compute_version_context(
        self, doc_ids: list[str] | None
    ) -> tuple[str, dict[str, int]]:
        if self._cache is None:
            return "", {}
        version_hash = "no_filter"
        epoch_map: dict[str, int] = {}
        if doc_ids:
            for did in doc_ids:
                epoch_raw = await self._cache.get(
                    "doc_epoch", did
                )
                epoch_map[did] = int(epoch_raw) if epoch_raw else 0
        else:
            epoch_map = {}
        active_key = hashlib.sha256(
            json.dumps(sorted(epoch_map.items())).encode()
        ).hexdigest()[:16]
        version_hash = active_key
        return version_hash, epoch_map

    async def run(
        self,
        request: QueryRequest,
        request_id: str = "",
    ) -> QueryResponse:
        timings: dict[str, int] = {}
        warnings: list[str] = []

        if self._input_guard is not None:
            t0 = time.perf_counter()
            input_result = await self._input_guard.scan_input(request.query)
            timings["guard_input_ms"] = int(
                (time.perf_counter() - t0) * 1000
            )
            if not input_result.is_valid:
                raise GuardError(
                    f"Input blocked: {', '.join(input_result.reasons)}"
                )

        version_hash, epoch_map = await self._compute_version_context(
            request.doc_ids
        )

        if self._cache is not None:
            cache_key = _answer_cache_key(
                query=request.query,
                doc_ids=request.doc_ids,
                version_ids=request.version_ids,
                top_k=request.top_k,
                top_n=request.top_n,
                version_hash=version_hash,
                epoch_map=epoch_map,
            )
            cached = await self._cache.get_json("answer", cache_key)
            if cached is not None:
                logger.info(
                    "query answered from cache",
                    request_id=request_id,
                )
                return QueryResponse(
                    answer=cached["answer"],
                    citations=[
                        CitationResponse(**c) for c in cached["citations"]
                    ],
                    request_id=request_id,
                    timings_ms={"cache_hit": 1},
                    warnings=[],
                )

        t0 = time.perf_counter()
        hits = await self._retriever.retrieve(
            request.query,
            doc_ids=request.doc_ids,
            version_ids=request.version_ids,
            top_k=request.top_k,
        )
        timings["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        reranked = await self._reranker.rerank(
            request.query, hits, top_n=request.top_n
        )
        timings["rerank_ms"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        answer = await self._generator.generate(request.query, reranked)
        timings["generate_ms"] = int((time.perf_counter() - t0) * 1000)

        if self._output_guard is not None and answer.citations:
            t0 = time.perf_counter()
            context_text = " ".join(c.chunk_text for c in answer.citations)
            output_result = await self._output_guard.scan_output(
                context_text, answer.text
            )
            timings["guard_output_ms"] = int(
                (time.perf_counter() - t0) * 1000
            )
            if not output_result.is_valid:
                warnings.append("low_groundedness")
            if output_result.score < 0.5:
                warnings.append(f"low_groundedness_score:{output_result.score:.2f}")

        citations = [
            CitationResponse(
                doc_id=c.doc_id,
                version_id=c.version_id,
                page=c.page,
                section_path=list(c.section_path),
                bbox=list(c.bbox),
                chunk_text=c.chunk_text,
                score=c.score,
            )
            for c in answer.citations
        ]

        if self._cache is not None:
            cache_key = _answer_cache_key(
                query=request.query,
                doc_ids=request.doc_ids,
                version_ids=request.version_ids,
                top_k=request.top_k,
                top_n=request.top_n,
                version_hash=version_hash,
                epoch_map=epoch_map,
            )
            await self._cache.set_json(
                "answer",
                cache_key,
                {
                    "answer": answer.text,
                    "citations": [c.model_dump() for c in citations],
                },
                self._cache_ttl_answer,
            )

        logger.info(
            "query completed",
            request_id=request_id,
            n_hits=len(hits),
            n_reranked=len(reranked),
            n_citations=len(citations),
            timings_ms=timings,
            warnings=warnings,
        )

        return QueryResponse(
            answer=answer.text,
            citations=citations,
            request_id=request_id,
            timings_ms=timings,
            warnings=warnings,
        )
