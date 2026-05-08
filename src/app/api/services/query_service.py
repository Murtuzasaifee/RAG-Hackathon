from __future__ import annotations

import hashlib
import json
import time

import structlog

from app.api.schemas import CitationResponse, QueryRequest, QueryResponse
from app.cache.protocols import CacheStore
from app.core.errors import GuardError
from app.generation.protocols import Generator
from app.retrieval.protocols import Reranker, Retriever
from app.security.protocols import InputGuard, OutputGuard

logger = structlog.get_logger("rag_hackathon.query_service")


def _answer_cache_key(
    query: str,
    doc_ids: list[str] | None,
    version_ids: list[str] | None,
    top_k: int,
    top_n: int,
    version_hash: str,
    epoch_map: dict[str, int],
    owner_id: str | None = None,
) -> str:
    parts = [
        query,
        json.dumps(sorted(doc_ids or [])),
        json.dumps(sorted(version_ids or [])),
        str(top_k),
        str(top_n),
        version_hash,
        json.dumps(sorted(epoch_map.items())),
        owner_id or "",
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

        logger.info(
            "query_service.initialized",
            input_guard_enabled=input_guard is not None,
            output_guard_enabled=output_guard is not None,
            cache_enabled=cache is not None,
            cache_ttl_answer=cache_ttl_answer,
        )

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
        owner_id: str | None = None,
    ) -> QueryResponse:
        timings: dict[str, int] = {}
        warnings: list[str] = []

        logger.info(
            "query.started",
            request_id=request_id,
            query_len=len(request.query),
            doc_ids=request.doc_ids,
            version_ids=request.version_ids,
            top_k=request.top_k,
            top_n=request.top_n,
            input_guard_active=self._input_guard is not None,
            output_guard_active=self._output_guard is not None,
        )

        if self._input_guard is not None:
            t0 = time.perf_counter()
            input_result = await self._input_guard.scan_input(request.query)
            timings["guard_input_ms"] = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "query.guard_input_done",
                request_id=request_id,
                is_valid=input_result.is_valid,
                score=input_result.score,
                reasons=input_result.reasons,
                duration_ms=timings["guard_input_ms"],
            )
            if not input_result.is_valid:
                logger.warning(
                    "query.guard_input_blocked",
                    request_id=request_id,
                    reasons=input_result.reasons,
                    score=input_result.score,
                    query_preview=request.query[:120],
                )
                reason_str = (
                    ", ".join(input_result.reasons)
                    if input_result.reasons
                    else "guard flagged with no scanner detail"
                )
                raise GuardError(f"Input blocked by LLM Guard — {reason_str}")
        else:
            logger.debug("query.guard_input_skipped", request_id=request_id)

        version_hash, epoch_map = await self._compute_version_context(
            request.doc_ids
        )
        logger.debug(
            "query.version_context",
            request_id=request_id,
            version_hash=version_hash,
            epoch_map=epoch_map,
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
                owner_id=owner_id,
            )
            logger.debug("query.cache_lookup", request_id=request_id, cache_key=cache_key[:16])
            cached = await self._cache.get_json("answer", cache_key)
            if cached is not None:
                logger.info(
                    "query.cache_hit",
                    request_id=request_id,
                    n_citations=len(cached.get("citations", [])),
                )
                return QueryResponse(
                    answer=cached["answer"],
                    citations=[
                        CitationResponse(**c) for c in cached["citations"]
                    ],
                    request_id=request_id,
                    timings_ms={"cache_hit": 1},
                    warnings=[],
                    cache_hit=True,
                )
            logger.debug("query.cache_miss", request_id=request_id)

        t0 = time.perf_counter()
        logger.info(
            "query.retrieve_started",
            request_id=request_id,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
            version_ids=request.version_ids,
        )
        hits = await self._retriever.retrieve(
            request.query,
            doc_ids=request.doc_ids,
            version_ids=request.version_ids,
            top_k=request.top_k,
            owner_id=owner_id,
        )
        timings["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "query.retrieve_done",
            request_id=request_id,
            n_hits=len(hits),
            duration_ms=timings["retrieve_ms"],
            top_scores=[round(h.score, 4) for h in hits[:5]],
        )

        if not hits and owner_id is not None:
            warnings.append("no_documents_found_for_your_account — ingest documents first or contact an admin")
            logger.info("query.no_hits_acl", request_id=request_id, owner_id=owner_id)

        t0 = time.perf_counter()
        logger.info(
            "query.rerank_started",
            request_id=request_id,
            n_hits=len(hits),
            top_n=request.top_n,
        )
        reranked = await self._reranker.rerank(
            request.query, hits, top_n=request.top_n
        )
        timings["rerank_ms"] = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "query.rerank_done",
            request_id=request_id,
            n_reranked=len(reranked),
            duration_ms=timings["rerank_ms"],
            top_scores=[round(h.score, 4) for h in reranked[:5]],
        )

        t0 = time.perf_counter()
        logger.info(
            "query.generate_started",
            request_id=request_id,
            n_context_chunks=len(reranked),
        )
        answer = await self._generator.generate(request.query, reranked)
        timings["generate_ms"] = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "query.generate_done",
            request_id=request_id,
            answer_chars=len(answer.text),
            n_citations=len(answer.citations),
            duration_ms=timings["generate_ms"],
        )

        if self._output_guard is not None and answer.citations:
            t0 = time.perf_counter()
            context_text = " ".join(c.chunk_text for c in answer.citations)
            logger.info(
                "query.guard_output_started",
                request_id=request_id,
                context_chars=len(context_text),
                answer_chars=len(answer.text),
                n_citation_chunks=len(answer.citations),
            )
            output_result = await self._output_guard.scan_output(
                context_text, answer.text
            )
            timings["guard_output_ms"] = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "query.guard_output_done",
                request_id=request_id,
                is_valid=output_result.is_valid,
                score=output_result.score,
                reasons=output_result.reasons,
                duration_ms=timings["guard_output_ms"],
            )
            if not output_result.is_valid:
                warnings.append("low_groundedness")
                logger.warning(
                    "query.guard_output_low_groundedness",
                    request_id=request_id,
                    score=output_result.score,
                    reasons=output_result.reasons,
                )
            if output_result.score < 0.5:
                warnings.append(f"low_groundedness_score:{output_result.score:.2f}")
        elif self._output_guard is None:
            logger.debug("query.guard_output_skipped", request_id=request_id, reason="disabled")
        else:
            logger.debug("query.guard_output_skipped", request_id=request_id, reason="no_citations")

        citations = [
            CitationResponse(
                doc_id=c.doc_id,
                version_id=c.version_id,
                page=c.page,
                section_path=list(c.section_path),
                bbox=list(c.bbox),
                page_bboxes=list(c.page_bboxes),
                chunk_text=c.chunk_text,
                chunk_type=c.chunk_type,
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
                owner_id=owner_id,
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
            logger.debug("query.cache_stored", request_id=request_id)

        total_ms = sum(timings.values())
        logger.info(
            "query.completed",
            request_id=request_id,
            n_hits=len(hits),
            n_reranked=len(reranked),
            n_citations=len(citations),
            timings_ms=timings,
            total_ms=total_ms,
            warnings=warnings,
        )

        return QueryResponse(
            answer=answer.text,
            citations=citations,
            request_id=request_id,
            timings_ms=timings,
            warnings=warnings,
        )
