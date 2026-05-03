from __future__ import annotations

import httpx
import structlog

from rag_hackathon.observability.tracing import stage_span
from rag_hackathon.security.protocols import ScanResult

logger = structlog.get_logger("rag_hackathon.security.guard")

_MAX_TOKENS_PER_SLICE = 512


def _chunk_text_for_nli(
    text: str, max_tokens: int = _MAX_TOKENS_PER_SLICE
) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i : i + max_tokens]))
    return chunks if chunks else [text]


def _parse_scanners(data: dict) -> tuple[bool, float, list[str]]:
    is_valid = data.get("is_valid", True)
    scanners = data.get("scanners", {})
    reasons: list[str] = []
    max_score = 0.0
    for name, result in scanners.items():
        if isinstance(result, dict):
            if not result.get("is_valid", True):
                reasons.append(name)
            if isinstance(result.get("score"), (int, float)) and not result.get(
                "is_valid", True
            ):
                max_score = max(max_score, float(result["score"]))
    return is_valid, max_score, reasons


class LLMGuardClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def scan_input(self, prompt: str) -> ScanResult:
        prompt_words = len(prompt.split())
        logger.info(
            "guard.input.scan_started",
            prompt_chars=len(prompt),
            prompt_words=prompt_words,
            base_url=self._base_url,
        )
        with stage_span("guard.input", prompt_len=len(prompt)):
            try:
                resp = await self._http.post(
                    "/scan/prompt",
                    json={"prompt": prompt},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning(
                    "guard.input.scan_failed_fail_open",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return ScanResult(is_valid=True, sanitized=prompt, score=0.0, reasons=[])

            is_valid, max_score, reasons = _parse_scanners(data)
            sanitized = data.get("sanitized_prompt", prompt)
            scanner_detail = {
                name: {"is_valid": res.get("is_valid"), "score": res.get("score")}
                for name, res in (data.get("scanners") or {}).items()
                if isinstance(res, dict)
            }

            logger.info(
                "guard.input.scan_complete",
                is_valid=is_valid,
                score=max_score,
                reasons=reasons,
                scanners=scanner_detail,
                sanitized_len=len(sanitized),
            )
            return ScanResult(
                is_valid=is_valid,
                sanitized=sanitized,
                score=max_score,
                reasons=reasons,
            )

    async def scan_output(self, prompt: str, output: str) -> ScanResult:
        context_chunks = _chunk_text_for_nli(prompt)
        logger.info(
            "guard.output.scan_started",
            context_chars=len(prompt),
            context_words=len(prompt.split()),
            output_chars=len(output),
            output_words=len(output.split()),
            n_slices=len(context_chunks),
        )
        with stage_span("guard.output", output_len=len(output)):
            try:
                slice_results = []
                for i, chunk in enumerate(context_chunks):
                    logger.debug(
                        "guard.output.slice_started",
                        slice=i + 1,
                        total_slices=len(context_chunks),
                        slice_words=len(chunk.split()),
                    )
                    result = await self._scan_output_slice(chunk, output)
                    logger.debug(
                        "guard.output.slice_done",
                        slice=i + 1,
                        total_slices=len(context_chunks),
                        is_valid=result.is_valid,
                        score=result.score,
                        reasons=result.reasons,
                    )
                    slice_results.append(result)
            except Exception as exc:
                logger.warning(
                    "guard.output.scan_failed_fail_open",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return ScanResult(is_valid=True, sanitized=output, score=0.0, reasons=[])

            all_valid = all(r.is_valid for r in slice_results)
            best_score = max((r.score for r in slice_results), default=0.0)
            all_reasons: list[str] = []
            sanitized = output
            for r in slice_results:
                all_reasons.extend(r.reasons)
                sanitized = r.sanitized
            reasons = list(dict.fromkeys(all_reasons))

            logger.info(
                "guard.output.scan_complete",
                is_valid=all_valid,
                score=best_score,
                n_slices=len(slice_results),
                reasons=reasons,
                groundedness_pass=all_valid,
            )
            return ScanResult(
                is_valid=all_valid,
                sanitized=sanitized,
                score=best_score,
                reasons=reasons,
            )

    async def _scan_output_slice(
        self, context_slice: str, output: str
    ) -> ScanResult:
        try:
            resp = await self._http.post(
                "/scan/output",
                json={"prompt": context_slice, "output": output},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "guard.output_slice.scan_failed_fail_open",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return ScanResult(is_valid=True, sanitized=output, score=0.0, reasons=[])

        is_valid, max_score, reasons = _parse_scanners(data)
        sanitized = data.get("sanitized_output", output)

        return ScanResult(
            is_valid=is_valid,
            sanitized=sanitized,
            score=max_score,
            reasons=reasons,
        )
