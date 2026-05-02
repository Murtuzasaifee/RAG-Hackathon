from __future__ import annotations

from unittest.mock import patch

from rag_hackathon.observability.tracing import stage_span


def test_stage_span_creates_span_without_request_id() -> None:
    with (
        patch("rag_hackathon.observability.tracing.logfire") as mock_logfire,
        stage_span("parse", doc_id="x"),
    ):
        mock_logfire.span.assert_called_once_with("parse", doc_id="x")


def test_stage_span_injects_request_id() -> None:
    from rag_hackathon.observability.logging import request_id_var

    request_id_var.set("req-abc")
    try:
        with (
            patch("rag_hackathon.observability.tracing.logfire") as mock_logfire,
            stage_span("chunk", job_id="j1"),
        ):
            mock_logfire.span.assert_called_once_with(
                "chunk", job_id="j1", request_id="req-abc"
            )
    finally:
        request_id_var.set(None)
