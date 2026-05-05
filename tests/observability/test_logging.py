from __future__ import annotations

import json

from app.observability.logging import (
    configure_logging,
    get_logger,
    request_id_var,
)


def test_configure_logging_emits_json(capsys: object) -> None:
    configure_logging()
    logger = get_logger("test")
    logger.info("hello", key="value")

    captured = capsys.readouterr()
    line = captured.err if captured.err else captured.out
    parsed = json.loads(line.strip().split("\n")[-1])

    assert parsed["event"] == "hello"
    assert parsed["key"] == "value"
    assert "timestamp" in parsed
    assert parsed["level"] == "info"


def test_request_id_injected_when_set(capsys: object) -> None:
    configure_logging()
    request_id_var.set("req-123")
    try:
        logger = get_logger("test")
        logger.info("with_request")

        captured = capsys.readouterr()
        line = captured.err if captured.err else captured.out
        parsed = json.loads(line.strip().split("\n")[-1])

        assert parsed["request_id"] == "req-123"
        assert parsed["event"] == "with_request"
    finally:
        request_id_var.set(None)


def test_log_without_request_id_still_valid_json(capsys: object) -> None:
    configure_logging()
    logger = get_logger("test")
    logger.info("no_context")

    captured = capsys.readouterr()
    line = captured.err if captured.err else captured.out
    parsed = json.loads(line.strip().split("\n")[-1])

    assert "request_id" not in parsed
    assert parsed["event"] == "no_context"
