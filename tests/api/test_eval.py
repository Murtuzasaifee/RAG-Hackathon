from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.app import create_app


def test_eval_run_returns_report():
    app = create_app()
    client = TestClient(app)

    mock_report = MagicMock()
    mock_report.total_questions = 2
    mock_report.failed_questions = 0
    mock_report.elapsed_seconds = 5.3
    mock_report.aggregate = {
        "faithfulness": 0.85,
        "context_precision": 0.9,
        "answer_relevancy": 0.8,
    }
    mock_report.results = [
        MagicMock(
            question="Q1?",
            scores={
                "faithfulness": 0.9,
                "context_precision": 0.95,
                "answer_relevancy": 0.85,
            },
            error=None,
        ),
        MagicMock(
            question="Q2?",
            scores={
                "faithfulness": 0.8,
                "context_precision": 0.85,
                "answer_relevancy": 0.75,
            },
            error=None,
        ),
    ]

    with patch(
        "app.api.routers.eval._run_eval_background",
        return_value=mock_report,
    ):
        resp = client.post("/eval/run")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_questions"] == 2
    assert data["failed_questions"] == 0
    assert data["elapsed_seconds"] == 5.3
    assert data["aggregate"]["faithfulness"] == 0.85
    assert len(data["per_question"]) == 2


def test_eval_run_with_failures():
    app = create_app()
    client = TestClient(app)

    mock_report = MagicMock()
    mock_report.total_questions = 1
    mock_report.failed_questions = 1
    mock_report.elapsed_seconds = 1.0
    mock_report.aggregate = {}
    mock_report.results = [
        MagicMock(
            question="Q?",
            scores={},
            error="Bifrost down",
        ),
    ]

    with patch(
        "app.api.routers.eval._run_eval_background",
        return_value=mock_report,
    ):
        resp = client.post("/eval/run")

    assert resp.status_code == 200
    data = resp.json()
    assert data["failed_questions"] == 1
    assert data["per_question"][0]["error"] == "Bifrost down"
