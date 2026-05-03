from __future__ import annotations

import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_hackathon.eval.ragas_runner import (
    EvalRecord,
    EvalReport,
    EvalResult,
    load_golden_set,
)

GOLDEN_SET_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "src" / "rag_hackathon" / "eval" / "golden_set.json"
)


class TestLoadGoldenSet:
    def test_loads_default_golden_set(self):
        records = load_golden_set()
        assert len(records) >= 5
        assert all(isinstance(r, EvalRecord) for r in records)
        assert all(r.question for r in records)
        assert all(r.expected_answer for r in records)

    def test_loads_custom_golden_set(self, tmp_path):
        data = [
            {
                "question": "What is X?",
                "expected_answer": "X is Y.",
                "doc_ids": ["doc1"],
                "expected_chunks": [],
            }
        ]
        p = tmp_path / "test_set.json"
        p.write_text(json.dumps(data))
        records = load_golden_set(p)
        assert len(records) == 1
        assert records[0].question == "What is X?"
        assert records[0].doc_ids == ["doc1"]

    def test_missing_doc_ids_defaults_empty(self, tmp_path):
        data = [{"question": "Q?", "expected_answer": "A."}]
        p = tmp_path / "test_set.json"
        p.write_text(json.dumps(data))
        records = load_golden_set(p)
        assert records[0].doc_ids == []


class TestEvalReport:
    def test_aggregate_computed(self):
        report = EvalReport(
            total_questions=2,
            results=[
                EvalResult(
                    question="Q1",
                    answer="A1",
                    contexts=["c1"],
                    scores={
                        "faithfulness": 0.9,
                        "context_precision": 0.8,
                        "answer_relevancy": 0.7,
                    },
                ),
                EvalResult(
                    question="Q2",
                    answer="A2",
                    contexts=["c2"],
                    scores={
                        "faithfulness": 0.5,
                        "context_precision": 0.6,
                        "answer_relevancy": 0.4,
                    },
                ),
            ],
        )
        for metric_name in (
            "faithfulness",
            "context_precision",
            "answer_relevancy",
        ):
            values = [
                r.scores[metric_name]
                for r in report.results
                if metric_name in r.scores
            ]
            if values:
                report.aggregate[metric_name] = round(
                    sum(values) / len(values), 4
                )

        assert report.aggregate["faithfulness"] == 0.7
        assert report.aggregate["context_precision"] == 0.7
        assert report.aggregate["answer_relevancy"] == 0.55


class TestRunEval:
    @pytest.mark.asyncio
    async def test_run_eval_queries_service(self):
        mock_qs = AsyncMock()
        mock_qs.run.return_value = MagicMock(
            answer="Test answer",
            citations=[MagicMock(chunk_text="chunk text")],
        )

        mock_llm_factory = MagicMock()
        mock_ragas_llm = MagicMock()
        mock_llm_factory.return_value = mock_ragas_llm

        mock_metric_instance = AsyncMock()
        mock_metric_instance.single_turn_ascore.return_value = 0.85
        mock_metric_instance.init = MagicMock()

        with patch(
            "rag_hackathon.eval.ragas_runner.load_golden_set"
        ) as mock_load, patch(
            "rag_hackathon.eval.ragas_runner.get_settings"
        ) as mock_settings, patch(
            "ragas.llms.llm_factory", mock_llm_factory
        ), patch(
            "ragas.metrics.Faithfulness",
            return_value=mock_metric_instance,
        ), patch(
            "ragas.metrics.ContextPrecision",
            return_value=mock_metric_instance,
        ), patch(
            "ragas.metrics.AnswerRelevancy",
            return_value=mock_metric_instance,
        ):
            mock_load.return_value = [
                EvalRecord(
                    question="Test Q?",
                    expected_answer="Test A.",
                    doc_ids=[],
                    expected_chunks=[],
                )
            ]
            mock_settings.return_value = MagicMock(
                llm_model="gpt-4o",
                bifrost_url="http://localhost:8080",
                openai_api_key="test-key",
            )

            from rag_hackathon.eval.ragas_runner import run_eval

            report = await run_eval(mock_qs)

        assert report.total_questions == 1
        assert report.failed_questions == 0
        assert report.results[0].answer == "Test answer"
        assert report.results[0].contexts == ["chunk text"]
        mock_qs.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_eval_handles_query_failure(self):
        mock_qs = AsyncMock()
        mock_qs.run.side_effect = RuntimeError("Bifrost down")

        with patch(
            "rag_hackathon.eval.ragas_runner.load_golden_set"
        ) as mock_load:
            mock_load.return_value = [
                EvalRecord(
                    question="Q?",
                    expected_answer="A.",
                    doc_ids=[],
                    expected_chunks=[],
                )
            ]

            from rag_hackathon.eval.ragas_runner import run_eval

            report = await run_eval(mock_qs)

        assert report.total_questions == 1
        assert report.failed_questions == 1
        assert report.results[0].error == "Bifrost down"
        assert len(report.aggregate) == 0
