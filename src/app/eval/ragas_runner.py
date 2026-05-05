from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field

import structlog

from app.api.schemas import QueryRequest
from app.api.services.query_service import QueryService
from app.core.settings import get_settings

logger = structlog.get_logger("app.eval")

GOLDEN_SET_PATH = pathlib.Path(__file__).parent / "golden_set.json"
RESULTS_DIR = pathlib.Path("eval-results")


@dataclass
class EvalRecord:
    question: str
    expected_answer: str
    doc_ids: list[str]
    expected_chunks: list[str]


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    total_questions: int = 0
    failed_questions: int = 0
    elapsed_seconds: float = 0.0


def load_golden_set(path: pathlib.Path | None = None) -> list[EvalRecord]:
    path = path or GOLDEN_SET_PATH
    with open(path) as f:
        raw = json.load(f)
    records = []
    for item in raw:
        records.append(EvalRecord(
            question=item["question"],
            expected_answer=item["expected_answer"],
            doc_ids=item.get("doc_ids", []),
            expected_chunks=item.get("expected_chunks", []),
        ))
    return records


async def run_eval(
    query_service: QueryService,
    golden_set_path: pathlib.Path | None = None,
) -> EvalReport:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    records = load_golden_set(golden_set_path)
    report = EvalReport(total_questions=len(records))
    t0 = time.perf_counter()

    results: list[EvalResult] = []
    for record in records:
        try:
            req = QueryRequest(
                query=record.question,
                doc_ids=record.doc_ids or None,
            )
            resp = await query_service.run(req)
            results.append(EvalResult(
                question=record.question,
                answer=resp.answer,
                contexts=[c.chunk_text for c in resp.citations],
            ))
        except Exception as exc:
            logger.error("eval_query_failed", question=record.question, error=str(exc))
            results.append(EvalResult(
                question=record.question,
                answer="",
                contexts=[],
                error=str(exc),
            ))
            report.failed_questions += 1

    valid_pairs = [
        (r, rec) for r, rec in zip(results, records, strict=False)
        if r.error is None
    ]

    if valid_pairs:
        settings = get_settings()

        from ragas import RunConfig
        from ragas.llms import llm_factory

        run_config = RunConfig(timeout=120, max_retries=3, max_workers=4)

        ragas_llm = llm_factory(
            model=settings.llm_model,
            run_config=run_config,
            base_url=f"{settings.bifrost_url}/openai",
        )

        metrics = {
            "faithfulness": Faithfulness(llm=ragas_llm),
            "context_precision": ContextPrecision(llm=ragas_llm),
            "answer_relevancy": AnswerRelevancy(llm=ragas_llm),
        }

        for _, metric in metrics.items():
            metric.init(run_config)

        for result, record in valid_pairs:
            sample = SingleTurnSample(
                user_input=record.question,
                response=result.answer,
                retrieved_contexts=result.contexts,
                reference=record.expected_answer,
            )
            for metric_name, metric in metrics.items():
                try:
                    score = await metric.single_turn_ascore(sample)
                    result.scores[metric_name] = round(score, 4)
                except Exception as exc:
                    logger.error(
                        "eval_metric_failed",
                        metric=metric_name,
                        question=record.question,
                        error=str(exc),
                    )
                    result.scores[metric_name] = 0.0

    report.results = results

    for metric_name in ("faithfulness", "context_precision", "answer_relevancy"):
        values = [
            r.scores[metric_name]
            for r in results
            if metric_name in r.scores
        ]
        if values:
            report.aggregate[metric_name] = round(sum(values) / len(values), 4)

    report.elapsed_seconds = round(time.perf_counter() - t0, 2)

    _write_results(report)

    return report


def _write_results(report: EvalReport) -> pathlib.Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "aggregate": report.aggregate,
        "total_questions": report.total_questions,
        "failed_questions": report.failed_questions,
        "elapsed_seconds": report.elapsed_seconds,
        "per_question": [
            {
                "question": r.question,
                "answer": r.answer[:200] + "..." if len(r.answer) > 200 else r.answer,
                "n_contexts": len(r.contexts),
                "scores": r.scores,
                "error": r.error,
            }
            for r in report.results
        ],
    }

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    lines = [
        f"# Eval Report — {timestamp}",
        "",
        f"**Questions:** {report.total_questions}",
        f"**Failed:** {report.failed_questions}",
        f"**Elapsed:** {report.elapsed_seconds}s",
        "",
        "## Aggregate Scores",
        "",
    ]
    for name, val in report.aggregate.items():
        lines.append(f"- **{name}**: {val:.4f}")
    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    for r in report.results:
        status = "FAIL" if r.error else "OK"
        lines.append(f"### [{status}] {r.question}")
        if r.error:
            lines.append(f"Error: {r.error}")
        else:
            for name, val in r.scores.items():
                lines.append(f"- {name}: {val:.4f}")
        lines.append("")

    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("eval_results_written", path=str(out_dir))
    return out_dir
