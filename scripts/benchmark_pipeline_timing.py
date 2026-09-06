from __future__ import annotations

"""Measure end-to-end pipeline stages on the local deterministic route."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * percentile)))
    return round(ordered[index], 6)


def summarize_timings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate profiling values without treating missing stages as zero."""

    stage_names = sorted(
        {
            str(stage)
            for row in rows
            for stage in (row.get("profiling") or {})
            if isinstance((row.get("profiling") or {}).get(stage), (int, float))
        }
    )
    stages: dict[str, dict[str, Any]] = {}
    for stage in stage_names:
        values = [
            float((row.get("profiling") or {})[stage])
            for row in rows
            if isinstance((row.get("profiling") or {}).get(stage), (int, float))
        ]
        stages[stage] = {
            "measured": len(values),
            "mean_seconds": round(statistics.fmean(values), 6) if values else None,
            "p50_seconds": _percentile(values, 0.50),
            "p95_seconds": _percentile(values, 0.95),
            "max_seconds": round(max(values), 6) if values else None,
        }
    return {
        "case_count": len(rows),
        "success_count": sum(row.get("status") == "SUCCESS" for row in rows),
        "error_count": sum(row.get("status") == "ERROR" for row in rows),
        "stages": stages,
    }


def run_benchmark(
    *,
    db_path: Path,
    chroma_path: Path,
    golden_path: Path,
    max_cases: int,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite timing report: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
    from src.reranker import PedagogicalGoldAssembler
    from src.retriever import DualMetricRetriever
    from src.storage_manager import DualEngineStorageManager

    golden = json.loads(golden_path.resolve(strict=True).read_text(encoding="utf-8"))
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    selected = golden[:max_cases]
    storage = DualEngineStorageManager(
        db_path=db_path.resolve(strict=True),
        chroma_dir=chroma_path.resolve(strict=True),
        embedding_backend="deterministic",
    )
    try:
        retriever = DualMetricRetriever(
            storage_manager=storage,
            query_rewrite_mode="deterministic",
            retrieval_mode="bm25_dense",
            bm25_weight=0.35,
        )
        assembler = PedagogicalGoldAssembler(
            max_prompt_tokens=2000,
            rerank_unit="card",
            evidence_selection_count=5,
        )
        pipeline = EndToEndPedagogicalRAGPipeline(
            retriever=retriever,
            assembler=assembler,
            chunk_strategy="card",
            generator_contract_version="v2",
        )
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(selected, start=1):
            started = time.perf_counter()
            try:
                response = pipeline.ask(case["question"], mode="deterministic")
                rows.append({
                    "case_index": index,
                    "status": "SUCCESS",
                    "profiling": dict(response._profiling),
                    "total_wall_seconds": round(time.perf_counter() - started, 6),
                    "audit_status": response.audit_status,
                    "citation_count": len(response.dialogue_citations),
                })
            except Exception as exc:
                rows.append({
                    "case_index": index,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "total_wall_seconds": round(time.perf_counter() - started, 6),
                })
    finally:
        storage.close()

    report = {
        "schema_version": "pipeline-timing-report/v1",
        "measurement_scope": "local deterministic generator; not external LLM production latency",
        "route": {
            "embedding_backend": "deterministic",
            "retrieval_mode": "bm25_dense",
            "bm25_weight": 0.35,
            "rerank_unit": "card",
            "generator_contract": "v2",
            "chunk_strategy": "card",
            "max_prompt_tokens": 2000,
        },
        "sources": {
            "db": str(db_path.resolve()),
            "chroma": str(chroma_path.resolve()),
            "golden": str(golden_path.resolve()),
        },
        "summary": summarize_timings(rows),
        "cases": rows,
    }
    (output_dir / "timing.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Pipeline Timing Report",
        "",
        f"Scope: **{report['measurement_scope']}**",
        "",
        f"- Cases: `{report['summary']['case_count']}`",
        f"- Success: `{report['summary']['success_count']}`",
        f"- Errors: `{report['summary']['error_count']}`",
        "",
        "| Stage | Measured | Mean (s) | P50 (s) | P95 (s) | Max (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, values in report["summary"]["stages"].items():
        lines.append(
            f"| {stage} | {values['measured']} | {values['mean_seconds']} | "
            f"{values['p50_seconds']} | {values['p95_seconds']} | {values['max_seconds']} |"
        )
    (output_dir / "timing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--golden", type=Path, default=Path("data/golden_test_set.json"))
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        db_path=args.db,
        chroma_path=args.chroma,
        golden_path=args.golden,
        max_cases=args.max_cases,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
