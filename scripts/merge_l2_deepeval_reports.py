from __future__ import annotations

"""Merge independently executed L2 DeepEval shards into one audited report."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def merge_reports(report_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    if not report_dirs:
        raise ValueError("at least one shard report directory is required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    reports: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    seen_cases: set[int] = set()
    expected_hash: str | None = None
    expected_route: dict[str, Any] | None = None
    expected_judge: dict[str, Any] | None = None
    for report_dir in report_dirs:
        report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        shard_traces = _read_jsonl(report_dir / "traces.jsonl")
        shard_metrics = _read_jsonl(report_dir / "metrics.jsonl")
        shard = report.get("case_shard", {})
        case_ids = {
            int(str(row["case_id"]).rsplit("-", 1)[-1]) for row in shard_traces
        }
        if not case_ids:
            raise ValueError(f"shard has no traces: {report_dir}")
        if seen_cases & case_ids:
            raise ValueError(f"overlapping case IDs across shards: {sorted(seen_cases & case_ids)}")
        tracks_by_case: dict[int, set[str]] = {}
        for row in shard_traces:
            case_index = int(str(row["case_id"]).rsplit("-", 1)[-1])
            tracks_by_case.setdefault(case_index, set()).add(str(row.get("track", "")))
        incomplete = [case for case, track_set in tracks_by_case.items() if track_set != {"gold", "real"}]
        if incomplete:
            raise ValueError(f"shard has incomplete Gold/Real pairs: {incomplete}")
        artifact_hash = str(report.get("artifact_hash_before", ""))
        if expected_hash is None:
            expected_hash = artifact_hash
        elif artifact_hash != expected_hash:
            raise ValueError("shards use different source artifact hashes")
        route = report.get("runtime_route", {})
        if expected_route is None:
            expected_route = route
        elif route != expected_route:
            raise ValueError("shards use different runtime routes")
        judge = report.get("judge_readiness", {})
        if expected_judge is None:
            expected_judge = judge
        elif judge.get("status") != expected_judge.get("status"):
            raise ValueError("shards have inconsistent DeepEval judge readiness")
        seen_cases.update(case_ids)
        reports.append(report)
        traces.extend(shard_traces)
        metrics.extend(shard_metrics)

    trace_cases = sorted(seen_cases)
    case_count = len(trace_cases)
    metric_aggregates: list[dict[str, Any]] = []
    for track in ("gold", "real"):
        metric_names = sorted({str(row["metric"]) for row in metrics if row.get("track") == track})
        for metric in metric_names:
            values = [
                float(row["score"])
                for row in metrics
                if row.get("track") == track
                and row.get("metric") == metric
                and row.get("status") == "SUCCESS"
                and row.get("score") is not None
            ]
            thresholds = {
                float(row["threshold"])
                for row in metrics
                if row.get("track") == track
                and row.get("metric") == metric
                and row.get("threshold") is not None
            }
            if len(thresholds) > 1:
                raise ValueError(f"metric {track}/{metric} has inconsistent thresholds")
            metric_aggregates.append(
                {
                    "track": track,
                    "metric": metric,
                    "measured": len(values),
                    "mean": statistics.mean(values) if values else None,
                    "threshold": next(iter(thresholds), None),
                }
            )
    errors = [row for row in traces if row.get("status") == "ERROR"] + [
        row for row in metrics if row.get("status") == "ERROR"
    ]
    judge_ready = bool(expected_judge and expected_judge.get("status") == "SUCCESS")
    all_thresholds_pass = bool(metric_aggregates) and all(
        item["mean"] is not None
        and item["threshold"] is not None
        and item["mean"] >= item["threshold"]
        for item in metric_aggregates
    )
    l1_precondition = reports[0].get("l1_precondition")
    if l1_precondition != "GO_TO_L2_TECHNICAL":
        release_decision = "BLOCKED_L1_PRECONDITION"
    elif not judge_ready:
        release_decision = "BLOCKED_JUDGE_UNMEASURED"
    elif errors:
        release_decision = "BLOCKED_ERRORS"
    elif not all_thresholds_pass:
        release_decision = "BLOCKED_METRIC_GATE"
    else:
        release_decision = "L2_PASSED"

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "l2-deepeval-report/v1",
        "run_id": output_dir.name,
        "shards": [str(path) for path in report_dirs],
        "case_shard": {"case_count": case_count, "case_ids": trace_cases},
        "artifact_hash_before": expected_hash,
        "artifact_hash_after": expected_hash,
        "artifact_mutated": False,
        "runtime_route": expected_route,
        "l1_precondition": l1_precondition,
        "judge_readiness": expected_judge,
        "tracks": {
            "gold_case_count": sum(row.get("track") == "gold" for row in traces),
            "real_case_count": sum(row.get("track") == "real" for row in traces),
        },
        "metric_aggregates": metric_aggregates,
        "trace_status_counts": {
            status: sum(row.get("status") == status for row in traces)
            for status in ("SUCCESS", "ERROR", "UNMEASURED")
        },
        "errors": errors,
        "release_decision": release_decision,
        "gold_real_interpretation": "Gold high/Real low indicates upstream context loss; both low indicates generator or data issue; interpret only when judge results are measured.",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "traces.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in traces),
        encoding="utf-8",
    )
    (output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metrics),
        encoding="utf-8",
    )
    lines = [
        "# L2 DeepEval Shard-Merged Report",
        "",
        f"Release decision: **{release_decision}**",
        f"Cases: `{case_count}`; Gold traces: `{report['tracks']['gold_case_count']}`; Real traces: `{report['tracks']['real_case_count']}`",
        f"Source artifact unchanged: `{report['artifact_mutated'] is False}`",
        "",
        "| Track | Metric | Measured | Mean | Threshold |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in metric_aggregates:
        lines.append(
            f"| {item['track']} | {item['metric']} | {item['measured']} | {item['mean']} | {item['threshold']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = merge_reports(
        [path.resolve(strict=True) for path in args.shard],
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "release_decision": report["release_decision"],
                "case_count": report["case_shard"]["case_count"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["release_decision"] == "L2_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
