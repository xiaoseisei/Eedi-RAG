from __future__ import annotations

"""Build a provenance-aware L1/L2 before-and-after comparison report."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _metric_map(report: dict[str, Any], prefix: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if prefix == "l1":
        rows = report.get("aggregate_metrics", [])
        if not rows:
            rows = [
                row
                for row in report.get("observations", [])
                if row.get("scope") == "aggregate"
            ]
        for row in rows:
            stage = str(row.get("stage", ""))
            name = str(row.get("metric_name", ""))
            value = row.get("value")
            if stage and name and value is not None:
                result[f"l1.{stage}.{name}"] = {
                    "value": value,
                    "status": row.get("status"),
                    "threshold": row.get("threshold"),
                }
    elif prefix == "l2":
        for row in report.get("metric_aggregates", []):
            track = str(row.get("track", ""))
            metric = str(row.get("metric", ""))
            if track and metric and row.get("mean") is not None:
                result[f"l2.{track}.{metric}"] = {
                    "value": row.get("mean"),
                    "status": "MEASURED",
                    "threshold": row.get("threshold"),
                }
    return result


def _extract_baseline_maps(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "l1" in baseline or "l2" in baseline:
        return {
            **_metric_map(baseline.get("l1", {}), "l1"),
            **_metric_map(baseline.get("l2", {}), "l2"),
        }
    return {
        **_metric_map(baseline, "l1"),
        **_metric_map(baseline, "l2"),
    }


def _delta(current: Any, previous: Any) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)):
        return None
    return round(float(current) - float(previous), 6)


def _relative_delta_percent(current: Any, previous: Any) -> float | None:
    delta = _delta(current, previous)
    if delta is None or float(previous) == 0.0:
        return None
    return round(delta / abs(float(previous)) * 100.0, 2)


def build_comparison_report(
    *,
    current_l1: Path,
    current_l2: Path,
    baseline: Path,
    output_dir: Path,
    timing_report: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite comparison report: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    l1 = _load(current_l1)
    l2 = _load(current_l2)
    baseline_data = _load(baseline)
    current_metrics = {**_metric_map(l1, "l1"), **_metric_map(l2, "l2")}
    baseline_metrics = _extract_baseline_maps(baseline_data)
    metrics: dict[str, dict[str, Any]] = {}
    for key in sorted(set(current_metrics) | set(baseline_metrics)):
        current = current_metrics.get(key, {})
        previous = baseline_metrics.get(key, {})
        current_value = current.get("value")
        previous_value = previous.get("value")
        metrics[key] = {
            "current": current_value,
            "baseline": previous_value,
            "delta": _delta(current_value, previous_value),
            "relative_delta_percent": _relative_delta_percent(
                current_value, previous_value
            ),
            "current_status": current.get("status", "UNMEASURED"),
            "baseline_status": previous.get("status", "UNMEASURED"),
            "threshold": current.get("threshold", previous.get("threshold")),
        }

    timing = _load(timing_report) if timing_report is not None else None
    report = {
        "schema_version": "l123-comparison-report/v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "comparison_policy": {
            "strict_same_sample": False,
            "note": "L1 and L2 snapshots can differ in dataset, route, token budget, and case count; deltas are directional unless these fields match.",
        },
        "provenance": {
            "current_l1_path": str(current_l1.resolve()),
            "current_l1_sha256": _sha256(current_l1),
            "current_l1_run_id": l1.get("run_id"),
            "current_l1_dataset_version": l1.get("dataset_version"),
            "current_l2_path": str(current_l2.resolve()),
            "current_l2_sha256": _sha256(current_l2),
            "current_l2_run_id": l2.get("run_id"),
            "current_l2_dataset_version": (l2.get("dataset") or {}).get("dataset_version"),
            "baseline_path": str(baseline.resolve()),
            "baseline_sha256": _sha256(baseline),
            "baseline_run_id": baseline_data.get("run_id"),
        },
        "runtime": {
            "current_l1": l1.get("system_config", {}),
            "current_l2": l2.get("runtime_route", {}),
            "timing_report": timing.get("summary") if timing else None,
        },
        "metrics": metrics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Eedi-RAG L1 + L2 Before/After Comparison",
        "",
        "> This report separates measured values from directional deltas. It does not convert different samples or routes into a strict A/B experiment.",
        "",
        "## Provenance",
        "",
        f"- Current L1: `{l1.get('run_id')}`",
        f"- Current L2: `{l2.get('run_id')}`",
        f"- Baseline: `{baseline_data.get('run_id')}`",
        f"- Strict same-sample comparison: `{report['comparison_policy']['strict_same_sample']}`",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Baseline | Current | Delta | Relative | Threshold | Current status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, value in metrics.items():
        relative = (
            "—"
            if value["relative_delta_percent"] is None
            else f"{value['relative_delta_percent']}%"
        )
        lines.append(
            f"| `{key}` | {value['baseline']} | {value['current']} | {value['delta']} | "
            f"{relative} | {value['threshold']} | "
            f"{value['current_status']} |"
        )
    lines.extend(["", "## Timing", ""])
    if timing is None:
        lines.append("- Timing report: `UNMEASURED`; pass `--timing-report` to include a fresh local profiling run.")
    else:
        lines.append(f"- Scope: **{timing.get('measurement_scope', 'unknown')}**")
        lines.extend([
            "",
            "| Stage | Mean (s) | P50 (s) | P95 (s) | Max (s) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for stage, values in (timing.get("summary", {}).get("stages", {}) or {}).items():
            lines.append(
                f"| `{stage}` | {values.get('mean_seconds')} | {values.get('p50_seconds')} | "
                f"{values.get('p95_seconds')} | {values.get('max_seconds')} |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-l1", type=Path, required=True)
    parser.add_argument("--current-l2", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--timing-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_comparison_report(
        current_l1=args.current_l1,
        current_l2=args.current_l2,
        baseline=args.baseline,
        output_dir=args.output_dir,
        timing_report=args.timing_report,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "metric_count": len(report["metrics"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
