from __future__ import annotations

import json
from pathlib import Path

from evals.contracts import L1ComponentReport, MetricStatus


def write_report(report: L1ComponentReport, output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / report.run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    _atomic_write_json(run_dir / "run.json", report.model_dump(mode="json"))
    _atomic_write_jsonl(
        run_dir / "observations.jsonl",
        [item.model_dump(mode="json") for item in report.observations],
    )
    failures = [
        item.model_dump(mode="json")
        for item in report.observations
        if item.status is not MetricStatus.SUCCESS
    ]
    _atomic_write_jsonl(run_dir / "failures.jsonl", failures)
    slices = [
        item.model_dump(mode="json") for item in report.observations if item.scope == "slice"
    ]
    _atomic_write_jsonl(run_dir / "slices.jsonl", slices)
    _atomic_write_text(run_dir / "summary.md", _render_summary(report, failures))
    return run_dir


def _render_summary(report: L1ComponentReport, failures: list[dict]) -> str:
    lines = [
        "# Eedi-RAG L1 Component Evaluation",
        "",
        f"Release status: **{report.release_status}**",
        "",
        f"- Run: `{report.run_id}`",
        f"- Dataset: `{report.dataset_version}`",
        f"- Qrels: `{report.qrels_version}`",
        f"- System config hash: `{report.system_config_hash}`",
        f"- Modules: {', '.join(report.executed_modules)}",
        "",
        "## Status counts",
        "",
    ]
    for status, count in report.status_counts.items():
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Non-success observations", ""])
    if not failures:
        lines.append("None.")
    else:
        lines.extend(["| Stage | Metric | Scope | Status | Value | Threshold | Reason |", "| --- | --- | --- | --- | ---: | ---: | --- |"])
        for item in failures:
            value = "" if item["value"] is None else str(item["value"])
            threshold = "" if item["threshold"] is None else str(item["threshold"])
            reason = (item.get("error_message") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {item['stage']} | {item['metric_name']} | {item['scope']} | {item['status']} | {value} | {threshold} | {reason} |"
            )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_json(path: Path, payload: object) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, content)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, content)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
