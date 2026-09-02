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
    return run_dir


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

