from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from evals.contracts import L1ComponentSuiteSpec
from evals.reporting import write_report
from evals.suite import run_component_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Eedi-RAG L1 component evaluation")
    parser.add_argument("--input", required=True, type=Path, help="strict l1-component-suite/v1 JSON")
    parser.add_argument("--output-root", default=Path("reports/eval"), type=Path)
    args = parser.parse_args(argv)

    try:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        spec = L1ComponentSuiteSpec.model_validate(raw)
        report = run_component_suite(spec)
        run_dir = write_report(report, args.output_root)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        parser.exit(1, f"L1 evaluation ERROR: {type(exc).__name__}: {exc}\n")

    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "release_status": report.release_status,
                "status_counts": report.status_counts,
                "report_dir": str(run_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.release_status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

