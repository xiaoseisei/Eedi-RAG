from __future__ import annotations

import json
from pathlib import Path


def test_l123_comparison_report_computes_deltas_and_preserves_provenance(tmp_path: Path) -> None:
    from scripts.build_l123_comparison_report import build_comparison_report

    l1 = tmp_path / "l1.json"
    l1.write_text(json.dumps({
        "run_id": "l1-new",
        "dataset_version": "l1-v2",
        "system_config_hash": "a" * 64,
        "release_status": "BLOCKED",
        "aggregate_metrics": [
            {"stage": "grounding", "metric_name": "citation_recall", "value": 0.8, "status": "FAILED"},
        ],
    }), encoding="utf-8")
    l2 = tmp_path / "l2.json"
    l2.write_text(json.dumps({
        "run_id": "l2-new",
        "schema_version": "l2-deepeval-report/v1",
        "artifact_hash_before": "a" * 64,
        "release_decision": "BLOCKED_L1_PRECONDITION",
        "metric_aggregates": [
            {"track": "real", "metric": "contextual_recall", "mean": 0.9},
        ],
    }), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "run_id": "old",
        "l1": {"observations": [
            {"scope": "aggregate", "stage": "grounding", "metric_name": "citation_recall", "value": 0.5},
        ]},
        "l2": {"metric_aggregates": [
            {"track": "real", "metric": "contextual_recall", "mean": 0.6},
        ]},
    }), encoding="utf-8")

    result = build_comparison_report(
        current_l1=l1,
        current_l2=l2,
        baseline=baseline,
        output_dir=tmp_path / "out",
    )

    assert result["metrics"]["l1.grounding.citation_recall"]["delta"] == 0.3
    assert result["metrics"]["l1.grounding.citation_recall"]["relative_delta_percent"] == 60.0
    assert result["metrics"]["l2.real.contextual_recall"]["delta"] == 0.3
    assert result["provenance"]["current_l1_run_id"] == "l1-new"
    assert (tmp_path / "out" / "report.md").exists()
