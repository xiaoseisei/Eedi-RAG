from __future__ import annotations

import json
from pathlib import Path


def test_l123_report_preserves_layer_decisions_and_hash_consistency(tmp_path: Path) -> None:
    from scripts.build_l123_analysis_report import build_report

    def write(name: str, value: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    l1_manifest = write("l1-manifest.json", {"dataset_version": "l1", "artifact_hashes": {"qwen_combined_sha256": "a" * 64}, "split": {"file": "split.json"}})
    (tmp_path / "split.json").write_text("{}", encoding="utf-8")
    l1_closeout = write("l1-closeout.json", {"decision": "BLOCKED", "status_counts": {}, "hard_blocker_count": 1, "aggregate_metrics": []})
    (tmp_path / "observations.jsonl").write_text("{}\n", encoding="utf-8")
    l2_dataset = write("l2-dataset.json", {"dataset_version": "l2"})
    l2_report = write("l2-report.json", {"release_decision": "BLOCKED_L1_PRECONDITION", "artifact_hash_before": "a" * 64, "judge_readiness": {"status": "SUCCESS"}, "tracks": {}, "metric_aggregates": [], "trace_status_counts": {}})
    (tmp_path / "traces.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    l3_dataset = write("l3-dataset.json", {"dataset_version": "l3"})
    l3_report = write("l3-report.json", {"release_decision": "L3_BUSINESS_UNMEASURED", "artifact_hash_before": "a" * 64, "case_count": 1, "metrics": {}})
    (tmp_path / "raw_cases.jsonl").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "out"

    report = build_report(
        output_dir=output,
        l1_closeout_path=l1_closeout,
        l1_manifest_path=l1_manifest,
        l2_report_path=l2_report,
        l2_dataset_manifest_path=l2_dataset,
        l3_report_path=l3_report,
        l3_dataset_manifest_path=l3_dataset,
    )

    assert report["delivery_decision"] == "L1_BLOCKED_L2_VALIDATED_L3_BUSINESS_UNMEASURED"
    assert report["artifact_hash_consistent"] is True
    assert (output / "report.md").exists()
