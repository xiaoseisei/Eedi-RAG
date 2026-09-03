from __future__ import annotations

import json
from pathlib import Path


def test_l3_dataset_declares_proxy_and_unmeasured_business_outcomes(tmp_path: Path) -> None:
    from scripts.run_l3_business_eval import build_l3_dataset

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([
        {
            "question": "Why?",
            "ground_truth": "Because place value was confused.",
            "category": "STUDENT_INSIGHT",
            "subject_path": "Number",
            "verbatim_grounding_quotes": [{"session_id": 1, "turn_id": 1, "speaker": "student", "quote_text": "I chose A"}],
        },
        {
            "question": "How?",
            "ground_truth": "Ask an aha question.",
            "category": "TUTOR_INTERVENTION",
            "subject_path": "Number",
            "verbatim_grounding_quotes": [{"session_id": 2, "turn_id": 2, "speaker": "tutor", "quote_text": "Which digit?"}],
        },
    ]), encoding="utf-8")
    l1_manifest = tmp_path / "l1.json"
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"query_assignments": {"eedi-l1-0001": "dev", "eedi-l1-0002": "holdout"}, "split_counts": {"dev": 1, "holdout": 1}, "session_leakage_count": 0}), encoding="utf-8")
    l1_manifest.write_text(json.dumps({"qrels_version": "derived", "split": {"file": split.name}}), encoding="utf-8")

    manifest = build_l3_dataset(output_dir=tmp_path / "out", golden_path=golden, l1_manifest_path=l1_manifest, artifact_hash="a" * 64)

    assert manifest["dataset_version"] == "eedi-l3-business-proxy-v1"
    cases = [json.loads(line) for line in (tmp_path / "out/business_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {case["business_data_status"] for case in cases} == {"proxy_eedi_no_nbcot_feedback"}
    assert "learning_gain" in manifest["system_config"]["real_business_outcomes"]
