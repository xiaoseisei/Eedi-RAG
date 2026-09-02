from __future__ import annotations

import json

from evals.contracts import (
    ComparisonOperator,
    DeepEvalCasePayload,
    DeepEvalSettings,
    L1ComponentReport,
    RunContext,
    observation_from_measurement,
)
from evals.deepeval_adapter import probe_deepeval
from evals.reporting import write_report


def _context() -> RunContext:
    return RunContext(
        run_id="run-report",
        dataset_version="component-fixture-v1",
        qrels_version="qrels-fixture-v1",
        system_config_hash="c" * 64,
    )


def test_report_is_blocked_by_failed_hard_gate_and_writes_auditable_files(tmp_path) -> None:
    observation = observation_from_measurement(
        context=_context(),
        case_id="storage-1",
        stage="storage",
        metric_name="metadata_contract_rate",
        value=0.5,
        threshold=1.0,
        operator=ComparisonOperator.GTE,
        hard_gate=True,
    )
    report = L1ComponentReport.from_observations(
        context=_context(),
        observations=[observation],
        executed_modules=["storage"],
    )

    assert report.release_status == "BLOCKED"
    run_dir = write_report(report, tmp_path)
    assert run_dir == tmp_path / "run-report"
    assert (run_dir / "run.json").exists()
    assert (run_dir / "observations.jsonl").exists()
    assert (run_dir / "failures.jsonl").exists()
    assert (run_dir / "summary.md").exists()

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    failure = json.loads((run_dir / "failures.jsonl").read_text(encoding="utf-8"))
    assert manifest["release_status"] == "BLOCKED"
    assert failure["metric_name"] == "metadata_contract_rate"
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Release status: **BLOCKED**" in summary
    assert "metadata_contract_rate" in summary


def test_deepeval_probe_marks_missing_dependency_unmeasured_without_fake_score() -> None:
    settings = DeepEvalSettings(
        evaluation_provider="openai-compatible",
        evaluation_model="judge-model",
        api_key_env="EEDI_EVAL_API_KEY",
    )
    readiness = probe_deepeval(
        settings,
        environ={"EEDI_EVAL_API_KEY": "secret-not-to-be-returned"},
        module_finder=lambda _: None,
    )

    assert readiness.status == "UNMEASURED"
    assert readiness.installed is False
    assert readiness.credentials_configured is True
    assert readiness.score is None
    assert "secret-not-to-be-returned" not in readiness.model_dump_json()


def test_deepeval_payload_uses_only_final_context_seen_by_generator() -> None:
    payload = DeepEvalCasePayload(
        case_id="generator-1",
        input="Why did the student choose 5.45?",
        actual_output="The student confused decimal-place rules.",
        expected_output="The student confused one and two decimal places.",
        final_retrieval_context=["final assembled context"],
    )

    assert payload.to_test_case_kwargs() == {
        "input": "Why did the student choose 5.45?",
        "actual_output": "The student confused decimal-place rules.",
        "expected_output": "The student confused one and two decimal places.",
        "retrieval_context": ["final assembled context"],
    }
