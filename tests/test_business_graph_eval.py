from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pathlib import Path
from src.business_graph_metrics import CaseEvaluation, evaluate_case
from src.business_graph_trace import BusinessGraphRuntimeTrace, StageStatus
from scripts import evaluate_business_graph
from scripts.evaluate_business_graph import validate_graph_provenance


ROOT = Path(__file__).resolve().parents[1]


def _case() -> dict[str, object]:
    return {
        "case_id": "eval-001",
        "query": "why?",
        "expected_router": {
            "intent": "QUESTION_TUTORING",
            "scope": "QUESTION",
            "computation_scope": "MICRO",
            "perspectives": ["STUDENT"],
        },
        "expected_plan": {
            "plan_type": "SOCRATIC_TUTORING",
            "selected_lever": {"id": "student_misconception_card_rag", "index": 1},
        },
        "expected_sql_scope": {"filters": {"question_id": 104614}, "keywords": ["question_id"]},
        "expected_graph_paths": [{"edges": ["HAS_MISCONCEPTION", "EVIDENCED_BY"]}],
        "expected_card_evidence": [{"card_id": "card-10", "source_session_id": 10}],
        "expected_window_evidence": [{"window_id": "session_10_win_1_7", "session_id": 10, "source_turn_ids": [1, 2, 3, 4, 5, 6, 7], "anchor_turn_ids": [4], "window_policy": "W7_S3"}],
        "expected_prompt": {"required_window_ids": ["session_10_win_1_7"], "required_sections": ["学生错因卡"], "rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "max_parent_cards": 5},
        "expected_citation_contract": {"required_evidence_turns": [{"session_id": 10, "turn_id": 4, "speaker": "student", "exact_quote": "exact quote"}]},
        "expected_status": "SUCCESS",
        "ideal_answer": {"required_claims": [], "forbidden_claims": []},
    }


def _trace(**overrides: object) -> BusinessGraphRuntimeTrace:
    payload: dict[str, object] = {
        "actual_router": {"intent": "QUESTION_TUTORING", "scope": "QUESTION", "computation_scope": "MICRO", "perspectives": ["STUDENT"]},
        "actual_plan": {"plan_type": "SOCRATIC_TUTORING", "selected_lever": {"id": "student_misconception_card_rag", "index": 1}},
        "capabilities": {}, "analytics_result": {"operation": "question_lookup", "filters": {"question_id": 104614}, "keywords": ["question_id"]},
        "graph_paths": [{"edge_types": ["HAS_MISCONCEPTION", "EVIDENCED_BY"], "session_id": 10}],
        "card_candidates": [{"chunk_id": "card-10", "metadata": {"session_id": 10}}],
        "parent_cards": [{"chunk_id": "card-10", "metadata": {"session_id": 10}}],
        "candidate_windows": [{"window_id": "session_10_win_1_7", "session_id": 10, "source_turn_ids": [1, 2, 3, 4, 5, 6, 7], "anchor_turn_ids": [4], "window_policy": "W7_S3"}],
        "ranked_windows": [{"window_id": "session_10_win_1_7", "session_id": 10, "source_turn_ids": [1, 2, 3, 4, 5, 6, 7], "anchor_turn_ids": [4], "rerank_rank": 1}],
        "selected_windows": [{"window_id": "session_10_win_1_7", "session_id": 10, "source_turn_ids": [1, 2, 3, 4, 5, 6, 7], "anchor_turn_ids": [4], "rerank_rank": 1}],
        "assembled_prompt": "学生错因卡\n7轮课堂上下文\n数据来源和限制\nsession_10_win_1_7",
        "citations": [{"session_id": 10, "turn_id": 4, "speaker": "student", "exact_quote": "exact quote"}],
        "answer": None, "stage_status": {"ROUTER": StageStatus.SUCCESS, "PLANNER": StageStatus.SUCCESS}, "errors": [], "timings": {},
    }
    payload.update(overrides)
    return BusinessGraphRuntimeTrace.model_validate(payload)


def test_evaluator_reports_router_mismatch_without_converting_it_to_pass() -> None:
    result = evaluate_case(_case(), _trace(actual_router={"intent": "DIFFICULT_CONCEPTS", "scope": "QUESTION", "computation_scope": "MICRO", "perspectives": ["STUDENT"]}))

    assert isinstance(result, CaseEvaluation)
    assert not result.checks["router_exact_match"]
    assert "ROUTER_INTENT_MISMATCH" in result.reason_codes
    assert result.first_failure_stage == "ROUTER"
    assert result.checks["source_artifact_unchanged"] is False
    assert result.measurement_status == "BLOCKED"


def test_evaluator_detects_wrong_window_and_prompt_coverage() -> None:
    wrong = [{"window_id": "other", "session_id": 10, "source_turn_ids": [1, 2], "anchor_turn_ids": [1], "rerank_rank": 1}]

    result = evaluate_case(_case(), _trace(candidate_windows=wrong, ranked_windows=wrong, selected_windows=wrong, assembled_prompt=""))

    assert not result.checks["candidate_window_recall"]
    assert not result.checks["prompt_window_coverage"]
    assert "WINDOW_NOT_RETRIEVED" in result.reason_codes


def test_evaluator_detects_quote_role_cross_session_and_fake_card() -> None:
    result = evaluate_case(
        _case(),
        _trace(
            card_candidates=[{"chunk_id": "fake-card", "metadata": {"session_id": 99}}],
            parent_cards=[{"chunk_id": "fake-card", "metadata": {"session_id": 99}}],
            citations=[{"session_id": 99, "turn_id": 4, "speaker": "tutor", "exact_quote": "altered"}],
        ),
    )

    assert not result.checks["card_evidence_validity"]
    assert not result.checks["citation_precision"]
    assert "CARD_NOT_RETRIEVED" in result.reason_codes
    assert "CITATION_UNAUTHORIZED" in result.reason_codes
    assert "QUOTE_MISMATCH" in result.reason_codes


def test_evaluator_preserves_negative_status_as_an_explicit_check() -> None:
    case = _case() | {"expected_status": "NO_EVIDENCE", "expected_card_evidence": [], "expected_window_evidence": []}
    result = evaluate_case(case, _trace(stage_status={"ROUTER": StageStatus.NO_EVIDENCE}))

    assert result.checks["status_accuracy"]


def test_unverified_citation_without_required_contract_does_not_default_to_pass() -> None:
    case = _case() | {
        "expected_citation_contract": {},
        "expected_card_evidence": [],
        "expected_window_evidence": [],
    }
    result = evaluate_case(
        case,
        _trace(citations=[{"session_id": 10, "turn_id": 4, "exact_quote": "quote"}]),
    )

    assert result.checks["citation_precision"] is False


def test_card_gate_requires_expected_card_to_reach_parent_cards() -> None:
    result = evaluate_case(
        _case(),
        _trace(
            card_candidates=[
                {"chunk_id": "card-10", "metadata": {"session_id": 10}},
                {"chunk_id": "other-card", "metadata": {"session_id": 11}},
            ],
            parent_cards=[{"chunk_id": "other-card", "metadata": {"session_id": 11}}],
        ),
    )

    assert not result.checks["card_evidence_validity"]
    assert "CARD_NOT_PARENT" in result.reason_codes


def test_candidate_graph_must_be_ready_and_source_bound_but_requires_reapproval() -> None:
    manifest = {
        "source_db_sha256": "f089c56f162666ab71b3beecef33a871f3e140608721b3f61f96f9ed384b6157",
        "graph_db_sha256": "0" * 64,
    }

    status = validate_graph_provenance(
        manifest,
        source_db="data/db/tutoring_knowledge.duckdb",
        graph_db="reports/staging/session-card-graph-v1/graph-concept-topology-20260909.duckdb",
    )

    assert status["ready_manifest"] is True
    assert status["source_hash_matches"] is True
    assert status["gold_graph_hash_matches"] is False
    assert status["approval_status"] == "PENDING_GOLD_REAPPROVAL"


def test_evaluator_classifier_loader_is_unmeasured_only_without_model(monkeypatch) -> None:
    monkeypatch.delenv("INTENT_MODEL", raising=False)

    assert evaluate_business_graph.load_intent_classifier_from_env() is None


def test_evaluator_classifier_loader_does_not_swallow_config_error(monkeypatch) -> None:
    monkeypatch.setenv("INTENT_MODEL", "Qwen/Qwen3.5-4B")
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("INTENT_API_KEY_ENV", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY_ENV", raising=False)

    with pytest.raises(ValueError, match="Intent classifier API key is not configured"):
        evaluate_business_graph.load_intent_classifier_from_env()


def test_evaluator_injects_environment_classifier_into_pipeline(monkeypatch, tmp_path) -> None:
    """The evaluator must run the same second-layer router as the CLI."""

    source_db = ROOT / "data/db/tutoring_knowledge.duckdb"
    graph_db = ROOT / "reports/staging/session-card-graph-v1/graph-taxonomy-verify-2-20260908.duckdb"
    chroma_dir = ROOT / "data/chroma"
    source_hash = evaluate_business_graph.sha256(source_db)
    graph_hash = evaluate_business_graph.sha256(graph_db)
    classifier = SimpleNamespace(model="Qwen/Qwen3.5-4B")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        evaluate_business_graph,
        "read_manifest",
        lambda _path: {
            "case_count": 83,
            "source_db_sha256": source_hash,
            "graph_db_sha256": graph_hash,
            "gold_sha256": "gold",
        },
    )
    monkeypatch.setattr(
        evaluate_business_graph,
        "load_gold_cases",
        lambda _path: [
            {
                "case_id": f"case-{index}",
                "query": "q",
                "category": "TEST",
                "expected_router": {"intent": "QUESTION_TUTORING"},
            }
            for index in range(83)
        ],
    )
    monkeypatch.setattr(
        evaluate_business_graph,
        "select_cases",
        lambda cases, _split: ([cases[0]], {"status": "MEASURED", "case_count": 1}),
    )
    snapshot = {
        "source_db": {"sha256": source_hash},
        "graph_db": {"sha256": graph_hash},
        "chroma_sqlite": {"sha256": "chroma"},
    }
    monkeypatch.setattr(evaluate_business_graph, "artifact_snapshot", lambda *args: snapshot)
    monkeypatch.setattr(
        evaluate_business_graph,
        "validate_graph_provenance",
        lambda *args, **kwargs: {
            "approval_status": "GOLD_APPROVED",
            "ready_manifest": True,
            "source_hash_matches": True,
            "gold_graph_hash_matches": True,
            "topology_tables_present": True,
        },
    )
    monkeypatch.setattr(
        evaluate_business_graph.OpenAICompatibleIntentClassifier,
        "try_from_env",
        classmethod(lambda cls, model=None: classifier),
    )

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            captured["intent_classifier"] = kwargs["intent_classifier"]

        def close(self):
            pass

    monkeypatch.setattr(evaluate_business_graph, "BusinessGraphPipeline", FakePipeline)
    monkeypatch.setattr(
        evaluate_business_graph,
        "evaluate_dataset",
        lambda cases, runtime: {
            "case_count": 1,
            "metrics": {"router_exact_match": 1.0},
            "hard_gates_passed": True,
            "measurement_status": "PASS",
            "first_failure_by_stage": {},
            "cases": [
                {
                    "case_id": "case-0",
                    "evaluation": {
                        "checks": {"router_exact_match": True},
                        "reason_codes": [],
                        "measurement_status": "PASS",
                        "first_failure_stage": None,
                    },
                }
            ],
        },
    )

    output = tmp_path / "report"
    result = evaluate_business_graph.main(
        [
            "--gold",
            str(ROOT / "data/business/business-graph-gold-v1.jsonl"),
            "--source-db",
            str(source_db),
            "--graph-db",
            str(graph_db),
            "--chroma-dir",
            str(chroma_dir),
            "--output",
            str(output),
        ]
    )

    assert result == 1  # deterministic run remains blocked by external model gates
    assert captured["intent_classifier"] is classifier
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["external_models"]["intent_classifier"]["status"] == "MEASURED"
