import importlib
import sys

import pytest
from pydantic import ValidationError

from src.evaluation import JudgeLLMOutput, RagasEvaluatorEngine
from src.models import DialogueCitation


VALID_JUDGE_OUTPUT = {
    "context_recall": 0.8,
    "context_precision": 0.7,
    "faithfulness": 0.9,
    "answer_relevance": 0.75,
    "socratic_quality": 0.6,
    "root_cause_diagnosis": "retrieval gap",
    "detailed_critique": "add a directly relevant source",
}


def test_judge_schema_requires_every_promised_field_and_forbids_extras():
    missing = dict(VALID_JUDGE_OUTPUT)
    missing.pop("socratic_quality")
    with pytest.raises(ValidationError):
        JudgeLLMOutput.model_validate(missing)

    extra = {**VALID_JUDGE_OUTPUT, "invented_score": 1.0}
    with pytest.raises(ValidationError):
        JudgeLLMOutput.model_validate(extra)


def test_missing_llm_configuration_fails_fast(monkeypatch):
    for name in ("LLM_API_KEY", "JUDGE_LLM_MODEL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        RagasEvaluatorEngine()


def test_heuristic_fallback_requires_opt_in_and_is_degraded(monkeypatch):
    for name in ("LLM_API_KEY", "JUDGE_LLM_MODEL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    evaluator = RagasEvaluatorEngine(allow_heuristic_fallback=True)
    result = evaluator.evaluate_case("why x", "because x", ["x"], "x")

    assert result["eval_mode"] == "HEURISTIC_FALLBACK"
    assert result["eval_status"] == "DEGRADED"


def test_llm_failure_raises_without_explicit_fallback(monkeypatch):
    evaluator = RagasEvaluatorEngine(api_key="test", model_name="judge")
    monkeypatch.setattr(evaluator, "_evaluate_with_llm", lambda *args: None)

    with pytest.raises(RuntimeError, match="LLM judge"):
        evaluator.evaluate_case("q", "a", ["c"], "g")


def test_citation_audit_requires_exact_session_turn_speaker_and_text(monkeypatch):
    evaluator = RagasEvaluatorEngine(allow_heuristic_fallback=True)
    quotes = [{
        "session_id": 7,
        "turn_id": 9,
        "speaker": "student",
        "quote_text": "Mathematics needs a common denominator.",
    }]
    exact = DialogueCitation(
        session_id=7,
        turn_id=9,
        speaker="student",
        quote_text="Mathematics needs a common denominator.",
    )
    one_character = DialogueCitation(session_id=7, turn_id=9, speaker="student", quote_text="a")
    wrong_speaker = DialogueCitation(
        session_id=7,
        turn_id=9,
        speaker="tutor",
        quote_text="Mathematics needs a common denominator.",
    )

    assert evaluator._audit_citations([], quotes, [exact]) == 1.0
    wrong_session = exact.model_copy(update={"session_id": 8})
    assert evaluator._audit_citations([], quotes, [wrong_session]) == 0.0
    assert evaluator._audit_citations([], quotes, [one_character]) == 0.0
    assert evaluator._audit_citations([], quotes, [wrong_speaker]) == 0.0


def test_import_does_not_load_dotenv(monkeypatch):
    import dotenv

    called = False

    def unexpected_load(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(dotenv, "load_dotenv", unexpected_load)
    sys.modules.pop("src.evaluation", None)
    importlib.import_module("src.evaluation")
    assert called is False


def test_benchmark_helpers_do_not_leak_gold_and_ignore_unmeasured_values():
    from scripts.eval_ragas_benchmark import _extract_predicted_intent, _mean_measured

    class ResponseWithoutPrediction:
        pass

    assert _extract_predicted_intent(ResponseWithoutPrediction()) is None
    assert _mean_measured([{"intent_match": None}], "intent_match") is None
    assert _mean_measured([{"intent_match": None}, {"intent_match": 1.0}], "intent_match") == 1.0


def test_benchmark_report_records_provenance_and_preserves_unmeasured_intent(
    monkeypatch
):
    import json
    import shutil
    import uuid
    from pathlib import Path
    from types import SimpleNamespace

    import scripts.eval_ragas_benchmark as benchmark

    class FakeStorage:
        def __init__(self, **kwargs):
            assert kwargs["embedding_backend"] == "deterministic"

        def close(self):
            pass

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def ask(self, **kwargs):
            assert kwargs["mode"] == "deterministic"
            return SimpleNamespace(
                retrieved_sources_debug=[{"document_text": "real source"}],
                answer_content="real answer",
                misconception_diagnosis="",
                dialogue_citations=[],
            )

    class FakeEvaluator:
        def __init__(self, **kwargs):
            assert kwargs["allow_heuristic_fallback"] is True

        def configuration(self):
            return {
                "backend": "deterministic-keyword-heuristic",
                "model": None,
                "allow_heuristic_fallback": True,
            }

        def evaluate_case(self, **kwargs):
            assert kwargs["actual_intent"] is None
            return {
                "context_recall": 0.5,
                "context_precision": 0.5,
                "faithfulness": 0.5,
                "answer_relevance": 0.5,
                "socratic_quality": None,
                "ragas_score": 0.5,
                "root_cause_diagnosis": "degraded deterministic score",
                "critique": "",
                "eval_mode": "HEURISTIC_FALLBACK",
                "eval_status": "DEGRADED",
                "citation_hit": 0.0,
                "intent_match": None,
            }

    monkeypatch.setattr(benchmark, "DualEngineStorageManager", FakeStorage)
    def fake_retriever(**kwargs):
        assert kwargs["query_rewrite_mode"] == "deterministic"
        return object()

    monkeypatch.setattr(benchmark, "DualMetricRetriever", fake_retriever)
    monkeypatch.setattr(benchmark, "PedagogicalGoldAssembler", lambda **kwargs: object())
    monkeypatch.setattr(benchmark, "EndToEndPedagogicalRAGPipeline", FakePipeline)
    monkeypatch.setattr(benchmark, "RagasEvaluatorEngine", FakeEvaluator)

    work_dir = Path(".audit-tmp") / f"evaluation-{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True)
    try:
        golden = work_dir / "golden.json"
        report_json = work_dir / "report.json"
        report_md = work_dir / "report.md"
        golden.write_text(json.dumps([{
            "question": "q",
            "ground_truth": "g",
            "category": "GOLD_ONLY",
            "subject_path": "Math",
            "verbatim_grounding_quotes": [],
        }]), encoding="utf-8")

        report = benchmark.run_ragas_benchmark(
            golden_test_set_path=str(golden),
            out_report_json=str(report_json),
            out_report_md=str(report_md),
            allow_heuristic_fallback=True,
            generation_mode="deterministic",
        )

        assert report["generated_at"].endswith("+00:00")
        assert report["eval_mode"] == "HEURISTIC_FALLBACK"
        assert report["eval_status"] == "DEGRADED"
        assert report["evaluation_config"]["backend"] == "deterministic-keyword-heuristic"
        assert report["generation_config"]["mode"] == "deterministic"
        assert report["generation_config"]["embedding_backend"] == "deterministic"
        assert report["generation_config"]["query_rewrite_mode"] == "deterministic"
        assert report["average_metrics"]["intent_accuracy"] is None
        assert report["case_details"][0]["actual_intent"] is None
    finally:
        shutil.rmtree(work_dir)
