"""Tests for versioned, success-only extraction caching."""

import json
from pathlib import Path

from src.extraction_cache import ExtractionCache, compute_contract_hash, compute_session_hash
from src.models import ExtractedPIU


def _successful_piu() -> ExtractedPIU:
    return ExtractedPIU.model_validate(
        {
            "session_id": 10,
            "question_id": 20,
            "misconception": {
                "session_id": 10,
                "question_id": 20,
                "subject_path": "Number > Rounding",
                "misconception_name": "位值混淆",
                "deep_mechanism": "学生混淆目标位与后一位。",
                "confusion_triggers": ["1dp"],
                "verbatim_student_quotes": ["5.45"],
                "source_turn_ids": [1],
            },
            "tutor_strategy": {
                "session_id": 10,
                "question_id": 20,
                "pedagogical_goal": "明确舍入位值。",
                "strategy_category": "Socratic_Questioning",
                "key_aha_question": "Which digit decides?",
                "scaffolding_steps": ["圈出目标位"],
                "talk_moves": [],
                "resolution_outcome": "学生完成修正。",
                "source_turn_ids": [2],
            },
            "extraction_status": "success",
        }
    )


def test_session_hash_covers_the_complete_session_payload() -> None:
    session = {"intervention_id": 10, "turns": [{"turn_id": 1, "text": "5.45"}]}
    changed_session = {"intervention_id": 10, "turns": [{"turn_id": 1, "text": "5.46"}]}

    assert compute_session_hash(session) != compute_session_hash(changed_session)
    assert compute_session_hash(session) == compute_session_hash(
        {"turns": [{"text": "5.45", "turn_id": 1}], "intervention_id": 10}
    )


def test_contract_hash_changes_when_any_contract_input_changes() -> None:
    base = {"prompt_version": "evidence-v2", "schema_version": "cards-v1", "model": "qwen"}

    assert compute_contract_hash(base) != compute_contract_hash({**base, "model": "qwen-new"})
    assert compute_contract_hash(base) != compute_contract_hash({**base, "prompt_version": "evidence-v3"})


def test_cache_round_trip_requires_matching_session_and_contract(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "extractions.json")
    session = {"intervention_id": 10, "turns": []}
    contract = {"prompt_version": "v1", "schema_version": "cards-v1", "model": "qwen"}
    result = _successful_piu()

    assert cache.put(session, contract, result) is True
    assert cache.get(session, contract) == result
    assert cache.get({"intervention_id": 10, "turns": [{"turn_id": 1}]}, contract) is None
    assert cache.get(session, {**contract, "model": "other"}) is None


def test_cache_rejects_failed_result_and_does_not_create_a_hit(tmp_path: Path) -> None:
    cache_path = tmp_path / "extractions.json"
    cache = ExtractionCache(cache_path)
    session = {"intervention_id": 10}
    contract = {"schema_version": "cards-v1"}
    failed = ExtractedPIU(
        session_id=10,
        question_id=20,
        misconception=None,
        tutor_strategy=None,
        extraction_status="failed",
    )

    assert cache.put(session, contract, failed) is False
    assert not cache_path.exists()
    assert cache.get(session, contract) is None


def test_cache_records_failure_but_never_returns_it_as_a_hit(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "extractions.json")
    session = {"intervention_id": 10}
    contract = {"schema_version": "cards-v1"}

    cache.put_failure(session, contract, "TimeoutError", "provider timeout")

    document = json.loads((tmp_path / "extractions.json").read_text(encoding="utf-8"))
    assert next(iter(document["entries"].values()))["status"] == "failed"
    assert cache.get(session, contract) is None


def test_cache_write_is_valid_json_and_recovery_from_partial_file_is_a_miss(tmp_path: Path) -> None:
    cache_path = tmp_path / "extractions.json"
    cache = ExtractionCache(cache_path)
    session = {"intervention_id": 10}
    contract = {"schema_version": "cards-v1"}
    cache.put(session, contract, _successful_piu())

    json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.write_text('{"schema_version":', encoding="utf-8")
    assert cache.get(session, contract) is None
