"""Contract tests for resumable, cache-keyed card ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.models import (
    CleanedSession,
    DialogueTurn,
    ExtractedPIU,
    Question,
    StudentMisconceptionProfile,
    SubjectHierarchy,
    TutorStrategyProfile,
)


def _session(session_id: int) -> CleanedSession:
    turns = [
        DialogueTurn(
            turn_id=1,
            speaker="student",
            is_tutor=False,
            raw_messages=["I think B"],
            text="I think B",
        ),
        DialogueTurn(
            turn_id=2,
            speaker="tutor",
            is_tutor=True,
            raw_messages=["Why?"],
            text="Why?",
        ),
    ]
    return CleanedSession(
        intervention_id=session_id,
        question_id=100 + session_id,
        tutor_id=7,
        subjects=SubjectHierarchy(paths=["Number > Arithmetic"]),
        question=Question(
            question_id=100 + session_id,
            question_text="Evaluate 2 + 3",
            options={"A": "4", "B": "5"},
        ),
        turns=turns,
        total_turns=2,
        student_turn_count=1,
        tutor_turn_count=1,
        has_valid_tutoring=True,
    )


def _extracted(session: CleanedSession) -> ExtractedPIU:
    return ExtractedPIU(
        session_id=session.intervention_id,
        question_id=session.question_id,
        misconception=StudentMisconceptionProfile(
            session_id=session.intervention_id,
            question_id=session.question_id,
            subject_path="Number > Arithmetic",
            misconception_name="Arithmetic confusion",
            deep_mechanism="The operation order is unclear.",
            confusion_triggers=["order"],
            verbatim_student_quotes=["I think B"],
            source_turn_ids=[1],
        ),
        tutor_strategy=TutorStrategyProfile(
            session_id=session.intervention_id,
            question_id=session.question_id,
            pedagogical_goal="Clarify the operation.",
            strategy_category="Socratic_Questioning",
            key_aha_question="Why?",
            scaffolding_steps=["Ask for the reason."],
            talk_moves=["<Press for Accuracy>"],
            resolution_outcome="Student checks the operation.",
            source_turn_ids=[2],
        ),
        extraction_status="success",
    )


@dataclass(frozen=True)
class _FakeKey:
    session_id: int
    prompt_version: str


@dataclass
class _Hit:
    extracted: ExtractedPIU
    evidence_metadata: dict


class _FakeCache:
    def __init__(self, *_args, **_kwargs):
        self.records: dict[_FakeKey, _Hit | dict] = {}
        self.failures: list[tuple[object, str, str]] = []

    def get(self, key):
        return self.records.get(key)

    def put_success(self, key, extracted, evidence_metadata):
        self.records[key] = _Hit(extracted, evidence_metadata)

    def put_failure(self, key, error_type, error_message):
        self.failures.append((key, error_type, error_message))


def _run(monkeypatch, tmp_path: Path, sessions, cache, extractor, **kwargs):
    import scripts.incremental_card_ingest as ingest

    monkeypatch.setattr(ingest, "ExtractionCache", _FakeCache, raising=False)
    return ingest.run_incremental_ingestion(
        sessions=sessions,
        cache=cache,
        cache_dir=tmp_path,
        evidence_index_builder=lambda session, **_: {"session_id": session.intervention_id},
        binder=lambda extracted, _index, **_: extracted,
        extractor=extractor,
        **kwargs,
    )


def test_unchanged_rerun_reuses_cache_without_semantic_llm(monkeypatch, tmp_path):
    session = _session(1)
    cache = _FakeCache()
    calls: list[int] = []

    def extractor(current, **_kwargs):
        calls.append(current.intervention_id)
        return _extracted(current)

    first = _run(monkeypatch, tmp_path, [session], cache, extractor, mode="full")
    second = _run(monkeypatch, tmp_path, [session], cache, extractor, mode="changed-only")

    assert first["llm_processed_count"] == 1
    assert second["llm_processed_count"] == 0
    assert second["reused_count"] == 1
    assert calls == [1]
    assert second["session_statuses"]["1"]["status"] == "REUSED"


def test_prompt_version_change_invalidates_only_matching_cache(monkeypatch, tmp_path):
    sessions = [_session(1), _session(2)]
    cache = _FakeCache()
    calls: list[int] = []

    def extractor(current, **_kwargs):
        calls.append(current.intervention_id)
        return _extracted(current)

    _run(monkeypatch, tmp_path, sessions, cache, extractor, mode="full", prompt_version="v1")
    result = _run(
        monkeypatch,
        tmp_path,
        sessions,
        cache,
        extractor,
        mode="changed-only",
        prompt_version="v2",
    )

    assert result["llm_processed_count"] == 2
    assert result["reused_count"] == 0
    assert calls == [1, 2, 1, 2]


def test_targeted_session_ids_do_not_process_other_sessions(monkeypatch, tmp_path):
    sessions = [_session(1), _session(2)]
    cache = _FakeCache()
    calls: list[int] = []

    def extractor(current, **_kwargs):
        calls.append(current.intervention_id)
        return _extracted(current)

    result = _run(
        monkeypatch,
        tmp_path,
        sessions,
        cache,
        extractor,
        mode="changed-only",
        session_ids=[2],
    )

    assert result["llm_processed_count"] == 1
    assert result["source_session_count"] == 1
    assert calls == [2]


def test_audit_only_rebinds_cache_and_never_calls_semantic_llm(monkeypatch, tmp_path):
    session = _session(1)
    cache = _FakeCache()
    calls: list[int] = []

    def extractor(current, **_kwargs):
        calls.append(current.intervention_id)
        return _extracted(current)

    _run(monkeypatch, tmp_path, [session], cache, extractor, mode="full")
    calls.clear()
    result = _run(monkeypatch, tmp_path, [session], cache, extractor, mode="audit-only")

    assert result["audit_count"] == 1
    assert result["llm_processed_count"] == 0
    assert result["failed_count"] == 0
    assert calls == []
    assert result["session_statuses"]["1"]["status"] == "AUDITED"


def test_binding_policy_change_rebinds_without_semantic_llm(monkeypatch, tmp_path):
    session = _session(1)
    cache = _FakeCache()
    calls: list[int] = []

    def extractor(current, **_kwargs):
        calls.append(current.intervention_id)
        return _extracted(current)

    _run(monkeypatch, tmp_path, [session], cache, extractor, mode="full", binding_policy="policy-v1")
    calls.clear()
    result = _run(
        monkeypatch,
        tmp_path,
        [session],
        cache,
        extractor,
        mode="changed-only",
        binding_policy="policy-v2",
    )

    assert result["llm_processed_count"] == 0
    assert result["reused_count"] == 1
    assert calls == []
    assert result["binding_contract"]["evidence_binding_policy"] == "policy-v2"


def test_failed_session_is_recorded_and_can_resume_without_reprocessing_successes(
    monkeypatch, tmp_path
):
    sessions = [_session(1), _session(2)]
    cache = _FakeCache()
    fail_session_2 = {2}
    calls: list[int] = []

    def extractor(current, **_kwargs):
        calls.append(current.intervention_id)
        if current.intervention_id in fail_session_2:
            raise RuntimeError("provider timeout")
        return _extracted(current)

    failed = _run(monkeypatch, tmp_path, sessions, cache, extractor, mode="changed-only")
    assert failed["failed_count"] == 1
    assert failed["session_statuses"]["2"]["status"] == "FAILED"
    assert failed["status"] == "FAILED"

    fail_session_2.clear()
    resumed = _run(monkeypatch, tmp_path, sessions, cache, extractor, mode="changed-only")
    assert resumed["reused_count"] == 1
    assert resumed["llm_processed_count"] == 1
    assert resumed["failed_count"] == 0
    assert calls == [1, 2, 2]


def test_rejects_unknown_mode_and_non_positive_workers(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="mode"):
        _run(monkeypatch, tmp_path, [_session(1)], _FakeCache(), lambda *_a, **_k: _extracted(_session(1)), mode="bad")
    with pytest.raises(ValueError, match="workers"):
        _run(monkeypatch, tmp_path, [_session(1)], _FakeCache(), lambda *_a, **_k: _extracted(_session(1)), mode="full", workers=0)
