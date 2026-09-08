"""Tests for full knowledge-base batch promotion and audits."""

from __future__ import annotations

import json
from pathlib import Path

from src.knowledge_import import (
    BatchAuditError,
    audit_batch,
    audit_full_storage,
    load_cleaned_sessions,
    partition_sessions,
    promote_batch,
)
from src.models import (
    CleanedSession,
    DialogueTurn,
    ExtractedPIU,
    Question,
    StudentMisconceptionProfile,
    SubjectHierarchy,
    TutorStrategyProfile,
)
from src.reranker import PedagogicalGoldAssembler
from src.storage_manager import DualEngineStorageManager
from src.chunker import SessionChunker


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


def test_load_and_partition_sessions_are_strict_and_deterministic(tmp_path: Path):
    sessions = [_session(3), _session(1), _session(2)]
    source = tmp_path / "sessions.jsonl"
    source.write_text(
        "\n".join(json.dumps(item.model_dump(mode="json")) for item in sessions) + "\n",
        encoding="utf-8",
    )

    loaded = load_cleaned_sessions(source)
    assert [item.intervention_id for item in loaded] == [1, 2, 3]
    batches = partition_sessions(loaded, 2)
    assert [[item.intervention_id for item in batch] for batch in batches] == [[1, 2], [3]]


def test_promote_batch_writes_cards_windows_and_passes_audit():
    sessions = [_session(1), _session(2)]
    extracted = [_extracted(item) for item in sessions]
    storage = DualEngineStorageManager(db_path=":memory:", in_memory=True, embedding_backend="deterministic")
    try:
        audit = promote_batch(storage, sessions, extracted, SessionChunker(window_size=2, step=1, default_strategy="hybrid"))
        assert audit.status == "PASSED"
        assert audit.expected_sessions == audit.actual_sessions == 2
        assert audit.expected_misconception_cards == audit.actual_misconception_cards == 2
        assert audit.expected_strategy_cards == audit.actual_strategy_cards == 2
        assert audit.expected_windows == audit.actual_windows == 2
        assert not audit.missing_session_ids
        assert not audit.missing_window_ids
    finally:
        storage.close()


def test_audit_detects_missing_card_after_partial_write():
    sessions = [_session(1)]
    storage = DualEngineStorageManager(db_path=":memory:", in_memory=True, embedding_backend="deterministic")
    try:
        storage.ingest_sessions(sessions)
        audit = audit_batch(
            storage,
            sessions,
            expected_window_ids=["session_1_win_1_2"],
        )
        assert audit.status == "FAILED"
        assert audit.missing_misconception_card_ids == ("session_1_misconception",)
        assert audit.missing_strategy_card_ids == ("session_1_tutor_strategy",)
    finally:
        storage.close()


def test_full_storage_audit_checks_exact_source_session_set():
    storage = DualEngineStorageManager(db_path=":memory:", in_memory=True, embedding_backend="deterministic")
    try:
        sessions = [_session(1)]
        storage.ingest_sessions(sessions)
        result = audit_full_storage(storage, sessions)
        assert result["status"] == "FAILED"
        assert result["checks"]["session_id_set"] is True
        assert result["checks"]["misconception_card_count"] is False
        assert result["checks"]["window_count"] is True
    finally:
        storage.close()
