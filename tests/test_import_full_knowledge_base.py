"""Integration tests for resumable full import orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.import_full_knowledge_base import run_full_import
from src.chunker import SessionChunker
from src.knowledge_import import promote_batch
from src.models import (
    CleanedSession,
    DialogueTurn,
    ExtractedPIU,
    Question,
    StudentMisconceptionProfile,
    SubjectHierarchy,
    TutorStrategyProfile,
)
from src.storage_manager import DualEngineStorageManager


def _session(session_id: int) -> CleanedSession:
    turns = [
        DialogueTurn(turn_id=1, speaker="student", is_tutor=False, raw_messages=["I think B"], text="I think B"),
        DialogueTurn(turn_id=2, speaker="tutor", is_tutor=True, raw_messages=["Why?"], text="Why?"),
    ]
    return CleanedSession(
        intervention_id=session_id,
        question_id=100 + session_id,
        tutor_id=7,
        subjects=SubjectHierarchy(paths=["Number > Arithmetic"]),
        question=Question(question_id=100 + session_id, question_text="Evaluate 2 + 3", options={"A": "4", "B": "5"}),
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


def _write_source(path: Path, sessions: list[CleanedSession]) -> None:
    path.write_text("\n".join(json.dumps(item.model_dump(mode="json")) for item in sessions) + "\n", encoding="utf-8")


def test_full_import_commits_batches_and_skips_committed_batch_on_resume(tmp_path: Path):
    sessions = [_session(2), _session(1), _session(3)]
    source = tmp_path / "sessions.jsonl"
    _write_source(source, sessions)
    storage = DualEngineStorageManager(db_path=":memory:", in_memory=True, embedding_backend="deterministic")
    calls: list[tuple[int, ...]] = []

    def ingest(**kwargs):
        batch = tuple(item.intervention_id for item in kwargs["sessions"])
        calls.append(batch)
        cards = [_extracted(item) for item in kwargs["sessions"]]
        output = Path(kwargs["cache_dir"]) / f"cards-{len(calls)}.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(card.model_dump(mode="json")) + "\n" for card in cards), encoding="utf-8")
        return {"status": "SUCCESS", "failed_count": 0, "cards_path": str(output)}

    try:
        first = run_full_import(
            source_path=source,
            state_dir=tmp_path / "state",
            batch_size=2,
            workers=2,
            max_batches=1,
            storage=storage,
            ingestion_runner=ingest,
        )
        assert first["status"] == "PARTIAL"
        assert first["batches"][0]["status"] == "COMMITTED"
        assert first["batches"][1]["status"] == "PENDING"
        assert calls == [(1, 2)]

        second = run_full_import(
            source_path=source,
            state_dir=tmp_path / "state",
            batch_size=2,
            workers=2,
            storage=storage,
            ingestion_runner=ingest,
        )
        assert second["status"] == "SUCCESS"
        assert [item["status"] for item in second["batches"]] == ["COMMITTED", "COMMITTED"]
        assert calls == [(1, 2), (3,)]
    finally:
        storage.close()


def test_failed_batch_is_checkpointed_and_retried(tmp_path: Path):
    sessions = [_session(1)]
    source = tmp_path / "sessions.jsonl"
    _write_source(source, sessions)
    storage = DualEngineStorageManager(db_path=":memory:", in_memory=True, embedding_backend="deterministic")
    fail = {"value": True}

    def ingest(**kwargs):
        if fail["value"]:
            return {"status": "FAILED", "failed_count": 1, "errors": [{"session_id": 1}]}
        output = Path(kwargs["cache_dir"]) / "cards-ok.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_extracted(kwargs["sessions"][0]).model_dump(mode="json")) + "\n", encoding="utf-8")
        return {"status": "SUCCESS", "failed_count": 0, "cards_path": str(output)}

    try:
        with pytest.raises(Exception):
            run_full_import(source_path=source, state_dir=tmp_path / "state", storage=storage, ingestion_runner=ingest)
        failed = json.loads((tmp_path / "state" / "manifest.json").read_text(encoding="utf-8"))
        assert failed["batches"][0]["status"] == "FAILED"
        fail["value"] = False
        resumed = run_full_import(source_path=source, state_dir=tmp_path / "state", storage=storage, ingestion_runner=ingest)
        assert resumed["status"] == "SUCCESS"
        assert json.loads((tmp_path / "state" / "manifest.json").read_text(encoding="utf-8"))["status"] == "SUCCESS"
    finally:
        storage.close()
