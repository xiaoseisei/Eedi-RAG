from __future__ import annotations

from pathlib import Path

from evals.adapters.storage import capture_storage_snapshot
from evals.live_storage import hash_storage_artifacts


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Connection:
    def execute(self, query: str):
        normalized = " ".join(query.split())
        if "FROM tutoring_sessions" in normalized:
            return _Rows([(10,)])
        if "FROM misconception_chunks" in normalized:
            return _Rows([("m10", 10)])
        if "FROM tutor_strategy_chunks" in normalized:
            return _Rows([])
        if "FROM sliding_window_chunks" in normalized:
            return _Rows([])
        if "FROM session_dialogue_turns" in normalized:
            return _Rows([("10_1", 10)])
        raise AssertionError(f"unexpected query: {query}")


class _Collection:
    def __init__(self, payload):
        self._payload = payload

    def get(self, *, include):
        assert include == ["metadatas"]
        return self._payload


class _Chroma:
    def get_collection(self, name: str):
        if name == "student_misconceptions":
            return _Collection(
                {
                    "ids": ["m10"],
                    "metadatas": [
                        {
                            "session_id": 10,
                            "question_id": 20,
                            "subject_path": "Number",
                            "misconception_name": "place value",
                            "deep_mechanism": "confusion",
                            "confusion_triggers": '["decimal places"]',
                            "verbatim_student_quotes": '["5.45"]',
                            "source_turn_ids": "[1]",
                        }
                    ],
                }
            )
        return _Collection({"ids": [], "metadatas": []})


class _Manager:
    duck_conn = _Connection()
    chroma_client = _Chroma()


def test_storage_adapter_captures_existing_handles_without_ingestion() -> None:
    snapshot = capture_storage_snapshot(_Manager(), case_id="fixture")

    assert snapshot.case_id == "fixture"
    assert snapshot.valid_parent_ids == {10}
    assert snapshot.collections[0].expected_ids == {"m10"}
    assert snapshot.collections[0].index_records[0].parent_id == 10
    assert snapshot.additional_relational_records[0].record_id == "10_1"


def test_storage_artifact_hash_changes_with_content_not_absolute_root(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_db = first_root / "store.duckdb"
    second_db = second_root / "store.duckdb"
    first_chroma = first_root / "chroma"
    second_chroma = second_root / "chroma"
    first_chroma.mkdir()
    second_chroma.mkdir()
    first_db.write_bytes(b"db")
    second_db.write_bytes(b"db")
    (first_chroma / "index.bin").write_bytes(b"index")
    (second_chroma / "index.bin").write_bytes(b"index")

    first_hash = hash_storage_artifacts(first_db, first_chroma)
    assert first_hash == hash_storage_artifacts(second_db, second_chroma)

    (second_chroma / "index.bin").write_bytes(b"changed")
    assert first_hash != hash_storage_artifacts(second_db, second_chroma)

