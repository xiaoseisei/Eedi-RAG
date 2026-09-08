from __future__ import annotations

from pathlib import Path

import json

import duckdb


def test_qwen_index_upserts_large_collection_in_chroma_safe_batches() -> None:
    from scripts.rebuild_qwen_embedding_index import _upsert_rows

    class RecordingCollection:
        def __init__(self) -> None:
            self.calls: list[dict[str, list]] = []

        def upsert(self, *, ids, documents, metadatas) -> None:
            self.calls.append({
                "ids": list(ids),
                "documents": list(documents),
                "metadatas": list(metadatas),
            })

    rows = [
        {"id": f"w-{index}", "document": f"doc-{index}", "metadata": {"index": index}}
        for index in range(11_337)
    ]
    collection = RecordingCollection()

    _upsert_rows(collection, rows)

    assert [len(call["ids"]) for call in collection.calls] == [5_000, 5_000, 1_337]
    assert [item for call in collection.calls for item in call["ids"]] == [f"w-{index}" for index in range(11_337)]


def test_qwen_index_loader_reads_duckdb_without_chroma(tmp_path: Path) -> None:
    from scripts.rebuild_qwen_embedding_index import _load_documents

    db = tmp_path / "source.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE tutoring_sessions (intervention_id BIGINT, question_text VARCHAR)")
    con.execute("CREATE TABLE misconception_chunks (chunk_id VARCHAR, session_id BIGINT, question_id BIGINT, subject_path VARCHAR, misconception_name VARCHAR, error_choice VARCHAR, deep_mechanism VARCHAR, confusion_triggers VARCHAR[], verbatim_student_quotes VARCHAR[], source_turn_ids INTEGER[])")
    con.execute("CREATE TABLE tutor_strategy_chunks (chunk_id VARCHAR, session_id BIGINT, question_id BIGINT, subject_path VARCHAR, pedagogical_goal VARCHAR, strategy_category VARCHAR, key_aha_question VARCHAR, scaffolding_steps VARCHAR[], analogy_or_metaphor VARCHAR, talk_moves VARCHAR[], resolution_outcome VARCHAR, source_turn_ids INTEGER[])")
    con.execute("CREATE TABLE sliding_window_chunks (chunk_id VARCHAR, session_id BIGINT, question_id BIGINT, subject_path VARCHAR, window_start_turn INTEGER, window_end_turn INTEGER, content VARCHAR, source_turn_ids INTEGER[])")
    con.execute("INSERT INTO tutoring_sessions VALUES (1, 'Round 5.4')")
    con.execute("INSERT INTO misconception_chunks VALUES ('m1',1,2,'Number','rounding','', 'mechanism', ['trigger'], ['quote'], [1])")
    con.execute("INSERT INTO tutor_strategy_chunks VALUES ('s1',1,2,'Number','goal','Scaffolding','aha',['step'],'', ['move'],'outcome',[2])")
    con.execute("INSERT INTO sliding_window_chunks VALUES ('w1',1,2,'Number',1,1,'[Turn 1] [Student]: quote',[1])")
    con.close()
    docs = _load_documents(db)
    assert {key: len(value) for key, value in docs.items()} == {
        "student_misconceptions": 1,
        "tutor_strategies": 1,
        "fallback_windows": 1,
    }
    assert "Round 5.4" in docs["student_misconceptions"][0]["document"]


def test_freeze_closeout_inputs_records_hashes_splits_and_config(tmp_path: Path) -> None:
    from scripts.freeze_l1_closeout_inputs import freeze_closeout_inputs
    from scripts.build_l1_component_datasets import _directory_sha256, _file_sha256

    db = tmp_path / "source.duckdb"
    db.write_bytes(b"db")
    deterministic = tmp_path / "deterministic"
    qwen = tmp_path / "qwen"
    rollback = tmp_path / "rollback"
    for path, payload in ((deterministic, b"d"), (qwen, b"q"), (rollback, b"r")):
        path.mkdir()
        (path / "index.bin").write_bytes(payload)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([
        {
            "question": f"question {index}",
            "category": "STUDENT_INSIGHT",
            "verbatim_grounding_quotes": [{"session_id": index, "turn_id": 1, "speaker": "student", "quote": "x"}],
        }
        for index in range(1, 6)
    ]), encoding="utf-8")
    qwen_build = tmp_path / "chroma.build.json"
    qwen_build.write_text(json.dumps({
        "provider": {"model": "Qwen/Qwen3-Embedding-0.6B", "dimension": 1024, "index_version": "v1"},
        "collection_counts": {"fallback_windows": 5},
        "source_duckdb_sha256": _file_sha256(db),
        "target_chroma_sha256": _directory_sha256(qwen),
    }), encoding="utf-8")
    output = tmp_path / "frozen"

    manifest_path = freeze_closeout_inputs(
        output_dir=output,
        golden_path=golden,
        db_path=db,
        deterministic_chroma=deterministic,
        qwen_chroma=qwen,
        qwen_build_manifest=qwen_build,
        rollback_chroma=rollback,
        experiment_config={"fusion_strategy": "raw_first", "top_k": 20},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queries = [json.loads(line) for line in (output / "queries.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["artifact_mutated_during_freeze"] is False
    assert manifest["split"]["session_leakage_count"] == 0
    assert set(item["split"] for item in queries) == {"dev", "holdout"}
    assert manifest["experiment_config"]["fusion_strategy"] == "raw_first"
    assert len(manifest["system_config_hash"]) == 64


def test_freeze_closeout_inputs_rejects_qwen_hash_drift(tmp_path: Path) -> None:
    from scripts.freeze_l1_closeout_inputs import freeze_closeout_inputs
    from scripts.build_l1_component_datasets import _directory_sha256, _file_sha256

    db = tmp_path / "source.duckdb"
    db.write_bytes(b"db")
    directories = [tmp_path / name for name in ("deterministic", "qwen", "rollback")]
    for path in directories:
        path.mkdir()
        (path / "index.bin").write_bytes(b"index")
    golden = tmp_path / "golden.json"
    golden.write_text("[]", encoding="utf-8")
    build = tmp_path / "chroma.build.json"
    build.write_text(json.dumps({
        "source_duckdb_sha256": _file_sha256(db),
        "target_chroma_sha256": _directory_sha256(directories[1]),
    }), encoding="utf-8")
    (directories[1] / "housekeeping.bin").write_bytes(b"changed")

    with __import__("pytest").raises(ValueError, match="no longer matches"):
        freeze_closeout_inputs(
            output_dir=tmp_path / "out",
            golden_path=golden,
            db_path=db,
            deterministic_chroma=directories[0],
            qwen_chroma=directories[1],
            qwen_build_manifest=build,
            rollback_chroma=directories[2],
            experiment_config={"top_k": 20},
        )
