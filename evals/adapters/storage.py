from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from evals.contracts import IndexedRecord, StorageCollectionSnapshot, StorageEvalSnapshot


COLLECTION_SPECS = {
    "student_misconceptions": {
        "table": "misconception_chunks",
        "required": [
            "session_id",
            "question_id",
            "subject_path",
            "misconception_name",
            "deep_mechanism",
            "confusion_triggers",
            "verbatim_student_quotes",
            "source_turn_ids",
        ],
    },
    "tutor_strategies": {
        "table": "tutor_strategy_chunks",
        "required": [
            "session_id",
            "question_id",
            "subject_path",
            "strategy_category",
            "key_aha_question",
            "pedagogical_goal",
            "scaffolding_steps",
            "talk_moves",
            "resolution_outcome",
            "source_turn_ids",
        ],
    },
    "fallback_windows": {
        "table": "sliding_window_chunks",
        "required": [
            "session_id",
            "question_id",
            "subject_path",
            "window_start_turn",
            "window_end_turn",
            "source_turn_ids",
        ],
    },
}


def capture_storage_snapshot(storage_manager: Any, *, case_id: str = "storage-live") -> StorageEvalSnapshot:
    """Capture a real, read-only audit view from an initialized storage manager."""

    return _capture_from_handles(
        storage_manager.duck_conn,
        storage_manager.chroma_client,
        case_id=case_id,
    )


def capture_storage_snapshot_from_paths(
    db_path: str | Path,
    chroma_dir: str | Path,
    *,
    case_id: str = "storage-live",
) -> StorageEvalSnapshot:
    """Open existing artifacts without creating tables or collections."""

    database_path = Path(db_path).resolve(strict=True)
    chroma_path = Path(chroma_dir).resolve(strict=True)
    if not database_path.is_file():
        raise ValueError(f"DuckDB artifact is not a file: {database_path}")
    if not chroma_path.is_dir():
        raise ValueError(f"Chroma artifact is not a directory: {chroma_path}")

    import duckdb

    connection = duckdb.connect(database=str(database_path), read_only=True)
    try:
        index_records = _read_chroma_metadata_sqlite(chroma_path)
        return _capture_from_index_records(connection, index_records, case_id=case_id)
    finally:
        connection.close()


def _capture_from_handles(connection: Any, chroma_client: Any, *, case_id: str) -> StorageEvalSnapshot:
    index_records_by_collection: dict[str, list[IndexedRecord]] = {}
    for collection_name in COLLECTION_SPECS:
        collection = chroma_client.get_collection(collection_name)
        raw_index = collection.get(include=["metadatas"])
        index_ids = raw_index.get("ids") or []
        metadatas = raw_index.get("metadatas") or [{} for _ in index_ids]
        if len(index_ids) != len(metadatas):
            raise ValueError(f"Chroma collection {collection_name} returned misaligned IDs and metadata")
        records = []
        for record_id, metadata in zip(index_ids, metadatas, strict=True):
            safe_metadata = metadata or {}
            parent_id = safe_metadata.get("session_id")
            records.append(
                IndexedRecord(
                    record_id=str(record_id),
                    parent_id=int(parent_id) if parent_id is not None else None,
                    metadata=safe_metadata,
                )
            )
        index_records_by_collection[collection_name] = records
    return _capture_from_index_records(connection, index_records_by_collection, case_id=case_id)


def _capture_from_index_records(
    connection: Any,
    index_records_by_collection: dict[str, list[IndexedRecord]],
    *,
    case_id: str,
) -> StorageEvalSnapshot:
    parent_ids = {row[0] for row in connection.execute("SELECT intervention_id FROM tutoring_sessions").fetchall()}
    collections: list[StorageCollectionSnapshot] = []
    for collection_name, spec in COLLECTION_SPECS.items():
        relational_rows = connection.execute(
            f"SELECT chunk_id, session_id FROM {spec['table']} ORDER BY chunk_id"
        ).fetchall()
        relational_records = [
            IndexedRecord(record_id=str(record_id), parent_id=parent_id, metadata={})
            for record_id, parent_id in relational_rows
        ]
        if collection_name not in index_records_by_collection:
            raise ValueError(f"required Chroma collection is missing: {collection_name}")
        index_records = index_records_by_collection[collection_name]
        collections.append(
            StorageCollectionSnapshot(
                name=collection_name,
                expected_ids={item.record_id for item in relational_records},
                relational_records=relational_records,
                index_records=index_records,
                required_metadata_fields=spec["required"],
            )
        )

    turn_rows = connection.execute(
        "SELECT turn_pk, intervention_id FROM session_dialogue_turns ORDER BY turn_pk"
    ).fetchall()
    turn_records = [
        IndexedRecord(record_id=str(record_id), parent_id=parent_id, metadata={})
        for record_id, parent_id in turn_rows
    ]
    return StorageEvalSnapshot(
        case_id=case_id,
        valid_parent_ids=parent_ids,
        collections=collections,
        additional_relational_records=turn_records,
    )


def _read_chroma_metadata_sqlite(chroma_path: Path) -> dict[str, list[IndexedRecord]]:
    """Read Chroma metadata through SQLite immutable mode, never PersistentClient."""

    sqlite_path = (chroma_path / "chroma.sqlite3").resolve(strict=True)
    required_tables = {"collections", "segments", "embeddings", "embedding_metadata"}
    connection = sqlite3.connect(sqlite_path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        existing_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?",
                ("table",),
            ).fetchall()
        }
        missing_tables = required_tables - existing_tables
        if missing_tables:
            raise ValueError(
                "unsupported Chroma SQLite schema; missing tables: " + ", ".join(sorted(missing_tables))
            )
        collection_names = {
            str(row[0]) for row in connection.execute("SELECT name FROM collections").fetchall()
        }
        rows = connection.execute(
            """
            SELECT
                c.name,
                e.embedding_id,
                m.key,
                m.string_value,
                m.int_value,
                m.float_value,
                m.bool_value
            FROM collections AS c
            JOIN segments AS s
              ON s.collection = c.id AND s.scope = 'METADATA'
            JOIN embeddings AS e
              ON e.segment_id = s.id
            LEFT JOIN embedding_metadata AS m
              ON m.id = e.id
            ORDER BY c.name, e.embedding_id, m.key
            """
        ).fetchall()
    finally:
        connection.close()

    metadata_by_record: dict[tuple[str, str], dict[str, Any]] = {}
    for collection_name, record_id, key, string_value, int_value, float_value, bool_value in rows:
        metadata = metadata_by_record.setdefault((str(collection_name), str(record_id)), {})
        if key is None:
            continue
        if key in metadata:
            raise ValueError(f"duplicate Chroma metadata key {key!r} for {collection_name}/{record_id}")
        metadata[str(key)] = _chroma_scalar(string_value, int_value, float_value, bool_value)

    result: dict[str, list[IndexedRecord]] = {name: [] for name in collection_names}
    for (collection_name, record_id), metadata in metadata_by_record.items():
        parent_id = metadata.get("session_id")
        result.setdefault(collection_name, []).append(
            IndexedRecord(
                record_id=record_id,
                parent_id=int(parent_id) if parent_id is not None else None,
                metadata=metadata,
            )
        )
    return result


def _chroma_scalar(
    string_value: str | None,
    int_value: int | None,
    float_value: float | None,
    bool_value: int | None,
) -> Any:
    populated = [value is not None for value in (string_value, int_value, float_value, bool_value)]
    if sum(populated) != 1:
        raise ValueError("Chroma metadata scalar must have exactly one populated value column")
    if string_value is not None:
        return string_value
    if int_value is not None:
        return int_value
    if float_value is not None:
        return float_value
    return bool(bool_value)
