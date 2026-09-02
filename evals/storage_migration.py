from __future__ import annotations

"""Build and apply a Storage metadata migration without mutating source artifacts.

The source Chroma directory is treated as immutable.  ``plan_metadata_migration``
only reads it; ``apply_metadata_migration`` first copies it to a new directory
and updates that copy.  This boundary is intentional because opening a Chroma
PersistentClient can write housekeeping files even for an apparent read.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import duckdb

from evals.adapters.storage import COLLECTION_SPECS, _read_chroma_metadata_sqlite


def expected_metadata_from_duckdb(db_path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the complete Chroma metadata contract derived from DuckDB rows."""

    connection = duckdb.connect(database=str(Path(db_path).resolve(strict=True)), read_only=True)
    try:
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for collection, spec in COLLECTION_SPECS.items():
            table = spec["table"]
            rows = connection.execute(f"SELECT * FROM {table} ORDER BY chunk_id").fetchall()
            columns = [item[0] for item in connection.execute(f"DESCRIBE {table}").fetchall()]
            result[collection] = {
                str(row[columns.index("chunk_id")]): _metadata_row(collection, dict(zip(columns, row, strict=True)))
                for row in rows
            }
        return result
    finally:
        connection.close()


def _metadata_row(collection: str, row: dict[str, Any]) -> dict[str, Any]:
    # Restrict writes to the versioned required contract. Optional DuckDB
    # columns (for example ``analogy_or_metaphor``) remain untouched so a
    # migration cannot turn an absent optional value into a noisy full-index
    # rewrite.
    fields = COLLECTION_SPECS[collection]["required"]
    metadata: dict[str, Any] = {}
    for field in fields:
        value = row.get(field)
        if isinstance(value, (list, tuple)):
            value = json.dumps(list(value), ensure_ascii=False)
        elif value is None:
            value = ""
        metadata[field] = value
    return metadata


def plan_metadata_migration(
    db_path: str | Path,
    chroma_dir: str | Path,
) -> dict[str, Any]:
    """Create a deterministic, JSON-serialisable migration diff."""

    expected = expected_metadata_from_duckdb(db_path)
    actual_records = _read_chroma_metadata_sqlite(Path(chroma_dir).resolve(strict=True))
    actual = {
        collection: {record.record_id: record.metadata for record in records}
        for collection, records in actual_records.items()
    }
    collections: dict[str, Any] = {}
    total_changes = 0
    for collection in COLLECTION_SPECS:
        expected_rows = expected.get(collection, {})
        actual_rows = actual.get(collection, {})
        missing_ids = sorted(set(expected_rows) - set(actual_rows))
        unexpected_ids = sorted(set(actual_rows) - set(expected_rows))
        changed: list[dict[str, Any]] = []
        for record_id in sorted(set(expected_rows) & set(actual_rows)):
            before = actual_rows[record_id]
            after = expected_rows[record_id]
            fields = {
                field: {"before": before.get(field), "after": value}
                for field, value in after.items()
                if before.get(field) != value
            }
            if fields:
                changed.append({"id": record_id, "fields": fields})
        total = len(missing_ids) + len(unexpected_ids) + len(changed)
        total_changes += total
        collections[collection] = {
            "expected_count": len(expected_rows),
            "actual_count": len(actual_rows),
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "changed": changed,
            "change_count": total,
        }
    return {
        "schema": "storage-metadata-migration/v1",
        "source": {"db": str(Path(db_path).resolve()), "chroma": str(Path(chroma_dir).resolve())},
        "total_changes": total_changes,
        "collections": collections,
    }


def apply_metadata_migration(
    db_path: str | Path,
    source_chroma: str | Path,
    target_chroma: str | Path,
) -> dict[str, Any]:
    """Copy source Chroma to a new directory and update only the copy."""

    source = Path(source_chroma).resolve(strict=True)
    target = Path(target_chroma).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing target: {target}")
    shutil.copytree(source, target)
    expected = expected_metadata_from_duckdb(db_path)
    import chromadb

    client = chromadb.PersistentClient(path=str(target))
    updated: dict[str, int] = {}
    for collection, rows in expected.items():
        coll = client.get_collection(collection)
        if rows:
            coll.update(ids=list(rows), metadatas=list(rows.values()))
        updated[collection] = len(rows)
    return {"schema": "storage-metadata-migration/v1", "target": str(target), "updated": updated}
