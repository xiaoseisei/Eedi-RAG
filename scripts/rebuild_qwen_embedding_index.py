from __future__ import annotations

"""Rebuild a Chroma index with the configured SiliconFlow Qwen embedding API.

The DuckDB source is opened read-only and the target Chroma directory must not
exist.  The existing production Chroma directory is never opened, copied, or
modified by this command.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction
from src.storage_manager import (
    build_misconception_embedding_doc,
    build_tutor_strategy_embedding_doc,
)
from src.models import StudentMisconceptionProfile, TutorStrategyProfile


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Chroma target contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _combined_hash(db_path: Path, chroma_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"duckdb\0")
    digest.update(_file_sha256(db_path).encode("ascii"))
    digest.update(b"chroma\0")
    digest.update(_directory_sha256(chroma_path).encode("ascii"))
    return digest.hexdigest()


def _json_metadata(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return "" if value is None else value


def _load_documents(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    connection = duckdb.connect(database=str(db_path.resolve(strict=True)), read_only=True)
    try:
        questions = {
            int(row[0]): str(row[1] or "")
            for row in connection.execute(
                "SELECT intervention_id, question_text FROM tutoring_sessions"
            ).fetchall()
        }
        result: dict[str, list[dict[str, Any]]] = {
            "student_misconceptions": [],
            "tutor_strategies": [],
            "fallback_windows": [],
        }
        misc_cols = [row[0] for row in connection.execute("DESCRIBE misconception_chunks").fetchall()]
        for raw in connection.execute("SELECT * FROM misconception_chunks ORDER BY chunk_id").fetchall():
            row = dict(zip(misc_cols, raw, strict=True))
            card = StudentMisconceptionProfile(
                session_id=int(row["session_id"]),
                question_id=int(row["question_id"]),
                subject_path=row.get("subject_path") or "",
                misconception_name=row["misconception_name"],
                error_choice=row.get("error_choice") or None,
                deep_mechanism=row["deep_mechanism"],
                confusion_triggers=list(row.get("confusion_triggers") or []),
                verbatim_student_quotes=list(row.get("verbatim_student_quotes") or []),
                source_turn_ids=list(row.get("source_turn_ids") or []),
            )
            metadata = {
                field: _json_metadata(row.get(field))
                for field in (
                    "session_id", "question_id", "subject_path", "misconception_name",
                    "error_choice", "deep_mechanism", "confusion_triggers",
                    "verbatim_student_quotes", "source_turn_ids",
                )
            }
            result["student_misconceptions"].append({
                "id": str(row["chunk_id"]),
                "document": build_misconception_embedding_doc(
                    card,
                    question_text=questions.get(int(row["session_id"]), ""),
                    subject_path=row.get("subject_path"),
                ),
                "metadata": metadata,
            })

        strat_cols = [row[0] for row in connection.execute("DESCRIBE tutor_strategy_chunks").fetchall()]
        for raw in connection.execute("SELECT * FROM tutor_strategy_chunks ORDER BY chunk_id").fetchall():
            row = dict(zip(strat_cols, raw, strict=True))
            card = TutorStrategyProfile(
                session_id=int(row["session_id"]),
                question_id=int(row["question_id"]),
                pedagogical_goal=row["pedagogical_goal"],
                strategy_category=row["strategy_category"],
                key_aha_question=row["key_aha_question"],
                scaffolding_steps=list(row.get("scaffolding_steps") or []),
                analogy_or_metaphor=row.get("analogy_or_metaphor") or None,
                talk_moves=list(row.get("talk_moves") or []),
                resolution_outcome=row["resolution_outcome"],
                source_turn_ids=list(row.get("source_turn_ids") or []),
            )
            metadata = {
                field: _json_metadata(row.get(field))
                for field in (
                    "session_id", "question_id", "subject_path", "pedagogical_goal",
                    "strategy_category", "key_aha_question", "scaffolding_steps",
                    "analogy_or_metaphor", "talk_moves", "resolution_outcome", "source_turn_ids",
                )
            }
            result["tutor_strategies"].append({
                "id": str(row["chunk_id"]),
                "document": build_tutor_strategy_embedding_doc(
                    card,
                    question_text=questions.get(int(row["session_id"]), ""),
                    subject_path=row.get("subject_path") or "",
                ),
                "metadata": metadata,
            })

        window_cols = [row[0] for row in connection.execute("DESCRIBE sliding_window_chunks").fetchall()]
        for raw in connection.execute("SELECT * FROM sliding_window_chunks ORDER BY chunk_id").fetchall():
            row = dict(zip(window_cols, raw, strict=True))
            result["fallback_windows"].append({
                "id": str(row["chunk_id"]),
                "document": str(row["content"]),
                "metadata": {
                    field: _json_metadata(row.get(field))
                    for field in (
                        "session_id", "question_id", "subject_path", "window_start_turn",
                        "window_end_turn", "source_turn_ids",
                    )
                },
            })
        return result
    finally:
        connection.close()


def _upsert_rows(
    collection: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 5_000,
) -> None:
    """Upsert rows in batches below Chroma's local maximum batch size."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection.upsert(
            ids=[row["id"] for row in batch],
            documents=[row["document"] for row in batch],
            metadatas=[row["metadata"] for row in batch],
        )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an isolated Qwen3 Embedding Chroma index")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--target", type=Path, required=True, help="new Chroma directory; must not exist")
    parser.add_argument("--batch-size", type=int, help="override EMBEDDING_BATCH_SIZE")
    args = parser.parse_args(argv)
    _load_dotenv_if_available()

    db_path = args.db.resolve(strict=True)
    target = args.target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing Chroma target: {target}")
    if not db_path.is_file():
        raise ValueError(f"DuckDB path is not a file: {db_path}")

    source_db_hash = _file_sha256(db_path)
    documents = _load_documents(db_path)
    provider = SiliconFlowQwen3EmbeddingFunction.from_env()
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        provider.batch_size = args.batch_size

    import chromadb

    contract = provider.contract()

    target.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target))
    started = time.perf_counter()
    collection_counts: dict[str, int] = {}
    for collection_name, rows in documents.items():
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "embedding_provider": contract.provider,
                "embedding_model": contract.model,
                "embedding_dimension": contract.dimension,
                "embedding_index_version": contract.index_version,
            },
            embedding_function=provider,
        )
        if rows:
            _upsert_rows(collection, rows)
        collection_counts[collection_name] = len(rows)

    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    del client
    elapsed = time.perf_counter() - started
    if not target.is_dir():
        raise RuntimeError("Chroma target disappeared after build")
    target_hash = _directory_sha256(target)
    if _file_sha256(db_path) != source_db_hash:
        raise RuntimeError("DuckDB artifact changed during embedding index build")
    report = {
        "schema_version": "embedding-index-build/v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "provider": {
            "name": provider.name(),
            "base_url": provider.base_url,
            "model": provider.model,
            "dimension": provider.dimension,
            "index_version": contract.index_version,
        },
        "collection_counts": collection_counts,
        "target": str(target),
        "target_chroma_sha256": target_hash,
        "source_duckdb_sha256": source_db_hash,
        "elapsed_seconds": round(elapsed, 3),
        "usage": {
            "request_count": provider.usage.request_count,
            "retry_count": provider.usage.retry_count,
            "input_tokens": provider.usage.input_tokens,
            "total_tokens": provider.usage.total_tokens,
        },
        "notes": [
            "DuckDB opened read_only; no production Chroma path was opened",
            "target directory was required to be absent and was created as a new index",
            "API key is intentionally excluded from this report",
        ],
    }
    report_path = target.parent / f"{target.name}.build.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
