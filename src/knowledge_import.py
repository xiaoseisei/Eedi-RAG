"""Batch promotion and checkpoint primitives for the full knowledge import."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.chunker import SessionChunker
from src.models import CleanedSession, ExtractedPIU


class BatchAuditError(RuntimeError):
    """A promoted batch does not match the authoritative source contract."""


@dataclass(frozen=True)
class BatchAudit:
    batch_id: str
    status: str
    expected_sessions: int
    actual_sessions: int
    expected_misconception_cards: int
    actual_misconception_cards: int
    expected_strategy_cards: int
    actual_strategy_cards: int
    expected_windows: int
    actual_windows: int
    expected_misconception_vectors: int
    actual_misconception_vectors: int
    expected_strategy_vectors: int
    actual_strategy_vectors: int
    expected_window_vectors: int
    actual_window_vectors: int
    missing_session_ids: tuple[int, ...] = ()
    missing_misconception_card_ids: tuple[str, ...] = ()
    missing_strategy_card_ids: tuple[str, ...] = ()
    missing_window_ids: tuple[str, ...] = ()
    missing_misconception_vector_ids: tuple[str, ...] = ()
    missing_strategy_vector_ids: tuple[str, ...] = ()
    missing_window_vector_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def load_cleaned_sessions(path: Path | str) -> list[CleanedSession]:
    """Load every non-empty JSONL line and reject malformed/duplicate input."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"cleaned session source does not exist: {source}")
    sessions: list[CleanedSession] = []
    seen: set[int] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                session = CleanedSession.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid CleanedSession at {source}:{line_number}: {exc}") from exc
            if session.intervention_id in seen:
                raise ValueError(f"duplicate intervention_id in source: {session.intervention_id}")
            seen.add(session.intervention_id)
            sessions.append(session)
    if not sessions:
        raise ValueError(f"source contains no CleanedSession records: {source}")
    return sorted(sessions, key=lambda item: item.intervention_id)


def source_sha256(path: Path | str) -> str:
    """Return the digest used to prove a checkpoint belongs to this source."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_sessions(
    sessions: Sequence[CleanedSession], batch_size: int = 100
) -> list[list[CleanedSession]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = sorted(sessions, key=lambda item: item.intervention_id)
    return [ordered[offset : offset + batch_size] for offset in range(0, len(ordered), batch_size)]


def _query_ids(storage: Any, query: str, params: Sequence[Any] = ()) -> set[str]:
    return {str(row[0]) for row in storage.duck_conn.execute(query, list(params)).fetchall()}


def _collection_ids(storage: Any, attribute: str, expected_ids: set[str]) -> set[str]:
    """Read IDs from a Chroma collection without depending on document payloads."""

    collection = getattr(storage, attribute, None)
    if collection is None:
        return set()
    if not expected_ids:
        return set()
    result = collection.get(ids=sorted(expected_ids))
    values = result.get("ids", []) if isinstance(result, Mapping) else []
    return {str(value) for value in values}


def audit_batch(
    storage: Any,
    sessions: Sequence[CleanedSession],
    *,
    batch_id: str = "batch",
    expected_window_ids: Iterable[str] | None = None,
) -> BatchAudit:
    """Compare authoritative DuckDB rows with exactly the batch's expected IDs."""

    session_ids = tuple(sorted(int(item.intervention_id) for item in sessions))
    expected_session_ids = {str(item) for item in session_ids}
    expected_misc = {f"session_{item}_misconception" for item in session_ids}
    expected_strategy = {f"session_{item}_tutor_strategy" for item in session_ids}
    if expected_window_ids is None:
        expected_windows: set[str] = set()
        for session in sessions:
            rows = storage.duck_conn.execute(
                "SELECT chunk_id FROM sliding_window_chunks WHERE session_id = ?",
                [session.intervention_id],
            ).fetchall()
            expected_windows.update(str(row[0]) for row in rows)
    else:
        expected_windows = {str(item) for item in expected_window_ids}

    placeholders = ",".join("?" for _ in session_ids)
    actual_sessions = _query_ids(
        storage,
        f"SELECT intervention_id FROM tutoring_sessions WHERE intervention_id IN ({placeholders})",
        session_ids,
    ) if session_ids else set()
    actual_misc = _query_ids(
        storage,
        f"SELECT chunk_id FROM misconception_chunks WHERE session_id IN ({placeholders})",
        session_ids,
    ) if session_ids else set()
    actual_strategy = _query_ids(
        storage,
        f"SELECT chunk_id FROM tutor_strategy_chunks WHERE session_id IN ({placeholders})",
        session_ids,
    ) if session_ids else set()
    actual_windows = _query_ids(
        storage,
        f"SELECT chunk_id FROM sliding_window_chunks WHERE session_id IN ({placeholders})",
        session_ids,
    ) if session_ids else set()

    actual_misc_vectors = _collection_ids(storage, "coll_misconceptions", expected_misc)
    actual_strategy_vectors = _collection_ids(storage, "coll_strategies", expected_strategy)
    actual_window_vectors = _collection_ids(storage, "coll_windows", expected_windows)

    missing_sessions = tuple(sorted(int(item) for item in expected_session_ids - actual_sessions))
    missing_misc = tuple(sorted(expected_misc - actual_misc))
    missing_strategy = tuple(sorted(expected_strategy - actual_strategy))
    missing_windows = tuple(sorted(expected_windows - actual_windows))
    missing_misc_vectors = tuple(sorted(expected_misc - actual_misc_vectors))
    missing_strategy_vectors = tuple(sorted(expected_strategy - actual_strategy_vectors))
    missing_window_vectors = tuple(sorted(expected_windows - actual_window_vectors))
    status = "PASSED" if not (
        missing_sessions
        or missing_misc
        or missing_strategy
        or missing_windows
        or missing_misc_vectors
        or missing_strategy_vectors
        or missing_window_vectors
    ) else "FAILED"
    return BatchAudit(
        batch_id=batch_id,
        status=status,
        expected_sessions=len(expected_session_ids),
        actual_sessions=len(actual_sessions),
        expected_misconception_cards=len(expected_misc),
        actual_misconception_cards=len(actual_misc & expected_misc),
        expected_strategy_cards=len(expected_strategy),
        actual_strategy_cards=len(actual_strategy & expected_strategy),
        expected_windows=len(expected_windows),
        actual_windows=len(actual_windows & expected_windows),
        expected_misconception_vectors=len(expected_misc),
        actual_misconception_vectors=len(actual_misc_vectors),
        expected_strategy_vectors=len(expected_strategy),
        actual_strategy_vectors=len(actual_strategy_vectors),
        expected_window_vectors=len(expected_windows),
        actual_window_vectors=len(actual_window_vectors),
        missing_session_ids=missing_sessions,
        missing_misconception_card_ids=missing_misc,
        missing_strategy_card_ids=missing_strategy,
        missing_window_ids=missing_windows,
        missing_misconception_vector_ids=missing_misc_vectors,
        missing_strategy_vector_ids=missing_strategy_vectors,
        missing_window_vector_ids=missing_window_vectors,
    )


def promote_batch(
    storage: Any,
    sessions: Sequence[CleanedSession],
    extracted: Sequence[ExtractedPIU],
    chunker: SessionChunker,
    *,
    batch_id: str = "batch",
) -> BatchAudit:
    """Promote validated cards and windows, then fail closed if audit misses rows."""

    if not sessions:
        raise ValueError("cannot promote an empty batch")
    session_map = {int(item.intervention_id): item for item in sessions}
    extracted_map = {int(item.session_id): item for item in extracted}
    if set(session_map) != set(extracted_map):
        missing = sorted(set(session_map) - set(extracted_map))
        extra = sorted(set(extracted_map) - set(session_map))
        raise BatchAuditError(f"batch extraction set mismatch: missing={missing}, extra={extra}")
    ordered_extracted = [extracted_map[sid] for sid in sorted(session_map)]
    chunks = chunker.batch_chunk_sessions(
        [session_map[sid] for sid in sorted(session_map)],
        extracted_map=extracted_map,
        strategy="hybrid",
    )
    storage.ingest_all(
        cleaned_sessions=[session_map[sid] for sid in sorted(session_map)],
        extracted_pius=ordered_extracted,
        chunks=chunks,
    )
    expected_window_ids = [chunk.chunk_id for chunk in chunks if chunk.chunk_type == "sliding_window"]
    audit = audit_batch(
        storage,
        sessions,
        batch_id=batch_id,
        expected_window_ids=expected_window_ids,
    )
    if audit.status != "PASSED":
        raise BatchAuditError(json.dumps(audit.to_dict(), ensure_ascii=False, sort_keys=True))
    return audit


def audit_full_storage(storage: Any, sessions: Sequence[CleanedSession]) -> dict[str, Any]:
    """Verify that the complete source session set is represented in storage."""

    expected = {str(int(item.intervention_id)) for item in sessions}
    actual = _query_ids(storage, "SELECT intervention_id FROM tutoring_sessions")
    missing = tuple(sorted(int(item) for item in expected - actual))
    unexpected = tuple(sorted(int(item) for item in actual - expected))
    expected_misc = {f"session_{sid}_misconception" for sid in expected}
    expected_strategy = {f"session_{sid}_tutor_strategy" for sid in expected}
    actual_misc = _query_ids(storage, "SELECT chunk_id FROM misconception_chunks")
    actual_strategy = _query_ids(storage, "SELECT chunk_id FROM tutor_strategy_chunks")
    actual_misc_vectors = _collection_ids(storage, "coll_misconceptions", expected_misc)
    actual_strategy_vectors = _collection_ids(storage, "coll_strategies", expected_strategy)
    expected_windows: set[str] = set()
    for session in sessions:
        rows = storage.duck_conn.execute(
            "SELECT chunk_id FROM sliding_window_chunks WHERE session_id = ?",
            [session.intervention_id],
        ).fetchall()
        expected_windows.update(str(row[0]) for row in rows)
    actual_windows = _query_ids(storage, "SELECT chunk_id FROM sliding_window_chunks")
    actual_window_vectors = _collection_ids(storage, "coll_windows", expected_windows)
    missing_misc = tuple(sorted(expected_misc - actual_misc))
    missing_strategy = tuple(sorted(expected_strategy - actual_strategy))
    missing_misc_vectors = tuple(sorted(expected_misc - actual_misc_vectors))
    missing_strategy_vectors = tuple(sorted(expected_strategy - actual_strategy_vectors))
    checks = {
        "session_id_set": not missing and not unexpected,
        "misconception_card_count": len(actual_misc) == len(expected_misc),
        "strategy_card_count": len(actual_strategy) == len(expected_strategy),
        "misconception_vector_count": len(actual_misc_vectors) == len(expected_misc),
        "strategy_vector_count": len(actual_strategy_vectors) == len(expected_strategy),
        "window_count": len(actual_windows) == len(expected_windows),
        "window_vector_count": len(actual_window_vectors) == len(expected_windows),
    }
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "expected_sessions": len(expected),
        "actual_sessions": len(actual),
        "expected_misconception_cards": len(expected_misc),
        "actual_misconception_cards": len(actual_misc),
        "expected_strategy_cards": len(expected_strategy),
        "actual_strategy_cards": len(actual_strategy),
        "actual_misconception_vectors": len(actual_misc_vectors),
        "actual_strategy_vectors": len(actual_strategy_vectors),
        "expected_windows": len(expected_windows),
        "actual_windows": len(actual_windows),
        "actual_window_vectors": len(actual_window_vectors),
        "missing_session_ids": missing,
        "unexpected_session_ids": unexpected,
        "missing_misconception_card_ids": missing_misc,
        "missing_strategy_card_ids": missing_strategy,
        "missing_misconception_vector_ids": missing_misc_vectors,
        "missing_strategy_vector_ids": missing_strategy_vectors,
        "missing_window_ids": tuple(sorted(expected_windows - actual_windows)),
        "missing_window_vector_ids": tuple(sorted(expected_windows - actual_window_vectors)),
        "checks": checks,
    }


__all__ = [
    "BatchAudit",
    "BatchAuditError",
    "audit_batch",
    "audit_full_storage",
    "load_cleaned_sessions",
    "partition_sessions",
    "promote_batch",
    "source_sha256",
]
