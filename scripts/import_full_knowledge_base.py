"""Import the complete cleaned session corpus in resumable 100-session batches.

The coordinator owns storage writes.  Worker threads used by the incremental
extractor only call the real semantic extractor and populate its success-only
cache; a batch is promoted to DuckDB/Chroma only after every session has a
validated semantic result and the post-write audit passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import SessionChunker
from src.knowledge_import import (
    BatchAuditError,
    audit_batch,
    audit_full_storage,
    load_cleaned_sessions,
    partition_sessions,
    promote_batch,
    source_sha256,
)
from src.models import CleanedSession, ExtractedPIU
from src.storage_manager import DualEngineStorageManager
from scripts.incremental_card_ingest import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SCHEMA_VERSION,
    run_incremental_ingestion,
)

logger = logging.getLogger(__name__)
STATE_SCHEMA_VERSION = "full-knowledge-import/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _read_cards(path: Path) -> list[ExtractedPIU]:
    if not path.exists():
        raise FileNotFoundError(f"incremental extraction did not produce cards: {path}")
    cards: list[ExtractedPIU] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                card = ExtractedPIU.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid extracted card at {path}:{line_number}: {exc}") from exc
            if card.extraction_status != "success" or card.misconception is None or card.tutor_strategy is None:
                raise ValueError(f"non-success extracted card at {path}:{line_number}")
            cards.append(card)
    return cards


def _new_manifest(
    *,
    source: Path,
    source_digest: str,
    sessions: Sequence[CleanedSession],
    batch_size: int,
    workers: int,
    model: str | None,
    prompt_version: str,
    schema_version: str,
    temperature: float,
    max_retries: int,
    embedding_backend: str,
) -> dict[str, Any]:
    batches = partition_sessions(sessions, batch_size)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "source": str(source.resolve()),
        "source_sha256": source_digest,
        "source_session_count": len(sessions),
        "batch_size": batch_size,
        "workers": workers,
        "embedding_backend": embedding_backend,
        "contract": {
            "model": model or "",
            "prompt_version": prompt_version,
            "schema_version": schema_version,
            "temperature": temperature,
            "max_retries": max_retries,
        },
        "status": "RUNNING",
        "batches": [
            {
                "batch_id": f"batch-{index:04d}",
                "ordinal": index,
                "session_ids": [int(session.intervention_id) for session in batch],
                "status": "PENDING",
                "attempts": 0,
            }
            for index, batch in enumerate(batches, 1)
        ],
    }


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    source: Path,
    source_digest: str,
    sessions: Sequence[CleanedSession],
    batch_size: int,
    prompt_version: str,
    schema_version: str,
    model: str | None,
    temperature: float,
    max_retries: int,
    embedding_backend: str,
) -> None:
    if manifest.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version does not match this importer")
    if manifest.get("source_sha256") != source_digest:
        raise ValueError("source changed since checkpoint; start a new state directory")
    if manifest.get("source_session_count") != len(sessions):
        raise ValueError("source session count changed since checkpoint")
    if manifest.get("batch_size") != batch_size:
        raise ValueError("batch_size changed since checkpoint; resume requires the original value")
    stored_embedding_backend = manifest.get("embedding_backend")
    if stored_embedding_backend is not None and stored_embedding_backend != embedding_backend:
        raise ValueError("embedding backend changed since checkpoint; use a new state directory")
    expected_ids = [int(session.intervention_id) for session in sessions]
    actual_ids = [sid for batch in manifest.get("batches", []) for sid in batch.get("session_ids", [])]
    if actual_ids != expected_ids:
        raise ValueError("source session ordering/IDs changed since checkpoint")
    contract = manifest.get("contract", {})
    expected = {
        "model": model or "",
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "temperature": temperature,
        "max_retries": max_retries,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("extraction contract changed since checkpoint; resume requires the original contract")
    if Path(str(manifest.get("source", ""))).resolve() != source.resolve():
        raise ValueError("checkpoint belongs to a different source path")


def run_full_import(
    *,
    source_path: Path | str = PROJECT_ROOT / "data" / "cleaned_sessions.jsonl",
    db_path: Path | str = PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb",
    chroma_dir: Path | str = PROJECT_ROOT / "data" / "chroma",
    state_dir: Path | str = PROJECT_ROOT / "reports" / "staging" / "full-import",
    batch_size: int = 100,
    workers: int = 8,
    resume: bool = True,
    start_batch: int = 1,
    max_batches: int | None = None,
    model: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    temperature: float = 0.1,
    max_retries: int = 3,
    embedding_backend: str | None = None,
    storage: Any | None = None,
    storage_factory: Callable[[], Any] | None = None,
    ingestion_runner: Callable[..., Mapping[str, Any]] = run_incremental_ingestion,
) -> dict[str, Any]:
    """Run or resume the full import, stopping when the selected range ends."""

    if batch_size <= 0 or workers <= 0 or start_batch <= 0:
        raise ValueError("batch_size, workers and start_batch must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    # The existing production Chroma collections were built with the explicit
    # deterministic backend.  Preserve that contract by default; choosing a
    # different provider must be an explicit full-index migration decision.
    effective_embedding_backend = embedding_backend or os.environ.get("EMBEDDING_BACKEND") or "deterministic"
    if effective_embedding_backend not in {"deterministic", "siliconflow"}:
        raise ValueError("embedding_backend must be deterministic or siliconflow")
    source = Path(source_path).resolve()
    state = Path(state_dir).resolve()
    sessions = load_cleaned_sessions(source)
    digest = source_sha256(source)
    state.mkdir(parents=True, exist_ok=True)
    manifest_path = state / "manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_manifest(
            manifest,
            source=source,
            source_digest=digest,
            sessions=sessions,
            batch_size=batch_size,
            prompt_version=prompt_version,
            schema_version=schema_version,
            model=model or os.environ.get("LLM_MODEL"),
            temperature=temperature,
            max_retries=max_retries,
            embedding_backend=effective_embedding_backend,
        )
    else:
        manifest = _new_manifest(
            source=source,
            source_digest=digest,
            sessions=sessions,
            batch_size=batch_size,
            workers=workers,
            model=model or os.environ.get("LLM_MODEL"),
            prompt_version=prompt_version,
            schema_version=schema_version,
            temperature=temperature,
            max_retries=max_retries,
            embedding_backend=effective_embedding_backend,
        )
        _atomic_json(manifest_path, manifest)

    batches = partition_sessions(sessions, batch_size)
    end_batch = len(batches) if max_batches is None else min(len(batches), start_batch + max_batches - 1)
    manifest.setdefault("embedding_backend", effective_embedding_backend)
    manifest["status"] = "RUNNING"
    manifest["updated_at"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    owned_storage = storage is None
    if storage is None:
        if storage_factory is not None:
            storage = storage_factory()
        else:
            storage = DualEngineStorageManager(
                db_path=db_path,
                chroma_dir=chroma_dir,
                embedding_backend=effective_embedding_backend,
            )
    cache_dir = state / "extraction-cache"
    chunker = SessionChunker(window_size=6, step=3, default_strategy="hybrid")

    try:
        for ordinal in range(start_batch, end_batch + 1):
            batch_entry = manifest["batches"][ordinal - 1]
            batch_sessions = batches[ordinal - 1]
            if batch_entry.get("status") == "COMMITTED":
                logger.info("跳过已提交批次 %s", batch_entry["batch_id"])
                continue
            batch_entry["attempts"] = int(batch_entry.get("attempts", 0)) + 1
            batch_entry["status"] = "EXTRACTING"
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            logger.info(
                "开始批次 %s: sessions %s-%s (%s 场)",
                batch_entry["batch_id"],
                batch_sessions[0].intervention_id,
                batch_sessions[-1].intervention_id,
                len(batch_sessions),
            )
            try:
                extraction = ingestion_runner(
                    sessions=batch_sessions,
                    cache_dir=cache_dir,
                    mode="changed-only",
                    workers=workers,
                    model=model,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    temperature=temperature,
                    max_retries=max_retries,
                )
                batch_entry["extraction"] = dict(extraction)
                if extraction.get("status") != "SUCCESS" or int(extraction.get("failed_count", 0)):
                    raise BatchAuditError(
                        f"batch extraction failed: failed_count={extraction.get('failed_count')}, "
                        f"errors={extraction.get('errors', [])}"
                    )
                cards = _read_cards(Path(str(extraction["cards_path"])))
                batch_entry["status"] = "PROMOTING"
                manifest["updated_at"] = _utc_now()
                _atomic_json(manifest_path, manifest)
                audit = promote_batch(
                    storage,
                    batch_sessions,
                    cards,
                    chunker,
                    batch_id=batch_entry["batch_id"],
                )
                batch_entry["audit"] = audit.to_dict()
                batch_entry["status"] = "COMMITTED"
                batch_entry["committed_at"] = _utc_now()
                logger.info("批次 %s 检查通过并提交", batch_entry["batch_id"])
            except Exception as exc:
                batch_entry["status"] = "FAILED"
                batch_entry["error"] = f"{type(exc).__name__}: {exc}"
                manifest["status"] = "FAILED"
                manifest["updated_at"] = _utc_now()
                _atomic_json(manifest_path, manifest)
                raise
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
    finally:
        if owned_storage and storage is not None:
            storage.close()

    full_audit = None
    all_committed = all(item.get("status") == "COMMITTED" for item in manifest["batches"])
    if all_committed:
        # The write connection has been closed; use a fresh read connection for
        # the complete source-set audit.
        audit_storage = storage
        owns_audit_storage = False
        if owned_storage:
            audit_storage = storage_factory() if storage_factory is not None else DualEngineStorageManager(
                db_path=db_path,
                chroma_dir=chroma_dir,
                embedding_backend=effective_embedding_backend,
            )
            owns_audit_storage = True
        try:
            full_audit = audit_full_storage(audit_storage, sessions)
        finally:
            if owns_audit_storage and audit_storage is not None:
                audit_storage.close()
        manifest["full_audit"] = full_audit
        if full_audit["status"] != "PASSED":
            manifest["status"] = "FAILED"
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            raise BatchAuditError(json.dumps(full_audit, ensure_ascii=False, sort_keys=True))
    manifest["status"] = "SUCCESS" if all_committed else "PARTIAL"
    manifest["updated_at"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable concurrent full knowledge-base import")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "cleaned_sessions.jsonl")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb")
    parser.add_argument("--chroma-dir", type=Path, default=PROJECT_ROOT / "data" / "chroma")
    parser.add_argument("--embedding-backend", choices=("deterministic", "siliconflow"), default=None)
    parser.add_argument("--state-dir", type=Path, default=PROJECT_ROOT / "reports" / "staging" / "full-import")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_full_import(
            source_path=args.source,
            db_path=args.db,
            chroma_dir=args.chroma_dir,
            embedding_backend=args.embedding_backend,
            state_dir=args.state_dir,
            batch_size=args.batch_size,
            workers=args.workers,
            resume=not args.no_resume,
            start_batch=args.start_batch,
            max_batches=args.max_batches,
            model=args.model,
            prompt_version=args.prompt_version,
            schema_version=args.schema_version,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
    except Exception as exc:
        logger.error("完整知识库导入失败: %s", exc)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
