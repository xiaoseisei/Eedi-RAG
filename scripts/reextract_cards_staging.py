from __future__ import annotations

"""Re-extract the historical 100-card corpus into isolated staging artifacts.

The production DuckDB/Chroma paths are never opened for writing. Extraction is
atomic at the corpus level: no staging storage is created until every selected
session has produced two strict, grounded cards.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import SessionChunker
from src.extract_knowledge import EXTRACTION_PROMPT_VERSION, extract_knowledge_from_session
from src.evidence_index import bind_card_to_evidence_index, build_evidence_index
from src.models import CleanedSession, ExtractedPIU
from src.storage_manager import DualEngineStorageManager


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
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
    digest.update(_sha256(db_path).encode("ascii"))
    digest.update(b"chroma\0")
    digest.update(_directory_sha256(chroma_path).encode("ascii"))
    return digest.hexdigest()


def _load_sessions(path: Path) -> dict[int, CleanedSession]:
    sessions: dict[int, CleanedSession] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            session = CleanedSession.model_validate(json.loads(line))
            sessions[session.intervention_id] = session
    return sessions


def _db_session_ids(path: Path) -> list[int]:
    import duckdb

    connection = duckdb.connect(database=str(path.resolve(strict=True)), read_only=True)
    try:
        return [int(row[0]) for row in connection.execute("SELECT intervention_id FROM tutoring_sessions ORDER BY intervention_id").fetchall()]
    finally:
        connection.close()


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_reextraction(
    *,
    sessions_path: Path,
    production_db: Path,
    staging_dir: Path,
    expected_count: int = 100,
    model: str | None = None,
    base_url: str | None = None,
    max_retries: int = 2,
    timeout_seconds: float = 180.0,
    workers: int = 1,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite staging directory: {staging_dir}")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    api_key = os.getenv("LLM_API_KEY")
    effective_model = model or os.getenv("LLM_MODEL")
    effective_base_url = base_url or os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("LLM_API_KEY 未配置，拒绝伪造全量抽取")
    if not effective_model:
        raise RuntimeError("LLM_MODEL 未配置，拒绝隐式模型默认值")
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("缺少 openai 依赖，无法执行真实全量抽取") from exc

    sessions = _load_sessions(sessions_path)
    source_ids = _db_session_ids(production_db)
    if len(source_ids) != expected_count:
        raise ValueError(f"production DB session count mismatch: expected={expected_count}, actual={len(source_ids)}")
    missing = [session_id for session_id in source_ids if session_id not in sessions]
    if missing:
        raise ValueError(f"cleaned_sessions 缺少生产 DB 会话: {missing[:20]}")
    selected = [sessions[session_id] for session_id in source_ids]

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url
    client = openai.OpenAI(**client_kwargs)
    chunker = SessionChunker(window_size=6, step=3, default_strategy="hybrid")
    extracted_by_session: dict[int, ExtractedPIU] = {}
    chunks_by_session: dict[int, list[Any]] = {}
    audit_rows_by_session: dict[int, dict[str, Any]] = {}
    failures_by_session: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()

    def extract_one(session: CleanedSession) -> tuple[ExtractedPIU, list[Any], dict[str, Any]]:
        case_started = time.perf_counter()
        piu = extract_knowledge_from_session(
            session,
            llm_client=client,
            model=effective_model,
            max_retries=max_retries,
            temperature=temperature,
            evidence_audit=False,
            semantic_only=True,
        )
        if piu.extraction_status != "success" or piu.misconception is None or piu.tutor_strategy is None:
            raise RuntimeError("抽取结果不是完整 success 双卡片")
        session_map = {session.intervention_id: session}
        evidence_index = build_evidence_index(session, window_size=6, step=3)
        piu = piu.model_copy(update={
            "misconception": bind_card_to_evidence_index(piu.misconception, evidence_index),
            "tutor_strategy": bind_card_to_evidence_index(piu.tutor_strategy, evidence_index),
        })
        # Reuse the same strict storage validator before staging write.
        DualEngineStorageManager._validate_extracted_pius([piu], session_map)
        session_chunks = chunker.chunk_session(session, piu)
        audit = {
            "session_id": session.intervention_id,
            "status": "SUCCESS",
            "misconception_source_turn_ids": piu.misconception.source_turn_ids,
            "tutor_strategy_source_turn_ids": piu.tutor_strategy.source_turn_ids,
            "elapsed_seconds": round(time.perf_counter() - case_started, 3),
        }
        return piu, session_chunks, audit

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="card-extract") as executor:
        futures = {executor.submit(extract_one, session): session for session in selected}
        completed = 0
        try:
            for future in as_completed(futures):
                session = futures[future]
                try:
                    piu, session_chunks, audit = future.result()
                except Exception as exc:
                    failures_by_session[session.intervention_id] = {
                        "session_id": session.intervention_id,
                        "status": "FAILED",
                        "error": str(exc).replace("\n", " ")[:2000],
                    }
                    completed += 1
                    print(f"[{completed}/{len(selected)}] Session #{session.intervention_id} FAILED (recorded; continuing)", flush=True)
                    continue
                extracted_by_session[session.intervention_id] = piu
                chunks_by_session[session.intervention_id] = session_chunks
                audit_rows_by_session[session.intervention_id] = audit
                completed += 1
                print(f"[{completed}/{len(selected)}] Session #{session.intervention_id} SUCCESS", flush=True)
        finally:
            # Context-manager shutdown waits for in-flight provider calls; no
            # staging persistence occurs until every future has completed.
            pass

    extracted = [extracted_by_session[session.intervention_id] for session in selected]
    chunks = [chunk for session in selected for chunk in chunks_by_session[session.intervention_id]]
    audit_rows = [audit_rows_by_session.get(session.intervention_id) or failures_by_session[session.intervention_id] for session in selected]

    # Preserve a resumable checkpoint after the full traversal. Successful
    # cards can be retained while only failed sessions receive targeted repair.
    partial_path = staging_dir / "extracted_pius_partial.jsonl"
    _write_jsonl(partial_path, [item.model_dump() for item in extracted])
    audit_path = staging_dir / "extraction_audit.jsonl"
    _write_jsonl(audit_path, audit_rows)
    if failures_by_session:
        failure_path = staging_dir / "extraction_failures.jsonl"
        _write_jsonl(failure_path, [failures_by_session[sid] for sid in sorted(failures_by_session)])
        failure_manifest = {
            "schema_version": "card-reextraction-staging-partial/v1",
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "status": "PARTIAL_BLOCKED_PENDING_REPAIR",
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "model": effective_model,
            "evidence_audit": False,
            "workers": workers,
            "temperature": temperature,
            "session_count": len(selected),
            "success_count": len(extracted),
            "failure_count": len(failures_by_session),
            "failed_session_ids": sorted(failures_by_session),
            "files": {
                "partial_cards": _sha256(partial_path),
                "audit": _sha256(audit_path),
                "failures": _sha256(failure_path),
            },
            "note": "No staging DB/Chroma created until targeted repair clears all failures; production artifacts untouched.",
        }
        (staging_dir / "manifest.json").write_text(json.dumps(failure_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return failure_manifest

    if len(extracted) != expected_count:
        raise RuntimeError(f"抽取数量不完整: expected={expected_count}, actual={len(extracted)}")

    # Persist raw cards/audit first; storage creation remains isolated staging.
    extracted_path = staging_dir / "extracted_pius.jsonl"
    _write_jsonl(extracted_path, [item.model_dump() for item in extracted])
    staging_db = staging_dir / "tutoring_knowledge.duckdb"
    staging_chroma = staging_dir / "chroma"
    storage = DualEngineStorageManager(
        db_path=staging_db,
        chroma_dir=staging_chroma,
        embedding_backend="deterministic",
    )
    try:
        storage.ingest_all(cleaned_sessions=selected, extracted_pius=extracted, chunks=chunks)
        snapshot = storage.get_storage_audit_snapshot()
    finally:
        storage.close()

    source_db_hash = _sha256(production_db)
    manifest = {
        "schema_version": "card-reextraction-staging/v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "SUCCESS",
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "model": effective_model,
        "evidence_audit": False,
        "workers": workers,
        "temperature": temperature,
        "session_count": len(selected),
        "extracted_card_count": len(extracted) * 2,
        "window_chunk_count": len(chunks),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "source": {
            "sessions": {"path": str(sessions_path.resolve()), "sha256": _sha256(sessions_path)},
            "production_db": {"path": str(production_db.resolve()), "sha256": source_db_hash, "session_ids": source_ids},
        },
        "staging": {
            "db": {"path": str(staging_db.resolve()), "sha256": _sha256(staging_db)},
            "chroma": {"path": str(staging_chroma.resolve()), "sha256": _directory_sha256(staging_chroma)},
            "combined_sha256": _combined_hash(staging_db, staging_chroma),
            "snapshot": snapshot,
        },
        "files": {name: _sha256(staging_dir / name) for name in ("extracted_pius.jsonl", "extraction_audit.jsonl")},
        "limitations": [
            "Staging uses deterministic embeddings; a Qwen index must be built separately before Qwen retrieval comparison.",
            "Cards are generated by real LLM and strict grounding, but pointer completeness remains a technical proxy until human qrels review.",
            "Production DB/Chroma were never opened for writing.",
        ],
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=PROJECT_ROOT / "data" / "cleaned_sessions.jsonl")
    parser.add_argument("--production-db", type=Path, default=PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb")
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=1, help="并发 LLM 抽取 worker 数；默认 1")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        manifest = run_reextraction(
            sessions_path=args.sessions.resolve(strict=True),
            production_db=args.production_db.resolve(strict=True),
            staging_dir=args.staging_dir.resolve(),
            expected_count=args.expected_count,
            model=args.model,
            base_url=args.base_url,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
            workers=args.workers,
            temperature=args.temperature,
        )
    except Exception as exc:
        # Persist an explicit blocked/failed marker, never partial cards or a
        # success manifest. This makes an interrupted run auditable/resumable.
        failure_dir = args.staging_dir.resolve()
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "card-reextraction-staging-failure/v1",
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "status": "BLOCKED_PROVIDER_OR_EXTRACTION",
            "error": str(exc).replace("\n", " ")[:2000],
            "traceback": traceback.format_exc(limit=8),
            "production_db": str(args.production_db.resolve()),
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "model": args.model or os.getenv("LLM_MODEL"),
            "workers": args.workers,
            "max_retries": args.max_retries,
            "note": "No staging DB/Chroma was created; production DB/Chroma were not opened for writing.",
        }
        (failure_dir / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"staging_dir": str(args.staging_dir), "status": failure["status"], "error": failure["error"]}, ensure_ascii=False), file=sys.stderr)
        return 1
    result = {
        "staging_dir": str(args.staging_dir),
        "status": manifest["status"],
        "session_count": manifest["session_count"],
        "success_count": manifest.get("success_count"),
        "failure_count": manifest.get("failure_count"),
    }
    if "staging" in manifest:
        result["combined_sha256"] = manifest["staging"]["combined_sha256"]
    print(json.dumps(result, ensure_ascii=False))
    return 0 if manifest["status"] == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
