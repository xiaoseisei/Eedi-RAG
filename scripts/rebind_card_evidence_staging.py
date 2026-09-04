from __future__ import annotations

"""Deterministically rebind existing cards to complete raw evidence in staging.

No LLM is called.  Existing card semantics and vector documents are preserved;
only source-turn pointers and Chroma provenance metadata are updated in an
isolated copy.  Promotion is deliberately outside this command.
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.live_storage import hash_storage_artifacts
from src.evidence_index import (
    EVIDENCE_BINDING_POLICY,
    EVIDENCE_INDEX_VERSION,
    bind_card_to_evidence_index,
    build_evidence_index,
)
from src.models import (
    CleanedSession,
    ExtractedPIU,
    StudentMisconceptionProfile,
    TutorStrategyProfile,
)

logger = logging.getLogger(__name__)


def _load_sessions_from_db(db_path: Path) -> dict[int, CleanedSession]:
    from scripts.incremental_card_ingest import _load_sessions_from_db as loader

    return {session.intervention_id: session for session in loader(db_path)}


def _load_cards(db_path: Path) -> dict[int, ExtractedPIU]:
    import duckdb

    connection = duckdb.connect(database=str(db_path.resolve(strict=True)), read_only=True)
    try:
        misconceptions = connection.execute(
            """
            SELECT session_id, question_id, subject_path, misconception_name,
                   error_choice, deep_mechanism, confusion_triggers,
                   verbatim_student_quotes, source_turn_ids
            FROM misconception_chunks ORDER BY session_id
            """
        ).fetchall()
        strategies = connection.execute(
            """
            SELECT session_id, question_id, pedagogical_goal, strategy_category,
                   key_aha_question, scaffolding_steps, analogy_or_metaphor,
                   talk_moves, resolution_outcome, source_turn_ids
            FROM tutor_strategy_chunks ORDER BY session_id
            """
        ).fetchall()
    finally:
        connection.close()

    by_session: dict[int, dict[str, Any]] = {}
    for row in misconceptions:
        sid = int(row[0])
        by_session.setdefault(sid, {})["misconception"] = StudentMisconceptionProfile(
            session_id=sid,
            question_id=int(row[1]),
            subject_path=str(row[2] or ""),
            misconception_name=str(row[3]),
            error_choice=str(row[4] or "") or None,
            deep_mechanism=str(row[5]),
            confusion_triggers=list(row[6] or []),
            verbatim_student_quotes=list(row[7] or []),
            source_turn_ids=[int(value) for value in (row[8] or [])],
        )
    for row in strategies:
        sid = int(row[0])
        by_session.setdefault(sid, {})["tutor_strategy"] = TutorStrategyProfile(
            session_id=sid,
            question_id=int(row[1]),
            pedagogical_goal=str(row[2]),
            strategy_category=str(row[3]),
            key_aha_question=str(row[4]),
            scaffolding_steps=list(row[5] or []),
            analogy_or_metaphor=str(row[6] or "") or None,
            talk_moves=list(row[7] or []),
            resolution_outcome=str(row[8]),
            source_turn_ids=[int(value) for value in (row[9] or [])],
        )
    result: dict[int, ExtractedPIU] = {}
    for sid, cards in by_session.items():
        if set(cards) != {"misconception", "tutor_strategy"}:
            raise ValueError(f"Session {sid} does not contain both card slots")
        result[sid] = ExtractedPIU(
            session_id=sid,
            question_id=cards["misconception"].question_id,
            misconception=cards["misconception"],
            tutor_strategy=cards["tutor_strategy"],
            extraction_status="success",
        )
    return result


def _update_chroma_metadata(chroma_path: Path, cards: dict[int, ExtractedPIU]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    collections = {
        "student_misconceptions": "misconception",
        "tutor_strategies": "tutor_strategy",
    }
    try:
        for collection_name, slot in collections.items():
            collection = client.get_collection(collection_name)
            ids: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for sid in sorted(cards):
                card = getattr(cards[sid], slot)
                assert card is not None
                chunk_id = f"session_{sid}_{'misconception' if slot == 'misconception' else 'tutor_strategy'}"
                current = collection.get(ids=[chunk_id], include=["metadatas"])
                if not current.get("ids"):
                    raise ValueError(f"Chroma collection {collection_name} missing {chunk_id}")
                metadata = dict((current.get("metadatas") or [{}])[0] or {})
                metadata.update({
                    "source_turn_ids": json.dumps(card.source_turn_ids),
                    "source_index_ids": json.dumps(card.source_index_ids, ensure_ascii=False),
                    "evidence_binding_policy": card.evidence_binding_policy or "",
                    "evidence_index_version": card.evidence_index_version or "",
                    "evidence_index_hash": card.evidence_index_hash or "",
                })
                ids.append(chunk_id)
                metadatas.append(metadata)
            collection.update(ids=ids, metadatas=metadatas)
    finally:
        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception as exc:  # pragma: no cover - cleanup varies by Chroma version
            logger.warning("无法立即清理 staging Chroma client: %s", exc)


def _update_duckdb(db_path: Path, cards: dict[int, ExtractedPIU]) -> None:
    import duckdb

    connection = duckdb.connect(database=str(db_path))
    try:
        for sid in sorted(cards):
            item = cards[sid]
            assert item.misconception is not None and item.tutor_strategy is not None
            connection.execute(
                "UPDATE misconception_chunks SET source_turn_ids = ? WHERE session_id = ?",
                [item.misconception.source_turn_ids, sid],
            )
            connection.execute(
                "UPDATE tutor_strategy_chunks SET source_turn_ids = ? WHERE session_id = ?",
                [item.tutor_strategy.source_turn_ids, sid],
            )
        connection.commit()
    finally:
        connection.close()


def run_rebind(
    *,
    production_db: Path,
    production_chroma: Path,
    output_dir: Path,
    session_ids: list[int] | None = None,
    window_size: int = 6,
    step: int = 3,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    production_db = production_db.resolve(strict=True)
    production_chroma = production_chroma.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = hash_storage_artifacts(production_db, production_chroma)
    sessions = _load_sessions_from_db(production_db)
    cards = _load_cards(production_db)
    selected_ids = sorted(session_ids if session_ids is not None else cards)
    missing = [sid for sid in selected_ids if sid not in sessions or sid not in cards]
    if missing:
        raise ValueError(f"selected sessions missing from source cards/sessions: {missing}")

    indexes: dict[int, Any] = {}
    bound_cards: dict[int, ExtractedPIU] = {}
    changes: list[dict[str, Any]] = []
    for sid in selected_ids:
        index = build_evidence_index(
            sessions[sid],
            window_size=window_size,
            step=step,
            evidence_index_version=EVIDENCE_INDEX_VERSION,
        )
        item = cards[sid]
        assert item.misconception is not None and item.tutor_strategy is not None
        bound = item.model_copy(update={
            "misconception": bind_card_to_evidence_index(item.misconception, index),
            "tutor_strategy": bind_card_to_evidence_index(item.tutor_strategy, index),
        })
        indexes[sid] = index
        bound_cards[sid] = bound
        changes.append({
            "session_id": sid,
            "misconception_before": item.misconception.source_turn_ids,
            "misconception_after": bound.misconception.source_turn_ids,
            "strategy_before": item.tutor_strategy.source_turn_ids,
            "strategy_after": bound.tutor_strategy.source_turn_ids,
            "evidence_index_hash": index.evidence_index_hash,
            "source_index_count": len(index.windows),
        })

    staging_db = output_dir / "tutoring_knowledge.duckdb"
    staging_chroma = output_dir / "chroma"
    shutil.copy2(production_db, staging_db)
    shutil.copytree(production_chroma, staging_chroma)
    _update_duckdb(staging_db, bound_cards)
    _update_chroma_metadata(staging_chroma, bound_cards)

    index_path = output_dir / "evidence-index.jsonl"
    index_path.write_text(
        "".join(json.dumps(index.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for index in indexes.values()),
        encoding="utf-8",
    )
    changes_path = output_dir / "card-rebind.jsonl"
    changes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in changes),
        encoding="utf-8",
    )
    staging_hash = hash_storage_artifacts(staging_db, staging_chroma)
    manifest = {
        "schema_version": "card-evidence-rebind-staging/v1",
        "status": "SUCCESS",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "session_count": len(selected_ids),
        "card_count": len(selected_ids) * 2,
        "llm_calls": 0,
        "window_size": window_size,
        "step": step,
        "evidence_index_version": EVIDENCE_INDEX_VERSION,
        "evidence_binding_policy": EVIDENCE_BINDING_POLICY,
        "source_hash_before": source_hash,
        "staging_hash_after": staging_hash,
        "source_artifact_mutated": False,
        "files": {
            "db": str(staging_db),
            "chroma": str(staging_chroma),
            "evidence_index": str(index_path),
            "card_rebind": str(changes_path),
        },
        "reproduction_command": " ".join(sys.argv),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-ids", default=None, help="comma-separated IDs; default is all card sessions")
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args(argv)
    ids = None if args.session_ids is None else [int(value) for value in args.session_ids.split(",") if value.strip()]
    manifest = run_rebind(
        production_db=args.db,
        production_chroma=args.chroma,
        output_dir=args.output_dir,
        session_ids=ids,
        window_size=args.window_size,
        step=args.step,
    )
    print(json.dumps({
        "status": manifest["status"],
        "session_count": manifest["session_count"],
        "card_count": manifest["card_count"],
        "llm_calls": manifest["llm_calls"],
        "source_hash_before": manifest["source_hash_before"],
        "staging_hash_after": manifest["staging_hash_after"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
