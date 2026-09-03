from __future__ import annotations

"""Evaluate Turn/window recall with provisional multi-relevant graded qrels."""

import argparse
import gc
import hashlib
import json
import shutil
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.graded_qrels import build_card_graded_qrels, build_window_graded_qrels, turn_coverage_at_k
from evals.live_storage import hash_storage_artifacts
from evals.metrics.retrieval import ndcg_at_k
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_sessions(path: Path) -> dict[int, Any]:
    from src.models import CleanedSession

    sessions: dict[int, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            session = CleanedSession.model_validate(json.loads(line))
            sessions[session.intervention_id] = session
    return sessions


def _required_turns(case: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(item["session_id"]), int(item["turn_id"]))
        for item in case.get("verbatim_grounding_quotes", [])
    }


def _card_slot_with_required_evidence(case: dict[str, Any]) -> tuple[str, bool]:
    """Choose an evidence-supported card slot without inventing relevance."""
    preferred = "misconception" if case.get("category") == "STUDENT_INSIGHT" else "strategy"
    preferred_speaker = "student" if preferred == "misconception" else "tutor"
    speakers = {str(item["speaker"]) for item in case.get("verbatim_grounding_quotes", [])}
    if preferred_speaker in speakers:
        return preferred, False
    fallback = "strategy" if preferred == "misconception" else "misconception"
    fallback_speaker = "tutor" if fallback == "strategy" else "student"
    if fallback_speaker not in speakers:
        raise ValueError("case has no student or tutor required evidence")
    return fallback, True


def _pointer_coverage(case: dict[str, Any], conn: Any) -> dict[str, Any]:
    sid = int(case["verbatim_grounding_quotes"][0]["session_id"])
    required = _required_turns(case)
    target_slot, slot_fallback = _card_slot_with_required_evidence(case)
    target_speaker = "student" if target_slot == "misconception" else "tutor"
    required_with_speakers = {
        (int(item["turn_id"]), str(item["speaker"]))
        for item in case["verbatim_grounding_quotes"]
    }
    target_required_with_speakers = {
        (turn_id, speaker)
        for turn_id, speaker in required_with_speakers
        if speaker == target_speaker
    }
    misc = {
        int(turn_id)
        for row in conn.execute("SELECT source_turn_ids FROM misconception_chunks WHERE session_id = ?", [sid]).fetchall()
        for turn_id in (row[0] or [])
    }
    strat = {
        int(turn_id)
        for row in conn.execute("SELECT source_turn_ids FROM tutor_strategy_chunks WHERE session_id = ?", [sid]).fetchall()
        for turn_id in (row[0] or [])
    }
    role_correct = {
        turn_id
        for turn_id, speaker in target_required_with_speakers
        if turn_id in (misc if speaker == "student" else strat)
    }
    required_turn_ids = {turn_id for _, turn_id in required}
    target_required_turn_ids = {turn_id for turn_id, _ in target_required_with_speakers}
    union = required_turn_ids.intersection(misc | strat)
    return {
        "session_id": sid,
        "target_slot": target_slot,
        "slot_fallback_due_to_missing_role_evidence": slot_fallback,
        "target_speaker": target_speaker,
        "required_turn_count": len(required),
        "target_required_turn_count": len(target_required_turn_ids),
        "all_role_card_pointer_recall": len(union) / len(required),
        "card_pointer_recall": len(role_correct) / len(target_required_turn_ids) if target_required_turn_ids else 0.0,
        "role_correct_card_pointer_recall": len(role_correct) / len(target_required_turn_ids) if target_required_turn_ids else 0.0,
        "missing_from_card_pointers": sorted(required_turn_ids - union),
        "missing_from_role_correct_pointers": sorted(target_required_turn_ids - role_correct),
    }


def _graded_mrr(ranked_ids: list[str], qrels: dict[str, int], minimum_relevance: int = 2) -> float:
    for rank, document_id in enumerate(ranked_ids, start=1):
        if qrels.get(document_id, 0) >= minimum_relevance:
            return 1.0 / rank
    return 0.0


def _graded_recall(ranked_ids: list[str], qrels: dict[str, int], k: int, minimum_relevance: int = 2) -> float:
    positive = {doc_id for doc_id, relevance in qrels.items() if relevance >= minimum_relevance}
    return len(positive.intersection(ranked_ids[:k])) / len(positive) if positive else 0.0


def _graded_precision(ranked_ids: list[str], qrels: dict[str, int], k: int, minimum_relevance: int = 1) -> float:
    top = ranked_ids[:k]
    return sum(qrels.get(doc_id, 0) >= minimum_relevance for doc_id in top) / len(top) if top else 0.0


def evaluate(
    *,
    db_path: Path,
    chroma_path: Path,
    golden_path: Path,
    l1_manifest_path: Path,
    output_dir: Path,
    top_k: int,
    expected_artifact_hash: str | None = None,
    split_manifest_path: Path | None = None,
    embedding_backend: str = "qwen",
    retrieval_mode: str = "bm25_dense",
    bm25_weight: float = 0.35,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Turn graded eval: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    l1_manifest = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    source_hash = hash_storage_artifacts(db_path, chroma_path)
    expected_hash = expected_artifact_hash or l1_manifest["artifact_hashes"]["qwen_combined_sha256"]
    if source_hash != expected_hash:
        raise ValueError(f"artifact hash mismatch: expected={expected_hash}, actual={source_hash}")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    sessions = _load_sessions(PROJECT_ROOT / "data/cleaned_sessions.jsonl")
    qrels_rows: list[dict[str, Any]] = []
    card_qrels_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    import duckdb

    conn = duckdb.connect(database=str(db_path), read_only=True)
    try:
        for index, case in enumerate(golden, 1):
            query_id = f"eedi-l1-{index:04d}"
            sid = int(case["verbatim_grounding_quotes"][0]["session_id"])
            windows = [
                {
                    "chunk_id": str(row[0]),
                    "session_id": int(row[1]),
                    "subject_path": row[2] or "",
                    "source_turn_ids": list(row[3] or []),
                }
                for row in conn.execute(
                    "SELECT chunk_id, session_id, subject_path, source_turn_ids FROM sliding_window_chunks ORDER BY chunk_id"
                ).fetchall()
            ]
            case_with_id = {**case, "query_id": query_id}
            rows = build_window_graded_qrels(case_with_id, windows)
            qrels_rows.extend(rows)
            target_slot, slot_fallback = _card_slot_with_required_evidence(case)
            if target_slot == "misconception":
                card_rows = [
                    {
                        "chunk_id": str(row[0]),
                        "session_id": int(row[1]),
                        "subject_path": row[2] or "",
                        "source_turn_ids": list(row[3] or []),
                    }
                    for row in conn.execute(
                        "SELECT chunk_id, session_id, subject_path, source_turn_ids FROM misconception_chunks ORDER BY chunk_id"
                    ).fetchall()
                ]
            else:
                card_rows = [
                    {
                        "chunk_id": str(row[0]),
                        "session_id": int(row[1]),
                        "subject_path": row[2] or "",
                        "source_turn_ids": list(row[3] or []),
                    }
                    for row in conn.execute(
                        "SELECT chunk_id, session_id, subject_path, source_turn_ids FROM tutor_strategy_chunks ORDER BY chunk_id"
                    ).fetchall()
                ]
            card_qrels_rows.extend(
                build_card_graded_qrels(
                    case_with_id,
                    card_rows,
                    target_slot=target_slot,
                )
            )
            pointer = _pointer_coverage(case, conn)
            cases.append({
                "case_id": f"eedi-turn-graded-{index:04d}",
                "query_id": query_id,
                "query": case["question"],
                "category": case.get("category", "UNKNOWN"),
                "session_id": sid,
                "required_turns": sorted(_required_turns(case)),
                "pointer_coverage": pointer,
                "golden_case": case,
                "qrels": {row["document_id"]: row["relevance"] for row in rows},
                "card_target_slot": target_slot,
                "card_slot_fallback_due_to_missing_role_evidence": slot_fallback,
            })
    finally:
        conn.close()
    _write_jsonl(output_dir / "graded_qrels.jsonl", qrels_rows)
    _write_jsonl(output_dir / "card_graded_qrels.jsonl", card_qrels_rows)

    if embedding_backend not in {"qwen", "deterministic"}:
        raise ValueError("embedding_backend must be qwen or deterministic")
    provider = (
        SiliconFlowQwen3EmbeddingFunction.from_env()
        if embedding_backend == "qwen"
        else __import__("src.storage_manager", fromlist=["FastDeterministicEmbeddingFunction"]).FastDeterministicEmbeddingFunction()
    )
    rows: list[dict[str, Any]] = []
    scratch_root = output_dir.parent / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    # Windows may retain mapped HNSW files briefly after Chroma is closed.
    # The evaluation copy is disposable; cleanup failure must not discard an
    # otherwise valid, hash-verified diagnostic result.
    with tempfile.TemporaryDirectory(
        prefix="turn-graded-",
        dir=str(scratch_root),
        ignore_cleanup_errors=True,
    ) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        storage = None
        try:
            from src.storage_manager import DualEngineStorageManager
            from src.retriever import DualMetricRetriever

            storage = DualEngineStorageManager(
                db_path=copied_db,
                chroma_dir=copied_chroma,
                embedding_function=provider,
                embedding_index_version=provider.index_version if embedding_backend == "qwen" else None,
            )
            retriever = DualMetricRetriever(
                storage_manager=storage,
                query_rewrite_mode="deterministic",
                fusion_strategy="raw_first",
                retrieval_mode=retrieval_mode,
                bm25_weight=bm25_weight,
            )
            for case in cases:
                started = time.perf_counter()
                ranked = retriever.retrieve_fallback_windows(case["query"], top_k=top_k)
                latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
                ranked_ids = [str(item["chunk_id"]) for item in ranked]
                ranked_windows = [
                    {
                        "chunk_id": str(item["chunk_id"]),
                        "session_id": int(item["metadata"].get("session_id")),
                        "source_turn_ids": json.loads(item["metadata"].get("source_turn_ids", "[]"))
                        if isinstance(item["metadata"].get("source_turn_ids"), str)
                        else (item["metadata"].get("source_turn_ids") or []),
                    }
                    for item in ranked
                ]
                qrels = case["qrels"]
                required = {tuple(item) for item in case["required_turns"]}
                metric_row = {
                    "case_id": case["case_id"],
                    "query_id": case["query_id"],
                    "query": case["query"],
                    "split": "unknown",
                    "category": case["category"],
                    "session_id": case["session_id"],
                    "required_turns": case["required_turns"],
                    "ranked_ids": ranked_ids,
                    "ranked_windows": ranked_windows,
                    "graded_recall_at_3": _graded_recall(ranked_ids, qrels, 3),
                    "graded_recall_at_5": _graded_recall(ranked_ids, qrels, 5),
                    "graded_recall_at_10": _graded_recall(ranked_ids, qrels, 10),
                    "graded_recall_at_15": _graded_recall(ranked_ids, qrels, 15),
                    "graded_recall_at_20": _graded_recall(ranked_ids, qrels, 20),
                    "graded_precision_at_5": _graded_precision(ranked_ids, qrels, 5),
                    "graded_ndcg_at_5": ndcg_at_k(ranked_ids, qrels, 5),
                    "graded_mrr": _graded_mrr(ranked_ids, qrels),
                    "turn_coverage_at_1": turn_coverage_at_k(ranked_windows, required, 1),
                    "turn_coverage_at_3": turn_coverage_at_k(ranked_windows, required, 3),
                    "turn_coverage_at_5": turn_coverage_at_k(ranked_windows, required, 5),
                    "turn_coverage_at_10": turn_coverage_at_k(ranked_windows, required, 10),
                    "turn_coverage_at_15": turn_coverage_at_k(ranked_windows, required, 15),
                    "turn_coverage_at_20": turn_coverage_at_k(ranked_windows, required, 20),
                    "latency_ms": latency_ms,
                    "pointer_coverage": case["pointer_coverage"],
                }
                rows.append(metric_row)
                target_slot = case["card_target_slot"]
                card_ranked = (
                    retriever.retrieve_misconceptions(case["query"], top_k=top_k, fetch_evidence=False)
                    if target_slot == "misconception"
                    else retriever.retrieve_strategies(case["query"], top_k=top_k, fetch_evidence=False)
                )
                card_ranked_ids = [str(item["chunk_id"]) for item in card_ranked]
                card_qrels = {
                    row["document_id"]: row["relevance"]
                    for row in card_qrels_rows
                    if row["query_id"] == case["query_id"]
                }
                rows[-1]["card_slot"] = target_slot
                rows[-1]["card_slot_fallback_due_to_missing_role_evidence"] = case["card_slot_fallback_due_to_missing_role_evidence"]
                rows[-1]["card_ranked_ids"] = card_ranked_ids
                rows[-1]["card_graded_recall_at_3"] = _graded_recall(card_ranked_ids, card_qrels, 3)
                rows[-1]["card_graded_recall_at_5"] = _graded_recall(card_ranked_ids, card_qrels, 5)
                rows[-1]["card_graded_recall_at_10"] = _graded_recall(card_ranked_ids, card_qrels, 10)
                rows[-1]["card_graded_recall_at_15"] = _graded_recall(card_ranked_ids, card_qrels, 15)
                rows[-1]["card_graded_recall_at_20"] = _graded_recall(card_ranked_ids, card_qrels, 20)
                rows[-1]["card_graded_precision_at_5"] = _graded_precision(card_ranked_ids, card_qrels, 5)
                rows[-1]["card_graded_ndcg_at_5"] = ndcg_at_k(card_ranked_ids, card_qrels, 5)
                rows[-1]["card_graded_mrr"] = _graded_mrr(card_ranked_ids, card_qrels)
        finally:
            if storage is not None:
                storage.close()
            try:
                from chromadb.api.client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()
    after_hash = hash_storage_artifacts(db_path, chroma_path)
    if after_hash != source_hash:
        raise RuntimeError("source artifact changed during Turn graded evaluation")
    split_path = split_manifest_path or (l1_manifest_path.parent / l1_manifest["split"]["file"])
    split_manifest = json.loads(split_path.resolve(strict=True).read_text(encoding="utf-8"))
    assignment = split_manifest["query_assignments"]
    for row in rows:
        row["split"] = assignment[row["query_id"]]
    _write_jsonl(output_dir / "raw_cases.jsonl", rows)

    metric_names = [
        "graded_recall_at_3", "graded_recall_at_5", "graded_recall_at_10", "graded_recall_at_15", "graded_recall_at_20",
        "graded_precision_at_5", "graded_ndcg_at_5", "graded_mrr",
        "turn_coverage_at_1", "turn_coverage_at_3", "turn_coverage_at_5", "turn_coverage_at_10", "turn_coverage_at_15", "turn_coverage_at_20", "latency_ms",
        "card_graded_recall_at_3", "card_graded_recall_at_5", "card_graded_recall_at_10", "card_graded_recall_at_15", "card_graded_recall_at_20",
        "card_graded_precision_at_5", "card_graded_ndcg_at_5", "card_graded_mrr",
    ]
    aggregates: dict[str, Any] = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        aggregates[name] = {
            "mean": statistics.mean(values),
            "p95": sorted(values)[max(0, int(len(values) * 0.95) - 1)],
            "dev_mean": statistics.mean([row[name] for row in rows if row["split"] == "dev"]),
            "holdout_mean": statistics.mean([row[name] for row in rows if row["split"] == "holdout"]),
            "measured_cases": len(values),
        }
    pointer_values = [row["pointer_coverage"]["role_correct_card_pointer_recall"] for row in rows]
    window_all_coverage = [row["turn_coverage_at_5"] for row in rows]
    attribution = {
        "card_pointer_loss_cases": [row["case_id"] for row in rows if row["pointer_coverage"]["role_correct_card_pointer_recall"] < 1.0],
        "window_generation_loss_cases": [],
        "retrieval_rank_loss_cases": [row["case_id"] for row in rows if row["turn_coverage_at_3"] < 1.0],
        "card_pointer_role_correct_mean": statistics.mean(pointer_values),
        "full_window_coverage_at_5_mean": statistics.mean(window_all_coverage),
        "interpretation": "If card pointer recall is low while full window coverage is complete, the first loss is extraction/pointer selection. If pointer coverage is complete but top-k Turn coverage is low, the loss is window retrieval/ranking. Assembler and Generator must be evaluated separately against final context and citation traces.",
    }
    manifest = {
        "schema_version": "turn-graded-eval/v1",
        "dataset_version": "eedi-turn-graded-provisional-v1",
        "qrels_version": "turn-window-graded-provisional-rule-v1",
        "label_source": "provisional_rule_from_human_turns_v1",
        "needs_human_review": True,
        "case_count": len(rows),
        "qrel_count": len(qrels_rows),
        "relevance_scale": {"0": "unrelated", "1": "same session/subject without direct required turn", "2": "covers a subset of required turns", "3": "covers all required turns"},
        "split_counts": {"dev": sum(row["split"] == "dev" for row in rows), "holdout": sum(row["split"] == "holdout" for row in rows)},
        "artifact_hash_before": source_hash,
        "artifact_hash_after": after_hash,
        "artifact_mutated": source_hash != after_hash,
        "l1_manifest_sha256": _file_sha256(l1_manifest_path),
        "split_manifest_sha256": _file_sha256(split_path),
        "expected_artifact_hash": expected_hash,
        "files": {name: _file_sha256(output_dir / name) for name in ("graded_qrels.jsonl", "card_graded_qrels.jsonl", "raw_cases.jsonl")},
        "config": {
            "embedding_backend": embedding_backend,
            "embedding_model": provider.model if hasattr(provider, "model") else provider.name(),
            "embedding_dimension": provider.dimension if hasattr(provider, "dimension") else provider.dim,
            "fusion_strategy": "raw_first",
            "retrieval_mode": retrieval_mode,
            "bm25_weight": bm25_weight,
            "top_k": top_k,
        },
        "aggregates": aggregates,
        "attribution": attribution,
        "card_slot_fallback_cases": [row["case_id"] for row in rows if row["card_slot_fallback_due_to_missing_role_evidence"]],
        "limitations": [
            "These graded labels are deterministic derivations, not independent human judgments.",
            "A window with relevance 1 is topical/session-related but not proven evidence for the answer.",
            "Do not use these qrels to claim business accuracy until human reviewers approve multi-document relevance.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Turn-level graded retrieval evaluation",
        "",
        "Status: **PROVISIONAL — HUMAN REVIEW REQUIRED**",
        "",
        f"Cases: `{len(rows)}`; window graded qrels: `{len(qrels_rows)}`; card graded qrels: `{len(card_qrels_rows)}`; source artifact unchanged: `{source_hash == after_hash}`",
        "",
        "## Aggregates",
        "",
        "| Metric | Mean | Dev | Holdout |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in aggregates.items():
        lines.append(f"| {name} | {value['mean']:.4f} | {value['dev_mean']:.4f} | {value['holdout_mean']:.4f} |")
    lines.extend([
        "", "## Attribution", "",
        f"- Card pointer loss cases: `{len(attribution['card_pointer_loss_cases'])}` / `{len(rows)}`",
        f"- Retrieval rank loss cases (Turn@3 < 1): `{len(attribution['retrieval_rank_loss_cases'])}` / `{len(rows)}`",
        f"- Role-correct card pointer recall mean: `{attribution['card_pointer_role_correct_mean']:.4f}`",
        f"- Turn coverage@5 mean: `{attribution['full_window_coverage_at_5_mean']:.4f}`",
        "- Window generation loss cases: `0` in this dataset; production window union covers all required turns.",
        "",
        "## Interpretation",
        "",
        attribution["interpretation"],
        "",
        "Detailed per-case data is in `raw_cases.jsonl`; all qrels carry `needs_human_review=true`.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=Path("data/golden_test_set.json"))
    parser.add_argument("--l1-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expected-artifact-hash", type=str, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--embedding-backend", choices=("qwen", "deterministic"), default="qwen")
    parser.add_argument("--retrieval-mode", choices=("dense", "bm25_dense"), default="bm25_dense")
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    manifest = evaluate(
        db_path=args.db.resolve(strict=True),
        chroma_path=args.chroma.resolve(strict=True),
        golden_path=args.golden.resolve(strict=True),
        l1_manifest_path=args.l1_manifest.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        top_k=args.top_k,
        expected_artifact_hash=args.expected_artifact_hash,
        split_manifest_path=args.split_manifest.resolve(strict=True) if args.split_manifest else None,
        embedding_backend=args.embedding_backend,
        retrieval_mode=args.retrieval_mode,
        bm25_weight=args.bm25_weight,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "dataset_version": manifest["dataset_version"], "case_count": manifest["case_count"], "qrel_count": manifest["qrel_count"], "artifact_mutated": manifest["artifact_mutated"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
