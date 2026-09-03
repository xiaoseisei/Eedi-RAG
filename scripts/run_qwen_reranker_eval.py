from __future__ import annotations

"""Evaluate Qwen3-Reranker on a BM25+Dense candidate pool."""

import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.graded_qrels import build_card_graded_qrels
from evals.live_storage import hash_storage_artifacts
from evals.metrics.retrieval import ndcg_at_k
from scripts.run_turn_graded_eval import _card_slot_with_required_evidence, _graded_mrr, _graded_precision, _graded_recall
from src.reranker_provider import SiliconFlowQwen3Reranker
from src.storage_manager import FastDeterministicEmbeddingFunction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_turns(case: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(item["session_id"]), int(item["turn_id"])) for item in case.get("verbatim_grounding_quotes", [])}


def _target_required_turns(case: dict[str, Any], slot: str) -> set[tuple[int, int]]:
    speaker = "student" if slot == "misconception" else "tutor"
    return {
        (int(item["session_id"]), int(item["turn_id"]))
        for item in case.get("verbatim_grounding_quotes", [])
        if str(item.get("speaker")) == speaker
    }


def _turn_coverage(cards: list[dict[str, Any]], required: set[tuple[int, int]], k: int) -> float:
    if not required:
        return 0.0
    seen: set[tuple[int, int]] = set()
    for card in cards[:k]:
        sid = int(card.get("metadata", {}).get("session_id"))
        raw_ids = card.get("metadata", {}).get("source_turn_ids", [])
        if isinstance(raw_ids, str):
            raw_ids = json.loads(raw_ids)
        seen.update((sid, int(turn_id)) for turn_id in (raw_ids or []))
    return len(required.intersection(seen)) / len(required)


def evaluate(
    *,
    db_path: Path,
    chroma_path: Path,
    golden_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    pool_size: int,
    embedding_backend: str = "deterministic",
    bm25_weight: float = 0.35,
    limit: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite reranker eval: {output_dir}")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    golden = json.loads(golden_path.resolve(strict=True).read_text(encoding="utf-8"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        golden = golden[:limit]
    split_assignments = json.loads(split_manifest_path.resolve(strict=True).read_text(encoding="utf-8"))["query_assignments"]
    source_hash = hash_storage_artifacts(db_path, chroma_path)
    reranker = SiliconFlowQwen3Reranker.from_env()
    with tempfile.TemporaryDirectory(prefix="qwen-reranker-", dir=str(output_dir.parent), ignore_cleanup_errors=True) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        from src.retriever import DualMetricRetriever
        from src.storage_manager import DualEngineStorageManager

        embedding = FastDeterministicEmbeddingFunction() if embedding_backend == "deterministic" else __import__("src.embedding_provider", fromlist=["SiliconFlowQwen3EmbeddingFunction"]).SiliconFlowQwen3EmbeddingFunction.from_env()
        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_function=embedding,
            embedding_index_version=None,
        )
        retriever = DualMetricRetriever(
            storage_manager=storage,
            query_rewrite_mode="deterministic",
            fusion_strategy="raw_first",
            retrieval_mode="bm25_dense",
            bm25_weight=bm25_weight,
        )
        rows: list[dict[str, Any]] = []
        try:
            conn = storage.duck_conn
            try:
                for index, case in enumerate(golden, start=1):
                    query_id = f"eedi-l1-{index:04d}"
                    slot, slot_fallback = _card_slot_with_required_evidence(case)
                    table = "misconception_chunks" if slot == "misconception" else "tutor_strategy_chunks"
                    card_rows = [
                        {"chunk_id": str(row[0]), "session_id": int(row[1]), "subject_path": row[2] or "", "source_turn_ids": list(row[3] or [])}
                        for row in conn.execute(f"SELECT chunk_id, session_id, subject_path, source_turn_ids FROM {table} ORDER BY chunk_id").fetchall()
                    ]
                    case_with_id = {**case, "query_id": query_id}
                    qrels = {row["document_id"]: row["relevance"] for row in build_card_graded_qrels(case_with_id, card_rows, target_slot=slot)}
                    started = time.perf_counter()
                    candidates = retriever.retrieve_misconceptions(case["question"], top_k=pool_size, fetch_evidence=True) if slot == "misconception" else retriever.retrieve_strategies(case["question"], top_k=pool_size, fetch_evidence=True)
                    retrieval_ms = round((time.perf_counter() - started) * 1000.0, 3)
                    try:
                        rerank_documents = []
                        for item in candidates:
                            evidence_text = "\n".join(
                                f"[Turn {turn.get('turn_id')}] [{turn.get('speaker')}] {turn.get('text', '')}"
                                for turn in (item.get("evidence_turns", []) or [])
                            )
                            rerank_documents.append(
                                f"{item.get('document', '')}\n【直接证据】\n{evidence_text}"
                                if evidence_text else str(item.get("document", ""))
                            )
                        reranked_scores = reranker.rerank(case["question"], rerank_documents, top_n=len(candidates))
                        reranked = []
                        for rank, score in enumerate(reranked_scores, start=1):
                            candidate = dict(candidates[score.index])
                            candidate["reranker_score"] = score.relevance_score
                            candidate["reranker_rank"] = rank
                            candidate["reranker_model"] = reranker.name()
                            reranked.append(candidate)
                        status = "SUCCESS"
                        error = None
                    except Exception as exc:
                        reranked = []
                        status = "FAILED"
                        error = str(exc).replace("\n", " ")[:1000]
                    ranked_ids = [str(item["chunk_id"]) for item in reranked]
                    required = _target_required_turns(case, slot)
                    row: dict[str, Any] = {
                        "case_id": f"eedi-reranker-{index:04d}",
                        "query_id": query_id,
                        "query": case["question"],
                        "category": case.get("category", "UNKNOWN"),
                        "split": split_assignments.get(query_id, "unknown"),
                        "session_id": int(case["verbatim_grounding_quotes"][0]["session_id"]),
                        "target_slot": slot,
                        "slot_fallback": slot_fallback,
                        "pool_size": pool_size,
                        "retrieval_mode": "bm25_dense",
                        "bm25_weight": bm25_weight,
                        "retrieval_ms": retrieval_ms,
                        "status": status,
                        "error": error,
                        "pool_ranked_ids": [str(item["chunk_id"]) for item in candidates],
                        "reranked_ids": ranked_ids,
                    }
                    if status == "SUCCESS":
                        row.update({
                            "pool_card_recall_at_5": _graded_recall(row["pool_ranked_ids"], qrels, 5),
                            "pool_card_recall_at_10": _graded_recall(row["pool_ranked_ids"], qrels, 10),
                            "pool_card_recall_at_15": _graded_recall(row["pool_ranked_ids"], qrels, 15),
                            "pool_card_recall_at_20": _graded_recall(row["pool_ranked_ids"], qrels, 20),
                            "pool_card_mrr": _graded_mrr(row["pool_ranked_ids"], qrels),
                            "card_recall_at_5": _graded_recall(ranked_ids, qrels, 5),
                            "card_recall_at_10": _graded_recall(ranked_ids, qrels, 10),
                            "card_recall_at_15": _graded_recall(ranked_ids, qrels, 15),
                            "card_recall_at_20": _graded_recall(ranked_ids, qrels, 20),
                            "card_precision_at_5": _graded_precision(ranked_ids, qrels, 5),
                            "card_ndcg_at_5": ndcg_at_k(ranked_ids, qrels, 5),
                            "card_mrr": _graded_mrr(ranked_ids, qrels),
                            "pool_card_turn_coverage_at_5": _turn_coverage(candidates, required, 5),
                            "pool_card_turn_coverage_at_10": _turn_coverage(candidates, required, 10),
                            "pool_card_turn_coverage_at_15": _turn_coverage(candidates, required, 15),
                            "pool_card_turn_coverage_at_20": _turn_coverage(candidates, required, 20),
                            "card_turn_coverage_at_5": _turn_coverage(reranked, required, 5),
                            "card_turn_coverage_at_10": _turn_coverage(reranked, required, 10),
                            "card_turn_coverage_at_15": _turn_coverage(reranked, required, 15),
                            "card_turn_coverage_at_20": _turn_coverage(reranked, required, 20),
                        })
                    rows.append(row)
            finally:
                pass
        finally:
            storage.close()
            try:
                from chromadb.api.client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()
    after_hash = hash_storage_artifacts(db_path, chroma_path)
    if after_hash != source_hash:
        raise RuntimeError("source artifact changed during reranker evaluation")
    successful = [row for row in rows if row["status"] == "SUCCESS"]
    metric_names = [
        "pool_card_recall_at_5", "pool_card_recall_at_10", "pool_card_recall_at_15", "pool_card_recall_at_20", "pool_card_mrr",
        "card_recall_at_5", "card_recall_at_10", "card_recall_at_15", "card_recall_at_20",
        "card_precision_at_5", "card_ndcg_at_5", "card_mrr",
        "pool_card_turn_coverage_at_5", "pool_card_turn_coverage_at_10", "pool_card_turn_coverage_at_15", "pool_card_turn_coverage_at_20",
        "card_turn_coverage_at_5", "card_turn_coverage_at_10", "card_turn_coverage_at_15", "card_turn_coverage_at_20",
    ]
    aggregates = {
        name: {
            "mean": sum(float(row[name]) for row in successful) / len(successful) if successful else None,
            "measured_cases": len(successful),
            "dev_mean": sum(float(row[name]) for row in successful if row["split"] == "dev") / max(1, sum(row["split"] == "dev" for row in successful)),
            "holdout_mean": sum(float(row[name]) for row in successful if row["split"] == "holdout") / max(1, sum(row["split"] == "holdout" for row in successful)),
        }
        for name in metric_names
    }
    summary = {
        "schema_version": "qwen-reranker-eval/v1",
        "status": "MEASURED" if successful else "FAILED",
        "scope": "BM25+Dense candidate pool reranked by SiliconFlow Qwen3-Reranker-0.6B",
        "model": reranker.model,
        "pool_size": pool_size,
        "embedding_backend": embedding_backend,
        "retrieval_mode": "bm25_dense",
        "bm25_weight": bm25_weight,
        "case_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(rows) - len(successful),
        "aggregates": aggregates,
        "reranker_telemetry": reranker.telemetry,
        "artifact_hash_before": source_hash,
        "artifact_hash_after": after_hash,
        "artifact_mutated": source_hash != after_hash,
        "limitations": [
            "qrels remain provisional rule-derived labels requiring human review",
            "reranker scores are provider outputs and do not by themselves prove citation or answer quality",
        ],
    }
    (output_dir / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Qwen3-Reranker-0.6B Evaluation",
        "",
        f"Status: **{summary['status']}**",
        f"- Pool: BM25+Dense Top-{pool_size}; model: `{reranker.model}`",
        f"- Cases success/failure: `{summary['success_count']}/{summary['failure_count']}`",
        "",
        "| Metric | Mean | Dev | Holdout |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in aggregates.items():
        lines.append(f"| {name} | {value['mean']} | {value['dev_mean']:.4f} | {value['holdout_mean']:.4f} |")
    lines.extend(["", "Detailed per-case reranked order is in `results.jsonl`."])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data" / "golden_test_set.json")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=15)
    parser.add_argument("--embedding-backend", choices=("deterministic", "qwen"), default="deterministic")
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    summary = evaluate(
        db_path=args.db.resolve(strict=True),
        chroma_path=args.chroma.resolve(strict=True),
        golden_path=args.golden.resolve(strict=True),
        split_manifest_path=args.split_manifest.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        pool_size=args.pool_size,
        embedding_backend=args.embedding_backend,
        bm25_weight=args.bm25_weight,
        limit=args.limit,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "status": summary["status"], "success_count": summary["success_count"], "failure_count": summary["failure_count"]}, ensure_ascii=False))
    return 0 if summary["status"] == "MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
