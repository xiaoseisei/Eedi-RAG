from __future__ import annotations

"""Full, same-qrels comparison of card-first and content-first chunk routes."""

import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.graded_qrels import build_card_graded_qrels, build_window_graded_qrels
from evals.live_storage import hash_storage_artifacts
from evals.metrics.retrieval import ndcg_at_k
from scripts.run_turn_graded_eval import _card_slot_with_required_evidence, _graded_mrr, _graded_precision, _graded_recall
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction
from src.reranker_provider import SiliconFlowQwen3Reranker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.resolve(strict=True).read_text(encoding="utf-8").splitlines() if line.strip()]


def _required(case: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(item["session_id"]), int(item["turn_id"])) for item in case.get("verbatim_grounding_quotes", [])}


def _parse_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in json.loads(value)]
    return [int(item) for item in (value or [])]


def _covered(candidates: list[dict[str, Any]], k: int) -> set[tuple[int, int]]:
    covered: set[tuple[int, int]] = set()
    for candidate in candidates[:k]:
        metadata = candidate.get("metadata", {})
        session_id = int(metadata.get("session_id"))
        covered.update((session_id, turn_id) for turn_id in _parse_ids(metadata.get("source_turn_ids", [])))
    return covered


def _turn_coverage(candidates: list[dict[str, Any]], required: set[tuple[int, int]], k: int) -> float:
    return len(_covered(candidates, k) & required) / len(required) if required else 0.0


def _candidate_text(candidate: dict[str, Any]) -> str:
    document = str(candidate.get("document", ""))
    evidence = candidate.get("evidence_turns", []) or []
    evidence_text = "\n".join(
        f"[Turn {turn.get('turn_id')}] [{turn.get('speaker')}] {turn.get('text', '')}"
        for turn in evidence
    )
    return f"{document}\n【直接证据】\n{evidence_text}" if evidence_text else document


def _rerank(provider: SiliconFlowQwen3Reranker, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    scores = provider.rerank(query, [_candidate_text(candidate) for candidate in candidates], top_n=len(candidates))
    result: list[dict[str, Any]] = []
    for rank, score in enumerate(scores, start=1):
        candidate = dict(candidates[score.index])
        candidate["reranker_score"] = score.relevance_score
        candidate["reranker_rank"] = rank
        candidate["reranker_model"] = provider.name()
        candidate["reranking_applied"] = True
        result.append(candidate)
    return result


def _all_collection_rows(connection: Any, table: str) -> list[dict[str, Any]]:
    return [
        {"chunk_id": str(row[0]), "session_id": int(row[1]), "subject_path": row[2] or "", "source_turn_ids": list(row[3] or [])}
        for row in connection.execute(f"SELECT chunk_id, session_id, subject_path, source_turn_ids FROM {table} ORDER BY chunk_id").fetchall()
    ]


def _metrics_for_rankings(
    *,
    pool: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    qrels: dict[str, int],
    required_turns: set[tuple[int, int]],
    target_turns: set[tuple[int, int]] | None = None,
    ks: tuple[int, ...] = (5, 10, 15),
) -> dict[str, Any]:
    pool_ids = [str(item["chunk_id"]) for item in pool]
    ranked_ids = [str(item["chunk_id"]) for item in ranked]
    result: dict[str, Any] = {}
    target_turns = target_turns or required_turns
    for k in ks:
        result[f"pool_recall_at_{k}"] = _graded_recall(pool_ids, qrels, k)
        result[f"ranked_recall_at_{k}"] = _graded_recall(ranked_ids, qrels, k)
        result[f"pool_turn_coverage_at_{k}"] = _turn_coverage(pool, required_turns, k)
        result[f"ranked_turn_coverage_at_{k}"] = _turn_coverage(ranked, required_turns, k)
        result[f"pool_target_turn_coverage_at_{k}"] = _turn_coverage(pool, target_turns, k)
        result[f"ranked_target_turn_coverage_at_{k}"] = _turn_coverage(ranked, target_turns, k)
    result.update({
        "pool_mrr": _graded_mrr(pool_ids, qrels),
        "ranked_mrr": _graded_mrr(ranked_ids, qrels),
        "pool_ndcg_at_5": ndcg_at_k(pool_ids, qrels, 5),
        "ranked_ndcg_at_5": ndcg_at_k(ranked_ids, qrels, 5),
        "pool_precision_at_5": _graded_precision(pool_ids, qrels, 5),
        "ranked_precision_at_5": _graded_precision(ranked_ids, qrels, 5),
    })
    return result


def evaluate_strategy(
    *,
    strategy: str,
    db_path: Path,
    chroma_path: Path,
    golden_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    pool_size: int,
    bm25_weight: float,
    use_reranker: bool,
) -> dict[str, Any]:
    if strategy not in {"card", "fallback"}:
        raise ValueError("strategy must be card or fallback")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite strategy eval: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    golden = json.loads(golden_path.resolve(strict=True).read_text(encoding="utf-8"))
    split_assignments = json.loads(split_manifest_path.resolve(strict=True).read_text(encoding="utf-8"))["query_assignments"]
    source_hash = hash_storage_artifacts(db_path, chroma_path)
    embedding = SiliconFlowQwen3EmbeddingFunction.from_env()
    reranker = SiliconFlowQwen3Reranker.from_env() if use_reranker else None
    with tempfile.TemporaryDirectory(prefix=f"chunk-{strategy}-", dir=str(output_dir.parent), ignore_cleanup_errors=True) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        import duckdb
        from src.retriever import DualMetricRetriever
        from src.reranker import PedagogicalGoldAssembler
        from src.storage_manager import DualEngineStorageManager

        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_function=embedding,
            embedding_index_version=embedding.index_version,
        )
        retriever = DualMetricRetriever(
            storage_manager=storage,
            query_rewrite_mode="deterministic",
            fusion_strategy="raw_first",
            retrieval_mode="bm25_dense",
            bm25_weight=bm25_weight,
        )
        assembler = PedagogicalGoldAssembler(
            chunk_strategy=strategy,
            model_reranker=None,
            reranker_pool_size=pool_size,
        )
        conn = storage.duck_conn
        rows: list[dict[str, Any]] = []
        try:
            windows = _all_collection_rows(conn, "sliding_window_chunks")
            cards_cache = {
                "misconception": _all_collection_rows(conn, "misconception_chunks"),
                "strategy": _all_collection_rows(conn, "tutor_strategy_chunks"),
            }
            for index, case in enumerate(golden, start=1):
                query = case["question"]
                query_id = f"eedi-l1-{index:04d}"
                slot, slot_fallback = _card_slot_with_required_evidence(case)
                if strategy == "fallback":
                    qrel_rows = build_window_graded_qrels({**case, "query_id": query_id}, windows)
                    pool = retriever.retrieve_fallback_windows(query, top_k=pool_size, fetch_evidence=True)
                else:
                    qrel_rows = build_card_graded_qrels({**case, "query_id": query_id}, cards_cache[slot], target_slot=slot)
                    pool = retriever.retrieve_misconceptions(query, top_k=pool_size, fetch_evidence=True) if slot == "misconception" else retriever.retrieve_strategies(query, top_k=pool_size, fetch_evidence=True)
                qrels = {row["document_id"]: row["relevance"] for row in qrel_rows}
                ranked = _rerank(reranker, query, pool) if reranker is not None else pool
                retrieval_results = {
                    "chunk_strategy": strategy,
                    "rewritten_queries": {"extracted_keywords": []},
                }
                if strategy == "fallback":
                    retrieval_results.update({"windows": ranked, "misconceptions": [], "strategies": []})
                else:
                    retrieval_results.update({"windows": [], "misconceptions": ranked if slot == "misconception" else [], "strategies": ranked if slot == "strategy" else []})
                assembly_started = time.perf_counter()
                context = assembler.assemble(raw_query=query, retrieval_results=retrieval_results)
                assembly_ms = round((time.perf_counter() - assembly_started) * 1000.0, 3)
                required = _required(case)
                context_keys = {(int(turn["session_id"]), int(turn["turn_id"])) for turn in context.evidence_turns}
                target_required = {
                    (int(item["session_id"]), int(item["turn_id"]))
                    for item in case.get("verbatim_grounding_quotes", [])
                    if strategy == "fallback" or str(item.get("speaker")) == ("student" if slot == "misconception" else "tutor")
                }
                rows.append({
                    "case_id": f"chunk-{strategy}-{index:04d}",
                    "query_id": query_id,
                    "split": split_assignments.get(query_id, "unknown"),
                    "strategy": strategy,
                    "category": case.get("category", "UNKNOWN"),
                    "session_id": int(case["verbatim_grounding_quotes"][0]["session_id"]),
                    "target_slot": slot,
                    "slot_fallback": slot_fallback,
                    "pool_size": pool_size,
                    "use_reranker": use_reranker,
                    "retrieval_mode": "bm25_dense",
                    "bm25_weight": bm25_weight,
                    "pool_ids": [str(item["chunk_id"]) for item in pool],
                    "ranked_ids": [str(item["chunk_id"]) for item in ranked],
                    "evidence_count": len(context.evidence_turns),
                    "context_token_count": context.estimated_token_count,
                    "budget_violation": context.budget_violation,
                    "assembly_ms": assembly_ms,
                    "final_evidence_recall": len(context_keys & required) / len(required) if required else 0.0,
                    "target_card_or_window_evidence_recall": len(context_keys & target_required) / len(target_required) if target_required else 0.0,
                    **_metrics_for_rankings(
                        pool=pool,
                        ranked=ranked,
                        qrels=qrels,
                        required_turns=required,
                        target_turns=target_required,
                    ),
                })
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
        raise RuntimeError("source artifact changed during chunk strategy evaluation")
    successful = rows
    metric_names = [
        "pool_recall_at_5", "pool_recall_at_10", "pool_recall_at_15",
        "ranked_recall_at_5", "ranked_recall_at_10", "ranked_recall_at_15",
        "pool_turn_coverage_at_5", "pool_turn_coverage_at_10", "pool_turn_coverage_at_15",
        "ranked_turn_coverage_at_5", "ranked_turn_coverage_at_10", "ranked_turn_coverage_at_15",
        "pool_target_turn_coverage_at_5", "pool_target_turn_coverage_at_10", "pool_target_turn_coverage_at_15",
        "ranked_target_turn_coverage_at_5", "ranked_target_turn_coverage_at_10", "ranked_target_turn_coverage_at_15",
        "pool_mrr", "ranked_mrr", "pool_ndcg_at_5", "ranked_ndcg_at_5",
        "pool_precision_at_5", "ranked_precision_at_5", "final_evidence_recall",
        "target_card_or_window_evidence_recall", "context_token_count", "assembly_ms",
    ]
    aggregates: dict[str, Any] = {}
    for name in metric_names:
        values = [float(row[name]) for row in successful]
        aggregates[name] = {
            "mean": mean(values) if values else None,
            "dev_mean": mean([float(row[name]) for row in successful if row["split"] == "dev"]) if any(row["split"] == "dev" for row in successful) else None,
            "holdout_mean": mean([float(row[name]) for row in successful if row["split"] == "holdout"]) if any(row["split"] == "holdout" for row in successful) else None,
            "measured_cases": len(values),
        }
    summary = {
        "schema_version": "chunk-strategy-full-eval/v1",
        "status": "MEASURED",
        "strategy": strategy,
        "embedding_model": embedding.model,
        "embedding_backend": "qwen",
        "retrieval_mode": "bm25_dense",
        "bm25_weight": bm25_weight,
        "pool_size": pool_size,
        "use_reranker": use_reranker,
        "reranker_model": reranker.model if reranker else None,
        "strategy_semantics": (
            "card retrieval over role-specific misconception/strategy cards with source_turn pointer backtracking"
            if strategy == "card"
            else "content-first retrieval over immutable fallback sliding windows with direct Turn evidence"
        ),
        "case_count": len(rows),
        "aggregates": aggregates,
        "reranker_telemetry": reranker.telemetry if reranker else None,
        "artifact_hash_before": source_hash,
        "artifact_hash_after": after_hash,
        "artifact_mutated": source_hash != after_hash,
        "limitations": [
            "qrels are provisional rule-derived labels and require human review",
            "deterministic generation is used for final evidence coverage; LLM semantic answer quality is evaluated separately by L2",
            "pool_turn_coverage measures all required Turns; pool_target_turn_coverage is the role-aware card comparison",
        ],
    }
    (output_dir / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Chunk Strategy Full Evaluation",
        "",
        f"Status: **{summary['status']}**; strategy=`{strategy}`; reranker=`{use_reranker}`",
        "",
        "| Metric | Mean | Dev | Holdout |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in aggregates.items():
        lines.append(f"| {name} | {value['mean']} | {value['dev_mean']} | {value['holdout_mean']} |")
    lines.extend(["", "Detailed per-case rankings and evidence are in `results.jsonl`."])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("card", "fallback"), required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data" / "golden_test_set.json")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=15)
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args(argv)
    summary = evaluate_strategy(
        strategy=args.strategy,
        db_path=args.db.resolve(strict=True),
        chroma_path=args.chroma.resolve(strict=True),
        golden_path=args.golden.resolve(strict=True),
        split_manifest_path=args.split_manifest.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        pool_size=args.pool_size,
        bm25_weight=args.bm25_weight,
        use_reranker=not args.no_reranker,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "strategy": args.strategy, "status": summary["status"], "case_count": summary["case_count"], "artifact_mutated": summary["artifact_mutated"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
