from __future__ import annotations

"""Run the complete L1 v2 suite against an isolated Qwen embedding index."""

import argparse
import gc
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.contracts import (
    AssemblerEvalCase,
    ChunkStructuralEvalCase,
    ChunkingEvalCase,
    FusionEvalCase,
    GroundingEvalCase,
    L1ComponentSuiteSpec,
    RewriteEvalCase,
    RetrievalEvalCase,
    RunContext,
    canonical_system_config_hash,
)
from evals.live_storage import hash_storage_artifacts
from evals.adapters.storage import capture_storage_snapshot_from_paths
from evals.reporting import write_report
from evals.suite import run_component_suite
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _load_golden() -> list[dict]:
    return json.loads((PROJECT_ROOT / "data/golden_test_set.json").read_text(encoding="utf-8"))


def _target_id(case: dict) -> tuple[str, str]:
    quotes = case.get("verbatim_grounding_quotes") or []
    if not quotes:
        raise ValueError("golden case has no verbatim grounding quotes")
    intent = case.get("category", "UNKNOWN")
    slot = "misconception" if intent == "STUDENT_INSIGHT" else "strategy"
    return f"session_{quotes[0]['session_id']}_{slot if slot == 'misconception' else 'tutor_strategy'}", slot


def build_retrieval_cases(golden: list[dict], retriever, *, top_k: int) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    for index, case in enumerate(golden, 1):
        target_id, slot = _target_id(case)
        collection = "student_misconceptions" if slot == "misconception" else "tutor_strategies"
        import time

        started = time.perf_counter()
        candidates = (
            retriever.retrieve_misconceptions(case["question"], top_k=top_k, fetch_evidence=False)
            if slot == "misconception"
            else retriever.retrieve_strategies(case["question"], top_k=top_k, fetch_evidence=False)
        )
        cases.append(RetrievalEvalCase(
            case_id=f"golden-{index:02d}-qwen-retrieval",
            query=case["question"],
            ranked_ids=[str(item["chunk_id"]) for item in candidates],
            qrels={target_id: 3},
            latency_ms=(time.perf_counter() - started) * 1000.0,
            slices={"category": case.get("category", "unknown"), "collection": collection, "embedding": "qwen3-0.6b"},
        ))
    return cases


def _attach_split(cases: list, split_assignments: dict[str, str]) -> None:
    for index, case in enumerate(cases, 1):
        query_id = f"eedi-l1-{index:04d}"
        if query_id not in split_assignments:
            raise ValueError(f"split manifest is missing {query_id}")
        case.slices = {**case.slices, "split": split_assignments[query_id]}


def _write_closeout(
    *,
    run_dir: Path,
    report,
    dataset_manifest: dict,
    source_hash: str,
    after_hash: str,
    reproduction_command: str,
) -> dict:
    hard_blockers = [
        item.model_dump(mode="json")
        for item in report.observations
        if item.hard_gate and item.status.value != "SUCCESS"
    ]
    decision = "GO_TO_L2_TECHNICAL" if not hard_blockers else "BLOCKED"
    stage_actions = {
        "storage": "repair index/relational parity or metadata and rebuild a new isolated index",
        "chunk_structure": "repair production chunk pointers/boundaries before retrieval tuning",
        "chunking": "inspect failed window ranking and query/document representation",
        "grounding": "inspect the exact citation quad and final-evidence authorization",
        "rewrite": "obtain human/DeepEval labels or repair protected-token preservation",
        "fusion": "retain raw_first or validate another non-degrading strategy on dev/holdout",
        "retrieval": "improve candidate representation/recall on the frozen qrels",
        "assembler": "inspect retriever_miss versus assembler_drop and slot retention",
    }
    blocker_rows = []
    for item in hard_blockers:
        blocker_rows.append({
            "case_id": item["case_id"],
            "stage": item["stage"],
            "metric": item["metric_name"],
            "scope": item["scope"],
            "value": item["value"],
            "threshold": item["threshold"],
            "status": item["status"],
            "evidence_refs": item.get("evidence_refs", []),
            "action": stage_actions.get(item["stage"], "inspect the recorded case trace"),
        })
    aggregate = [
        item.model_dump(mode="json")
        for item in report.observations
        if item.scope == "aggregate"
    ]
    payload = {
        "schema_version": "l1-closeout/v1",
        "decision": decision,
        "business_quality_passed": False,
        "business_quality_reason": dataset_manifest["qrels_provenance"]["limitation"],
        "run_id": report.run_id,
        "dataset_version": report.dataset_version,
        "qrels_version": report.qrels_version,
        "system_config_hash": report.system_config_hash,
        "status_counts": report.status_counts,
        "artifact_hash_before": source_hash,
        "artifact_hash_after": after_hash,
        "artifact_mutated_during_evaluation": source_hash != after_hash,
        "api_usage": report.system_config.get("embedding_telemetry", {}),
        "api_cost": {
            "status": "UNMEASURED",
            "reason": "provider pricing was not configured; token usage is recorded without fabricating a cost",
        },
        "aggregate_metrics": aggregate,
        "hard_blocker_count": len(blocker_rows),
        "hard_blockers": blocker_rows,
        "reproduction_command": reproduction_command,
        "l2": {
            "admitted": decision == "GO_TO_L2_TECHNICAL",
            "started": False,
            "reason": (
                "L1 hard gates passed; L2 may start in a separate run"
                if decision == "GO_TO_L2_TECHNICAL"
                else "L1 hard gates failed; DeepEval entry is prohibited"
            ),
        },
    }
    (run_dir / "closeout.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage_order = ["storage", "chunk_structure", "chunking", "grounding", "rewrite", "fusion", "retrieval", "assembler"]
    lines = [
        "# Eedi-RAG L1 Closeout",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Run: `{report.run_id}`",
        f"- Dataset: `{report.dataset_version}`",
        f"- Qrels: `{report.qrels_version}`",
        f"- Artifact hash unchanged: `{source_hash == after_hash}`",
        f"- Hard blockers: `{len(blocker_rows)}`",
        "- Business quality claim: `NOT_ALLOWED` (qrels are not independent multi-document graded judgments)",
        "",
    ]
    for stage in stage_order:
        lines.extend([f"## {stage}", ""])
        rows = [item for item in aggregate if item["stage"] == stage]
        if not rows:
            lines.append("No aggregate observation.")
        else:
            lines.extend(["| Metric | Status | Value | Threshold | Hard gate |", "| --- | --- | ---: | ---: | --- |"])
            for item in rows:
                lines.append(
                    f"| {item['metric_name']} | {item['status']} | {item['value']} | {item['threshold']} | {item['hard_gate']} |"
                )
        lines.append("")
    lines.extend(["## Reproduce", "", f"`{reproduction_command}`", ""])
    (run_dir / "closeout.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run complete L1 v2 evaluation on Qwen3-Embedding-0.6B")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, required=True, help="isolated Qwen Chroma index")
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    _load_dotenv_if_available()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    db_path = args.db.resolve(strict=True)
    chroma_path = args.chroma.resolve(strict=True)
    dataset_manifest_path = args.dataset_manifest.resolve(strict=True)
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    split_file = dataset_manifest_path.parent / dataset_manifest["split"]["file"]
    split_manifest = json.loads(split_file.read_text(encoding="utf-8"))
    split_assignments = split_manifest["query_assignments"]
    provider = SiliconFlowQwen3EmbeddingFunction.from_env()
    if provider.model != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError(f"expected Qwen/Qwen3-Embedding-0.6B, got {provider.model!r}")

    source_hash = hash_storage_artifacts(db_path, chroma_path)
    expected_hash = dataset_manifest["artifact_hashes"]["qwen_combined_sha256"]
    if source_hash != expected_hash:
        raise ValueError(f"Qwen source hash does not match frozen dataset: expected={expected_hash} actual={source_hash}")
    storage_snapshot = capture_storage_snapshot_from_paths(db_path, chroma_path, case_id="storage-qwen-closeout")
    if hash_storage_artifacts(db_path, chroma_path) != source_hash:
        raise RuntimeError("immutable storage audit changed the Qwen source artifact")
    run_id = args.run_id or datetime.now(UTC).strftime("l1-qwen3-0.6b-full-%Y%m%dT%H%M%SZ")
    golden = _load_golden()
    output_root = args.output_root.resolve()
    scratch_root = output_root / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="l1-qwen-full-", dir=str(scratch_root), ignore_cleanup_errors=True) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)

        from scripts.build_l1_component_datasets import (
            build_assembler_cases,
            build_chunk_retrieval_cases,
            build_chunk_structure_cases,
            build_fusion_cases,
            build_grounding_cases,
            build_rewrite_cases,
            load_sessions,
        )
        from src.query_rewriter import MultiPerspectiveQueryRewriter
        from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
        from src.reranker import PedagogicalGoldAssembler
        from src.retriever import DualMetricRetriever
        from src.storage_manager import DualEngineStorageManager

        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_function=provider,
            embedding_index_version=provider.index_version,
        )
        try:
            retriever = DualMetricRetriever(
                storage_manager=storage,
                query_rewrite_mode="deterministic",
                alpha=0.5,
                fusion_strategy="raw_first",
            )
            assembler = PedagogicalGoldAssembler(lambda_diversity=0.7, max_prompt_tokens=1500)
            pipeline = EndToEndPedagogicalRAGPipeline(retriever=retriever, assembler=assembler)
            rewriter = MultiPerspectiveQueryRewriter(mode="deterministic")
            sessions = load_sessions()
            indexed_session_ids = {
                int(row[0])
                for row in storage.duck_conn.execute("SELECT DISTINCT session_id FROM sliding_window_chunks").fetchall()
            }
            retrieval_cases = build_retrieval_cases(golden, retriever, top_k=args.top_k)
            grounding_cases = [GroundingEvalCase.model_validate(item) for item in build_grounding_cases(golden, sessions, pipeline)]
            chunk_structure_cases = [ChunkStructuralEvalCase.model_validate(item) for item in build_chunk_structure_cases(sessions, indexed_session_ids)]
            chunking_cases = [ChunkingEvalCase.model_validate(item) for item in build_chunk_retrieval_cases(golden, retriever)]
            rewrite_cases = [RewriteEvalCase.model_validate(item) for item in build_rewrite_cases(golden, sessions, retriever, rewriter)]
            fusion_cases = [FusionEvalCase.model_validate(item) for item in build_fusion_cases(golden, retriever)]
            assembler_cases = [AssemblerEvalCase.model_validate(item) for item in build_assembler_cases(golden, sessions, retriever, assembler, top_k_each=20)]
            for cases in (retrieval_cases, grounding_cases, chunking_cases, rewrite_cases, fusion_cases, assembler_cases):
                _attach_split(cases, split_assignments)
            for case in chunk_structure_cases:
                case.slices = {**case.slices, "split": "all_indexed"}
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
        raise RuntimeError("source Qwen artifact changed during complete L1 evaluation")

    system_config = {
        "component": "l1-closeout-all-components",
        "embedding_backend": "siliconflow_openai_compatible",
        "embedding_model": provider.model,
        "embedding_dimension": provider.dimension,
        "embedding_telemetry": provider.telemetry,
        "embedding_provider": provider.name(),
        "embedding_base_url": provider.base_url,
        "embedding_index_version": provider.index_version,
        "embedding_batch_size": provider.batch_size,
        "embedding_timeout_seconds": provider.timeout_seconds,
        "embedding_cache_key_fields": ["provider", "model", "index_version", "query_text"],
        "embedding_cache_success_only": True,
        "query_rewrite_mode": "deterministic_rules",
        "retrieval_backend": "dense-chroma-plus-dialogue-numeric-formula-exact",
        "fusion_strategy": "raw_first",
        "rrf_k": 60,
        "mmr_lambda": 0.7,
        "assembler_top_k_each": 20,
        "assembler_max_prompt_tokens": 1500,
        "top_k": args.top_k,
        "artifact_hash": source_hash,
        "source_access": "isolated-artifact-copy",
        "dataset_manifest_sha256": __import__("hashlib").sha256(dataset_manifest_path.read_bytes()).hexdigest(),
    }
    context = RunContext(
        run_id=run_id,
        dataset_version=dataset_manifest["dataset_version"],
        qrels_version=dataset_manifest["qrels_version"],
        system_config_hash=canonical_system_config_hash(system_config),
        system_config=system_config,
    )
    report = run_component_suite(L1ComponentSuiteSpec(
        schema_version="l1-component-suite/v2",
        context=context,
        storage=storage_snapshot,
        grounding_cases=grounding_cases,
        chunking_cases=chunking_cases,
        chunk_structural_cases=chunk_structure_cases,
        rewrite_cases=rewrite_cases,
        fusion_cases=fusion_cases,
        retrieval_cases=retrieval_cases,
        assembler_cases=assembler_cases,
    ))
    run_dir = write_report(report, output_root)
    (run_dir / "embedding_usage.json").write_text(json.dumps({
        "provider": provider.name(),
        "model": provider.model,
        "dimension": provider.dimension,
        "usage": provider.telemetry,
        "artifact_hash": source_hash,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": report.run_id,
        "release_status": report.release_status,
        "status_counts": report.status_counts,
        "report_dir": str(run_dir),
        "artifact_hash": source_hash,
        "embedding_model": provider.model,
        "embedding_dimension": provider.dimension,
        "embedding_usage": provider.telemetry,
    }, ensure_ascii=False))
    reproduction_command = (
        f".venv\\Scripts\\python.exe scripts\\run_l1_qwen_full_eval.py --db {args.db} "
        f"--chroma {args.chroma} --dataset-manifest {args.dataset_manifest} "
        f"--output-root {args.output_root} --run-id <new-unique-run-id> --top-k {args.top_k}"
    )
    closeout = _write_closeout(
        run_dir=run_dir,
        report=report,
        dataset_manifest=dataset_manifest,
        source_hash=source_hash,
        after_hash=after_hash,
        reproduction_command=reproduction_command,
    )
    print(json.dumps({"closeout_decision": closeout["decision"], "hard_blockers": closeout["hard_blocker_count"]}, ensure_ascii=False))
    return 0 if closeout["decision"] == "GO_TO_L2_TECHNICAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
