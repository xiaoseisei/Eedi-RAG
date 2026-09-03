from __future__ import annotations

"""Build and evaluate the L3 business/proxy feedback loop without fake outcomes."""

import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.contracts import EvidenceRef
from evals.live_storage import hash_storage_artifacts
from evals.metrics.retrieval import mean_reciprocal_rank, recall_at_k
from scripts.build_l1_component_datasets import build_document_qrels
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_l3_dataset(*, output_dir: Path, golden_path: Path, l1_manifest_path: Path, artifact_hash: str) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite L3 dataset: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    l1_manifest = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads((l1_manifest_path.parent / l1_manifest["split"]["file"]).read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    for index, item in enumerate(golden, 1):
        case_id = f"eedi-l3-{index:04d}"
        query_id = f"eedi-l1-{index:04d}"
        source_sessions = sorted({int(q["session_id"]) for q in item.get("verbatim_grounding_quotes", [])})
        rows = build_document_qrels(item, query_id=query_id)
        qrels.extend([{**row, "case_id": case_id} for row in rows])
        cases.append({
            "case_id": case_id,
            "query": item["question"],
            "intent": item.get("category", "UNKNOWN"),
            "business_task": "student_insight" if item.get("category") == "STUDENT_INSIGHT" else "tutor_intervention",
            "split": split_manifest["query_assignments"][query_id],
            "source_session_ids": source_sessions,
            "expected_facts": [item["ground_truth"]],
            "required_evidence_quads": item.get("verbatim_grounding_quotes", []),
            "acceptance_criteria": [
                "target document is recalled by the real retrieval path",
                "final context contains the required evidence turns",
                "citation authorization is verified when a generator is present",
                "pedagogical action fields are present for tutor intervention cases",
            ],
            "business_data_status": "proxy_eedi_no_nbcot_feedback",
        })
    for name, rows in (("business_cases.jsonl", cases), ("qrels.jsonl", qrels)):
        (output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "layer": "L3",
        "data_mode": "Eedi_proxy_only",
        "real_business_outcomes": ["researcher_adoption", "task_completion", "learning_gain", "online_feedback"],
        "unmeasured_outcome_policy": "explicit_unmeasured_no_proxy_substitution",
        "retrieval_top_k": 20,
        "assembler_top_k_each": 5,
    }
    manifest = {
        "schema_version": "l3-business-proxy-dataset/v1",
        "dataset_version": "eedi-l3-business-proxy-v1",
        "qrels_version": l1_manifest["qrels_version"],
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "case_count": len(cases),
        "qrel_count": len(qrels),
        "split_counts": split_manifest["split_counts"],
        "session_leakage_count": split_manifest["session_leakage_count"],
        "artifact_hash": artifact_hash,
        "l1_manifest_sha256": _sha256(l1_manifest_path),
        "files": {
            name: _sha256(output_dir / name)
            for name in ("business_cases.jsonl", "qrels.jsonl", "split_manifest.json")
        },
        "system_config": config,
        "system_config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "limitations": [
            "Eedi is a technical proxy and not NBCOT business data",
            "no independent researcher adoption labels",
            "no online user feedback or learning-gain measurements",
            "qrels are derived from human quote evidence and are not multi-document graded judgments",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# L3 business proxy dataset\n\n"
        "This is an Eedi technical proxy for business-loop diagnostics. Adoption, task completion, learning gain, and online feedback are intentionally unmeasured until real researcher/user data is supplied.\n",
        encoding="utf-8",
    )
    return manifest


def _required_turn_keys(case: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(item["session_id"]), int(item["turn_id"])) for item in case.get("required_evidence_quads", [])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--l1-manifest", type=Path, required=True)
    parser.add_argument("--l1-closeout", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    l1_closeout = json.loads(args.l1_closeout.resolve(strict=True).read_text(encoding="utf-8"))
    db_path = args.db.resolve(strict=True)
    chroma_path = args.chroma.resolve(strict=True)
    artifact_hash = hash_storage_artifacts(db_path, chroma_path)
    l1_manifest_path = args.l1_manifest.resolve(strict=True)
    l1_manifest = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    if artifact_hash != l1_manifest["artifact_hashes"]["qwen_combined_sha256"]:
        raise ValueError("L3 artifact hash does not match frozen L1 manifest")
    dataset_manifest = build_l3_dataset(
        output_dir=args.dataset_output.resolve(),
        golden_path=PROJECT_ROOT / "data/golden_test_set.json",
        l1_manifest_path=l1_manifest_path,
        artifact_hash=artifact_hash,
    )
    cases = [json.loads(line) for line in (args.dataset_output.resolve() / "business_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    provider = SiliconFlowQwen3EmbeddingFunction.from_env()
    scratch_root = args.output_root.resolve() / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="l3-proxy-", dir=str(scratch_root), ignore_cleanup_errors=True) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        storage = None
        try:
            from src.storage_manager import DualEngineStorageManager
            from src.retriever import DualMetricRetriever
            from src.reranker import PedagogicalGoldAssembler

            storage = DualEngineStorageManager(
                db_path=copied_db,
                chroma_dir=copied_chroma,
                embedding_function=provider,
                embedding_index_version=provider.index_version,
            )
            retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic", fusion_strategy="raw_first")
            assembler = PedagogicalGoldAssembler(lambda_diversity=0.7, max_prompt_tokens=1500)
            for case in cases:
                started = time.perf_counter()
                collection = "student_misconceptions" if case["business_task"] == "student_insight" else "tutor_strategies"
                target_suffix = "misconception" if collection == "student_misconceptions" else "tutor_strategy"
                qrels = {f"session_{case['source_session_ids'][0]}_{target_suffix}": 3}
                ranked = (
                    retriever.retrieve_misconceptions(case["query"], top_k=args.top_k, fetch_evidence=False)
                    if collection == "student_misconceptions"
                    else retriever.retrieve_strategies(case["query"], top_k=args.top_k, fetch_evidence=False)
                )
                ranked_ids = [str(item["chunk_id"]) for item in ranked]
                retrieval = retriever.retrieve_multi_perspective_rrf(case["query"], top_k_each=5, fetch_evidence=True)
                context = assembler.assemble(case["query"], retrieval)
                final_ids = [str(item["chunk_id"]) for item in (context.selected_misconception, context.selected_strategy) if item]
                final_turn_keys = {(int(item["session_id"]), int(item["turn_id"])) for item in context.evidence_turns}
                required_keys = _required_turn_keys(case)
                selected = context.selected_misconception if collection == "student_misconceptions" else context.selected_strategy
                selected_meta = selected.get("metadata", {}) if selected else {}
                actionability = bool(
                    (selected_meta.get("misconception_name") and selected_meta.get("deep_mechanism"))
                    if collection == "student_misconceptions"
                    else (selected_meta.get("key_aha_question") and selected_meta.get("pedagogical_goal"))
                )
                rows.append({
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "query": case["query"],
                    "business_task": case["business_task"],
                    "ranked_ids": ranked_ids,
                    "final_context_ids": final_ids,
                    "candidate_recall_at_20": float(bool(set(qrels).intersection(ranked_ids))),
                    "recall_at_5": recall_at_k(ranked_ids, qrels, 5),
                    "mrr": mean_reciprocal_rank(ranked_ids, qrels),
                    "final_document_recall": float(bool(set(qrels).intersection(final_ids))),
                    "final_evidence_recall": len(required_keys.intersection(final_turn_keys)) / len(required_keys) if required_keys else 0.0,
                    "actionability": float(actionability),
                    "citation_audit": "UNMEASURED_GENERATOR_NOT_RUN",
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                })
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
    if after_hash != artifact_hash:
        raise RuntimeError("L3 evaluation changed the source artifact")
    run_dir = args.output_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "raw_cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    metrics: dict[str, Any] = {}
    for name in ("candidate_recall_at_20", "recall_at_5", "mrr", "final_document_recall", "final_evidence_recall", "actionability"):
        values = [float(row[name]) for row in rows]
        metrics[name] = {
            "mean": sum(values) / len(values) if values else None,
            "measured_cases": len(values),
            "dev_mean": sum(row[name] for row in rows if row["split"] == "dev") / max(1, sum(row["split"] == "dev" for row in rows)),
            "holdout_mean": sum(row[name] for row in rows if row["split"] == "holdout") / max(1, sum(row["split"] == "holdout" for row in rows)),
            "status": "DIAGNOSTIC_PROXY",
        }
    metrics.update({
        "researcher_adoption": {"status": "UNMEASURED", "reason": "no human researcher feedback dataset"},
        "task_completion": {"status": "UNMEASURED", "reason": "no user task outcome events"},
        "learning_gain": {"status": "UNMEASURED", "reason": "Eedi proxy contains no pre/post learning outcomes"},
        "online_feedback": {"status": "UNMEASURED", "reason": "no online telemetry or ratings"},
    })
    report = {
        "schema_version": "l3-business-proxy-report/v1",
        "run_id": args.run_id,
        "dataset": dataset_manifest,
        "l1_precondition": l1_closeout.get("decision"),
        "artifact_hash_before": artifact_hash,
        "artifact_hash_after": after_hash,
        "artifact_mutated": artifact_hash != after_hash,
        "case_count": len(rows),
        "metrics": metrics,
        "raw_case_file": "raw_cases.jsonl",
        "release_decision": "L3_BUSINESS_UNMEASURED",
        "release_reason": "real researcher/user/learning outcomes are unavailable; Eedi proxy metrics cannot be promoted to business quality",
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# L3 Business Proxy Report",
        "",
        "Release decision: **L3_BUSINESS_UNMEASURED**",
        "",
        f"- Cases: `{len(rows)}` (dev={sum(row['split'] == 'dev' for row in rows)}, holdout={sum(row['split'] == 'holdout' for row in rows)})",
        f"- Source artifact unchanged: `{artifact_hash == after_hash}`",
        "- Eedi is a technical proxy; adoption, task completion, learning gain, and online feedback are not measured.",
        "",
        "## Proxy metrics",
        "",
        "| Metric | Mean | Dev | Holdout | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, value in metrics.items():
        if "mean" in value:
            lines.append(f"| {name} | {value['mean']:.4f} | {value['dev_mean']:.4f} | {value['holdout_mean']:.4f} | {value['status']} |")
        else:
            lines.append(f"| {name} | — | — | — | {value['status']} |")
    lines.extend(["", "Raw per-case evidence: `raw_cases.jsonl`", ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "release_decision": report["release_decision"], "case_count": len(rows), "report_dir": str(run_dir), "artifact_hash": artifact_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
