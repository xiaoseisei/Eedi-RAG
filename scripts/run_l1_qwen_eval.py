from __future__ import annotations

"""Run Retrieval/Fusion/Assembler L1 evaluation against a Qwen Chroma index."""

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.contracts import (
    FusionEvalCase,
    L1ComponentSuiteSpec,
    RetrievalEvalCase,
    RunContext,
    canonical_system_config_hash,
)
from evals.live_storage import hash_storage_artifacts
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


def build_retrieval_cases(golden: list[dict], retriever, *, top_k: int = 20) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    for index, case in enumerate(golden, 1):
        target_id, slot = _target_id(case)
        collection = "student_misconceptions" if slot == "misconception" else "tutor_strategies"
        started = time.perf_counter()
        if slot == "misconception":
            candidates = retriever.retrieve_misconceptions(case["question"], top_k=top_k, fetch_evidence=False)
        else:
            candidates = retriever.retrieve_strategies(case["question"], top_k=top_k, fetch_evidence=False)
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [str(item["chunk_id"]) for item in candidates]
        cases.append(RetrievalEvalCase(
            case_id=f"golden-{index:02d}-qwen-retrieval",
            query=case["question"],
            ranked_ids=ranked_ids,
            qrels={target_id: 3},
            latency_ms=latency_ms,
            slices={"category": case.get("category", "unknown"), "collection": collection, "embedding": "qwen3-0.6b"},
        ))
    return cases


def _copy_inputs(db_path: Path, chroma_path: Path, scratch: Path) -> tuple[Path, Path]:
    copied_db = scratch / "tutoring_knowledge.duckdb"
    copied_chroma = scratch / "chroma"
    shutil.copy2(db_path, copied_db)
    shutil.copytree(chroma_path, copied_chroma)
    return copied_db, copied_chroma


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run L1 Retrieval/Fusion/Assembler on Qwen3-Embedding-0.6B")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, required=True, help="isolated Qwen Chroma index")
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    db_path = args.db.resolve(strict=True)
    chroma_path = args.chroma.resolve(strict=True)
    if not chroma_path.is_dir():
        parser.error(f"--chroma must be an existing directory: {chroma_path}")

    _load_dotenv_if_available()
    provider = SiliconFlowQwen3EmbeddingFunction.from_env()
    if provider.model != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError(
            "Qwen L1 evaluation requires EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B; "
            f"got {provider.model!r}"
        )
    source_hash = hash_storage_artifacts(db_path, chroma_path)
    run_id = args.run_id or datetime.now(UTC).strftime("l1-qwen3-0.6b-%Y%m%dT%H%M%SZ")
    golden = _load_golden()
    system_config = {
        "component": "retrieval-fusion-assembler",
        "embedding_backend": "siliconflow_openai_compatible",
        "embedding_model": provider.model,
        "embedding_dimension": provider.dimension,
        "query_rewrite_mode": "deterministic_rules",
        "retrieval_backend": "dense-only-chroma",
        "mmr_lambda": 0.7,
        "top_k": args.top_k,
        "artifact_hash": source_hash,
        "source_access": "isolated-artifact-copy",
    }
    context = RunContext(
        run_id=run_id,
        dataset_version="golden30-v2-qwen3-0.6b",
        qrels_version="golden-verbatim-v2-derived-document-v1",
        system_config_hash=canonical_system_config_hash(system_config),
        system_config=system_config,
    )

    output_root = args.output_root.resolve()
    scratch_root = output_root / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    # Chroma's local HNSW backend may keep mapped index files locked briefly on
    # Windows after the client is closed.  Cleanup errors in this disposable
    # evaluation copy must not erase the measured result; the scratch copy is
    # intentionally retained under reports/runtime for later cleanup.
    with tempfile.TemporaryDirectory(
        prefix="l1-qwen-",
        dir=str(scratch_root),
        ignore_cleanup_errors=True,
    ) as temp_name:
        copied_db, copied_chroma = _copy_inputs(db_path, chroma_path, Path(temp_name))
        from src.retriever import DualMetricRetriever
        from src.reranker import PedagogicalGoldAssembler
        from src.storage_manager import DualEngineStorageManager
        from scripts.build_l1_component_datasets import build_assembler_cases, build_fusion_cases, load_sessions

        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_function=provider,
            embedding_index_version=provider.index_version,
        )
        try:
            retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic", alpha=0.5)
            assembler = PedagogicalGoldAssembler(lambda_diversity=0.7)
            sessions = load_sessions()
            retrieval_cases = build_retrieval_cases(golden, retriever, top_k=args.top_k)
            fusion_cases = [FusionEvalCase.model_validate(item) for item in build_fusion_cases(golden, retriever)]
            assembler_cases = [
                # build_assembler_cases returns strict JSON dictionaries from the same trace.
                __import__("evals.contracts", fromlist=["AssemblerEvalCase"]).AssemblerEvalCase.model_validate(item)
                for item in build_assembler_cases(golden, sessions, retriever, assembler)
            ]
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
        raise RuntimeError("source Qwen artifact changed during L1 evaluation")

    # Complete provenance only after all API calls have finished: dimension and
    # usage are discovered from real responses, never guessed in advance.
    context.system_config["embedding_dimension"] = provider.dimension
    context.system_config["embedding_usage"] = provider.usage.__dict__
    context.system_config_hash = canonical_system_config_hash(context.system_config)

    report = run_component_suite(L1ComponentSuiteSpec(
        schema_version="l1-component-suite/v2",
        context=context,
        retrieval_cases=retrieval_cases,
        fusion_cases=fusion_cases,
        assembler_cases=assembler_cases,
    ))
    run_dir = write_report(report, output_root)
    (run_dir / "embedding_usage.json").write_text(
        json.dumps({
            "provider": provider.name(),
            "model": provider.model,
            "dimension": provider.dimension,
            "usage": provider.usage.__dict__,
            "artifact_hash": source_hash,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_id": report.run_id,
        "release_status": report.release_status,
        "status_counts": report.status_counts,
        "report_dir": str(run_dir),
        "artifact_hash": source_hash,
        "embedding_model": provider.model,
        "embedding_dimension": provider.dimension,
        "embedding_usage": provider.usage.__dict__,
    }, ensure_ascii=False))
    return 0 if report.release_status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
