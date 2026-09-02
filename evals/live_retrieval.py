from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from evals.adapters.retrieval import capture_legacy_retrieval_cases
from evals.contracts import L1ComponentReport, RunContext, canonical_system_config_hash
from evals.live_storage import hash_storage_artifacts
from evals.reporting import write_report
from evals.runners.retrieval import RetrievalRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run legacy retrieval diagnostics on isolated artifact copies")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--mode", choices=["raw", "rrf", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="diagnostic smoke limit; omit for all 50")
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", help="unique run ID; defaults to a UTC timestamp")
    args = parser.parse_args(argv)

    db_path = args.db.resolve(strict=True)
    chroma_path = args.chroma.resolve(strict=True)
    source_hash = hash_storage_artifacts(db_path, chroma_path)
    run_id = args.run_id or datetime.now(UTC).strftime("l1-retrieval-legacy50-%Y%m%dT%H%M%SZ")
    modes = ["raw", "rrf"] if args.mode == "both" else [args.mode]
    system_config = {
        "component": "retrieval",
        "dataset": "legacy-benchmark-50",
        "qrels": "unique-target-session-v1",
        "embedding_backend": "deterministic",
        "modes": modes,
        "top_k": args.top_k,
        "source_access": "isolated-artifact-copy",
    }
    context = RunContext(
        run_id=run_id,
        dataset_version=f"legacy-benchmark-50-artifacts-{source_hash[:16]}",
        qrels_version="legacy-unique-session-v1",
        system_config_hash=canonical_system_config_hash(system_config),
        system_config=system_config,
    )
    started_at = datetime.now(UTC)
    cases = []
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root = output_root / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="retrieval-", dir=str(scratch_root)) as work_dir_name:
        work_dir = Path(work_dir_name)
        copied_db = work_dir / "tutoring_knowledge.duckdb"
        copied_chroma = work_dir / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)

        from src.retriever import DualMetricRetriever
        from src.storage_manager import DualEngineStorageManager

        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_backend="deterministic",
        )
        retriever = None
        try:
            retriever = DualMetricRetriever(
                storage_manager=storage,
                query_rewrite_mode="deterministic",
                alpha=0.5,
            )
            for mode in modes:
                cases.extend(
                    capture_legacy_retrieval_cases(
                        retriever,
                        mode=mode,  # type: ignore[arg-type]
                        top_k=args.top_k,
                        limit=args.limit,
                    )
                )
        finally:
            storage.close()
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
            retriever = None
            del storage
            gc.collect()

    after_hash = hash_storage_artifacts(db_path, chroma_path)
    if source_hash != after_hash:
        raise RuntimeError("source artifact changed during isolated retrieval evaluation")
    observations = RetrievalRunner().run(context, cases)
    report = L1ComponentReport.from_observations(
        context=context,
        observations=observations,
        executed_modules=["retrieval"],
        started_at=started_at,
    )
    run_dir = write_report(report, output_root)
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "case_count": len(cases),
                "artifact_hash": source_hash,
                "release_status": report.release_status,
                "status_counts": report.status_counts,
                "report_dir": str(run_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.release_status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
