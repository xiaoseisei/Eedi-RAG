from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from evals.adapters.storage import capture_storage_snapshot_from_paths
from evals.contracts import L1ComponentReport, RunContext, canonical_system_config_hash
from evals.reporting import write_report
from evals.runners.storage import StorageRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit existing Eedi-RAG DuckDB/Chroma artifacts read-only")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", help="unique run ID; defaults to a UTC timestamp")
    args = parser.parse_args(argv)

    db_path = args.db.resolve(strict=True)
    chroma_path = args.chroma.resolve(strict=True)
    artifact_hash = hash_storage_artifacts(db_path, chroma_path)
    run_id = args.run_id or datetime.now(UTC).strftime("l1-storage-%Y%m%dT%H%M%SZ")
    system_config = {
        "component": "storage",
        "audit_contract": "storage-l1/v1",
        "required_collections": [
            "student_misconceptions",
            "tutor_strategies",
            "fallback_windows",
        ],
        "source_access": {
            "duckdb": "read_only",
            "chroma_sqlite": "mode=ro&immutable=1",
        },
    }
    context = RunContext(
        run_id=run_id,
        dataset_version=f"storage-artifacts-{artifact_hash[:16]}",
        qrels_version="not-applicable",
        system_config_hash=canonical_system_config_hash(system_config),
        system_config=system_config,
    )
    started_at = datetime.now(UTC)
    snapshot = capture_storage_snapshot_from_paths(db_path, chroma_path)
    after_capture_hash = hash_storage_artifacts(db_path, chroma_path)
    if after_capture_hash != artifact_hash:
        raise RuntimeError(
            "read-only storage capture changed an artifact; evaluation aborted to avoid reporting a moving snapshot"
        )
    observations = StorageRunner().run(context, snapshot)
    report = L1ComponentReport.from_observations(
        context=context,
        observations=observations,
        executed_modules=["storage"],
        started_at=started_at,
    )
    run_dir = write_report(report, args.output_root)
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "artifact_hash": artifact_hash,
                "release_status": report.release_status,
                "status_counts": report.status_counts,
                "report_dir": str(run_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.release_status == "PASSED" else 2


def hash_storage_artifacts(db_path: Path, chroma_path: Path) -> str:
    digest = hashlib.sha256()
    chroma_files = sorted(item for item in chroma_path.rglob("*") if item.is_file())
    if not chroma_files:
        raise ValueError(f"Chroma directory has no files: {chroma_path}")
    files = [("duckdb", db_path), *[(item.relative_to(chroma_path).as_posix(), item) for item in chroma_files]]
    for logical_name, path in files:
        digest.update(logical_name.encode("utf-8"))
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
