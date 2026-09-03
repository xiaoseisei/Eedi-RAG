from __future__ import annotations

"""Freeze immutable L1 closeout queries, qrels, splits, and experiment provenance."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.contracts import canonical_system_config_hash
from evals.live_storage import hash_storage_artifacts
from scripts.build_l1_component_datasets import (
    _directory_sha256,
    _file_sha256,
    build_document_qrels,
    build_split_manifest,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def freeze_closeout_inputs(
    *,
    output_dir: Path,
    golden_path: Path,
    db_path: Path,
    deterministic_chroma: Path,
    qwen_chroma: Path,
    qwen_build_manifest: Path,
    rollback_chroma: Path,
    experiment_config: dict[str, Any],
) -> Path:
    paths = [golden_path, db_path, qwen_build_manifest]
    for path in paths:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    for path in (deterministic_chroma, qwen_chroma, rollback_chroma):
        if not path.resolve().is_dir():
            raise NotADirectoryError(path)
    qwen_build = json.loads(qwen_build_manifest.read_text(encoding="utf-8"))
    current_db_hash = _file_sha256(db_path.resolve(strict=True))
    current_qwen_hash = _directory_sha256(qwen_chroma.resolve(strict=True))
    expected_db_hash = qwen_build.get("source_duckdb_sha256")
    expected_qwen_hash = qwen_build.get("target_chroma_sha256")
    if not expected_db_hash or not expected_qwen_hash:
        raise ValueError("Qwen build manifest must contain source_duckdb_sha256 and target_chroma_sha256")
    if current_db_hash != expected_db_hash or current_qwen_hash != expected_qwen_hash:
        raise ValueError(
            "Qwen artifact no longer matches its build manifest: "
            f"db expected={expected_db_hash} actual={current_db_hash}; "
            f"chroma expected={expected_qwen_hash} actual={current_qwen_hash}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen dataset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_db = db_path.resolve(strict=True)
    artifact_hashes_before = {
        "duckdb_sha256": _file_sha256(resolved_db),
        "deterministic_chroma_sha256": _directory_sha256(deterministic_chroma.resolve(strict=True)),
        "deterministic_combined_sha256": hash_storage_artifacts(resolved_db, deterministic_chroma),
        "qwen_chroma_sha256": _directory_sha256(qwen_chroma.resolve(strict=True)),
        "qwen_combined_sha256": hash_storage_artifacts(resolved_db, qwen_chroma),
        "rollback_chroma_sha256": _directory_sha256(rollback_chroma.resolve(strict=True)),
        "rollback_combined_sha256": hash_storage_artifacts(resolved_db, rollback_chroma),
    }
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    if not isinstance(golden, list) or not golden:
        raise ValueError("golden dataset must be a non-empty JSON array")
    split_manifest = build_split_manifest(golden)

    query_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    for index, case in enumerate(golden, 1):
        query_id = f"eedi-l1-{index:04d}"
        source_sessions = sorted({int(item["session_id"]) for item in case.get("verbatim_grounding_quotes", [])})
        query_rows.append({
            "query_id": query_id,
            "query": case["question"],
            "intent": case.get("category", "UNKNOWN"),
            "split": split_manifest["query_assignments"][query_id],
            "source_session_ids": source_sessions,
            "slices": {
                "category": case.get("category", "unknown"),
                "language": "zh-query/en-evidence",
                "contains_numeric_or_formula": bool(__import__("re").search(r"\d|[+\-*/×÷=<>]", case["question"])),
            },
        })
        qrel_rows.extend(build_document_qrels(case, query_id=query_id))

    queries_path = output_dir / "queries.jsonl"
    qrels_path = output_dir / "qrels.jsonl"
    split_path = output_dir / "split_manifest.json"
    _write_jsonl(queries_path, query_rows)
    _write_jsonl(qrels_path, qrel_rows)
    _write_json(split_path, split_manifest)

    manifest = {
        "schema_version": "l1-closeout-inputs/v1",
        "dataset_version": "eedi-golden30-l1-closeout-v1",
        "qrels_version": "golden-verbatim-v2-derived-document-v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "case_count": len(query_rows),
        "qrel_count": len(qrel_rows),
        "split": {
            "file": split_path.name,
            "sha256": _file_sha256(split_path),
            "strategy": split_manifest["strategy"],
            "split_counts": split_manifest["split_counts"],
            "session_split_counts": split_manifest["session_split_counts"],
            "session_leakage_count": split_manifest["session_leakage_count"],
        },
        "files": {
            queries_path.name: _file_sha256(queries_path),
            qrels_path.name: _file_sha256(qrels_path),
            golden_path.name: _file_sha256(golden_path.resolve(strict=True)),
            qwen_build_manifest.name: _file_sha256(qwen_build_manifest.resolve(strict=True)),
        },
        "artifact_hashes": artifact_hashes_before,
        "qwen_index_contract": qwen_build.get("provider"),
        "qwen_collection_counts": qwen_build.get("collection_counts"),
        "rollback": {
            "directory": str(rollback_chroma.resolve(strict=True)),
            "chroma_sha256": artifact_hashes_before["rollback_chroma_sha256"],
        },
        "experiment_config": experiment_config,
        "system_config_hash": canonical_system_config_hash(experiment_config),
        "qrels_provenance": {
            "document_labels": "derived_from_human_quote",
            "multi_relevance_annotated": False,
            "business_quality_claim_allowed": False,
            "limitation": "single primary document labels derived from human quote/session evidence; no independent multi-document graded judgments",
        },
    }

    artifact_hashes_after = {
        "duckdb_sha256": _file_sha256(resolved_db),
        "deterministic_chroma_sha256": _directory_sha256(deterministic_chroma.resolve(strict=True)),
        "deterministic_combined_sha256": hash_storage_artifacts(resolved_db, deterministic_chroma),
        "qwen_chroma_sha256": _directory_sha256(qwen_chroma.resolve(strict=True)),
        "qwen_combined_sha256": hash_storage_artifacts(resolved_db, qwen_chroma),
        "rollback_chroma_sha256": _directory_sha256(rollback_chroma.resolve(strict=True)),
        "rollback_combined_sha256": hash_storage_artifacts(resolved_db, rollback_chroma),
    }
    if artifact_hashes_after != artifact_hashes_before:
        raise RuntimeError("source artifact changed while freezing closeout inputs")
    manifest["artifact_hashes_after"] = artifact_hashes_after
    manifest["artifact_mutated_during_freeze"] = False
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=Path("data/golden_test_set.json"))
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--deterministic-chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--qwen-chroma", type=Path, required=True)
    parser.add_argument("--qwen-build-manifest", type=Path, required=True)
    parser.add_argument("--rollback-chroma", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--assembler-top-k-each", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=1500)
    args = parser.parse_args(argv)
    positive = {
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_timeout_seconds": args.embedding_timeout_seconds,
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "assembler_top_k_each": args.assembler_top_k_each,
        "max_prompt_tokens": args.max_prompt_tokens,
    }
    if any(value <= 0 for value in positive.values()):
        parser.error(f"experiment values must be positive: {positive}")
    config = {
        **positive,
        "embedding_provider": "openai_compatible",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dimension": 1024,
        "embedding_index_version": "qwen3-embedding-0.6b-v1",
        "embedding_cache_key_fields": ["provider", "model", "index_version", "query_text"],
        "cache_success_only": True,
        "query_rewrite_mode": "deterministic_rules",
        "fusion_strategy": "raw_first",
        "fallback_exact_channel": "dialogue_weighted_numeric_formula_v1",
        "mmr_lambda": 0.7,
    }
    manifest = freeze_closeout_inputs(
        output_dir=args.output,
        golden_path=args.golden,
        db_path=args.db,
        deterministic_chroma=args.deterministic_chroma,
        qwen_chroma=args.qwen_chroma,
        qwen_build_manifest=args.qwen_build_manifest,
        rollback_chroma=args.rollback_chroma,
        experiment_config=config,
    )
    print(json.dumps({"manifest": str(manifest), "status": "FROZEN"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
