from __future__ import annotations

"""Sweep BM25+Dense candidate cutoffs and select a rerank pool size.

Each cutoff is evaluated against the same frozen graded qrels. The selector
prefers graded recall, then nDCG, then MRR, and finally precision; it does not
pretend that an unreviewed qrel is business truth.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.live_storage import hash_storage_artifacts
from scripts.run_turn_graded_eval import evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_sweep(
    *,
    db_path: Path,
    chroma_path: Path,
    golden_path: Path,
    l1_manifest_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    cutoffs: list[int],
    embedding_backend: str,
    retrieval_mode: str,
    bm25_weight: float,
    recall_tolerance: float = 0.02,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite sweep directory: {output_dir}")
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must contain positive values")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be unique")
    if retrieval_mode not in {"dense", "bm25_dense"}:
        raise ValueError("retrieval_mode must be dense or bm25_dense")
    if recall_tolerance < 0:
        raise ValueError("recall_tolerance must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_hash = hash_storage_artifacts(db_path, chroma_path)
    results: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        result_dir = output_dir / f"top{cutoff}"
        manifest = evaluate(
            db_path=db_path,
            chroma_path=chroma_path,
            golden_path=golden_path,
            l1_manifest_path=l1_manifest_path,
            output_dir=result_dir,
            top_k=cutoff,
            expected_artifact_hash=expected_hash,
            split_manifest_path=split_manifest_path,
            embedding_backend=embedding_backend,
            retrieval_mode=retrieval_mode,
            bm25_weight=bm25_weight,
        )
        effective_k = min(cutoff, 20)
        agg = manifest["aggregates"]
        results.append({
            "cutoff": cutoff,
            "effective_k": effective_k,
            "output_dir": str(result_dir.resolve()),
            "manifest_sha256": _sha256(result_dir / "manifest.json") if (result_dir / "manifest.json").is_file() else None,
            "aggregates": agg,
            "recall_at_cutoff": agg[f"graded_recall_at_{effective_k}"]["mean"],
            "turn_coverage_at_cutoff": agg[f"turn_coverage_at_{effective_k}"]["mean"],
            "card_recall_at_cutoff": agg[f"card_graded_recall_at_{effective_k}"]["mean"],
            "attribution": manifest["attribution"],
            "artifact_hash_before": manifest["artifact_hash_before"],
            "artifact_hash_after": manifest["artifact_hash_after"],
            "artifact_mutated": manifest["artifact_mutated"],
        })
    actual_after = hash_storage_artifacts(db_path, chroma_path)
    if actual_after != expected_hash:
        raise RuntimeError(f"source artifact changed during BM25 sweep: before={expected_hash}, after={actual_after}")

    def key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        aggregates = item["aggregates"]
        return (
            float(item["recall_at_cutoff"]),
            float(aggregates["graded_ndcg_at_5"]["mean"]),
            float(aggregates["graded_mrr"]["mean"]),
            float(aggregates["graded_precision_at_5"]["mean"]),
            -float(item["cutoff"]),
        )

    max_recall = max(float(item["recall_at_cutoff"]) for item in results)
    eligible = [item for item in results if max_recall - float(item["recall_at_cutoff"]) <= recall_tolerance]
    selected = min(eligible, key=lambda item: item["cutoff"])
    summary = {
        "schema_version": "bm25-dense-sweep/v1",
        "status": "PROVISIONAL_MEASURED",
        "scope": "same 30-case frozen technical qrels; labels require human review",
        "embedding_backend": embedding_backend,
        "retrieval_mode": retrieval_mode,
        "bm25_weight": bm25_weight,
        "dense_weight": 1.0 - bm25_weight,
        "recall_tolerance": recall_tolerance,
        "cutoffs": cutoffs,
        "selected_rerank_pool_size": selected["cutoff"],
        "selection_rule": "choose the smallest cutoff within recall_tolerance of maximum recall_at_cutoff; tie-break by nDCG, MRR, Precision",
        "source_artifact_hash_before": expected_hash,
        "source_artifact_hash_after": actual_after,
        "artifact_mutated": expected_hash != actual_after,
        "results": results,
        "limitations": [
            "The qrels are provisional rule-derived labels, not independent human judgments.",
            "Top-K cutoff is selected for retrieval pool quality; it is not a claim that every returned document is relevant.",
            "A reranker must be evaluated separately on the selected pool; this sweep does not call the reranker API.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# BM25 + Dense cutoff sweep",
        "",
        "Status: **PROVISIONAL_MEASURED — HUMAN REVIEW REQUIRED**",
        "",
        f"Embedding: `{embedding_backend}`; retrieval mode: `{retrieval_mode}`; BM25 weight: `{bm25_weight}`",
        f"Selected rerank pool size: **Top-{selected['cutoff']}** (max Recall gap allowed: `{recall_tolerance}`)",
        "",
        "| Cutoff | Recall@cutoff | Graded nDCG@5 | Graded MRR | Precision@5 | Card Recall@cutoff | Turn coverage@cutoff |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        agg = item["aggregates"]
        def metric_mean(name: str) -> float:
            value = agg.get(name, {}).get("mean")
            return float(value) if value is not None else 0.0
        lines.append(
            f"| {item['cutoff']} | {item['recall_at_cutoff']:.4f} | {metric_mean('graded_ndcg_at_5'):.4f} | "
            f"{metric_mean('graded_mrr'):.4f} | {metric_mean('graded_precision_at_5'):.4f} | "
            f"{item['card_recall_at_cutoff']:.4f} | {item['turn_coverage_at_cutoff']:.4f} |"
        )
    lines.extend([
        "",
        "Selection is for the reranker candidate pool. It must not be interpreted as final business quality until qrels are human-reviewed.",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data" / "golden_test_set.json")
    parser.add_argument("--l1-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", type=str, default="5,10,15,20")
    parser.add_argument("--embedding-backend", choices=("qwen", "deterministic"), default="qwen")
    parser.add_argument("--retrieval-mode", choices=("dense", "bm25_dense"), default="bm25_dense")
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--recall-tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)
    cutoffs = [int(item.strip()) for item in args.cutoffs.split(",") if item.strip()]
    summary = run_sweep(
        db_path=args.db.resolve(strict=True),
        chroma_path=args.chroma.resolve(strict=True),
        golden_path=args.golden.resolve(strict=True),
        l1_manifest_path=args.l1_manifest.resolve(strict=True),
        split_manifest_path=args.split_manifest.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        cutoffs=cutoffs,
        embedding_backend=args.embedding_backend,
        retrieval_mode=args.retrieval_mode,
        bm25_weight=args.bm25_weight,
        recall_tolerance=args.recall_tolerance,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "selected_rerank_pool_size": summary["selected_rerank_pool_size"], "artifact_mutated": summary["artifact_mutated"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
