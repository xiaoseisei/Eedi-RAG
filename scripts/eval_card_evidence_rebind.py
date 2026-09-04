from __future__ import annotations

"""Compare card-hit Turn evidence coverage before/after deterministic rebinding."""

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_turn_graded_eval import _card_slot_with_required_evidence


def _required(case: dict[str, Any], slot: str) -> set[tuple[int, int]]:
    expected = "student" if slot == "misconception" else "tutor"
    return {
        (int(item["session_id"]), int(item["turn_id"]))
        for item in case.get("verbatim_grounding_quotes", [])
        if str(item.get("speaker")) == expected
    }


def _ids(candidate: dict[str, Any]) -> list[int]:
    raw = candidate.get("metadata", {}).get("source_turn_ids", [])
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [int(value) for value in (raw or [])]


def _coverage(candidates: list[dict[str, Any]], required: set[tuple[int, int]], k: int) -> float:
    covered = {
        (int(candidate.get("metadata", {}).get("session_id")), turn_id)
        for candidate in candidates[:k]
        for turn_id in _ids(candidate)
    }
    return len(covered & required) / len(required) if required else 0.0


def _precision(candidates: list[dict[str, Any]], required: set[tuple[int, int]], k: int) -> float:
    pointers = {
        (int(candidate.get("metadata", {}).get("session_id")), turn_id)
        for candidate in candidates[:k]
        for turn_id in _ids(candidate)
    }
    return len(pointers & required) / len(pointers) if pointers else 0.0


def _evaluate_snapshot(
    *,
    db_path: Path,
    chroma_path: Path,
    golden: list[dict[str, Any]],
    top_k: int,
    bm25_weight: float,
) -> dict[str, Any]:
    from src.reranker import PedagogicalGoldAssembler
    from src.retriever import DualMetricRetriever
    from src.storage_manager import DualEngineStorageManager

    with tempfile.TemporaryDirectory(
        prefix="card-evidence-eval-",
        dir=str(chroma_path.parent),
        ignore_cleanup_errors=True,
    ) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_backend="deterministic",
        )
        retriever = DualMetricRetriever(
            storage_manager=storage,
            query_rewrite_mode="deterministic",
            fusion_strategy="raw_first",
            retrieval_mode="bm25_dense",
            bm25_weight=bm25_weight,
        )
        assembler = PedagogicalGoldAssembler(chunk_strategy="card", max_prompt_tokens=1500)
        rows: list[dict[str, Any]] = []
        try:
            for index, case in enumerate(golden, start=1):
                slot, slot_fallback = _card_slot_with_required_evidence(case)
                required = _required(case, slot)
                if slot == "misconception":
                    candidates = retriever.retrieve_misconceptions(case["question"], top_k=top_k, fetch_evidence=True)
                else:
                    candidates = retriever.retrieve_strategies(case["question"], top_k=top_k, fetch_evidence=True)

                retrieval_results = {
                    "chunk_strategy": "card",
                    "misconceptions": candidates if slot == "misconception" else [],
                    "strategies": candidates if slot == "strategy" else [],
                    "windows": [],
                    "rewritten_queries": {"extracted_keywords": []},
                }
                context = assembler.assemble(case["question"], retrieval_results)
                context_keys = {
                    (int(turn["session_id"]), int(turn["turn_id"]))
                    for turn in context.evidence_turns
                }
                rows.append({
                    "case_id": f"card-evidence-{index:04d}",
                    "session_id": int(case["verbatim_grounding_quotes"][0]["session_id"]),
                    "category": case.get("category", "UNKNOWN"),
                    "slot": slot,
                    "slot_fallback": slot_fallback,
                    "required_turn_count": len(required),
                    "card_pointer_count": len({
                        (int(item.get("metadata", {}).get("session_id")), turn_id)
                        for item in candidates
                        for turn_id in _ids(item)
                    }),
                    "assembler_context_turn_count": len(context_keys),
                    "assembler_context_evidence_recall": len(context_keys & required) / len(required) if required else 0.0,
                    "assembler_context_evidence_precision": len(context_keys & required) / len(context_keys) if context_keys else 0.0,
                    **{
                        f"card_evidence_recall_at_{k}": _coverage(candidates, required, k)
                        for k in (1, 3, 5, 10, 20)
                    },
                    **{
                        f"card_evidence_precision_at_{k}": _precision(candidates, required, k)
                        for k in (1, 3, 5, 10, 20)
                    },
                })
        finally:
            storage.close()
            try:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()
    metrics = [
        "card_evidence_recall_at_1",
        "card_evidence_recall_at_3",
        "card_evidence_recall_at_5",
        "card_evidence_recall_at_10",
        "card_evidence_recall_at_20",
        "card_evidence_precision_at_1",
        "card_evidence_precision_at_3",
        "card_evidence_precision_at_5",
        "card_evidence_precision_at_10",
        "card_evidence_precision_at_20",
        "assembler_context_evidence_recall",
        "assembler_context_evidence_precision",
        "card_pointer_count",
        "assembler_context_turn_count",
    ]
    return {
        "case_count": len(rows),
        "aggregates": {metric: {"mean": mean(float(row[metric]) for row in rows), "measured_cases": len(rows)} for metric in metrics},
        "rows": rows,
    }


def evaluate(
    *,
    before_db: Path,
    before_chroma: Path,
    after_db: Path,
    after_chroma: Path,
    golden_path: Path,
    output_dir: Path,
    top_k: int = 20,
    bm25_weight: float = 0.35,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    golden = json.loads(golden_path.resolve(strict=True).read_text(encoding="utf-8"))
    before = _evaluate_snapshot(
        db_path=before_db.resolve(strict=True),
        chroma_path=before_chroma.resolve(strict=True),
        golden=golden,
        top_k=top_k,
        bm25_weight=bm25_weight,
    )
    after = _evaluate_snapshot(
        db_path=after_db.resolve(strict=True),
        chroma_path=after_chroma.resolve(strict=True),
        golden=golden,
        top_k=top_k,
        bm25_weight=bm25_weight,
    )
    deltas = {}
    for metric, value in before["aggregates"].items():
        old = value["mean"]
        new = after["aggregates"][metric]["mean"]
        deltas[metric] = {"before": old, "after": new, "delta": new - old}
    payload = {
        "schema_version": "card-evidence-rebind-eval/v1",
        "status": "MEASURED",
        "top_k": top_k,
        "bm25_weight": bm25_weight,
        "case_count": len(golden),
        "before": {"aggregates": before["aggregates"]},
        "after": {"aggregates": after["aggregates"]},
        "deltas": deltas,
        "limitations": [
            "required evidence comes from provisional human-quote-derived labels",
            "pointer precision is conservative because rebinding intentionally favors complete non-noise role evidence",
            "this evaluates card-hit evidence, not LLM answer quality",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "before.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in before["rows"]), encoding="utf-8")
    (output_dir / "after.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in after["rows"]), encoding="utf-8")
    lines = [
        "# Card-hit Turn Evidence Rebind Evaluation",
        "",
        f"Cases: `{len(golden)}`; Top-K: `{top_k}`; BM25 weight: `{bm25_weight}`",
        "",
        "| Metric | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, delta in deltas.items():
        lines.append(f"| {metric} | {delta['before']:.4f} | {delta['after']:.4f} | {delta['delta']:+.4f} |")
    lines.extend([
        "",
        "Required evidence labels are provisional; inspect `before.jsonl` and `after.jsonl` for case-level coverage.",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-db", type=Path, required=True)
    parser.add_argument("--before-chroma", type=Path, required=True)
    parser.add_argument("--after-db", type=Path, required=True)
    parser.add_argument("--after-chroma", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data/golden_test_set.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    args = parser.parse_args(argv)
    payload = evaluate(
        before_db=args.before_db,
        before_chroma=args.before_chroma,
        after_db=args.after_db,
        after_chroma=args.after_chroma,
        golden_path=args.golden,
        output_dir=args.output_dir,
        top_k=args.top_k,
        bm25_weight=args.bm25_weight,
    )
    print(json.dumps({"status": payload["status"], "case_count": payload["case_count"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
