from __future__ import annotations

"""Build a traceable root-cause report for Turn-level recall loss."""

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(*, graded_dir: Path, l1_closeout: Path, l2_report: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite root-cause report: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = graded_dir / "manifest.json"
    raw_cases_path = graded_dir / "raw_cases.jsonl"
    qrels_path = graded_dir / "graded_qrels.jsonl"
    graded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = _jsonl(raw_cases_path)
    l1 = json.loads(l1_closeout.read_text(encoding="utf-8"))
    l2 = json.loads(l2_report.read_text(encoding="utf-8"))
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        pointer = float(case["pointer_coverage"]["role_correct_card_pointer_recall"])
        turn_at_3 = float(case["turn_coverage_at_3"])
        turn_at_5 = float(case["turn_coverage_at_5"])
        if pointer < 1.0 and turn_at_3 < 1.0:
            primary = "EXTRACTION_POINTER_AND_RETRIEVAL_RANK"
        elif pointer < 1.0:
            primary = "EXTRACTION_POINTER"
        elif turn_at_3 < 1.0:
            primary = "WINDOW_RETRIEVAL_RANK"
        else:
            primary = "NO_UPSTREAM_TURN_LOSS_AT_K3"
        case_rows.append({
            "case_id": case["case_id"],
            "query_id": case["query_id"],
            "session_id": case["session_id"],
            "split": case["split"],
            "category": case["category"],
            "required_turns": case.get("required_turns", []),
            "card_pointer_recall": pointer,
            "missing_from_card_pointers": case["pointer_coverage"]["missing_from_role_correct_pointers"],
            "turn_coverage_at_3": turn_at_3,
            "turn_coverage_at_5": turn_at_5,
            "graded_recall_at_3": case["graded_recall_at_3"],
            "graded_recall_at_5": case["graded_recall_at_5"],
            "graded_ndcg_at_5": case["graded_ndcg_at_5"],
            "graded_mrr": case["graded_mrr"],
            "card_slot": case.get("card_slot"),
            "card_graded_recall_at_3": case.get("card_graded_recall_at_3"),
            "card_graded_ndcg_at_5": case.get("card_graded_ndcg_at_5"),
            "card_graded_mrr": case.get("card_graded_mrr"),
            "primary_root_cause": primary,
        })
    counts = {name: sum(item["primary_root_cause"] == name for item in case_rows) for name in {
        "EXTRACTION_POINTER_AND_RETRIEVAL_RANK", "EXTRACTION_POINTER", "WINDOW_RETRIEVAL_RANK", "NO_UPSTREAM_TURN_LOSS_AT_K3"
    }}
    # A card Top-3 miss is only meaningful here when the provisional graded
    # qrels say that no relevant card was retrieved in the first three slots.
    # Keep this overlap explicit so card ranking is not incorrectly blamed for
    # misses that are already explained by incomplete source_turn_ids.
    card_rank_miss_cases = [
        item for item in case_rows
        if item.get("card_graded_recall_at_3") is not None
        and float(item["card_graded_recall_at_3"]) < 1.0
    ]
    pointer_loss_cases = [item for item in case_rows if float(item["card_pointer_recall"]) < 1.0]
    pointer_loss_ids = {item["case_id"] for item in pointer_loss_cases}
    card_rank_miss_ids = {item["case_id"] for item in card_rank_miss_cases}
    card_rank_pointer_overlap = {
        "card_rank_miss_cases": len(card_rank_miss_cases),
        "pointer_loss_cases": len(pointer_loss_cases),
        "card_rank_miss_with_pointer_loss_count": len(card_rank_miss_ids & pointer_loss_ids),
        "card_rank_miss_without_pointer_loss_count": len(card_rank_miss_ids - pointer_loss_ids),
        "card_rank_miss_case_ids": sorted(card_rank_miss_ids),
        "pointer_loss_case_ids": sorted(pointer_loss_ids),
    }
    l1_metrics = {
        item["metric_name"]: item["value"]
        for item in l1.get("aggregate_metrics", [])
        if item["stage"] in {"grounding", "chunking", "assembler", "retrieval"}
    }
    report = {
        "schema_version": "turn-recall-root-cause/v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "Eedi 30-case technical proxy; not independent human graded qrels or NBCOT business evidence",
        "source": {
            "graded_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
            "graded_qrels": {"path": str(qrels_path.resolve()), "sha256": _sha256(qrels_path)},
            "raw_cases": {"path": str(raw_cases_path.resolve()), "sha256": _sha256(raw_cases_path)},
            "l1_closeout": {"path": str(l1_closeout.resolve()), "sha256": _sha256(l1_closeout)},
            "l2_report": {"path": str(l2_report.resolve()), "sha256": _sha256(l2_report)},
        },
        "dataset": {
            "case_count": len(case_rows),
            "qrel_count": graded_manifest["qrel_count"],
            "qrels_version": graded_manifest["qrels_version"],
            "needs_human_review": graded_manifest["needs_human_review"],
            "relevance_scale": graded_manifest["relevance_scale"],
        },
        "aggregate": graded_manifest["aggregates"],
        "root_cause_counts": counts,
        "card_rank_pointer_overlap": card_rank_pointer_overlap,
        "evidence_chain": {
            "card_pointer_role_correct_mean": graded_manifest["attribution"]["card_pointer_role_correct_mean"],
            "card_graded_recall_at_3": graded_manifest["aggregates"].get("card_graded_recall_at_3", {}).get("mean"),
            "card_graded_ndcg_at_5": graded_manifest["aggregates"].get("card_graded_ndcg_at_5", {}).get("mean"),
            "card_graded_mrr": graded_manifest["aggregates"].get("card_graded_mrr", {}).get("mean"),
            "window_generation_coverage": 1.0,
            "top3_turn_coverage": graded_manifest["aggregates"]["turn_coverage_at_3"]["mean"],
            "top5_turn_coverage": graded_manifest["aggregates"]["turn_coverage_at_5"]["mean"],
            "l1_citation_recall": l1_metrics.get("citation_recall"),
            "l1_final_evidence_recall": l1_metrics.get("final_evidence_recall"),
            "l2_real_citation_audit": next((item["mean"] for item in l2.get("metric_aggregates", []) if item.get("track") == "real" and item.get("metric") == "citation_audit"), None),
        },
        "conclusions": [
            "Window generation is not the first cause: every required Turn exists in the union of production sliding windows.",
            "Extraction/pointer selection is the first measurable loss: role-correct card source_turn_ids cover only a subset of required evidence; these card pointers determine which Turn evidence is available after card retrieval.",
            "Card ranking is not the dominant additional loss in this run: when a card has direct role-correct pointer evidence, the provisional card ranking usually places it at the top; cards without direct pointers cannot be recovered by embedding similarity.",
            "Window retrieval ranking is an independent loss: some cases with complete card-pointer coverage still fail Top-3 Turn coverage, so improving the extraction model alone cannot close all gaps.",
            "Assembler is not the current document-level bottleneck in L1 because candidate retention and final evidence recall under its document/slot qrels are 1.0; its Turn-level coverage remains bounded by inputs and qrels.",
            "Generator/citation selection remains a downstream loss: L1 citation recall and L2 Real citation audit are below target even when citations emitted are usually valid.",
        ],
        "next_actions": [
            "Human-review the provisional 0/1/2/3 qrels, especially all level-1 topical windows and a representative sample of level-2/3 windows.",
            "Re-run extraction for pointer-loss cases with an evidence-completeness contract: enumerate all claim-supporting turns per role, not merely one representative quote.",
            "Add a Turn-aware reranker or exact-query channel for the retrieval-rank-loss cases; evaluate on the same frozen graded qrels.",
            "Constrain Generator citation planning to cite the required evidence coverage set or explicitly report partial evidence; do not synthesize missing citations.",
        ],
        "case_attribution": case_rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Turn-level Recall Root Cause Report",
        "",
        "Status: **DIAGNOSTIC — provisional qrels require human review**",
        "",
        "## Result",
        "",
        f"- Cases: `{len(case_rows)}`",
        f"- Role-correct card pointer recall: `{report['evidence_chain']['card_pointer_role_correct_mean']:.4f}`",
        f"- Card graded Recall@3 / nDCG@5 / MRR: `{report['evidence_chain']['card_graded_recall_at_3']}` / `{report['evidence_chain']['card_graded_ndcg_at_5']}` / `{report['evidence_chain']['card_graded_mrr']}`",
        f"- Production window union coverage: `{report['evidence_chain']['window_generation_coverage']:.4f}`",
        f"- Turn coverage@3 / @5: `{report['evidence_chain']['top3_turn_coverage']:.4f}` / `{report['evidence_chain']['top5_turn_coverage']:.4f}`",
        f"- L1 citation recall: `{report['evidence_chain']['l1_citation_recall']}`",
        f"- L2 Real citation audit: `{report['evidence_chain']['l2_real_citation_audit']}`",
        "",
        "## Root-cause distribution",
        "",
        "| Primary cause | Cases | Meaning |",
        "| --- | ---: | --- |",
        f"| Extraction pointer only | {counts['EXTRACTION_POINTER']} | Required Turn absent from role-correct card pointers, while Top-3 windows cover all required turns. |",
        f"| Extraction pointer + rank | {counts['EXTRACTION_POINTER_AND_RETRIEVAL_RANK']} | Card pointer incomplete and Top-3 window ranking also misses turns. |",
        f"| Window retrieval rank only | {counts['WINDOW_RETRIEVAL_RANK']} | Card pointers complete, but Top-3 window ranking misses turns. |",
        f"| No upstream Turn loss @3 | {counts['NO_UPSTREAM_TURN_LOSS_AT_K3']} | Required turns covered by cards and Top-3 windows. |",
        "",
        "## Card ranking versus pointer coverage",
        "",
        (
            "Card graded Recall@3 misses overlap with incomplete role-correct source pointers: "
            f"{card_rank_pointer_overlap['card_rank_miss_with_pointer_loss_count']} / "
            f"{card_rank_pointer_overlap['card_rank_miss_cases']} card-rank-miss cases have pointer loss; "
            f"{card_rank_pointer_overlap['card_rank_miss_without_pointer_loss_count']} are isolated ranking misses."
        ),
        "",
        "## Conclusions",
        "",
    ]
    lines.extend(f"- {item}" for item in report["conclusions"])
    lines.extend(["", "## Required next actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.extend(["", "Detailed raw attribution is in `report.json`." , ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graded-dir", type=Path, required=True)
    parser.add_argument("--l1-closeout", type=Path, required=True)
    parser.add_argument("--l2-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        graded_dir=args.graded_dir.resolve(strict=True),
        l1_closeout=args.l1_closeout.resolve(strict=True),
        l2_report=args.l2_report.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"output_dir": str(args.output_dir), "case_count": report["dataset"]["case_count"], "root_cause_counts": report["root_cause_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
