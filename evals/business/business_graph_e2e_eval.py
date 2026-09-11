"""Deterministic validation and stage-gate reporting for Business Graph Gold v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business_graph_gold import load_gold_cases


HARD_GATES = (
    "router_exact_match", "plan_exact_match", "scope_confinement",
    "card_evidence_validity", "window_coverage", "prompt_window_coverage",
    "citation_fidelity", "status_correctness",
)


def _check_case_shape(case: dict[str, Any]) -> dict[str, Any]:
    status = case["expected_status"]
    windows = case["expected_window_evidence"]
    citations = case["expected_citation_contract"].get("required_evidence_turns", [])
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "router_exact_match": bool(all(case["expected_router"].get(k) is not None for k in ("intent", "scope", "computation_scope", "perspectives"))),
        "plan_exact_match": bool(case["expected_plan"].get("plan_type") and case["expected_plan"].get("selected_lever", {}).get("id")),
        "scope_confinement": bool(isinstance(case["expected_sql_scope"].get("filters"), dict) and isinstance(case["expected_sql_scope"].get("keywords"), list)),
        "card_evidence_validity": status != "SUCCESS" or not case["expected_card_evidence"] or bool(case["expected_graph_paths"]),
        "window_coverage": status != "SUCCESS" or all(w.get("window_policy") == "W7_S3" and w.get("source_turn_ids") and set(w.get("anchor_turn_ids", [])) <= set(w["source_turn_ids"]) for w in windows),
        "prompt_window_coverage": set(case["expected_prompt"].get("required_window_ids", [])) <= {w["window_id"] for w in windows},
        "citation_fidelity": status != "SUCCESS" or all(e.get("session_id") and e.get("turn_id") and e.get("speaker") and e.get("exact_quote") for e in citations),
        "status_correctness": status in {"SUCCESS", "NO_EVIDENCE", "UNSUPPORTED", "FAILED"},
    }


def evaluate_gold(path: Path) -> dict[str, Any]:
    cases = load_gold_cases(path)
    rows = [_check_case_shape(case) for case in cases]
    metrics = {gate: sum(bool(row[gate]) for row in rows) / len(rows) for gate in HARD_GATES}
    return {"schema_version": "business-graph-gold-eval/v1", "case_count": len(rows), "metrics": metrics, "hard_gates_passed": all(value == 1.0 for value in metrics.values()), "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("data/business/business-graph-gold-v1.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_gold(args.gold.resolve(strict=True))
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["hard_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
