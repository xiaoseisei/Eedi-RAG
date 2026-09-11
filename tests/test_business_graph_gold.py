from pathlib import Path

from src.business_graph_gold import load_gold_cases
from evals.business.business_graph_e2e_eval import evaluate_gold


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/business/business-graph-gold-v1.jsonl"


def test_gold_loads_and_covers_all_requested_categories():
    cases = load_gold_cases(GOLD)
    categories = {case["category"] for case in cases}
    assert len(cases) == 83
    assert {"NO_EVIDENCE", "UNSUPPORTED", "FAILED"} <= categories
    assert all(case["expected_prompt"]["rerank_unit"] == "WINDOW" for case in cases)


def test_gold_structural_gates_pass():
    report = evaluate_gold(GOLD)
    assert report["case_count"] == 83
    assert report["hard_gates_passed"] is True
