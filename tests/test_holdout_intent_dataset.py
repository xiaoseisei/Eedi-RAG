import json
from pathlib import Path
import pytest
from src.business_models import QueryIntent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HLDOUT_PATH = PROJECT_ROOT / "data" / "business" / "holdout_intent_eval_v1.jsonl"
GOLD_PATH = POJECT_ROOT = PROJECT_ROOT / "data" / "business" / "business-graph-gold-v1.jsonl"

def test_holdout_dataset_integrity():
    assert HLDOUT_PATH.exists(), f"Holdout dataset missing at {HLDOUT_PATH}"
    records = []
    with open(HLDOUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    assert len(records) >= 40, f"Expected at least 40 records, got {len(records)}"
    
    valid_categories = {
        "CLEAR_RULE",
        "CONFLICT_NEED_LLM",
        "PARAPHRASE_NEED_LLM",
        "OUT_OF_BOUNDS_UNSUPPORTED",
    }
    valid_routes = {"RESOLVED", "NEED_LLM", "UNSUPPORTED"}
    valid_intents = {i.value for i in QueryIntent}
    
    seen_ids = set()
    category_counts = {}
    
    for r in records:
        assert "id" in r and r["id"] not in seen_ids
        seen_ids.add(r["id"])
        assert r["category"] in valid_categories
        assert r["expected_tier1_route"] in valid_routes
        assert r["gold_intent"] in valid_intents
        assert len(r["query"]) >= 3
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat in valid_categories:
        assert category_counts.get(cat, 0) >= 8, f"{cat} has insufficient records: {category_counts.get(cat, 0)}"


def test_holdout_dataset_decoupled_from_gold():
    gold_queries = set()
    if GOLD_PATH.exists():
        with open(GOLD_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    gold_queries.add(item["query"].strip().lower())

    with open(HLDOUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                q = item["query"].strip().lower()
                assert q not in gold_queries, f"Holdout query leaks from Gold: {q}"