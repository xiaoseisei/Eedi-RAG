from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_business_graph_dataset import export_dataset, sha256
from scripts.evaluate_business_graph import load_exported_cases
from src.business_graph_gold import load_gold_cases


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/business/business-graph-gold-v1.jsonl"


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_export_is_lossless_across_four_evaluator_views(tmp_path: Path) -> None:
    output = tmp_path / "business-graph-v1"
    manifest = export_dataset(GOLD, output)
    source = load_gold_cases(GOLD)

    queries = _rows(output / "queries.jsonl")
    facts = _rows(output / "expected_facts.jsonl")
    paths = _rows(output / "expected_graph_paths.jsonl")
    evidence = _rows(output / "expected_evidence.jsonl")
    assert len(queries) == len(facts) == len(paths) == len(evidence) == len(source) == 83
    assert manifest["source_gold_sha256"] == sha256(GOLD)
    assert [row["case_id"] for row in queries] == [case["case_id"] for case in source]
    for index, case in enumerate(source):
        assert queries[index]["query"] == case["query"]
        assert facts[index]["expected_router"] == case["expected_router"]
        assert facts[index]["expected_plan"] == case["expected_plan"]
        assert facts[index]["expected_sql_scope"] == case["expected_sql_scope"]
        assert paths[index]["expected_graph_paths"] == case["expected_graph_paths"]
        assert evidence[index]["expected_card_evidence"] == case["expected_card_evidence"]
        assert evidence[index]["expected_window_evidence"] == case["expected_window_evidence"]
        assert evidence[index]["expected_citation_contract"] == case["expected_citation_contract"]


def test_export_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "business-graph-v1"
    export_dataset(GOLD, output)
    with pytest.raises(FileExistsError):
        export_dataset(GOLD, output)


def test_evaluator_loader_round_trips_the_exported_dataset(tmp_path: Path) -> None:
    output = tmp_path / "business-graph-v1"
    export_dataset(GOLD, output)

    cases = load_exported_cases(output)
    source = load_gold_cases(GOLD)
    assert len(cases) == len(source) == 83
    assert [case["case_id"] for case in cases] == [case["case_id"] for case in source]
    assert cases[0]["query"] == source[0]["query"]
    assert cases[0]["expected_graph_paths"] == source[0]["expected_graph_paths"]
