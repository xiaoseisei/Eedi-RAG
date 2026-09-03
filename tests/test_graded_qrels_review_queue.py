from __future__ import annotations

import json
from pathlib import Path


def test_review_queue_includes_top_ranked_and_direct_evidence(tmp_path: Path) -> None:
    from scripts.build_graded_qrels_review_queue import build_review_queue

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    raw = {
        "case_id": "case-1", "query_id": "q-1", "query": "question", "required_turns": [[1, 1]],
        "ranked_ids": ["w0", "w2"], "card_ranked_ids": ["c0"],
    }
    (dataset / "raw_cases.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    window_rows = [
        {"query_id": "q-1", "document_id": "w0", "relevance": 0, "rationale": "noise", "evidence_turns": [], "label_source": "provisional_rule"},
        {"query_id": "q-1", "document_id": "w2", "relevance": 2, "rationale": "direct", "evidence_turns": [{"session_id": 1, "turn_id": 1}], "label_source": "provisional_rule"},
    ]
    card_rows = [{"query_id": "q-1", "document_id": "c0", "relevance": 3, "rationale": "direct", "evidence_turns": [{"session_id": 1, "turn_id": 1}], "label_source": "provisional_rule"}]
    for name, rows in (("graded_qrels.jsonl", window_rows), ("card_graded_qrels.jsonl", card_rows)):
        (dataset / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (dataset / "manifest.json").write_text("{}", encoding="utf-8")

    summary = build_review_queue(dataset_dir=dataset, output_dir=tmp_path / "review", top_k=1)
    queue = [json.loads(line) for line in (tmp_path / "review/review_queue.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["row_count"] == 3
    assert {item["document_id"] for item in queue} == {"w0", "w2", "c0"}
    assert all(item["review_status"] == "PENDING" for item in queue)
