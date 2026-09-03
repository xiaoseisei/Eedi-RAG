from __future__ import annotations

import pytest

from evals.graded_qrels import (
    build_window_graded_qrels,
    graded_precision_at_k,
    graded_recall_at_k,
    turn_coverage_at_k,
)


def test_window_qrels_are_multi_relevant_graded_and_explicitly_provisional() -> None:
    case = {
        "case_id": "case-1",
        "subject_path": "Number > Rounding",
        "verbatim_grounding_quotes": [
            {"session_id": 10, "turn_id": 10},
            {"session_id": 10, "turn_id": 12},
        ],
    }
    windows = [
        {"chunk_id": "w-all", "session_id": 10, "subject_path": "Number > Rounding", "source_turn_ids": [10, 11, 12]},
        {"chunk_id": "w-one", "session_id": 10, "subject_path": "Number > Rounding", "source_turn_ids": [12, 13]},
        {"chunk_id": "w-related", "session_id": 20, "subject_path": "Number > Rounding", "source_turn_ids": [1, 2]},
        {"chunk_id": "w-noise", "session_id": 30, "subject_path": "Algebra", "source_turn_ids": [1]},
    ]

    rows = build_window_graded_qrels(case, windows)
    qrels = {row["document_id"]: row["relevance"] for row in rows}

    assert qrels == {"w-all": 3, "w-one": 2, "w-related": 1, "w-noise": 0}
    assert len([value for value in qrels.values() if value >= 2]) == 2
    assert all(row["needs_human_review"] is True for row in rows)
    assert all(row["label_source"].startswith("provisional_rule") for row in rows)


def test_graded_metrics_and_turn_coverage_use_all_relevant_documents() -> None:
    qrels = {"w-all": 3, "w-one": 2, "w-related": 1, "w-noise": 0}
    ranked = ["w-one", "w-noise", "w-all", "w-related"]
    assert graded_recall_at_k(ranked, qrels, 3, minimum_relevance=2) == 1.0
    assert graded_precision_at_k(ranked, qrels, 4, minimum_relevance=1) == 0.75
    windows = [
        {"session_id": 10, "source_turn_ids": [12]},
        {"session_id": 30, "source_turn_ids": [1]},
        {"session_id": 10, "source_turn_ids": [10]},
    ]
    assert turn_coverage_at_k(windows, {(10, 10), (10, 12)}, 3) == 1.0


@pytest.mark.parametrize("bad_case", [{}, {"case_id": "x", "verbatim_grounding_quotes": []}])
def test_graded_qrels_reject_missing_required_turns(bad_case) -> None:
    with pytest.raises(ValueError):
        build_window_graded_qrels(bad_case, [{"chunk_id": "w", "session_id": 1, "source_turn_ids": [1]}])
