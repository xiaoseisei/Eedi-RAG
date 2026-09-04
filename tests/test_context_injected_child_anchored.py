from __future__ import annotations

from scripts.eval_context_injected_child_anchored import _coverage_aware_select


def _child(chunk_id: str, session_id: int, rank: int, score: float, turns: list[int]) -> dict:
    return {
        "chunk_id": chunk_id,
        "reranker_rank": rank,
        "reranker_score": score,
        "metadata": {"session_id": session_id, "source_turn_ids": turns},
    }


def test_coverage_aware_selection_respects_per_session_quota() -> None:
    candidates = [
        _child("s1-a", 1, 1, 0.99, [1, 2, 3]),
        _child("s1-b", 1, 2, 0.98, [3, 4, 5]),
        _child("s2-a", 2, 3, 0.80, [10, 11, 12]),
    ]

    selected = _coverage_aware_select(
        candidates,
        selection_count=3,
        per_session_quota=1,
        anchor_keys={(1, 1), (1, 4), (2, 10)},
    )

    assert len(selected) == 2
    assert {item["metadata"]["session_id"] for item in selected} == {1, 2}


def test_coverage_aware_selection_is_deterministic_and_uses_anchor_gain() -> None:
    candidates = [
        _child("low-rank-new-anchor", 1, 2, 0.90, [9, 10]),
        _child("high-rank-overlap", 1, 1, 0.90, [1, 2]),
    ]
    kwargs = {
        "selection_count": 1,
        "per_session_quota": 2,
        "anchor_keys": {(1, 9)},
    }

    first = _coverage_aware_select(candidates, **kwargs)
    second = _coverage_aware_select(candidates, **kwargs)

    assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
    assert first[0]["chunk_id"] == "low-rank-new-anchor"
