from __future__ import annotations

import json

import pytest

from src.business_graph_trace import (
    BusinessGraphRuntimeTrace,
    StageStatus,
    run_business_case,
)


class RecordingRuntime:
    def __init__(self, trace: BusinessGraphRuntimeTrace) -> None:
        self.trace = trace
        self.queries: list[str] = []

    def ask_case(self, query: str) -> BusinessGraphRuntimeTrace:
        self.queries.append(query)
        return self.trace


def _case(status: str = "SUCCESS") -> dict[str, object]:
    return {
        "case_id": "trace-case-001",
        "query": "Why did the student round 5.4598 to 5.45?",
        "expected_status": status,
        "ideal_answer": {"required_claims": ["secret gold claim"]},
    }


def _trace(status: StageStatus = StageStatus.SUCCESS) -> BusinessGraphRuntimeTrace:
    return BusinessGraphRuntimeTrace(
        actual_router={"intent": "QUESTION_TUTORING"},
        actual_plan={"plan_type": "SOCRATIC_TUTORING"},
        capabilities={"DIALOGUE_EVIDENCE": {"available": True}},
        analytics_result=None,
        graph_paths=[],
        card_candidates=[{"card_id": "session_10_misconception"}],
        parent_cards=[{"card_id": "session_10_misconception"}],
        candidate_windows=[
            {
                "window_id": "session_10_win_4_10",
                "session_id": 10,
                "source_turn_ids": [4, 5, 6, 7, 8, 9, 10],
                "anchor_turn_ids": [10],
                "rerank_rank": None,
            }
        ],
        ranked_windows=[],
        selected_windows=[],
        assembled_prompt="authorized prompt",
        citations=[],
        answer=None,
        stage_status={"ROUTER": status},
        errors=[],
        timings={"router": 0.001},
    )


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("SUCCESS", StageStatus.SUCCESS),
        ("NO_EVIDENCE", StageStatus.NO_EVIDENCE),
        ("UNSUPPORTED", StageStatus.UNSUPPORTED),
        ("FAILED", StageStatus.FAILED),
    ],
)
def test_run_business_case_passes_only_query_to_runtime_and_preserves_explicit_status(
    expected: str,
    actual: StageStatus,
) -> None:
    runtime = RecordingRuntime(_trace(actual))

    trace = run_business_case(_case(expected), runtime)

    assert runtime.queries == ["Why did the student round 5.4598 to 5.45?"]
    assert trace.stage_status["ROUTER"] is actual
    assert trace.candidate_windows[0]["source_turn_ids"] == [4, 5, 6, 7, 8, 9, 10]
    assert trace.candidate_windows[0]["anchor_turn_ids"] == [10]


def test_trace_serialization_does_not_include_gold_expected_answer() -> None:
    runtime = RecordingRuntime(_trace())

    payload = run_business_case(_case(), runtime).to_json()

    assert "secret gold claim" not in payload
    assert json.loads(payload)["candidate_windows"][0]["window_id"] == "session_10_win_4_10"


def test_turns_cannot_be_declared_as_ranking_candidates() -> None:
    with pytest.raises(ValueError, match="Turn records cannot be ranking candidates"):
        BusinessGraphRuntimeTrace(
            actual_router={}, actual_plan={}, capabilities={}, analytics_result=None,
            graph_paths=[], card_candidates=[], parent_cards=[],
            candidate_windows=[{"turn_id": 10}], ranked_windows=[], selected_windows=[],
            assembled_prompt="", citations=[], answer=None, stage_status={}, errors=[], timings={},
        )
