from __future__ import annotations

from scripts.eval_anchored_logical_windows import (
    _assemble,
    _anchored_window_rank_text,
    filter_anchored_windows,
)


def _card(card_id: str, session_id: int, turns: list[int]) -> dict:
    return {
        "chunk_id": card_id,
        "metadata": {
            "session_id": session_id,
            "source_turn_ids": turns,
            "misconception_name": f"card {card_id}",
            "deep_mechanism": "compact parent explanation",
            "subject_path": "Number",
        },
    }


def _window(window_id: str, session_id: int, turns: list[int]) -> dict:
    return {
        "chunk_id": window_id,
        "document": "[Turn text from source]",
        "metadata": {
            "session_id": session_id,
            "source_turn_ids": turns,
            "parent_contexts": [],
        },
        "evidence_turns": [
            {"session_id": session_id, "turn_id": turn, "text": f"turn {turn}"}
            for turn in turns
        ],
    }


def test_filter_anchored_windows_only_keeps_source_pointer_overlap() -> None:
    parents = [_card("p1", 1, [3, 4])]
    windows = [
        _window("w-overlap", 1, [1, 2, 3, 4, 5, 6]),
        _window("w-no-overlap", 1, [7, 8, 9]),
        _window("w-other-session", 2, [3, 4]),
    ]

    anchored = filter_anchored_windows(windows, parents)

    assert [item["chunk_id"] for item in anchored] == ["w-overlap"]
    assert anchored[0]["metadata"]["anchor_overlap_turn_ids"] == [3, 4]
    assert anchored[0]["metadata"]["parent_contexts"][0]["card_id"] == "p1"


def test_filter_does_not_read_gold_or_invent_turns() -> None:
    parents = [_card("p1", 1, [3])]
    anchored = filter_anchored_windows([_window("w", 1, [2, 3, 4])], parents)

    assert anchored[0]["metadata"]["source_turn_ids"] == [2, 3, 4]
    assert anchored[0]["evidence_turns"][-1]["turn_id"] == 4
    assert "required" not in anchored[0]["metadata"]


def test_parent_metadata_is_reranker_context_not_window_document() -> None:
    candidate = _window("w", 1, [2, 3])
    candidate["metadata"]["parent_contexts"] = [
        {
            "session_id": "1",
            "subject_path": "Number",
            "parent_title": "Rounding",
            "parent_summary": "Use the next digit",
        }
    ]

    ranked_text = _anchored_window_rank_text(candidate)

    assert "Rounding" in ranked_text
    assert "[Logical Window]" in ranked_text
    assert candidate["document"] == "[Turn text from source]"


def test_assembler_keeps_window_reranker_rank_and_only_real_turn_evidence() -> None:
    parent = _card("p1", 1, [3])
    window = _window("w", 1, [2, 3, 4])
    window["reranker_rank"] = 7
    window["reranker_score"] = 0.42

    context = _assemble("why?", [window], [parent], "misconception")

    assert context.selected_evidence_units[0]["reranker_rank"] == 7
    assert "card p1" not in context.prompt_context_markdown
    assert {(turn["session_id"], turn["turn_id"]) for turn in context.evidence_turns} == {
        (1, 2),
        (1, 3),
        (1, 4),
    }
