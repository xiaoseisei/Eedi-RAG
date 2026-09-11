"""Task 3: controlled cross-session label assignment tests."""

import json
from pathlib import Path
import pytest

from src.graph_taxonomy import (
    DifficultySignal,
    LabelAssignment,
    Taxonomy,
    assign_exam_error_label,
    assign_content_opportunity,
    assign_misconception_label,
    assign_question_function,
    assign_strategy_labels,
    derive_difficulty_signals,
)


TAXONOMY_PATH = Path(__file__).parents[1] / "data" / "business" / "graph-taxonomy-v1.json"
TAXONOMY = Taxonomy(
    taxonomy_version="graph-taxonomy-v1",
    label_type="MISCONCEPTION",
    labels={"PLACE_VALUE_CONFUSION": "Place value confusion", "UNCLASSIFIED": "Unclassified"},
    signals={"PLACE_VALUE_CONFUSION": ["place value", "digit"], "UNCLASSIFIED": []},
)


def card():
    return {
        "chunk_id": "mc-1", "session_id": 10, "question_id": 20,
        "misconception_name": "place value confusion", "deep_mechanism": "digit confusion",
        "source_turn_ids": [1], "embedding_document": "original card document",
    }


def test_taxonomy_artifact_is_controlled_and_versioned():
    artifact = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    assert set(artifact) >= {"misconception", "exam_error", "tutor_strategy", "question_function"}
    label_lists = [artifact[key] for key in (
        "misconception", "exam_error", "tutor_strategy", "difficulty_signal", "content_opportunity"
    )]
    assert all("UNCLASSIFIED" in labels for labels in label_lists)


def test_label_assignment_is_separate_from_embedding_document():
    original_document = card()["embedding_document"]
    source = card()
    assignment = assign_misconception_label(source, TAXONOMY)
    assert assignment.review_status == "AUTO_ASSIGNED"
    assert assignment.assignment_source == "deterministic_rules"
    assert source["embedding_document"] == original_document


def test_unmatched_card_stays_unclassified():
    source = {**card(), "misconception_name": "ambiguous", "deep_mechanism": "unclear"}
    assignment = assign_misconception_label(source, TAXONOMY)
    assert assignment.canonical_label == "UNCLASSIFIED"
    assert assignment.assignment_source == "deterministic_rules"


def test_question_function_uses_first_declared_rule():
    assignment = assign_question_function("Can you explain why this answer is wrong?")
    assert assignment.canonical_label == "REQUEST_EXPLANATION"
    assert assignment.review_status == "AUTO_ASSIGNED"


def test_strategy_labels_are_independent_and_do_not_use_resolution_outcome():
    source = {"chunk_id": "st-1", "session_id": 10, "source_turn_ids": [2],
             "strategy_category": "Socratic_Questioning", "talk_moves": ["Questioning"],
             "resolution_outcome": "student revises"}
    assignments = assign_strategy_labels(source, TAXONOMY)
    assert assignments
    assert all(item.review_status == "AUTO_ASSIGNED" for item in assignments)


def test_exam_error_and_difficulty_assignments_keep_provenance():
    error = assign_exam_error_label({**card(), "error_choice": "B"}, TAXONOMY)
    assert error.source_session_id == 10
    difficulty = derive_difficulty_signals(
        {"intervention_id": 10},
        [{"turn_id": 1, "is_tutor": False, "text": "I don't know"},
         {"turn_id": 2, "is_tutor": True, "text": "Try again"}],
    )
    assert all(item.review_status == "AUTO_ASSIGNED" for item in difficulty)


def test_assignment_contract_is_strict_and_never_human_approves():
    assignment = assign_misconception_label(card(), TAXONOMY)
    assert isinstance(assignment, LabelAssignment)
    assert assignment.source_turn_ids == [1]
    assert assignment.taxonomy_version == "graph-taxonomy-v1"
    assert assignment.review_status != "HUMAN_APPROVED"


def test_llm_suggestion_cannot_be_promoted_to_human_approved():
    with pytest.raises(ValueError, match="HUMAN_APPROVED"):
        LabelAssignment(
            source_node_id="misconception-event:mc-1", label_type="MISCONCEPTION",
            canonical_label="PLACE_VALUE_CONFUSION", assignment_source="llm_suggestion",
            review_status="HUMAN_APPROVED", taxonomy_version="graph-taxonomy-v1",
            source_session_id=10, source_turn_ids=[1],
        )


def test_difficulty_signal_contract_is_strict_and_traceable():
    signals = derive_difficulty_signals(
        {"intervention_id": 10},
        [{"turn_id": 1, "is_tutor": False, "text": "I don't know"},
         {"turn_id": 2, "is_tutor": True, "text": "Try again"}],
    )
    assert signals
    assert all(isinstance(item, DifficultySignal) for item in signals)
    assert all(item.source_session_id == 10 and item.source_turn_ids for item in signals)


def test_content_opportunity_is_explicit_and_traceable():
    assignment = assign_content_opportunity(card(), TAXONOMY, source_node_id="misconception-event:mc-1")
    assert assignment.label_type == "CONTENT_OPPORTUNITY"
    assert assignment.source_node_id == "misconception-event:mc-1"
