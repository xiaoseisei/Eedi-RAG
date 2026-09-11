"""Task 2 session/card graph projection tests."""

import pytest

from src.graph_models import GraphNode
from src.graph_projection import project_session_graph
from scripts.build_session_card_graph import _deduplicate_nodes


def session_row():
    return {
        "intervention_id": 10,
        "question_id": 20,
        "subject_path": "  Algebra / Fractions ",
        "question_text": "What is 1/2 + 1/4?",
    }


def turns():
    return [
        {"turn_id": 1, "speaker": "student", "is_tutor": False, "text": "I add the bottoms.", "talk_moves": ["elicit"]},
        {"turn_id": 2, "speaker": "tutor", "is_tutor": True, "text": "What denominator works?", "talk_moves": ["Questioning"]},
    ]


def misconception_row():
    return {
        "chunk_id": "mc-1", "session_id": 10, "question_id": 20,
        "subject_path": "Algebra / Fractions", "misconception_name": "Add denominators",
        "error_choice": "B", "deep_mechanism": "Confuses parts", "confusion_triggers": ["addition"],
        "verbatim_student_quotes": ["I add the bottoms."], "source_turn_ids": [1],
    }


def strategy_row():
    return {
        "chunk_id": "st-1", "session_id": 10, "question_id": 20,
        "subject_path": "Algebra / Fractions", "pedagogical_goal": "Build denominator concept",
        "strategy_category": "Socratic questioning", "key_aha_question": "What denominator works?",
        "scaffolding_steps": ["Compare denominators"], "analogy_or_metaphor": "shared size",
        "talk_moves": ["Questioning"], "resolution_outcome": "student revises",
        "source_turn_ids": [2],
    }


def edge_types(projection):
    return {edge.edge_type for edge in projection.edges}


def test_projection_creates_one_session_two_card_events_and_source_edges():
    projection = project_session_graph(session_row(), misconception_row(), strategy_row(), turns())
    assert [node.node_type for node in projection.nodes].count("SESSION") == 1
    assert [node.node_type for node in projection.nodes].count("MISCONCEPTION_EVENT") == 1
    assert [node.node_type for node in projection.nodes].count("TUTOR_STRATEGY_EVENT") == 1
    assert edge_types(projection) == {
        "ANCHORS", "ABOUT_SUBJECT", "HAS_MISCONCEPTION", "HAS_STRATEGY",
        "EVIDENCED_BY", "USES_TALK_MOVE", "HAS_TURN", "TYPE_OF", "CO_OCCURS_WITH"
    }


def test_projection_materializes_shared_concepts_and_same_session_cooccurrence():
    projection = project_session_graph(session_row(), misconception_row(), strategy_row(), turns())

    concept_nodes = [node for node in projection.nodes if node.node_type.endswith("_CONCEPT")]
    assert {node.node_type for node in concept_nodes} == {"MISCONCEPTION_CONCEPT", "STRATEGY_CONCEPT"}
    assert {edge.edge_type for edge in projection.edges} >= {"TYPE_OF", "CO_OCCURS_WITH"}
    assert any(
        edge.edge_type == "TYPE_OF"
        and edge.source_node_id == "misconception-event:mc-1"
        and edge.target_node_id == "misconception-concept:FRACTION_DENOMINATOR_CONFUSION"
        for edge in projection.edges
    )
    assert any(
        edge.edge_type == "CO_OCCURS_WITH"
        and edge.source_node_id == "misconception-event:mc-1"
        and edge.target_node_id == "tutor-strategy-event:st-1"
        for edge in projection.edges
    )


def test_projection_rejects_wrong_role_card_pointer():
    bad_card = {**misconception_row(), "source_turn_ids": [2]}
    with pytest.raises(ValueError, match="student"):
        project_session_graph(session_row(), bad_card, strategy_row(), turns())


def test_projection_rejects_foreign_card_and_turn_pointers():
    with pytest.raises(ValueError, match="session_id"):
        project_session_graph(session_row(), {**misconception_row(), "session_id": 99}, strategy_row(), turns())
    with pytest.raises(ValueError, match="does not exist"):
        project_session_graph(session_row(), {**misconception_row(), "source_turn_ids": [9]}, strategy_row(), turns())


def test_projection_preserves_card_fields_and_normalizes_graph_keys():
    projection = project_session_graph(session_row(), misconception_row(), strategy_row(), turns())
    event = next(node for node in projection.nodes if node.node_type == "MISCONCEPTION_EVENT")
    assert event.properties["misconception_name"] == "Add denominators"
    assert event.properties["deep_mechanism"] == "Confuses parts"
    assert any(node.node_id == "subject:algebra/fractions" for node in projection.nodes)
    assert any(node.node_id == "talk-move:questioning" for node in projection.nodes)


def test_builder_deduplicates_shared_nodes_and_keeps_all_source_refs():
    first = GraphNode(
        node_id="subject:algebra/fractions", node_type="SUBJECT", label="algebra/fractions",
        properties={"subject_path": "Algebra / Fractions"}, source_session_id=10,
        source_turn_ids=[1, 2], derivation_version="v1", review_status="AUTO_ASSIGNED",
    )
    second = first.model_copy(update={"source_session_id": 11, "source_turn_ids": [3, 4]})
    merged = _deduplicate_nodes([first, second])
    assert len(merged) == 1
    assert merged[0].properties["source_refs"] == [
        {"source_session_id": 10, "source_turn_ids": [1, 2]},
        {"source_session_id": 11, "source_turn_ids": [3, 4]},
    ]


def test_builder_preserves_question_text_variants_for_shared_question_id():
    first = GraphNode(
        node_id="question:20", node_type="QUESTION", label="question 20",
        properties={"question_text": "Alex asks a question."}, source_session_id=10,
        source_turn_ids=[1, 2], derivation_version="v1", review_status="SOURCE",
    )
    second = first.model_copy(update={
        "source_session_id": 11,
        "source_turn_ids": [3, 4],
        "properties": {"question_text": "Nate asks a question."},
    })
    merged = _deduplicate_nodes([first, second])
    assert merged[0].properties["question_text"] == "Alex asks a question."
    assert merged[0].properties["source_question_texts"] == [
        "Alex asks a question.", "Nate asks a question."
    ]


def test_builder_keeps_first_question_source_ref():
    first = GraphNode(
        node_id="question:20", node_type="QUESTION", label="question 20",
        properties={"question_text": "Alex asks a question."}, source_session_id=10,
        source_turn_ids=[1, 2], derivation_version="v1", review_status="SOURCE",
    )
    second = first.model_copy(update={"source_session_id": 11, "source_turn_ids": [3, 4]})
    merged = _deduplicate_nodes([first, second])
    assert merged[0].properties["source_refs"][0] == {
        "source_session_id": 10, "source_turn_ids": [1, 2]
    }


def test_projection_rejects_cross_session_turn_and_duplicate_card_pointer():
    with pytest.raises(ValueError, match="another session"):
        project_session_graph(
            session_row(), misconception_row(), strategy_row(),
            [{**turns()[0], "intervention_id": 99}, turns()[1]],
        )


def test_projection_emits_persistable_difficulty_assignments():
    projection = project_session_graph(session_row(), misconception_row(), strategy_row(), turns())
    assert any(
        assignment.label_type == "DIFFICULTY_SIGNAL"
        for assignment in (projection.label_assignments or [])
    )


def test_projection_assignments_reference_real_graph_nodes():
    projection = project_session_graph(session_row(), misconception_row(), strategy_row(), turns())
    node_ids = {node.node_id for node in projection.nodes}
    assert all(item.source_node_id in node_ids for item in (projection.label_assignments or []))
    with pytest.raises(ValueError, match="must be unique"):
        project_session_graph(
            session_row(), {**misconception_row(), "source_turn_ids": [1, 1]},
            strategy_row(), turns(),
        )
