from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from src.graph_analytics import GraphAnalytics, GraphRepository
from src.graph_models import GraphProjectionManifest
from src.graph_store import GraphStore
from src.graph_taxonomy import LabelAssignment


@pytest.fixture
def graph_repo(tmp_path: Path):
    source = duckdb.connect(":memory:")
    source.execute("CREATE TABLE tutoring_sessions (intervention_id BIGINT, subject_path VARCHAR)")
    source.execute("CREATE TABLE session_dialogue_turns (intervention_id BIGINT, turn_id INTEGER, speaker VARCHAR, text VARCHAR)")
    source.execute("CREATE TABLE misconception_chunks (chunk_id VARCHAR, session_id BIGINT, subject_path VARCHAR, misconception_name VARCHAR, source_turn_ids INTEGER[])")
    source.execute("CREATE TABLE tutor_strategy_chunks (chunk_id VARCHAR, session_id BIGINT, subject_path VARCHAR, strategy_category VARCHAR, source_turn_ids INTEGER[])")
    source.executemany("INSERT INTO tutoring_sessions VALUES (?, ?)", [(1, "Number"), (2, "Number"), (3, "Algebra")])
    source.executemany("INSERT INTO session_dialogue_turns VALUES (?, ?, ?, ?)", [
        (1, 1, "student", "Why is this wrong?"), (1, 2, "tutor", "Explain the place value."),
        (2, 1, "student", "Why is this wrong?"), (2, 2, "tutor", "Ask for reasoning."),
        (3, 1, "student", "I am unsure."), (3, 2, "tutor", "Use a counter example."),
    ])
    source.executemany("INSERT INTO misconception_chunks VALUES (?, ?, ?, ?, ?)", [
        ("mc-1", 1, "Number", "Place value", [1]),
        ("mc-2", 2, "Number", "Place value", [1]),
        ("mc-3", 2, "Number", "Place value", [1]),
        ("mc-4", 3, "Algebra", "Sign", [1]),
    ])
    source.executemany("INSERT INTO tutor_strategy_chunks VALUES (?, ?, ?, ?, ?)", [
        ("st-1", 1, "Number", "ASK_FOR_REASONING", [2]),
        ("st-2", 2, "Number", "ASK_FOR_REASONING", [2]),
        ("st-3", 3, "Algebra", "USE_COUNTER_EXAMPLE", [2]),
    ])

    manifest = GraphProjectionManifest(
        graph_version="graph-fixture-v1", source_artifact_hash="a" * 64,
        node_count=0, edge_count=0, build_status="STAGING",
        built_at_utc=datetime.now(timezone.utc),
    )
    graph_path = tmp_path / "graph.duckdb"
    staging = GraphStore.create_staging(graph_path, manifest)
    staging._connection.execute("DELETE FROM graph_manifest")
    staging._connection.execute("INSERT INTO graph_manifest VALUES (?, ?, ?, ?, ?, ?, ?)",
                                ["graph-fixture-v1", "a" * 64, 0, 0, "READY", datetime.now(timezone.utc), None])
    nodes = [
        ("turn:1:1", "TURN", "turn 1", "{}", 1, None, [1], "v1", "SOURCE"),
        ("turn:2:1", "TURN", "turn 1", "{}", 2, None, [1], "v1", "SOURCE"),
        ("turn:3:1", "TURN", "turn 1", "{}", 3, None, [1], "v1", "SOURCE"),
        ("misconception-event:mc-1", "MISCONCEPTION_EVENT", "Place value", "{}", 1, "mc-1", [1], "v1", "SOURCE"),
        ("misconception-event:mc-2", "MISCONCEPTION_EVENT", "Place value", "{}", 2, "mc-2", [1], "v1", "SOURCE"),
        ("misconception-event:mc-3", "MISCONCEPTION_EVENT", "Place value", "{}", 2, "mc-3", [1], "v1", "SOURCE"),
        ("misconception-event:mc-4", "MISCONCEPTION_EVENT", "Sign", "{}", 3, "mc-4", [1], "v1", "SOURCE"),
        ("tutor-strategy-event:st-1", "TUTOR_STRATEGY_EVENT", "ASK_FOR_REASONING", "{}", 1, "st-1", [2], "v1", "SOURCE"),
        ("tutor-strategy-event:st-2", "TUTOR_STRATEGY_EVENT", "ASK_FOR_REASONING", "{}", 2, "st-2", [2], "v1", "SOURCE"),
        ("tutor-strategy-event:st-3", "TUTOR_STRATEGY_EVENT", "USE_COUNTER_EXAMPLE", "{}", 3, "st-3", [2], "v1", "SOURCE"),
    ]
    staging._connection.executemany("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", nodes)
    assignments = [
        ("q1", "turn:1:1", "QUESTION_FUNCTION", "REQUEST_EXPLANATION", 1, [1]),
        ("q2", "turn:2:1", "QUESTION_FUNCTION", "REQUEST_EXPLANATION", 2, [1]),
        ("q3", "turn:3:1", "QUESTION_FUNCTION", "UNCLASSIFIED", 3, [1]),
        ("m1", "misconception-event:mc-1", "MISCONCEPTION", "PLACE_VALUE_CONFUSION", 1, [1]),
        ("m2", "misconception-event:mc-2", "MISCONCEPTION", "PLACE_VALUE_CONFUSION", 2, [1]),
        ("m3", "misconception-event:mc-3", "MISCONCEPTION", "PLACE_VALUE_CONFUSION", 2, [1]),
        ("m4", "misconception-event:mc-4", "MISCONCEPTION", "UNCLASSIFIED", 3, [1]),
        ("e1", "misconception-event:mc-1", "EXAM_ERROR", "PROCEDURAL_ERROR", 1, [1]),
        ("e2", "misconception-event:mc-2", "EXAM_ERROR", "PROCEDURAL_ERROR", 2, [1]),
        ("s1", "tutor-strategy-event:st-1", "TUTOR_STRATEGY", "ASK_FOR_REASONING", 1, [2]),
        ("s2", "tutor-strategy-event:st-2", "TUTOR_STRATEGY", "ASK_FOR_REASONING", 2, [2]),
        ("s3", "tutor-strategy-event:st-3", "TUTOR_STRATEGY", "USE_COUNTER_EXAMPLE", 3, [2]),
        ("c1", "misconception-event:mc-1", "CONTENT_OPPORTUNITY", "ADD_worked_example", 1, [1]),
        ("c2", "misconception-event:mc-2", "CONTENT_OPPORTUNITY", "ADD_worked_example", 2, [1]),
        ("d1", "turn:1:1", "DIFFICULTY_SIGNAL", "REPEATED_ATTEMPTS", 1, [1]),
    ]
    staging._connection.executemany(
        "INSERT INTO graph_label_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(i, n, t, l, "deterministic_rules", "AUTO_ASSIGNED", "taxonomy-v1", sid, turns)
         for i, n, t, l, sid, turns in assignments],
    )
    staging.close()
    graph = GraphStore.open_ready(graph_path, "a" * 64)
    return GraphRepository(graph_store=graph, source_connection=source, dataset_version="dataset-fixture-v1")


def test_frequent_questions_count_distinct_sessions(graph_repo):
    result = GraphAnalytics(graph_repo).frequent_student_questions(top_k=10)
    assert result.rows[0]["distinct_session_count"] == 2
    assert result.rows[0]["event_count"] == 2
    assert result.rows[0]["session_share"] == pytest.approx(2 / 3)


def test_graph_analytics_keeps_unclassified_and_unmeasured_visible(graph_repo):
    result = GraphAnalytics(graph_repo).recurring_misconceptions(top_k=10)
    assert any(row["canonical_name"] == "UNCLASSIFIED" for row in result.rows)
    assert "UNRESOLVED_ENDING" in result.data_scope.unmeasured_metrics


def test_subject_filter_is_parameterized_and_scope_is_reported(graph_repo):
    result = GraphAnalytics(graph_repo).tutor_strategy_patterns(10, ["Number' OR 1=1 --"])
    assert result.rows == []
    assert result.result_status == "NO_MATCHING_SCOPE"
    assert result.status_reason == "explicit subject filters matched zero sessions"
    assert result.data_scope.matched_sessions == 0
    assert result.data_scope.subject_filters == ["Number' OR 1=1 --"]


def test_existing_scope_without_operator_facts_is_no_evidence(graph_repo):
    result = GraphAnalytics(graph_repo)._aggregate(
        "unpopulated_operator", "UNPOPULATED_LABEL", "STUDENT", 10, ["Number"]
    )

    assert result.rows == []
    assert result.data_scope.matched_sessions == 2
    assert result.result_status == "NO_EVIDENCE"
    assert result.status_reason == "sessions matched but no aggregate rows were produced"


def test_empty_source_dataset_is_no_evidence_not_unmeasured(graph_repo):
    graph_repo.source_connection.execute("DELETE FROM tutoring_sessions")

    result = GraphAnalytics(graph_repo).frequent_student_questions(10)

    assert result.rows == []
    assert result.data_scope.total_sessions_considered == 0
    assert result.result_status == "NO_EVIDENCE"
    assert result.status_reason == "no sessions available for analytics"


def test_pairs_only_join_events_from_same_session(graph_repo):
    result = GraphAnalytics(graph_repo).misconception_strategy_pairs(10)
    assert result.rows[0]["relation"] == "CO_OCCURS_WITH"
    assert result.rows[0]["distinct_session_count"] == 2
    assert all("causal" not in row for row in result.rows)


def test_content_opportunities_has_explicit_priority_and_refs(graph_repo):
    result = GraphAnalytics(graph_repo).content_opportunities(10)
    row = result.rows[0]
    assert row["missing_content_flag"] == 1
    assert row["repeated_error_count"] == 3
    assert row["priority_score"] == pytest.approx(3 * (1 - 2 / 3) + 1)
    assert {ref.session_id for ref in result.representative_refs} >= {1, 2}


def test_graph_analytics_does_not_write_to_ready_graph(graph_repo):
    before = graph_repo.graph_store._connection.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    GraphAnalytics(graph_repo).difficult_concepts(10)
    after = graph_repo.graph_store._connection.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    assert after == before


def test_difficulty_representative_refs_only_contain_student_turns(graph_repo):
    result = GraphAnalytics(graph_repo).difficult_concepts(top_k=10)

    assert result.representative_refs
    for ref in result.representative_refs:
        rows = graph_repo.source_connection.execute(
            "SELECT turn_id, speaker FROM session_dialogue_turns WHERE intervention_id=? AND turn_id IN (SELECT UNNEST(?))",
            [ref.session_id, ref.turn_ids],
        ).fetchall()
        assert rows
        assert all(str(speaker).casefold() == "student" for _, speaker in rows)
