"""Topology evaluator adversarial tests: real DB rows must be validated, not trusted."""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import duckdb
import pytest


EVAL_PATH = Path(__file__).parents[1] / "evals" / "graph" / "Topology _Construction_eval.py"
SPEC = importlib.util.spec_from_file_location("topology_construction_eval", EVAL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _evaluator():
    return MODULE.TopologyConstructionEvaluator.__new__(MODULE.TopologyConstructionEvaluator)


def test_pointer_validity_rejects_missing_graph_target_node():
    source = duckdb.connect(":memory:")
    source.execute("CREATE TABLE session_dialogue_turns (intervention_id BIGINT, turn_id INTEGER)")
    source.execute("INSERT INTO session_dialogue_turns VALUES (10, 1)")
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, label VARCHAR, properties_json VARCHAR, source_session_id BIGINT, source_card_id VARCHAR, source_turn_ids INTEGER[])")
    graph.execute("INSERT INTO graph_nodes VALUES ('turn:10:1', 'TURN', 'turn', '{}', 10, NULL, [1])")
    graph.execute("CREATE TABLE graph_edges (edge_id VARCHAR, source_node_id VARCHAR, target_node_id VARCHAR, source_session_id BIGINT, source_turn_ids INTEGER[], edge_type VARCHAR)")
    graph.execute("INSERT INTO graph_edges VALUES ('e1', 'missing-event', 'turn:10:1', 10, [1], 'EVIDENCED_BY')")

    result = _evaluator()._eval_pointer_validity(graph, source)

    assert result.value == 0.0
    assert result.status == MODULE.MetricStatus.FAILED
    source.close()
    graph.close()


def test_role_accuracy_rejects_unknown_source_event_instead_of_counting_it():
    source = duckdb.connect(":memory:")
    source.execute("CREATE TABLE session_dialogue_turns (intervention_id BIGINT, turn_id INTEGER, is_tutor BOOLEAN)")
    source.execute("INSERT INTO session_dialogue_turns VALUES (10, 1, FALSE)")
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, label VARCHAR, properties_json VARCHAR, source_session_id BIGINT, source_card_id VARCHAR, source_turn_ids INTEGER[])")
    graph.execute("INSERT INTO graph_nodes VALUES ('turn:10:1', 'TURN', 'turn', '{}', 10, NULL, [1])")
    graph.execute("CREATE TABLE graph_edges (edge_id VARCHAR, source_node_id VARCHAR, target_node_id VARCHAR, source_session_id BIGINT, source_turn_ids INTEGER[], edge_type VARCHAR)")
    graph.execute("INSERT INTO graph_edges VALUES ('e1', 'unknown-event', 'turn:10:1', 10, [1], 'EVIDENCED_BY')")

    result = _evaluator()._eval_role_accuracy(graph, source)

    assert result.value == 0.0
    assert result.status == MODULE.MetricStatus.FAILED
    source.close()
    graph.close()


def test_pointer_validity_rejects_event_from_different_session():
    source = duckdb.connect(":memory:")
    source.execute("CREATE TABLE session_dialogue_turns (intervention_id BIGINT, turn_id INTEGER)")
    source.execute("INSERT INTO session_dialogue_turns VALUES (10, 1)")
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, label VARCHAR, properties_json VARCHAR, source_session_id BIGINT, source_card_id VARCHAR, source_turn_ids INTEGER[])")
    graph.execute("INSERT INTO graph_nodes VALUES ('event:1', 'MISCONCEPTION_EVENT', 'mc', '{}', 99, 'card-1', [1])")
    graph.execute("INSERT INTO graph_nodes VALUES ('turn:10:1', 'TURN', 'turn', '{}', 10, NULL, [1])")
    graph.execute("CREATE TABLE graph_edges (edge_id VARCHAR, source_node_id VARCHAR, target_node_id VARCHAR, source_session_id BIGINT, source_turn_ids INTEGER[], edge_type VARCHAR)")
    graph.execute("INSERT INTO graph_edges VALUES ('e1', 'event:1', 'turn:10:1', 10, [1], 'EVIDENCED_BY')")

    result = _evaluator()._eval_pointer_validity(graph, source)

    assert result.value == 0.0
    assert result.status == MODULE.MetricStatus.FAILED
    source.close()
    graph.close()


def test_taxonomy_coverage_does_not_accept_unknown_labels():
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_label_assignments (source_node_id VARCHAR, label_type VARCHAR, canonical_label VARCHAR, source_session_id BIGINT, source_turn_ids INTEGER[])")
    graph.execute("INSERT INTO graph_label_assignments VALUES ('misconception-event:1', 'MISCONCEPTION', 'NOT_IN_TAXONOMY', 1, [1])")
    graph.execute("INSERT INTO graph_label_assignments VALUES ('tutor-strategy-event:1', 'TUTOR_STRATEGY', 'UNCLASSIFIED', 1, [2])")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, label VARCHAR, properties_json VARCHAR, source_session_id BIGINT, source_card_id VARCHAR, source_turn_ids INTEGER[], derivation_version VARCHAR, review_status VARCHAR)")
    graph.execute("INSERT INTO graph_nodes VALUES ('misconception-event:1', 'MISCONCEPTION_EVENT', 'mc', '{}', 1, 'mc-1', [1], 'v1', 'SOURCE')")
    graph.execute("INSERT INTO graph_nodes VALUES ('tutor-strategy-event:1', 'TUTOR_STRATEGY_EVENT', 'st', '{}', 1, 'st-1', [2], 'v1', 'SOURCE')")

    result = _evaluator()._eval_taxonomy_coverage(graph)

    assert result.details["invalid_label_count"] == 1
    assert result.details["core_cards_classified"] == 0
    graph.close()


def test_edge_endpoint_integrity_rejects_missing_endpoint():
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR)")
    graph.execute("INSERT INTO graph_nodes VALUES ('session:1')")
    graph.execute("CREATE TABLE graph_edges (edge_id VARCHAR, source_node_id VARCHAR, target_node_id VARCHAR)")
    graph.execute("INSERT INTO graph_edges VALUES ('e1', 'session:1', 'missing:1')")

    result = _evaluator()._eval_edge_endpoint_integrity(graph)

    assert result.value == 0.0
    assert result.status == MODULE.MetricStatus.FAILED
    graph.close()


def test_taxonomy_label_validity_is_a_hard_failure_for_unknown_label():
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_label_assignments (source_node_id VARCHAR, label_type VARCHAR, canonical_label VARCHAR)")
    graph.execute("INSERT INTO graph_label_assignments VALUES ('misconception-event:1', 'MISCONCEPTION', 'NOT_IN_TAXONOMY')")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, properties_json VARCHAR)")
    graph.execute("INSERT INTO graph_nodes VALUES ('misconception-event:1', 'MISCONCEPTION_EVENT', '{}')")

    result = _evaluator()._eval_taxonomy_label_validity(graph)

    assert result.value == 0.0
    assert result.status == MODULE.MetricStatus.FAILED
    assert result.hard_gate is True
    graph.close()


def test_taxonomy_label_validity_rejects_missing_required_assignment_type():
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_label_assignments (source_node_id VARCHAR, label_type VARCHAR, canonical_label VARCHAR)")
    graph.execute("INSERT INTO graph_label_assignments VALUES ('misconception-event:1', 'MISCONCEPTION', 'UNCLASSIFIED')")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, properties_json VARCHAR)")
    graph.execute("INSERT INTO graph_nodes VALUES ('misconception-event:1', 'MISCONCEPTION_EVENT', '{}')")

    result = _evaluator()._eval_taxonomy_label_validity(graph)

    assert result.value == 0.0
    assert result.status == MODULE.MetricStatus.FAILED
    graph.close()


def test_graph_content_hash_changes_when_graph_content_changes():
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, properties_json VARCHAR)")
    graph.execute("CREATE TABLE graph_edges (edge_id VARCHAR, source_node_id VARCHAR, target_node_id VARCHAR)")
    graph.execute("CREATE TABLE graph_label_assignments (assignment_id VARCHAR, canonical_label VARCHAR)")
    graph.execute("INSERT INTO graph_nodes VALUES ('n1', '{\"label\":\"before\"}')")
    graph.execute("INSERT INTO graph_edges VALUES ('e1', 'n1', 'n1')")
    graph.execute("INSERT INTO graph_label_assignments VALUES ('a1', 'UNCLASSIFIED')")
    first = _evaluator()._graph_content_sha256(graph)
    graph.execute("UPDATE graph_nodes SET properties_json=? WHERE node_id=?", ['{"label":"after"}', 'n1'])
    second = _evaluator()._graph_content_sha256(graph)

    assert first != second
    graph.close()


def test_reproducibility_fails_when_independent_graph_content_differs(tmp_path):
    source_path = tmp_path / "source.duckdb"
    source_path.write_bytes(b"source")
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, properties_json VARCHAR)")
    graph.execute("CREATE TABLE graph_edges (edge_id VARCHAR, source_node_id VARCHAR, target_node_id VARCHAR)")
    graph.execute("CREATE TABLE graph_label_assignments (assignment_id VARCHAR, canonical_label VARCHAR)")
    graph.execute("INSERT INTO graph_nodes VALUES ('n1', '{}')")
    graph.execute("INSERT INTO graph_edges VALUES ('e1', 'n1', 'n1')")
    graph.execute("INSERT INTO graph_label_assignments VALUES ('a1', 'UNCLASSIFIED')")
    result = _evaluator()._eval_reproducibility_hash(
        graph, source_path, _evaluator()._sha256_file(source_path),
        actual_node_count=1, actual_edge_count=1,
        manifest_node_count=1, manifest_edge_count=1,
        build_status="READY", rebuilt_graph_content_hash="different-content-hash",
    )

    assert result.value == 0.0
    assert result.status == MODULE.MetricStatus.FAILED
    graph.close()


def test_evaluate_rejects_multiple_manifest_rows(tmp_path):
    graph_path = tmp_path / "graph.duckdb"
    source_path = tmp_path / "source.duckdb"
    source = duckdb.connect(str(source_path))
    source.execute("CREATE TABLE session_dialogue_turns (intervention_id BIGINT, turn_id INTEGER, is_tutor BOOLEAN)")
    source.close()
    graph = duckdb.connect(str(graph_path))
    graph.execute("CREATE TABLE graph_manifest (graph_version VARCHAR, source_artifact_hash VARCHAR, node_count INTEGER, edge_count INTEGER, build_status VARCHAR)")
    graph.execute("INSERT INTO graph_manifest VALUES ('v1', ?, 0, 0, 'READY')", ["a" * 64])
    graph.execute("INSERT INTO graph_manifest VALUES ('v2', ?, 0, 0, 'READY')", ["b" * 64])
    graph.close()

    evaluator = MODULE.TopologyConstructionEvaluator(graph_path, source_path)
    with pytest.raises(ValueError, match="恰好一行"):
        evaluator.evaluate(run_id="manifest-test")


def test_find_latest_graph_db_ignores_non_ready_candidates(tmp_path):
    ready_path = tmp_path / "graph-ready.duckdb"
    failed_path = tmp_path / "graph-failed.duckdb"
    for path, status in ((ready_path, "READY"), (failed_path, "FAILED")):
        connection = duckdb.connect(str(path))
        connection.execute("CREATE TABLE graph_manifest (graph_version VARCHAR, source_artifact_hash VARCHAR, node_count INTEGER, edge_count INTEGER, build_status VARCHAR)")
        connection.execute("INSERT INTO graph_manifest VALUES ('v1', ?, 1, 1, ?)", ["a" * 64, status])
        connection.close()
    now = time.time()
    os.utime(failed_path, (now + 10, now + 10))

    selected = MODULE.find_latest_graph_db(tmp_path)

    assert selected == ready_path


def test_orphan_metric_blocks_on_empty_graph():
    graph = duckdb.connect(":memory:")
    graph.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, label VARCHAR)")
    graph.execute("CREATE TABLE graph_edges (source_node_id VARCHAR, target_node_id VARCHAR)")

    result = _evaluator()._eval_orphan_nodes(graph)

    assert result.status == MODULE.MetricStatus.BLOCKED
    assert result.total_cases == 0
    graph.close()


def test_projection_does_not_drop_null_session_or_malformed_turn(tmp_path):
    source = duckdb.connect(":memory:")
    source.execute("CREATE TABLE tutoring_sessions (intervention_id BIGINT)")
    source.execute("INSERT INTO tutoring_sessions VALUES (10)")
    source.execute("CREATE TABLE misconception_chunks (session_id BIGINT, chunk_id VARCHAR)")
    source.execute("CREATE TABLE tutor_strategy_chunks (session_id BIGINT, chunk_id VARCHAR)")
    source.execute("CREATE TABLE session_dialogue_turns (intervention_id BIGINT, turn_id INTEGER)")
    source.execute("INSERT INTO session_dialogue_turns VALUES (10, 1)")
    graph = duckdb.connect(":memory:")
    graph.execute("""CREATE TABLE graph_nodes (
        node_id VARCHAR, node_type VARCHAR, source_session_id BIGINT,
        source_card_id VARCHAR, source_turn_ids INTEGER[]
    )""")
    graph.execute("INSERT INTO graph_nodes VALUES ('session:10','SESSION',10,NULL,NULL)")
    graph.execute("INSERT INTO graph_nodes VALUES ('turn:10:bad','TURN',10,NULL,[1,2])")
    graph.execute("INSERT INTO graph_nodes VALUES ('orphan','TURN',NULL,NULL,[1])")
    graph.execute("""CREATE TABLE graph_edges (
        source_node_id VARCHAR, target_node_id VARCHAR, edge_type VARCHAR,
        source_session_id BIGINT
    )""")

    result = _evaluator()._eval_projection_completeness(graph, source)

    assert result.status == MODULE.MetricStatus.FAILED
    assert result.details["invalid_source_session_node_count"] == 1
    assert result.details["malformed_turn_node_count"] == 1
    source.close()
    graph.close()
