"""图投影 DuckDB sidecar 存储测试。"""

from pathlib import Path
from datetime import datetime, timezone

import duckdb
import pytest

from src.graph_models import GraphEdge, GraphNode, GraphProjectionManifest
from src.graph_store import GraphStore


def manifest(status="STAGING", node_count=0, edge_count=0):
    return GraphProjectionManifest(
        graph_version="graph-v1",
        source_artifact_hash="a" * 64,
        node_count=node_count,
        edge_count=edge_count,
        build_status=status,
        built_at_utc=datetime(2026, 9, 8, tzinfo=timezone.utc),
    )


def node(node_id="session:10"):
    return GraphNode(
        node_id=node_id,
        node_type="SESSION",
        label="session 10",
        source_session_id=10,
        derivation_version="graph-v1",
        review_status="SOURCE",
    )


def edge(edge_id="edge-1", source_id="session:10", target_id="subject:rounding"):
    return GraphEdge(
        edge_id=edge_id,
        source_node_id=source_id,
        target_node_id=target_id,
        edge_type="ABOUT_SUBJECT",
        source_session_id=10,
        source_turn_ids=[1],
        derivation_version="graph-v1",
        review_status="AUTO_ASSIGNED",
    )


def test_graph_store_rejects_stale_source_hash(tmp_path):
    store = GraphStore.create_staging(tmp_path / "graph.duckdb", manifest())
    store.mark_ready()
    store.close()
    with pytest.raises(ValueError, match="source hash mismatch"):
        GraphStore.open_ready(tmp_path / "graph.duckdb", expected_source_hash="b" * 64)


def test_replace_projection_is_transactional_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "graph.duckdb"
    subject = node("subject:rounding")
    store = GraphStore.create_staging(path, manifest(node_count=2, edge_count=1))
    store.replace_projection([node(), subject], [edge()])
    assert store.get_manifest().node_count == 2
    assert store.get_manifest().edge_count == 1
    with pytest.raises(ValueError, match="duplicate node ID"):
        store.replace_projection([node(), node()], [edge()])
    assert store.list_nodes() == [node(), subject]
    store.mark_ready()
    store.close()


def test_replace_projection_rejects_missing_edge_endpoint(tmp_path):
    store = GraphStore.create_staging(tmp_path / "graph.duckdb", manifest())
    edge_with_missing_endpoint = edge("edge-1", "session:10", "missing-node")
    with pytest.raises(ValueError, match="endpoint"):
        store.replace_projection([node()], [edge_with_missing_endpoint])
    store.close()


def test_replace_projection_persists_and_reads_label_assignments(tmp_path):
    from src.graph_taxonomy import LabelAssignment
    subject = node("subject:rounding")
    db_path = tmp_path / "graph.duckdb"
    store = GraphStore.create_staging(db_path, manifest(node_count=2, edge_count=1))
    assignment = LabelAssignment(
        source_node_id="session:10", label_type="DIFFICULTY_SIGNAL", canonical_label="UNCLASSIFIED",
        assignment_source="deterministic_rules", review_status="AUTO_ASSIGNED",
        taxonomy_version="graph-taxonomy-v1", source_session_id=10, source_turn_ids=[1],
    )
    store.replace_projection([node(), subject], [edge()], [assignment])
    assert store.list_label_assignments()[0] == assignment
    store.close()
    reopened = GraphStore(db_path, duckdb.connect(str(db_path)), read_only=False)
    assert reopened.list_label_assignments()[0] == assignment
    reopened.close()


def test_legacy_label_schema_migrates_provenance_from_graph_node(tmp_path):
    import json
    path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE graph_nodes (node_id VARCHAR, node_type VARCHAR, label VARCHAR, properties_json VARCHAR, source_session_id BIGINT, source_card_id VARCHAR, source_turn_ids INTEGER[], derivation_version VARCHAR, review_status VARCHAR)")
    connection.execute("CREATE TABLE graph_edges (edge_id VARCHAR, source_node_id VARCHAR, target_node_id VARCHAR, edge_type VARCHAR, source_session_id BIGINT, source_turn_ids INTEGER[], derivation_version VARCHAR, review_status VARCHAR)")
    connection.execute("CREATE TABLE graph_manifest (graph_version VARCHAR, source_artifact_hash VARCHAR, node_count INTEGER, edge_count INTEGER, build_status VARCHAR, built_at_utc TIMESTAMP, failure_reason VARCHAR)")
    connection.execute("CREATE TABLE graph_label_assignments (assignment_id VARCHAR, source_node_id VARCHAR, label_type VARCHAR, canonical_label VARCHAR, assignment_source VARCHAR, review_status VARCHAR, taxonomy_version VARCHAR)")
    connection.execute("INSERT INTO graph_nodes VALUES ('session:10','SESSION','session 10','{}',10,NULL,[1],'v1','SOURCE')")
    connection.execute("INSERT INTO graph_label_assignments VALUES ('a','session:10','DIFFICULTY_SIGNAL','UNCLASSIFIED','deterministic_rules','AUTO_ASSIGNED','v1')")
    connection.close()
    store = GraphStore(path, duckdb.connect(str(path)), read_only=False)
    store._create_schema()
    assert store.list_label_assignments()[0].source_session_id == 10
    store.close()


def test_replace_projection_rejects_duplicate_edge_ids(tmp_path):
    subject = node("subject:rounding")
    store = GraphStore.create_staging(tmp_path / "graph.duckdb", manifest(node_count=2, edge_count=2))
    with pytest.raises(ValueError, match="duplicate edge ID"):
        store.replace_projection([node(), subject], [edge(), edge()])
    store.close()


def test_mark_ready_rejects_manifest_count_mismatch(tmp_path):
    store = GraphStore.create_staging(tmp_path / "graph.duckdb", manifest())
    store.replace_projection([node(), node("subject:rounding")], [edge()])
    with pytest.raises(ValueError, match="manifest counts do not match"):
        store.mark_ready()
    assert store.get_manifest().build_status == "STAGING"
    store.close()


def test_ready_store_is_read_only_and_does_not_create_tables(tmp_path):
    path = tmp_path / "missing.duckdb"
    with pytest.raises(ValueError, match="ready manifest"):
        GraphStore.open_ready(path, expected_source_hash="a" * 64)
    assert not path.exists()


def test_open_ready_reads_projection_without_writes(tmp_path):
    path = tmp_path / "graph.duckdb"
    store = GraphStore.create_staging(path, manifest(node_count=2, edge_count=1))
    store.replace_projection([node(), node("subject:rounding")], [edge()])
    store.mark_ready()
    store.close()

    ready = GraphStore.open_ready(path, expected_source_hash="a" * 64)
    assert ready.list_nodes()[0].node_id == "session:10"
    assert ready.list_edges()[0].edge_id == "edge-1"
    with pytest.raises((PermissionError, duckdb.IOException, RuntimeError)):
        ready.mark_ready()
    ready.close()
