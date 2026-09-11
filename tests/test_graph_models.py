"""图投影契约测试。"""

import pytest
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError

from src.graph_models import GraphEdge, GraphNode, GraphProjectionManifest


def manifest(status="STAGING", failure_reason=None):
    return GraphProjectionManifest(
        graph_version="graph-v1",
        source_artifact_hash="a" * 64,
        node_count=0,
        edge_count=0,
        build_status=status,
        built_at_utc=datetime(2026, 9, 8, tzinfo=timezone.utc),
        failure_reason=failure_reason,
    )


def test_graph_edge_requires_source_session_and_derivation_version():
    with pytest.raises(ValidationError):
        GraphEdge(
            edge_id="edge-1",
            source_node_id="session:10",
            target_node_id="subject:rounding",
            edge_type="ABOUT_SUBJECT",
            source_session_id=None,
            source_turn_ids=[],
            derivation_version="graph-v1",
            review_status="AUTO_ASSIGNED",
        )


def test_graph_node_rejects_unknown_node_type_and_extra_fields():
    with pytest.raises(ValidationError):
        GraphNode(
            node_id="node-1",
            node_type="NOT_A_NODE",
            label="x",
            derivation_version="graph-v1",
            review_status="SOURCE",
        )

    with pytest.raises(ValidationError):
        GraphNode(
            node_id="node-1",
            node_type="SESSION",
            label="x",
            derivation_version="graph-v1",
            review_status="SOURCE",
            unexpected=True,
        )


def test_graph_edge_rejects_forbidden_relationships():
    with pytest.raises(ValidationError):
        GraphEdge(
            edge_id="edge-1",
            source_node_id="session:10",
            target_node_id="subject:rounding",
            edge_type="CAUSES",
            source_session_id=10,
            source_turn_ids=[1],
            derivation_version="graph-v1",
            review_status="AUTO_ASSIGNED",
        )


def test_manifest_failure_requires_reason_and_ready_has_no_failure():
    with pytest.raises(ValidationError):
        manifest(status="FAILED")
    with pytest.raises(ValidationError):
        manifest(status="READY", failure_reason="stale")


def test_manifest_hash_and_counts_are_strict():
    with pytest.raises(ValidationError):
        GraphProjectionManifest(
            graph_version="graph-v1", source_artifact_hash="short", node_count=0,
            edge_count=0, build_status="STAGING", built_at_utc=datetime(2026, 9, 8, tzinfo=timezone.utc)
        )
    with pytest.raises(ValidationError):
        GraphProjectionManifest(
            graph_version="graph-v1", source_artifact_hash="a" * 64, node_count=-1,
            edge_count=0, build_status="STAGING", built_at_utc=datetime(2026, 9, 8, tzinfo=timezone.utc)
        )


def test_manifest_requires_timezone_aware_utc_timestamp():
    base = dict(
        graph_version="graph-v1", source_artifact_hash="a" * 64,
        node_count=0, edge_count=0, build_status="STAGING",
    )
    with pytest.raises(ValidationError, match="timezone-aware UTC"):
        GraphProjectionManifest(**base, built_at_utc=datetime(2026, 9, 8))
    with pytest.raises(ValidationError, match="timezone-aware UTC"):
        GraphProjectionManifest(
            **base, built_at_utc=datetime(2026, 9, 8, tzinfo=timezone(timedelta(hours=8)))
        )
