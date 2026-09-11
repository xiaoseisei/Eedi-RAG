"""图聚合到 Card RAG 的授权桥接测试。"""

from datetime import datetime, timezone

import duckdb
import pytest

from src.graph_analytics import RepresentativeEvidenceRef
from src.graph_evidence_bridge import GraphEvidenceBridge
from src.graph_models import GraphNode, GraphProjectionManifest
from src.graph_store import GraphStore


@pytest.fixture
def graph_store(tmp_path):
    path = tmp_path / "graph.duckdb"
    store = GraphStore.create_staging(path, GraphProjectionManifest(
        graph_version="v1", source_artifact_hash="a" * 64, node_count=2, edge_count=0,
        build_status="STAGING", built_at_utc=datetime.now(timezone.utc),
    ))
    store.replace_projection([
        GraphNode(node_id="misconception-event:10", node_type="MISCONCEPTION_EVENT", label="mc",
                  source_session_id=10, source_card_id="card-10", source_turn_ids=[1],
                  derivation_version="v1", review_status="SOURCE"),
        GraphNode(node_id="turn:10:1", node_type="TURN", label="turn", source_session_id=10,
                  source_turn_ids=[1], derivation_version="v1", review_status="SOURCE"),
    ], [])
    store.mark_ready()
    store.close()
    return GraphStore.open_ready(path, "a" * 64)


@pytest.fixture
def source_storage():
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE session_dialogue_turns (intervention_id BIGINT, turn_id INTEGER, speaker VARCHAR, is_tutor BOOLEAN, text VARCHAR, talk_moves VARCHAR[], is_greeting_or_noise BOOLEAN)")
    connection.execute("INSERT INTO session_dialogue_turns VALUES (10,1,'student',false,'exact student quote',[],false)")
    return connection


def test_bridge_rejects_turn_not_declared_by_aggregate_event(graph_store, source_storage):
    ref = RepresentativeEvidenceRef(source_event_id="misconception-event:10", session_id=10,
                                     turn_ids=[99], perspective="STUDENT", subject_path="Number")
    with pytest.raises(ValueError, match="authorized"):
        GraphEvidenceBridge(graph_store, source_storage).materialize([ref])


def test_bridge_materializes_exact_audited_evidence(graph_store, source_storage):
    ref = RepresentativeEvidenceRef(source_event_id="misconception-event:10", session_id=10,
                                     turn_ids=[1], perspective="STUDENT", subject_path="Number")
    bundle = GraphEvidenceBridge(graph_store, source_storage).materialize([ref])
    assert bundle.pointers[0].exact_quote == "exact student quote"
    assert bundle.pointers[0].verified is True
    assert bundle.generator_context_token_count <= 4000
    assert bundle.audit_status == "AUDITED_100_VERIFIED"
