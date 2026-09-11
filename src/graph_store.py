"""独立 DuckDB 图投影 sidecar 的事务存储。"""

from __future__ import annotations

import json
import hashlib
from datetime import timezone
from pathlib import Path
from typing import Iterable

import duckdb

from src.graph_models import GraphEdge, GraphNode, GraphProjectionManifest
from src.graph_taxonomy import LabelAssignment


class GraphStore:
    """为图投影提供 staging 写入和 READY 只读查询。"""

    def __init__(self, db_path: Path, connection: duckdb.DuckDBPyConnection, *, read_only: bool):
        self.db_path = Path(db_path)
        self._connection = connection
        self._read_only = read_only

    @classmethod
    def create_staging(cls, db_path: str | Path, manifest: GraphProjectionManifest) -> "GraphStore":
        if manifest.build_status != "STAGING":
            raise ValueError("staging store requires STAGING manifest")
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(path))
        store = cls(path, connection, read_only=False)
        store._create_schema()
        connection.execute("DELETE FROM graph_manifest")
        store._insert_manifest(manifest)
        return store

    @classmethod
    def open_ready(cls, db_path: str | Path, expected_source_hash: str) -> "GraphStore":
        path = Path(db_path)
        if not path.exists():
            raise ValueError("ready manifest not found")
        try:
            connection = duckdb.connect(str(path), read_only=True)
            tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
            required = {"graph_manifest", "graph_nodes", "graph_edges", "graph_label_assignments"}
            if not required.issubset(tables):
                raise ValueError("ready manifest not found")
            rows = connection.execute(
                "SELECT graph_version, source_artifact_hash, node_count, edge_count, build_status, "
                "built_at_utc, failure_reason FROM graph_manifest WHERE build_status = 'READY'"
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("ready manifest must contain exactly one READY row")
            if rows[0][1] != expected_source_hash.lower():
                raise ValueError("source hash mismatch")
            return cls(path, connection, read_only=True)
        except Exception:
            if 'connection' in locals():
                connection.close()
            raise

    def _create_schema(self) -> None:
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS graph_manifest (
                graph_version VARCHAR NOT NULL, source_artifact_hash VARCHAR NOT NULL,
                node_count INTEGER NOT NULL, edge_count INTEGER NOT NULL,
                build_status VARCHAR NOT NULL, built_at_utc TIMESTAMP NOT NULL,
                failure_reason VARCHAR
            );
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id VARCHAR PRIMARY KEY, node_type VARCHAR NOT NULL, label VARCHAR NOT NULL,
                properties_json VARCHAR NOT NULL, source_session_id BIGINT,
                source_card_id VARCHAR, source_turn_ids INTEGER[], derivation_version VARCHAR NOT NULL,
                review_status VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id VARCHAR PRIMARY KEY, source_node_id VARCHAR NOT NULL, target_node_id VARCHAR NOT NULL,
                edge_type VARCHAR NOT NULL, source_session_id BIGINT NOT NULL, source_turn_ids INTEGER[] NOT NULL,
                derivation_version VARCHAR NOT NULL, review_status VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graph_label_assignments (
                assignment_id VARCHAR PRIMARY KEY, source_node_id VARCHAR NOT NULL, label_type VARCHAR NOT NULL,
                canonical_label VARCHAR NOT NULL, assignment_source VARCHAR NOT NULL,
                review_status VARCHAR NOT NULL, taxonomy_version VARCHAR NOT NULL,
                source_session_id BIGINT NOT NULL, source_turn_ids INTEGER[] NOT NULL
            );
        """)
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info('graph_label_assignments')").fetchall()}
        missing = {"source_session_id", "source_turn_ids"} - columns
        if missing:
            if "source_session_id" in missing:
                self._connection.execute("ALTER TABLE graph_label_assignments ADD COLUMN source_session_id BIGINT")
            if "source_turn_ids" in missing:
                self._connection.execute("ALTER TABLE graph_label_assignments ADD COLUMN source_turn_ids INTEGER[]")
            existing = self._connection.execute("SELECT COUNT(*) FROM graph_label_assignments").fetchone()[0]
            if existing:
                # 旧版 assignment 没有 provenance；只允许从同一 sidecar 的真实
                # graph_nodes 回填，无法回填时拒绝继续，绝不制造默认来源。
                self._connection.execute("""
                    UPDATE graph_label_assignments AS a
                    SET source_session_id = n.source_session_id,
                        source_turn_ids = n.source_turn_ids
                    FROM graph_nodes AS n
                    WHERE a.source_node_id = n.node_id
                      AND a.source_session_id IS NULL
                """)
                unresolved = self._connection.execute("""
                    SELECT COUNT(*) FROM graph_label_assignments
                    WHERE source_session_id IS NULL OR source_turn_ids IS NULL
                """).fetchone()[0]
                if unresolved:
                    raise ValueError(
                        "legacy graph_label_assignments rows lack recoverable provenance; rebuild graph sidecar"
                    )

    def _insert_manifest(self, manifest: GraphProjectionManifest) -> None:
        self._connection.execute(
            "INSERT INTO graph_manifest VALUES (?, ?, ?, ?, ?, ?, ?)",
            [manifest.graph_version, manifest.source_artifact_hash, manifest.node_count,
             manifest.edge_count, manifest.build_status, manifest.built_at_utc, manifest.failure_reason],
        )

    def replace_projection(self, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge],
                           label_assignments: Iterable[LabelAssignment] = ()) -> None:
        if self._read_only:
            raise RuntimeError("graph store is read-only")
        nodes = list(nodes)
        edges = list(edges)
        label_assignments = list(label_assignments)
        if len({item.node_id for item in nodes}) != len(nodes):
            raise ValueError("duplicate node ID")
        if len({item.edge_id for item in edges}) != len(edges):
            raise ValueError("duplicate edge ID")
        node_ids = {item.node_id for item in nodes}
        missing_endpoints = sorted({endpoint for edge in edges for endpoint in (edge.source_node_id, edge.target_node_id) if endpoint not in node_ids})
        if missing_endpoints:
            raise ValueError(f"edge endpoint does not exist: {missing_endpoints[0]}")
        for item in label_assignments:
            if item.source_node_id not in node_ids:
                raise ValueError(f"label assignment source node does not exist: {item.source_node_id}")
        try:
            self._connection.execute("BEGIN TRANSACTION")
            self._connection.execute("DELETE FROM graph_nodes")
            self._connection.execute("DELETE FROM graph_edges")
            self._connection.execute("DELETE FROM graph_label_assignments")
            for item in nodes:
                self._connection.execute("INSERT INTO graph_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    item.node_id, item.node_type, item.label, json.dumps(item.properties, ensure_ascii=False),
                    item.source_session_id, item.source_card_id, item.source_turn_ids,
                    item.derivation_version, item.review_status,
                ])
            for item in edges:
                self._connection.execute("INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                    item.edge_id, item.source_node_id, item.target_node_id, item.edge_type,
                    item.source_session_id, item.source_turn_ids, item.derivation_version, item.review_status,
                ])
            for item in label_assignments:
                payload = "\x1f".join((item.source_node_id, item.label_type, item.canonical_label,
                                       item.taxonomy_version, str(item.source_session_id),
                                       ",".join(map(str, item.source_turn_ids))))
                assignment_id = "assignment:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
                self._connection.execute("INSERT INTO graph_label_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    assignment_id, item.source_node_id, item.label_type, item.canonical_label,
                    item.assignment_source, item.review_status, item.taxonomy_version,
                    item.source_session_id, item.source_turn_ids,
                ])
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def mark_ready(self) -> None:
        if self._read_only:
            raise RuntimeError("graph store is read-only")
        manifest = self.get_manifest()
        actual_node_count = self._connection.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
        actual_edge_count = self._connection.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        if (manifest.node_count, manifest.edge_count) != (actual_node_count, actual_edge_count):
            raise ValueError("manifest counts do not match graph projection counts")
        self._connection.execute("UPDATE graph_manifest SET build_status = 'READY', failure_reason = NULL")

    def get_manifest(self) -> GraphProjectionManifest:
        row = self._connection.execute("SELECT * FROM graph_manifest LIMIT 1").fetchone()
        if row is None:
            raise ValueError("graph manifest not found")
        built_at_utc = row[5]
        # DuckDB TIMESTAMP is timezone-naive; this column is contractually UTC.
        if built_at_utc.tzinfo is None:
            built_at_utc = built_at_utc.replace(tzinfo=timezone.utc)
        return GraphProjectionManifest(
            graph_version=row[0], source_artifact_hash=row[1], node_count=row[2], edge_count=row[3],
            build_status=row[4], built_at_utc=built_at_utc, failure_reason=row[6],
        )

    def list_nodes(self) -> list[GraphNode]:
        rows = self._connection.execute("SELECT * FROM graph_nodes ORDER BY node_id").fetchall()
        return [GraphNode(node_id=r[0], node_type=r[1], label=r[2], properties=json.loads(r[3]),
                          source_session_id=r[4], source_card_id=r[5], source_turn_ids=r[6] or [],
                          derivation_version=r[7], review_status=r[8]) for r in rows]

    def list_edges(self) -> list[GraphEdge]:
        rows = self._connection.execute("SELECT * FROM graph_edges ORDER BY edge_id").fetchall()
        return [GraphEdge(edge_id=r[0], source_node_id=r[1], target_node_id=r[2], edge_type=r[3],
                          source_session_id=r[4], source_turn_ids=r[5], derivation_version=r[6],
                          review_status=r[7]) for r in rows]

    def list_label_assignments(self) -> list[LabelAssignment]:
        rows = self._connection.execute(
            "SELECT source_node_id, label_type, canonical_label, assignment_source, review_status, "
            "taxonomy_version, source_session_id, source_turn_ids FROM graph_label_assignments "
            "ORDER BY source_node_id, label_type, canonical_label"
        ).fetchall()
        return [LabelAssignment(source_node_id=r[0], label_type=r[1], canonical_label=r[2],
                                assignment_source=r[3], review_status=r[4], taxonomy_version=r[5],
                                source_session_id=r[6], source_turn_ids=r[7]) for r in rows]

    def close(self) -> None:
        self._connection.close()
