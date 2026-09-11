"""Build the session/card graph sidecar from a read-only source DuckDB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from src.graph_models import GraphProjectionManifest
from src.graph_projection import project_session_graph
from src.graph_store import GraphStore


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(connection, table: str, columns: str):
    cursor = connection.execute(f"SELECT {columns} FROM {table}")
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _deduplicate_nodes(nodes):
    """合并合法的跨会话共享节点，并保留全部来源引用。"""

    shared_types = {"QUESTION", "SUBJECT", "TALK_MOVE", "MISCONCEPTION_CONCEPT", "STRATEGY_CONCEPT"}
    by_id = {}
    for node in nodes:
        existing = by_id.get(node.node_id)
        if existing is None:
            by_id[node.node_id] = node
            continue
        if node.node_type not in shared_types or existing.node_type != node.node_type:
            raise ValueError(f"duplicate node ID: {node.node_id}")
        if node.node_type == "QUESTION":
            # 同一个 Eedi question_id 可能有不同学生姓名实例化的题干。
            # 保留首个确定性代表，同时完整记录每个真实文本变体，避免覆盖来源事实。
            variants = list(existing.properties.get("source_question_texts", []))
            for text in (existing.properties.get("question_text"), node.properties.get("question_text")):
                if text is not None and text not in variants:
                    variants.append(text)
            merged_properties = dict(existing.properties)
            merged_properties["source_question_texts"] = variants
            refs = list(existing.properties.get("source_refs", []))
            if not refs:
                refs.append({
                    "source_session_id": existing.source_session_id,
                    "source_turn_ids": list(existing.source_turn_ids),
                })
            refs.append({
                "source_session_id": node.source_session_id,
                "source_turn_ids": list(node.source_turn_ids),
            })
            merged_properties["source_refs"] = refs
            by_id[node.node_id] = existing.model_copy(update={"properties": merged_properties})
            continue
        elif node.node_type == "SUBJECT":
            if existing.properties.get("subject_path", "").casefold() != node.properties.get("subject_path", "").casefold():
                raise ValueError(f"conflicting subject node properties: {node.node_id}")

        refs = list(existing.properties.get("source_refs", []))
        if not refs:
            refs.append({
                "source_session_id": existing.source_session_id,
                "source_turn_ids": list(existing.source_turn_ids),
            })
        refs.append({
            "source_session_id": node.source_session_id,
            "source_turn_ids": list(node.source_turn_ids),
        })
        merged_properties = dict(existing.properties)
        merged_properties["source_refs"] = refs
        by_id[node.node_id] = existing.model_copy(update={"properties": merged_properties})
    return [by_id[node_id] for node_id in sorted(by_id)]


def build(source_db: Path, output: Path, graph_version: str) -> None:
    source_hash = sha256_file(source_db)
    store = None
    try:
        connection = duckdb.connect(str(source_db), read_only=True)
        try:
            sessions = rows(connection, "tutoring_sessions", "intervention_id, question_id, subject_path, question_text")
            turns = rows(connection, "session_dialogue_turns", "intervention_id, turn_id, speaker, is_tutor, text, talk_moves")
            misconceptions = rows(connection, "misconception_chunks", "chunk_id, session_id, question_id, subject_path, misconception_name, error_choice, deep_mechanism, confusion_triggers, verbatim_student_quotes, source_turn_ids")
            strategies = rows(connection, "tutor_strategy_chunks", "chunk_id, session_id, question_id, subject_path, pedagogical_goal, strategy_category, key_aha_question, scaffolding_steps, analogy_or_metaphor, talk_moves, resolution_outcome, source_turn_ids")
        finally:
            connection.close()
        by_session_turns = {}
        for turn in turns:
            by_session_turns.setdefault(turn["intervention_id"], []).append(turn)
        by_mc = {}
        for row in misconceptions:
            if row["session_id"] in by_mc:
                raise ValueError(f"multiple misconception cards for session {row['session_id']}")
            by_mc[row["session_id"]] = row
        by_st = {}
        for row in strategies:
            if row["session_id"] in by_st:
                raise ValueError(f"multiple strategy cards for session {row['session_id']}")
            by_st[row["session_id"]] = row
        missing_sessions = [
            session["intervention_id"] for session in sessions
            if session["intervention_id"] not in by_mc
            or session["intervention_id"] not in by_st
            or session["intervention_id"] not in by_session_turns
        ]
        if missing_sessions:
            raise ValueError(f"sessions missing paired cards or turns: {missing_sessions[:10]}")
        projections = [project_session_graph(session, by_mc[session["intervention_id"]], by_st[session["intervention_id"]], by_session_turns[session["intervention_id"]], graph_version=graph_version) for session in sessions]
        nodes = _deduplicate_nodes([node for projection in projections for node in projection.nodes])
        edges = [edge for projection in projections for edge in projection.edges]
        label_assignments = [
            assignment
            for projection in projections
            for assignment in (projection.label_assignments or [])
        ]
        manifest = GraphProjectionManifest(graph_version=graph_version, source_artifact_hash=source_hash,
                                           node_count=len(nodes), edge_count=len(edges), build_status="STAGING",
                                           built_at_utc=datetime.now(timezone.utc))
        store = GraphStore.create_staging(output, manifest)
        store.replace_projection(nodes, edges, label_assignments)
        if sha256_file(source_db) != source_hash:
            raise RuntimeError("source SHA-256 changed during build")
        store.mark_ready()
        store.close()
    except Exception as exc:
        if store is not None:
            store.close()
        failed = GraphProjectionManifest(graph_version=graph_version, source_artifact_hash=source_hash,
                                         node_count=0, edge_count=0, build_status="STAGING",
                                         built_at_utc=datetime.now(timezone.utc))
        failed_store = GraphStore.create_staging(output, failed)
        failed_store._connection.execute("UPDATE graph_manifest SET build_status='FAILED', failure_reason=?", [str(exc)])
        failed_store.close()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-version", required=True)
    args = parser.parse_args()
    try:
        build(args.source_db, args.output, args.graph_version)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "READY", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
