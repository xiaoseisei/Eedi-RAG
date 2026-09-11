"""Deterministic projection of one tutoring session and its two source cards."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from src.graph_models import GraphEdge, GraphNode
from src.graph_taxonomy import (DifficultySignal, LabelAssignment,
                                assign_misconception_label, assign_strategy_labels,
                                assign_exam_error_label, assign_question_function,
                                assign_content_opportunity, derive_difficulty_signals, load_taxonomies)


@dataclass(frozen=True)
class SessionGraphProjection:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    label_assignments: list[LabelAssignment] | None = None
    difficulty_signals: list[DifficultySignal] | None = None


def _normal(value: Any) -> str:
    if value is None:
        raise ValueError("graph key cannot be null")
    value = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_subject_path(value: str) -> str:
    return re.sub(r"\s*/\s*", "/", _normal(value))


def normalize_strategy_category(value: str) -> str:
    return re.sub(r"[^\w]+", "-", _normal(value), flags=re.UNICODE).strip("-")


def normalize_talk_move(value: str) -> str:
    normalized = _normal(value).replace("_", " ").replace("-", " ")
    variants = {"question": "questioning", "ask": "questioning", "prompt": "questioning"}
    return variants.get(normalized, re.sub(r"\s+", "-", normalized))


def stable_node_id(node_type: str, *parts: str | int) -> str:
    prefixes = {"SESSION": "session", "QUESTION": "question", "SUBJECT": "subject",
                "MISCONCEPTION_EVENT": "misconception-event", "TUTOR_STRATEGY_EVENT": "tutor-strategy-event",
                "MISCONCEPTION_CONCEPT": "misconception-concept", "STRATEGY_CONCEPT": "strategy-concept",
                "TURN": "turn", "TALK_MOVE": "talk-move"}
    prefix = prefixes.get(node_type, _normal(node_type).replace(" ", "-"))
    return ":".join([prefix, *[str(part) for part in parts]])


def stable_edge_id(edge_type: str, source_id: str, target_id: str) -> str:
    payload = "\x1f".join((edge_type, source_id, target_id)).encode("utf-8")
    return f"edge:{hashlib.sha256(payload).hexdigest()}"


def _required(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field {key}")
    return value


def _turn_map(turns: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for turn in turns:
        turn_id = int(_required(turn, "turn_id"))
        if turn_id in result:
            raise ValueError(f"duplicate turn_id {turn_id}")
        if "is_tutor" not in turn:
            raise ValueError("missing required field is_tutor")
        result[turn_id] = turn
    return result


def _edge(kind: str, source: GraphNode, target: GraphNode, session_id: int, turn_ids: list[int], version: str) -> GraphEdge:
    if not turn_ids:
        raise ValueError(f"edge {kind} requires source turns")
    return GraphEdge(edge_id=stable_edge_id(kind, source.node_id, target.node_id), source_node_id=source.node_id,
                     target_node_id=target.node_id, edge_type=kind, source_session_id=session_id,
                     source_turn_ids=turn_ids, derivation_version=version, review_status="AUTO_ASSIGNED")


def project_session_graph(session_row: dict[str, Any], misconception_row: dict[str, Any],
                          strategy_row: dict[str, Any], turns: Iterable[dict[str, Any]],
                          *, graph_version: str = "session-card-graph-v1",
                          taxonomies: dict[str, Any] | None = None) -> SessionGraphProjection:
    session_id = int(_required(session_row, "intervention_id"))
    question_id = int(_required(session_row, "question_id"))
    subject = normalize_subject_path(_required(session_row, "subject_path"))
    turn_by_id = _turn_map(turns)
    all_turn_ids = sorted(turn_by_id)
    if not all_turn_ids:
        raise ValueError("session must have authoritative turns")
    for turn_id, turn in turn_by_id.items():
        if "intervention_id" in turn and int(turn["intervention_id"]) != session_id:
            raise ValueError(f"turn {turn_id} belongs to another session")

    for name, card in (("misconception", misconception_row), ("strategy", strategy_row)):
        if int(_required(card, "session_id")) != session_id or int(_required(card, "question_id")) != question_id:
            raise ValueError(f"{name} card foreign key mismatch: session_id/question_id")
        if not _required(card, "source_turn_ids"):
            raise ValueError(f"{name} card requires source_turn_ids")

    def pointer_ids(card: dict[str, Any], role: str) -> list[int]:
        ids = [int(x) for x in card["source_turn_ids"]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{card.get('chunk_id', 'card')} source_turn_ids must be unique")
        for turn_id in ids:
            if turn_id not in turn_by_id:
                raise ValueError(f"source turn {turn_id} does not exist")
            if bool(turn_by_id[turn_id]["is_tutor"]) != (role == "tutor"):
                raise ValueError(f"{card.get('chunk_id', 'card')} source turn {turn_id} must be {role}")
        return ids

    misconception_turns = pointer_ids(misconception_row, "student")
    strategy_turns = pointer_ids(strategy_row, "tutor")
    mc_id = str(_required(misconception_row, "chunk_id"))
    st_id = str(_required(strategy_row, "chunk_id"))
    session = GraphNode(node_id=stable_node_id("SESSION", session_id), node_type="SESSION", label=f"session {session_id}",
                        properties=dict(session_row), source_session_id=session_id, source_turn_ids=all_turn_ids,
                        derivation_version=graph_version, review_status="SOURCE")
    question = GraphNode(node_id=stable_node_id("QUESTION", question_id), node_type="QUESTION", label=f"question {question_id}",
                         properties={"question_id": question_id, "question_text": session_row.get("question_text")},
                         source_session_id=session_id, source_turn_ids=all_turn_ids, derivation_version=graph_version, review_status="SOURCE")
    subject_node = GraphNode(node_id=stable_node_id("SUBJECT", subject), node_type="SUBJECT", label=subject,
                             properties={"subject_path": session_row["subject_path"]}, source_session_id=session_id,
                             source_turn_ids=all_turn_ids, derivation_version=graph_version, review_status="AUTO_ASSIGNED")
    mc = GraphNode(node_id=stable_node_id("MISCONCEPTION_EVENT", mc_id), node_type="MISCONCEPTION_EVENT",
                   label=str(misconception_row.get("misconception_name") or mc_id), properties=dict(misconception_row),
                   source_session_id=session_id, source_card_id=mc_id, source_turn_ids=misconception_turns,
                   derivation_version=graph_version, review_status="SOURCE")
    st = GraphNode(node_id=stable_node_id("TUTOR_STRATEGY_EVENT", st_id), node_type="TUTOR_STRATEGY_EVENT",
                   label=str(strategy_row.get("strategy_category") or st_id), properties=dict(strategy_row),
                   source_session_id=session_id, source_card_id=st_id, source_turn_ids=strategy_turns,
                   derivation_version=graph_version, review_status="SOURCE")
    turn_nodes = [GraphNode(node_id=stable_node_id("TURN", session_id, turn_id), node_type="TURN",
                           label=f"turn {turn_id}", properties=dict(turn), source_session_id=session_id,
                           source_turn_ids=[turn_id], derivation_version=graph_version, review_status="SOURCE")
                  for turn_id, turn in sorted(turn_by_id.items())]
    moves = sorted({normalize_talk_move(move) for card in (misconception_row, strategy_row)
                    for move in (card.get("talk_moves") or [])})
    move_nodes = [GraphNode(node_id=stable_node_id("TALK_MOVE", move), node_type="TALK_MOVE", label=move,
                            properties={"normalized_talk_move": move}, source_session_id=session_id,
                            source_turn_ids=all_turn_ids, derivation_version=graph_version, review_status="AUTO_ASSIGNED")
                  for move in moves]
    taxonomies = taxonomies or load_taxonomies()
    misconception_assignment = assign_misconception_label(
        misconception_row, taxonomies["MISCONCEPTION"], source_node_id=mc.node_id
    )
    strategy_assignments = assign_strategy_labels(
        strategy_row, taxonomies["TUTOR_STRATEGY"], source_node_id=st.node_id
    )
    misconception_concept = GraphNode(
        node_id=stable_node_id("MISCONCEPTION_CONCEPT", misconception_assignment.canonical_label),
        node_type="MISCONCEPTION_CONCEPT",
        label=misconception_assignment.canonical_label,
        properties={
            "canonical_label": misconception_assignment.canonical_label,
            "label_type": misconception_assignment.label_type,
            "taxonomy_version": misconception_assignment.taxonomy_version,
        },
        source_session_id=session_id,
        source_turn_ids=list(misconception_assignment.source_turn_ids),
        derivation_version=graph_version,
        review_status="AUTO_ASSIGNED",
    )
    strategy_concepts = [
        GraphNode(
            node_id=stable_node_id("STRATEGY_CONCEPT", assignment.canonical_label),
            node_type="STRATEGY_CONCEPT",
            label=assignment.canonical_label,
            properties={
                "canonical_label": assignment.canonical_label,
                "label_type": assignment.label_type,
                "taxonomy_version": assignment.taxonomy_version,
            },
            source_session_id=session_id,
            source_turn_ids=list(assignment.source_turn_ids),
            derivation_version=graph_version,
            review_status="AUTO_ASSIGNED",
        )
        for assignment in strategy_assignments
    ]
    nodes = [session, question, subject_node, mc, st, misconception_concept, *strategy_concepts, *turn_nodes, *move_nodes]
    edges = [_edge("ANCHORS", session, question, session_id, all_turn_ids, graph_version),
             _edge("ABOUT_SUBJECT", session, subject_node, session_id, all_turn_ids, graph_version),
             _edge("HAS_MISCONCEPTION", session, mc, session_id, misconception_turns, graph_version),
             _edge("HAS_STRATEGY", session, st, session_id, strategy_turns, graph_version),
             _edge("TYPE_OF", mc, misconception_concept, session_id, misconception_turns, graph_version)]
    edges.extend(
        _edge("TYPE_OF", st, concept, session_id, strategy_turns, graph_version)
        for concept in strategy_concepts
    )
    edges.append(_edge(
        "CO_OCCURS_WITH", mc, st, session_id,
        sorted(set(misconception_turns + strategy_turns)), graph_version,
    ))
    edges += [_edge("EVIDENCED_BY", mc, turn_nodes[all_turn_ids.index(t)], session_id, [t], graph_version) for t in misconception_turns]
    edges += [_edge("EVIDENCED_BY", st, turn_nodes[all_turn_ids.index(t)], session_id, [t], graph_version) for t in strategy_turns]
    edges += [_edge("HAS_TURN", session, turn, session_id, turn.source_turn_ids, graph_version) for turn in turn_nodes]
    for move_node in move_nodes:
        for card in (mc, st):
            if move_node.node_id.rsplit(":", 1)[1] in {normalize_talk_move(x) for x in (card.properties.get("talk_moves") or [])}:
                edges.append(_edge("USES_TALK_MOVE", card, move_node, session_id, card.source_turn_ids, graph_version))
    assignments = [misconception_assignment]
    assignments.append(assign_exam_error_label(misconception_row, taxonomies["EXAM_ERROR"], source_node_id=mc.node_id))
    assignments.append(assign_content_opportunity(misconception_row, taxonomies["CONTENT_OPPORTUNITY"], source_node_id=mc.node_id))
    assignments.extend(strategy_assignments)
    student_turns = [turn for turn_id, turn in sorted(turn_by_id.items()) if not bool(turn["is_tutor"]) and not bool(turn.get("is_greeting_or_noise", False))]
    for turn in student_turns:
        turn_id = int(turn["turn_id"])
        assignments.append(assign_question_function(
            str(turn.get("text", "")), source_node_id=stable_node_id("TURN", session_id, turn_id),
            source_session_id=session_id, source_turn_ids=[turn_id],
        ))
    difficulty = derive_difficulty_signals(session_row, list(turn_by_id.values()))
    assignments.extend(
        LabelAssignment(
            source_node_id=signal.source_node_id,
            label_type="DIFFICULTY_SIGNAL",
            canonical_label=signal.signal,
            assignment_source="deterministic_rules",
            review_status=signal.review_status,
            taxonomy_version=signal.taxonomy_version,
            source_session_id=signal.source_session_id,
            source_turn_ids=signal.source_turn_ids,
        )
        for signal in difficulty
    )
    return SessionGraphProjection(nodes=nodes, edges=edges, label_assignments=assignments,
                                  difficulty_signals=difficulty)
