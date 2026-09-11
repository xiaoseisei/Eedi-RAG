"""将图聚合结果安全桥接到已有 Card RAG 与精确对白审计。"""

from __future__ import annotations

import json
from typing import Any, Iterable

from pydantic import Field

from src.business_models import EvidencePointer, StrictBusinessModel
from src.generator_contract import EvidenceCatalogItem
from src.graph_analytics import RepresentativeEvidenceRef
from src.reranker import estimate_text_tokens


class AuthorizedGraphEvidenceBundle(StrictBusinessModel):
    pointers: list[EvidencePointer] = Field(min_length=1)
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    selected_parent_session_ids: list[int] = Field(min_length=1, max_length=5)
    selected_windows: list[dict[str, Any]] = Field(min_length=1)
    assembled_context: str = Field(min_length=1)
    evidence_catalog: dict[str, EvidenceCatalogItem]
    generator_context_token_count: int = Field(ge=1, le=4000)
    audit_status: str = "AUDITED_100_VERIFIED"


class GraphEvidenceBridge:
    """只允许 aggregate representative refs 指定的 graph/source 证据进入上下文。"""

    def __init__(self, graph_store: Any, source_storage: Any, retriever: Any | None = None,
                 assembler: Any | None = None) -> None:
        self.graph_store = graph_store
        self.source_storage = source_storage
        self.retriever = retriever
        self.assembler = assembler

    def _turns(self, session_id: int, turn_ids: list[int]) -> list[dict[str, Any]]:
        if hasattr(self.source_storage, "get_dialogue_turns"):
            rows = self.source_storage.get_dialogue_turns(session_id, turn_ids)
        else:
            cursor = self.source_storage.execute(
                "SELECT turn_id,speaker,is_tutor,text,talk_moves,is_greeting_or_noise "
                "FROM session_dialogue_turns WHERE intervention_id=? AND turn_id IN ("
                + ",".join("?" for _ in turn_ids) + ") ORDER BY turn_id",
                [session_id, *turn_ids],
            )
            rows = [dict(zip(("turn_id", "speaker", "is_tutor", "text", "talk_moves", "is_greeting_or_noise"), row))
                    for row in cursor.fetchall()]
        actual = {int(row["turn_id"]): row for row in rows}
        if set(actual) != set(turn_ids):
            raise ValueError("authorized source turns are missing")
        return [actual[turn_id] for turn_id in turn_ids]

    def materialize(self, refs: Iterable[RepresentativeEvidenceRef], *, raw_query: str = "") -> AuthorizedGraphEvidenceBundle:
        refs = list(refs)
        if not refs:
            raise ValueError("NO_EVIDENCE: no authorized representative exists")
        nodes = {node.node_id: node for node in self.graph_store.list_nodes()}
        pointers: list[EvidencePointer] = []
        windows: list[dict[str, Any]] = []
        graph_paths: list[dict[str, Any]] = []
        allowed_sessions = sorted({int(ref.session_id) for ref in refs})
        seen: set[tuple[int, int]] = set()
        for ref in refs:
            event = nodes.get(ref.source_event_id)
            if event is None or event.source_session_id != ref.session_id:
                raise ValueError(f"authorized event not found: {ref.source_event_id}")
            if not set(ref.turn_ids).issubset(set(event.source_turn_ids)):
                raise ValueError(f"turn is not authorized by aggregate event: {ref.source_event_id}")
            turns = self._turns(ref.session_id, list(ref.turn_ids))
            for turn in turns:
                expected_tutor = ref.perspective == "TUTOR"
                if bool(turn.get("is_tutor")) != expected_tutor:
                    raise ValueError(f"authorized turn has wrong speaker role: {ref.session_id}/{turn['turn_id']}")
                key = (ref.session_id, int(turn["turn_id"]))
                if key in seen:
                    continue
                seen.add(key)
                evidence_id = f"E{len(pointers) + 1:03d}"
                pointers.append(EvidencePointer(
                    evidence_id=evidence_id, source_event_id=ref.source_event_id,
                    source_card_id=event.source_card_id, session_id=ref.session_id,
                    turn_id=int(turn["turn_id"]), speaker="tutor" if expected_tutor else "student",
                    exact_quote=str(turn["text"]), subject_path=ref.subject_path, verified=True,
                ))
            windows.append({"session_id": ref.session_id, "source_event_id": ref.source_event_id,
                            "source_turn_ids": list(ref.turn_ids),
                            "turns": [dict(turn, session_id=ref.session_id) for turn in turns]})
            graph_paths.append({"source_event_id": ref.source_event_id, "session_id": ref.session_id,
                                "relation": "EVIDENCED_BY", "turn_ids": list(ref.turn_ids)})

        if self.retriever is not None:
            retrieval = self.retriever.retrieve_multi_perspective_rrf(
                raw_query or "authorized graph evidence", top_k_each=3, fetch_evidence=True,
                allowed_session_ids=allowed_sessions,
            )
            for lane in ("misconceptions", "strategies"):
                if any(int(item.get("metadata", {}).get("session_id", -1)) not in allowed_sessions
                       for item in retrieval.get(lane, [])):
                    raise ValueError("Card RAG returned evidence outside authorized Session scope")

        lines = ["【授权图证据】"]
        for pointer in pointers:
            lines.append(f"[{pointer.evidence_id}] Session {pointer.session_id} Turn {pointer.turn_id} {pointer.speaker}: {pointer.exact_quote}")
        lines.extend(json.dumps(window, ensure_ascii=False, sort_keys=True) for window in windows)
        context = "\n".join(lines)
        catalog = {pointer.evidence_id: EvidenceCatalogItem(
            evidence_id=pointer.evidence_id, session_id=pointer.session_id,
            turn_id=pointer.turn_id, speaker=pointer.speaker, quote_text=pointer.exact_quote,
        ) for pointer in pointers}
        return AuthorizedGraphEvidenceBundle(
            pointers=pointers, graph_paths=graph_paths, selected_parent_session_ids=allowed_sessions[:5],
            selected_windows=windows, assembled_context=context, evidence_catalog=catalog,
            generator_context_token_count=estimate_text_tokens(context),
        )


__all__ = ["AuthorizedGraphEvidenceBundle", "GraphEvidenceBridge"]
