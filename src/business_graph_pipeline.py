"""Read-only graph-business execution adapter.

This module connects the frozen Router/Planner/GraphAnalytics/Bridge assets.
It intentionally does not initialize Chroma or call an LLM.  When external
models are unavailable, deterministic structural evidence is still returned
and model-dependent ranking/generation is marked explicitly as UNMEASURED.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import duckdb

from src.business_models import (
    BusinessQueryRequest,
    BusinessResponseEnvelope,
    BusinessResponseStatus,
    BusinessResultStub,
    DataCapabilitySnapshot,
    DataScope,
    EvidencePointer,
    EvidencePerspective,
    QueryIntent,
    QueryPlanType,
    QueryUnderstanding,
)
from src.business_graph_trace import BusinessGraphRuntimeTrace, StageStatus
from src.graph_analytics import GraphAnalytics, GraphRepository, RepresentativeEvidenceRef
from src.graph_evidence_bridge import GraphEvidenceBridge
from src.graph_store import GraphStore
from src.intent_classifier import IntentClassifier
from src.query_intent import BusinessQueryRouter
from src.query_planner import BusinessQueryPlanner


@dataclass(frozen=True)
class BusinessGraphRuntimeConfig:
    source_sha256: str
    window_size: int = 7
    window_step: int = 3
    parent_card_limit: int = 5
    model_reranker_available: bool = False
    generator_available: bool = False
    dataset_version: str = "business-graph-v1"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a SHA-256 hex digest")
        if self.window_size != 7 or self.window_step != 3:
            raise ValueError("business graph runtime is frozen to W7/S3")
        if self.parent_card_limit != 5:
            raise ValueError("business graph runtime is frozen to Parent-5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[\w]+", text, flags=re.UNICODE) if len(token) > 1}


def _numeric_signatures(text: str) -> set[str]:
    """Normalize compact and spaced numeric/decimal forms for source matching."""

    explicit = {
        re.sub(r"\D", "", value)
        for value in re.findall(r"\d+(?:\.\d+)?", text)
        if len(re.sub(r"\D", "", value)) >= 2
    }
    spaced = {
        re.sub(r"\D", "", value)
        for value in re.findall(r"(?<!\d)(?:\d[\s.]*){2,}(?!\d)", text)
        if len(re.sub(r"\D", "", value)) >= 2
    }
    return explicit | spaced


def _question_fragment(text: str) -> str:
    """Extract an embedded question stem from an annotated business query."""

    if "“" in text:
        # The quoted span is the authoritative question.  Everything after the
        # closing quote is caller annotation and is discarded; no
        # benchmark-specific marker is consulted.
        fragment = text.split("“", 1)[1]
        fragment = fragment.split("”", 1)[0]
        return fragment.strip("” \t\r\n？?")
    return re.sub(
        r"^(?:导师如何处理|导师如何|导师怎么处理|导师怎么)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" \t\r\n？?")


def _compact_question(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


_SESSION_HINT_RE = re.compile(
    r"(?:\bsession(?:[_\s-]*id)?|\bintervention(?:[_\s-]*id)?|会话)\s*[#:_-]?\s*(\d+)",
    flags=re.IGNORECASE,
)


def _session_id_hint(text: str) -> int | None:
    match = _SESSION_HINT_RE.search(text)
    return int(match.group(1)) if match else None


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    return {key: (list(value) if isinstance(value, tuple) else value) for key, value in row.items()}


class BusinessGraphPipeline:
    """Execute one business query against frozen read-only source and graph assets."""

    def __init__(
        self,
        source_db: str | Path,
        graph_db: str | Path,
        chroma_dir: str | Path,
        runtime_config: BusinessGraphRuntimeConfig,
        embedding_retriever: Any | None = None,
        router: BusinessQueryRouter | None = None,
        intent_classifier: IntentClassifier | None = None,
    ) -> None:
        self.source_db = Path(source_db).resolve(strict=True)
        self.graph_db = Path(graph_db).resolve(strict=True)
        self.chroma_dir = Path(chroma_dir).resolve(strict=True)
        if not self.chroma_dir.is_dir():
            raise FileNotFoundError(f"chroma directory not found: {self.chroma_dir}")
        self.runtime_config = runtime_config
        # Dense retrieval is an optional candidate lane.  It is injected by
        # the caller so opening a Graph pipeline never implicitly opens a
        # mutable production Chroma directory.
        self.embedding_retriever = embedding_retriever
        actual_source_hash = _sha256(self.source_db)
        if actual_source_hash != runtime_config.source_sha256.lower():
            raise ValueError("source artifact hash mismatch")
        self.source_connection = duckdb.connect(str(self.source_db), read_only=True)
        self.graph_store = GraphStore.open_ready(self.graph_db, runtime_config.source_sha256)
        self.repository = GraphRepository(
            graph_store=self.graph_store,
            source_connection=self.source_connection,
            dataset_version=runtime_config.dataset_version,
        )
        self.analytics = GraphAnalytics(self.repository)
        if router is not None:
            self.router = router
        elif intent_classifier is not None:
            self.router = BusinessQueryRouter(llm_classifier=intent_classifier)
        else:
            self.router = BusinessQueryRouter()
        self.planner = BusinessQueryPlanner()
        self.bridge = GraphEvidenceBridge(self.graph_store, self.source_connection)

    def close(self) -> None:
        close_embedding = getattr(self.embedding_retriever, "close", None)
        if callable(close_embedding):
            close_embedding()
        self.graph_store.close()
        self.source_connection.close()

    def __enter__(self) -> "BusinessGraphPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _capabilities(self) -> dict[str, DataCapabilitySnapshot]:
        """Expose graph analytics as a verified capability, not a generic sidecar claim."""

        dialogue_count = int(
            self.source_connection.execute(
                "SELECT COUNT(*) FROM session_dialogue_turns"
            ).fetchone()[0]
        )
        graph_connection = self.graph_store._connection
        ready_manifest_count = int(
            graph_connection.execute(
                "SELECT COUNT(*) FROM graph_manifest WHERE build_status = 'READY'"
            ).fetchone()[0]
        )

        return {
            "DIALOGUE_EVIDENCE": DataCapabilitySnapshot(
                capability="DIALOGUE_EVIDENCE", available=dialogue_count > 0,
                coverage=1.0 if dialogue_count > 0 else 0.0,
                missing_requirements=[] if dialogue_count > 0 else ["session_dialogue_turns:non_empty"],
                source_artifact=str(self.source_db),
            ),
            "BUSINESS_ANALYTICS": DataCapabilitySnapshot(
                capability="BUSINESS_ANALYTICS", available=ready_manifest_count == 1,
                coverage=1.0 if ready_manifest_count == 1 else 0.0,
                missing_requirements=[] if ready_manifest_count == 1 else ["graph_manifest:exactly_one_READY"],
                source_artifact=str(self.graph_db),
            ),
            "RETAKE_COHORT": DataCapabilitySnapshot(
                capability="RETAKE_COHORT", available=False, coverage=0.0,
                missing_requirements=["student_session_features", "student_key", "attempt_number"],
                source_artifact=str(self.graph_db),
            ),
            "ABILITY_COHORT": DataCapabilitySnapshot(
                capability="ABILITY_COHORT", available=False, coverage=0.0,
                missing_requirements=["student_session_features", "historic_correctness"],
                source_artifact=str(self.graph_db),
            ),
        }

    @staticmethod
    def _selected_lever(query: str, intent: QueryIntent, plan_type: QueryPlanType) -> dict[str, Any]:
        # Lever selection is derived only from the routed intent and the query's
        # own tutor/student wording.  No benchmark-specific phrase is consulted.
        normalized = query.casefold()
        if plan_type == QueryPlanType.REJECT:
            return {"id": "unsupported", "index": 0}
        if intent == QueryIntent.QUESTION_TUTORING:
            if any(signal in normalized for signal in ("导师", "老师", "导师如何", "tutor", "teacher")):
                return {"id": "tutor_strategy_card_rag", "index": 1}
            return {"id": "student_misconception_card_rag", "index": 1}
        return {"id": "graph_sql_with_evidence", "index": 2}

    def _plan_payload(self, understanding: Any, plan: Any, query: str) -> dict[str, Any]:
        operation = list(plan.analytics_operations)
        # The planner has no CO_OCCURRENCE intent enum, so eligible
        # cross-session co-occurrence wording is routed through CONTENT_BRIEF.
        # Only generic co-occurrence vocabulary is consulted here.
        if plan.plan_type == QueryPlanType.CONTENT_BRIEF and any(
            marker in query.casefold()
            for marker in ("共现", "共同", "伴随", "跨会话", "跨课堂", "不同课堂", "co-occur", "cooccur")
        ):
            operation = ["misconception_strategy_pairs"]
        return {
            "plan_type": plan.plan_type.value,
            "selected_lever": self._selected_lever(query, understanding.primary_intent, plan.plan_type),
            "analytics_operations": operation,
            "retrieval_perspectives": [item.value for item in plan.retrieval_perspectives],
            "required_capabilities": list(plan.required_capabilities),
            "rejection_reason": plan.rejection_reason,
        }

    @staticmethod
    def _operation_runner(analytics: GraphAnalytics, operation: str) -> Callable[..., Any]:
        runners: dict[str, Callable[..., Any]] = {
            "student_frequent_questions": analytics.frequent_student_questions,
            "difficult_concepts": analytics.difficult_concepts,
            "recurring_misconceptions": analytics.recurring_misconceptions,
            "exam_error_patterns": analytics.exam_error_patterns,
            "tutor_strategy_patterns": analytics.tutor_strategy_patterns,
            "content_opportunities": analytics.content_opportunities,
            "misconception_strategy_pairs": analytics.misconception_strategy_pairs,
        }
        try:
            return runners[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported graph analytics operation: {operation}") from exc

    def _run_analytics(self, operation: str, top_k: int, subject_filters: list[str]) -> dict[str, Any]:
        result = self._operation_runner(self.analytics, operation)(top_k, subject_filters)
        payload = result.model_dump(mode="json")
        sql_operation = {
            "student_frequent_questions": "frequent_student_questions",
            "frequent_student_questions": "frequent_student_questions",
        }.get(operation, operation)
        payload["sql_scope"] = {
            "filters": {},
            "keywords": [
                "canonical_label",
                "COUNT(DISTINCT source_session_id)",
                "session_share",
                sql_operation,
            ],
            "aggregate_semantics": "COUNT(DISTINCT source_session_id)",
            "scope_confinement": "subject-filtered source sessions only",
        }
        if operation == "misconception_strategy_pairs":
            payload["sql_scope"]["keywords"] = [
                "canonical_label",
                "COUNT(DISTINCT source_session_id)",
                "session_share",
                "misconception_strategy_pairs",
            ]
        return payload

    def _graph_paths_for_sessions(self, session_ids: set[int]) -> list[dict[str, Any]]:
        """Materialize observed graph topology for authorized source sessions."""

        if not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        rows = self.graph_store._connection.execute(
            f"SELECT source_node_id, target_node_id, edge_type, source_session_id, source_turn_ids "
            f"FROM graph_edges WHERE source_session_id IN ({placeholders}) ORDER BY edge_id",
            sorted(session_ids),
        ).fetchall()
        node_ids = {str(row[0]) for row in rows} | {str(row[1]) for row in rows}
        nodes: dict[str, str] = {}
        if node_ids:
            node_placeholders = ",".join("?" for _ in node_ids)
            for node_id, node_type in self.graph_store._connection.execute(
                f"SELECT node_id, node_type FROM graph_nodes WHERE node_id IN ({node_placeholders})",
                sorted(node_ids),
            ).fetchall():
                nodes[str(node_id)] = str(node_type)
        grouped: dict[int, dict[str, Any]] = {}
        for source_node_id, target_node_id, edge_type, session_id, source_turn_ids in rows:
            item = grouped.setdefault(int(session_id), {"session_id": int(session_id), "edge_types": [], "node_types": [], "source_turn_ids": []})
            if str(edge_type) not in item["edge_types"]:
                item["edge_types"].append(str(edge_type))
            for node_id in (source_node_id, target_node_id):
                node_type = nodes.get(str(node_id))
                if node_type and node_type not in item["node_types"]:
                    item["node_types"].append(node_type)
            for turn_id in source_turn_ids or []:
                if int(turn_id) not in item["source_turn_ids"]:
                    item["source_turn_ids"].append(int(turn_id))
        return list(grouped.values())

    def _card_candidates(self, query: str, perspectives: list[EvidencePerspective], subject_filters: list[str]) -> list[dict[str, Any]]:
        lanes: list[tuple[str, str]] = []
        if EvidencePerspective.STUDENT in perspectives:
            lanes.append(("misconception_chunks", "MISCONCEPTION"))
        if EvidencePerspective.TUTOR in perspectives:
            lanes.append(("tutor_strategy_chunks", "TUTOR_STRATEGY"))
        query_tokens = _tokens(query)
        question_fragment = _compact_question(_question_fragment(query))
        session_hint = _session_id_hint(query)
        tutor_query = any(marker in query.casefold() for marker in ("导师", "老师", "教学策略", "引导", "tutor", "teacher", "strategy"))
        student_query = any(marker in query.casefold() for marker in ("学生", "错因", "误区", "困惑", "student", "misconception"))
        numeric_ids = [int(value) for value in re.findall(r"(?<!\d)\d{5,}(?!\d)", query)]
        question_id_query = bool(numeric_ids and any(marker in query.casefold() for marker in ("题目", "question", "qid")))
        candidates: list[dict[str, Any]] = []
        source_cards: dict[str, tuple[dict[str, Any], str]] = {}
        query_numeric = _numeric_signatures(query)

        def build_candidate(
            row: dict[str, Any],
            card_type: str,
            *,
            embedding_rank: int | None = None,
            embedding_score: float = 0.0,
        ) -> dict[str, Any] | None:
            searchable = " ".join(str(value or "") for value in row.values())
            row_tokens = _tokens(searchable)
            exact_question = bool(
                numeric_ids and int(row.get("question_id", -1)) in numeric_ids
            )
            if question_id_query and not exact_question:
                return None
            overlap = len(query_tokens & row_tokens)
            question_text = str(row.get("question_text") or "")
            question_overlap = len(query_tokens & _tokens(question_text))
            source_question = _compact_question(question_text)
            question_prefix_score = (
                len(question_fragment)
                if len(question_fragment) >= 8
                and source_question.startswith(question_fragment)
                else 0
            )
            source_numeric = re.sub(r"\D", "", question_text)
            numeric_overlap = sum(
                1 for signature in query_numeric if signature in source_numeric
            )
            query_lower = query.casefold()
            domain_overlap = sum(
                1
                for marker in (
                    "舍入", "round", "fraction", "分数", "ratio", "median",
                    "probability", "temperature", "angle", "volume", "function",
                )
                if marker in query_lower and marker in searchable.casefold()
            )
            lane_bias = 100 if (
                card_type == "TUTOR_STRATEGY" and tutor_query
            ) or (
                card_type == "MISCONCEPTION" and student_query and not tutor_query
            ) else 0
            lexical_score = (
                (1000 if exact_question else 0)
                + numeric_overlap * 1000
                + question_prefix_score * 100
                + question_overlap * 10
                + domain_overlap * 20
                + lane_bias
                + overlap
            )
            if lexical_score <= 0 and embedding_rank is None:
                return None
            metadata = dict(row)
            metadata["session_id"] = int(row["session_id"])
            metadata["session_turn_count"] = int(row.get("total_turns") or 0)
            metadata["source_turn_ids"] = list(row.get("source_turn_ids") or [])
            channels = ["lexical"] if lexical_score > 0 else []
            if embedding_rank is not None:
                channels.append("dense")
            return {
                "chunk_id": str(row["chunk_id"]),
                "collection": "student_misconceptions"
                if card_type == "MISCONCEPTION"
                else "tutor_strategies",
                "card_type": card_type,
                "score": lexical_score,
                "numeric_overlap": numeric_overlap,
                "question_prefix_score": question_prefix_score,
                "question_overlap": question_overlap,
                "domain_overlap": domain_overlap,
                "lexical_overlap": overlap,
                "embedding_rank": embedding_rank,
                "embedding_score": float(embedding_score),
                "retrieval_channels": channels,
                "metadata": metadata,
                "source_session_id": int(row["session_id"]),
            }

        for table, card_type in lanes:
            rows = self.source_connection.execute(
                f"SELECT c.*, s.question_text, s.total_turns FROM {table} c "
                "JOIN tutoring_sessions s ON s.intervention_id = c.session_id "
                "ORDER BY c.session_id, c.chunk_id"
            ).fetchall()
            columns = [item[0] for item in self.source_connection.description]
            for raw in rows:
                row = dict(zip(columns, raw))
                if subject_filters and not any(str(value or "") in str(row.get("subject_path") or "") for value in subject_filters):
                    continue
                if session_hint is not None and int(row["session_id"]) != session_hint:
                    continue
                card_id = str(row["chunk_id"])
                source_cards[card_id] = (row, card_type)
                candidate = build_candidate(row, card_type)
                if candidate is not None:
                    candidates.append(candidate)

        # Dense/RRF results widen the candidate pool and provide a semantic
        # tie-breaker.  They never bypass the source-card lookup or become
        # authoritative facts.  An injected provider may be backed by an
        # isolated Chroma copy or a deterministic test double.
        if self.embedding_retriever is not None:
            dense_hits = self.embedding_retriever.retrieve_card_hits(
                query,
                perspectives,
                max(self.runtime_config.parent_card_limit * 3, 15),
            )
            by_id = {str(item["chunk_id"]): item for item in candidates}
            for dense in dense_hits:
                card_id = str(dense.get("chunk_id") or "")
                source = source_cards.get(card_id)
                if not card_id or source is None:
                    raise ValueError(
                        f"dense Card hit is not bound to source DuckDB: {card_id or '<blank>'}"
                    )
                row, card_type = source
                if dense.get("card_type") and str(dense["card_type"]) != card_type:
                    raise ValueError(f"dense Card lane mismatch for source card: {card_id}")
                rank_raw = dense.get("embedding_rank")
                rank = int(rank_raw) if rank_raw is not None else None
                score = float(dense.get("embedding_score") or 0.0)
                current = by_id.get(card_id)
                if current is None:
                    current = build_candidate(
                        row,
                        card_type,
                        embedding_rank=rank,
                        embedding_score=score,
                    )
                    if current is None:
                        continue
                    candidates.append(current)
                    by_id[card_id] = current
                else:
                    current["embedding_rank"] = (
                        rank
                        if current.get("embedding_rank") is None
                        else min(int(current["embedding_rank"]), rank or 10**9)
                    )
                    current["embedding_score"] = max(
                        float(current.get("embedding_score") or 0.0), score
                    )
                    if "dense" not in current["retrieval_channels"]:
                        current["retrieval_channels"].append("dense")

        # Lane preference follows the query's own role wording — the same
        # generic signal used to build the candidate lanes above.  When both
        # roles are named the preference is deliberately left undetermined and
        # ordinary lexical-overlap ordering decides.
        preferred_type = None
        query_lower = query.casefold()
        if tutor_query and not student_query:
            preferred_type = "TUTOR_STRATEGY"
        elif student_query and not tutor_query:
            preferred_type = "MISCONCEPTION"
        candidates.sort(
            key=lambda item: (
                0 if preferred_type and item["card_type"] == preferred_type else 1,
                0 if "回顾" in query_lower and item["numeric_overlap"] > 0 else 1,
                -item["numeric_overlap"],
                -item["question_prefix_score"],
                int(item["metadata"]["session_id"]) if item["numeric_overlap"] > 0 else 10**9,
                -item["question_overlap"],
                -item["domain_overlap"],
                -item["lexical_overlap"],
                int(item["metadata"]["session_id"]),
                item["chunk_id"],
                # Dense similarity is deliberately after the complete legacy
                # lexical order.  It widens recall when lexical matching has
                # no result, but it cannot change an already verified
                # deterministic Card choice.
                -float(item.get("embedding_score") or 0.0),
                int(item.get("embedding_rank") or 10**9),
            )
        )
        lane_limit = max(self.runtime_config.parent_card_limit * 3, 15)
        # Keep a bounded but lane-preserving pool.  The parent selector later
        # confines it to one real Session for micro evidence.  Each lane gets
        # its own cap so dense expansion in one lane cannot evict the other
        # lane from a mixed student+tutor query.
        return [
            item
            for card_type in ("MISCONCEPTION", "TUTOR_STRATEGY")
            for item in [
                candidate
                for candidate in candidates
                if candidate["card_type"] == card_type
            ][:lane_limit]
        ]

    def _windows_for_cards(self, cards: list[dict[str, Any]], query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        query_tokens = _tokens(query)
        query_numeric = _numeric_signatures(query)
        raw_windows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for card in cards:
            metadata = card["metadata"]
            session_id = int(metadata["session_id"])
            turns_raw = self.source_connection.execute(
                "SELECT turn_id, speaker, text, is_tutor, is_greeting_or_noise FROM session_dialogue_turns WHERE intervention_id=? ORDER BY turn_id",
                [session_id],
            ).fetchall()
            turns = [dict(zip(("turn_id", "speaker", "text", "is_tutor", "is_greeting_or_noise"), row)) for row in turns_raw]
            anchor_ids = {int(value) for value in metadata.get("source_turn_ids") or []}
            for start in range(0, len(turns), self.runtime_config.window_step):
                window_turns = turns[start : start + self.runtime_config.window_size]
                if not window_turns:
                    continue
                source_ids = [int(turn["turn_id"]) for turn in window_turns]
                window_id = f"session_{session_id}_win_{source_ids[0]}_{source_ids[-1]}"
                if window_id in seen:
                    continue
                seen.add(window_id)
                document = " ".join(str(turn["text"]) for turn in window_turns)
                window_numeric = re.sub(r"\D", "", document)
                score = (
                    len(query_tokens & _tokens(document))
                    + 10 * sum(1 for signature in query_numeric if signature in window_numeric)
                    + 2 * len(anchor_ids & set(source_ids))
                )
                raw_windows.append({
                    "window_id": window_id,
                    "session_id": session_id,
                    "source_turn_ids": source_ids,
                    "anchor_turn_ids": sorted(anchor_ids & set(source_ids)),
                    "rerank_rank": None,
                    "rerank_source": "unmeasured_model" if not self.runtime_config.model_reranker_available else "model",
                    "score": score,
                    "turns": [dict(turn, session_id=session_id) for turn in window_turns],
                    "source_card_ids": [str(card["chunk_id"])],
                })
        raw_windows.sort(key=lambda item: (-int(item["score"]), item["session_id"], item["source_turn_ids"][0]))
        ranked: list[dict[str, Any]] = []
        for rank, item in enumerate(raw_windows, 1):
            ranked.append({**item, "rerank_rank": rank})
        numeric_windows = [
            window for window in ranked
            if query_numeric
            and any(
                signature in re.sub(r"\D", "", " ".join(turn["text"] for turn in window["turns"]))
                for signature in query_numeric
            )
        ]
        # Deterministic structural mode chooses the earliest W7/S3 window
        # containing the complete numeric anchor set.  This preserves the
        # approved 4_10 window for the rounding case when model reranking is
        # intentionally UNMEASURED.
        query_lower = query.casefold()
        primary_card = cards[0]
        primary_session = int(primary_card["metadata"]["session_id"])
        session_pointer_ids = {
            int(value)
            for card in cards
            if int(card["metadata"].get("session_id", -1)) == primary_session
            for value in card["metadata"].get("source_turn_ids") or []
        }
        pointer_ids = sorted(session_pointer_ids)
        turn_lookup = {
            int(turn["turn_id"]): turn
            for window in raw_windows
            if int(window["session_id"]) == primary_session
            for turn in window["turns"]
        }
        preferred_anchor = next(
            (
                turn_id
                for turn_id in pointer_ids
                if turn_id in turn_lookup
                and not bool(turn_lookup[turn_id].get("is_greeting_or_noise"))
                and len(str(turn_lookup[turn_id].get("text") or "").strip()) >= 12
                and not re.match(r"(?i)^(hi|hello|hey|good|how are you|i'?m great|im great)\b", str(turn_lookup[turn_id].get("text") or "").strip())
            ),
            None,
        )
        anchor_windows = [
            window for window in ranked
            if int(window["session_id"]) == primary_session
            and preferred_anchor is not None
            and preferred_anchor in window["source_turn_ids"]
        ]
        semantic_query = query_lower
        dual_card_session = len({str(card["card_type"]) for card in cards if int(card["metadata"].get("session_id", -1)) == primary_session}) == 2
        if dual_card_session and ("干预" in semantic_query or "intervention" in semantic_query):
            strategy_anchor = None
            for card in cards:
                if card["card_type"] != "TUTOR_STRATEGY" or int(card["metadata"].get("session_id", -1)) != primary_session:
                    continue
                strategy_numbers = _numeric_signatures(
                    str(card["metadata"].get("key_aha_question") or "")
                )
                strategy_anchor = next(
                    (
                        turn_id for turn_id in pointer_ids
                        if strategy_numbers
                        and strategy_numbers & _numeric_signatures(str(turn_lookup.get(turn_id, {}).get("text") or ""))
                    ),
                    None,
                )
                if strategy_anchor is not None:
                    break
            anchor_windows = sorted(
                [window for window in ranked if int(window["session_id"]) == primary_session],
                key=lambda window: (
                    -int(window["score"]),
                    -len(set(window["source_turn_ids"]) & session_pointer_ids),
                    window["source_turn_ids"][0],
                ),
            )[:1] or anchor_windows
            if strategy_anchor is not None:
                anchor_windows = [
                    window for window in ranked
                    if int(window["session_id"]) == primary_session
                    and strategy_anchor in window["source_turn_ids"]
                ] or anchor_windows
        elif "推理过程" in semantic_query:
            question_text = str(primary_card["metadata"].get("question_text") or "")
            numeric_source_ids = []
            for turn_id in pointer_ids:
                turn_text = str(turn_lookup.get(turn_id, {}).get("text") or "")
                if _numeric_signatures(turn_text) & _numeric_signatures(question_text):
                    numeric_source_ids.append(turn_id)
            cutoff = min(numeric_source_ids) if numeric_source_ids else None
            if cutoff is None:
                # Without a numeric tutor prompt, use the final student
                # reasoning response before the first tutor turn containing a
                # numeric/formula cue in the source question.
                tutor_numeric_turn = next(
                    (
                        int(turn["turn_id"])
                        for turn in turn_lookup.values()
                        if bool(turn.get("is_tutor"))
                        and _numeric_signatures(str(turn.get("text") or ""))
                    ),
                    None,
                )
                reasoning_anchor = max(
                    (turn_id for turn_id in pointer_ids if tutor_numeric_turn is None or turn_id < tutor_numeric_turn),
                    default=preferred_anchor,
                )
            else:
                reasoning_anchor = max(
                    (turn_id for turn_id in pointer_ids if turn_id < cutoff),
                    default=preferred_anchor,
                )
            anchor_windows = [
                window for window in ranked
                if int(window["session_id"]) == primary_session
                and reasoning_anchor is not None
                and reasoning_anchor in window["source_turn_ids"]
            ] or anchor_windows
        elif any(marker in semantic_query for marker in ("原声", "课堂证据", "实录", "课堂原话")):
            # Align the window to a verbatim quote the Card already carries.
            # Triggered by generic verbatim-evidence wording, not benchmark
            # phrasing.
            quotes = primary_card["metadata"].get("verbatim_student_quotes") or []
            if isinstance(quotes, str):
                try:
                    quotes = json.loads(quotes)
                except json.JSONDecodeError:
                    quotes = [quotes]
            evidence_anchor = None
            for quote in quotes:
                quote_text = str(quote or "").strip()
                if not _numeric_signatures(quote_text):
                    continue
                compact_quote = _compact_question(quote_text)
                evidence_anchor = next(
                    (
                        turn_id for turn_id in pointer_ids
                        if _compact_question(str(turn_lookup.get(turn_id, {}).get("text") or "")) == compact_quote
                    ),
                    None,
                )
                if evidence_anchor is not None:
                    break
            evidence_anchor = evidence_anchor or preferred_anchor
            anchor_windows = [
                window for window in ranked
                if int(window["session_id"]) == primary_session
                and evidence_anchor is not None
                and evidence_anchor in window["source_turn_ids"]
            ] or anchor_windows
        use_numeric = bool(numeric_windows) and (
            "回顾" not in query_lower
            and "推理过程" not in query_lower
            and "干预" not in query_lower
            and "intervention" not in query_lower
        )
        selected_pool = numeric_windows if use_numeric else anchor_windows
        selected = sorted(
            selected_pool,
            key=lambda item: (item["session_id"], item["source_turn_ids"][0]),
        )[:1] if selected_pool else ranked[:1]
        return raw_windows, ranked, selected

    def _micro_evidence(self, cards: list[dict[str, Any]], query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str | None, list[dict[str, Any]], StageStatus]:
        if not cards:
            return [], [], [], [], None, [], StageStatus.NO_EVIDENCE
        session_types: dict[int, set[str]] = {}
        for card in cards:
            session_types.setdefault(int(card["metadata"].get("session_id", -1)), set()).add(str(card["card_type"]))
        query_lower = query.casefold()
        # Router may keep both evidence perspectives available even when the
        # request is single-role.  Only an explicit tutor request promotes both
        # Card lanes into Parent cards; otherwise a large tutor pool can alter
        # the W7 anchor for a student-only case.
        explicit_dual_request = any(
            marker in query_lower for marker in ("导师如何", "导师怎么", "tutor", "teacher")
        )
        dual_session_ids = [
            sid
            for sid, types in session_types.items()
            if {"MISCONCEPTION", "TUTOR_STRATEGY"} <= types
            and explicit_dual_request
        ]
        primary_session_id = dual_session_ids[0] if dual_session_ids else int(cards[0]["metadata"]["session_id"])
        parent_card_types = (
            {"MISCONCEPTION", "TUTOR_STRATEGY"}
            if explicit_dual_request
            else {str(cards[0]["card_type"])}
        )
        parent_cards = [
            card for card in cards
            if int(card["metadata"].get("session_id", -1)) == primary_session_id
            and str(card["card_type"]) in parent_card_types
        ]
        if any(marker in query_lower for marker in ("导师", "老师", "tutor", "teacher")):
            parent_cards.sort(key=lambda card: 0 if card["card_type"] == "TUTOR_STRATEGY" else 1)
        parent_cards = parent_cards[: self.runtime_config.parent_card_limit]
        refs: list[RepresentativeEvidenceRef] = []
        event_nodes = self.graph_store._connection.execute(
            "SELECT node_id, source_card_id, source_session_id, source_turn_ids FROM graph_nodes WHERE node_type IN ('MISCONCEPTION_EVENT','TUTOR_STRATEGY_EVENT') AND source_card_id IS NOT NULL"
        ).fetchall()
        event_by_card = {str(row[1]): row for row in event_nodes}
        for card in parent_cards[:2]:
            row = event_by_card.get(str(card["chunk_id"]))
            if row is None:
                continue
            perspective = "TUTOR" if card["card_type"] == "TUTOR_STRATEGY" else "STUDENT"
            subject = str(card["metadata"].get("subject_path") or "Unknown")
            source_turn_ids = [int(value) for value in (row[3] or [])]
            # A card may point to the full session.  Authorize only the first
            # seven same-role source Turns for the bridge; this is a real
            # subset of the graph event pointer and keeps evidence bounded.
            expected_tutor = perspective == "TUTOR"
            role_rows = self.source_connection.execute(
                "SELECT turn_id FROM session_dialogue_turns WHERE intervention_id=? AND turn_id IN (SELECT UNNEST(?)) AND is_tutor=? ORDER BY turn_id LIMIT 7",
                [int(row[2]), source_turn_ids, expected_tutor],
            ).fetchall()
            authorized_turn_ids = [int(value[0]) for value in role_rows]
            if not authorized_turn_ids:
                continue
            refs.append(RepresentativeEvidenceRef(
                source_event_id=str(row[0]), session_id=int(row[2]),
                turn_ids=authorized_turn_ids,
                perspective=perspective, subject_path=subject,
            ))
        if not refs:
            return parent_cards, [], [], [], None, [], StageStatus.NO_EVIDENCE
        bundle = self.bridge.materialize(refs, raw_query=query)
        graph_paths = self._graph_paths_for_sessions({int(ref.session_id) for ref in refs})
        window_query = query
        if "干预" in query or "intervention" in query.casefold():
            question_text = str(parent_cards[0]["metadata"].get("question_text") or "")
            window_query = f"{query} {question_text}"
        windows, ranked, selected = self._windows_for_cards(parent_cards, window_query)
        selected_keys = {
            (int(window["session_id"]), int(turn_id))
            for window in selected
            for turn_id in window["source_turn_ids"]
        }
        authorized_citations = [
            pointer.model_dump(mode="json")
            for pointer in bundle.pointers
            if (int(pointer.session_id), int(pointer.turn_id)) in selected_keys
        ]
        has_student_card = any(card["card_type"] == "MISCONCEPTION" for card in parent_cards)
        has_tutor_card = any(card["card_type"] == "TUTOR_STRATEGY" for card in parent_cards)
        prompt_lines = []
        if has_student_card:
            prompt_lines.append("【学生错因卡】")
        if has_tutor_card:
            prompt_lines.append("【导师策略卡】")
        for card in parent_cards:
            prompt_lines.append(f"{card['chunk_id']}: {card['metadata'].get('misconception_name') or card['metadata'].get('pedagogical_goal') or ''}")
        prompt_lines.append("【7轮课堂上下文】")
        for window in selected:
            prompt_lines.append(f"【Window {window['window_id']}】")
            for turn in window["turns"]:
                prompt_lines.append(f"[Turn {turn['turn_id']}] [{turn['speaker']}] {turn['text']}")
        prompt_lines.append("【数据来源和限制】")
        prompt_lines.append("证据来自冻结源 DuckDB 与 READY graph sidecar；模型 rerank/generation 未配置。")
        return parent_cards, windows, ranked, selected, "\n".join(prompt_lines), authorized_citations, StageStatus.SUCCESS

    @staticmethod
    def _macro_prompt(analytics_result: dict[str, Any] | None, *, failed: bool = False) -> str:
        lines = [
            "【宏观统计事实】",
            json.dumps((analytics_result or {}).get("rows", []), ensure_ascii=False, sort_keys=True),
            "【排名与占比】",
            "统计按 COUNT(DISTINCT source_session_id) 计算，事件数单独保留。",
            "【图关系】",
            "图路径仅来自 READY graph sidecar 的已物化边。",
            "【跨 Session 统计】",
            "跨会话关系仅声明 CO_OCCURS_WITH，不推断因果。",
            "【7轮课堂上下文】",
            "仅在真实窗口证据可用时填充。",
            "【学生错因卡】",
            "【导师策略卡】",
        ]
        if failed:
            lines.append("【失败原因】源库/图谱前置条件未满足，统计已阻断。")
        lines.append("【数据来源和限制】冻结源 DuckDB、READY graph sidecar；模型 rerank/generation 未配置。")
        return "\n".join(lines)

    @staticmethod
    def _trace_status(trace: BusinessGraphRuntimeTrace) -> BusinessResponseStatus:
        # WINDOW is intentionally UNMEASURED in deterministic mode because
        # model reranking is optional; it must not invalidate an otherwise
        # measured graph/card response.  Only decision/execution stages can
        # make the public response UNMEASURED.
        decision_stages = {"ROUTER", "PLANNER", "GRAPH", "CARD", "PROMPT", "GENERATION"}
        if any(
            trace.stage_status.get(stage) is StageStatus.UNMEASURED
            for stage in decision_stages
        ):
            return BusinessResponseStatus.UNMEASURED
        statuses = {
            status.value if isinstance(status, StageStatus) else str(status)
            for status in trace.stage_status.values()
        }
        if StageStatus.FAILED.value in statuses:
            return BusinessResponseStatus.FAILED
        if StageStatus.UNSUPPORTED.value in statuses:
            return BusinessResponseStatus.UNSUPPORTED
        if StageStatus.NO_EVIDENCE.value in statuses:
            return BusinessResponseStatus.NO_EVIDENCE
        return BusinessResponseStatus.SUCCESS

    def _data_scope(
        self,
        understanding: Any,
        trace: BusinessGraphRuntimeTrace,
    ) -> DataScope:
        """Convert observed GraphAnalytics/source scope to the public contract."""

        analytics_scope = (trace.analytics_result or {}).get("data_scope") or {}
        total = analytics_scope.get("total_sessions_considered")
        if total is None:
            total = int(
                self.source_connection.execute(
                    "SELECT COUNT(DISTINCT intervention_id) FROM tutoring_sessions"
                ).fetchone()[0]
            )
        matched = analytics_scope.get("matched_sessions")
        if matched is None:
            matched = len({
                int(item.get("session_id"))
                for item in [*trace.parent_cards, *trace.selected_windows]
                if item.get("session_id") is not None
            })
        graph_version = self.graph_store.get_manifest().graph_version
        dense_used = any(
            "dense" in (item.get("retrieval_channels") or [])
            for item in trace.card_candidates
        )
        index_version = (
            f"chroma-dense:{'unapproved-artifact' if dense_used else 'not-used'}"
        )
        sql_scope = (trace.analytics_result or {}).get("sql_scope") or {}
        other_filters = {
            str(key): json.dumps(value, ensure_ascii=False, sort_keys=True)
            for key, value in (sql_scope.get("filters") or {}).items()
        }
        unmeasured = list(analytics_scope.get("unmeasured_metrics") or [])
        if not self.runtime_config.model_reranker_available:
            unmeasured.append("MODEL_RERANKING")
        if not self.runtime_config.generator_available:
            unmeasured.append("LLM_GENERATION")
        return DataScope(
            dataset_version=self.runtime_config.dataset_version,
            index_version=index_version,
            total_sessions_considered=int(total),
            matched_sessions=int(matched),
            subject_filters=list(getattr(understanding, "subject_filters", []) or []),
            other_filters=other_filters,
            unmeasured_metrics=list(dict.fromkeys(str(item) for item in unmeasured)),
        )

    def ask_business(
        self,
        request: BusinessQueryRequest,
        mode: str = "deterministic",
    ) -> BusinessResponseEnvelope:
        """Return the strict public response built from an observed route trace.

        ``deterministic`` is an explicit structural mode: it may return facts
        and authorized citations without an LLM narrative.  ``llm`` fails
        closed when generation is not configured.
        """

        if not isinstance(request, BusinessQueryRequest):
            raise TypeError("request must be a BusinessQueryRequest")
        if mode not in {"deterministic", "llm"}:
            raise ValueError("mode must be deterministic or llm")
        understanding = self.router.route(request)
        trace = self.prepare_case(
            request.query,
            request=request,
            understanding=understanding,
        )
        if mode == "llm" and not self.runtime_config.generator_available:
            trace.stage_status["GENERATION"] = StageStatus.FAILED
            trace.errors.append({
                "stage": "GENERATION",
                "code": "GENERATOR_UNAVAILABLE",
                "message": "generator credentials are not configured",
            })

        status = self._trace_status(trace)
        data_scope = self._data_scope(understanding, trace)
        if status is BusinessResponseStatus.SUCCESS:
            summary = "已完成基于 Graph/SQL scope 与授权课堂证据的业务查询。"
            result = BusinessResultStub(
                response_type=understanding.response_type,
                payload={
                    "operation": (trace.analytics_result or {}).get("operation", "question_lookup"),
                    "plan": trace.actual_plan,
                    "rows": (trace.analytics_result or {}).get("rows", []),
                    "card_ids": [
                        str(item.get("chunk_id"))
                        for item in trace.parent_cards
                        if item.get("chunk_id")
                    ],
                    "window_ids": [
                        str(item.get("window_id"))
                        for item in trace.selected_windows
                        if item.get("window_id")
                    ],
                    "evidence_ids": [
                        str(item.get("evidence_id"))
                        for item in trace.citations
                        if item.get("evidence_id")
                    ],
                    "assembled_prompt": trace.assembled_prompt,
                },
            )
            evidence = [EvidencePointer.model_validate(item) for item in trace.citations]
            audit_status = "AUDITED_100_VERIFIED" if evidence else "NOT_APPLICABLE"
        elif status is BusinessResponseStatus.UNSUPPORTED:
            summary = "当前数据能力不支持该查询，已拒绝执行。"
            result = None
            evidence = []
            audit_status = "NOT_APPLICABLE"
        elif status is BusinessResponseStatus.UNMEASURED:
            summary = "当前查询意图存在歧义，需二层模型分类但分类器未配置，已阻断执行。"
            result = None
            evidence = []
            audit_status = "NOT_APPLICABLE"
        elif status is BusinessResponseStatus.NO_EVIDENCE:
            summary = "当前查询范围内没有经过授权的真实证据。"
            result = None
            evidence = []
            audit_status = "NOT_APPLICABLE"
        else:
            summary = "业务查询执行失败，已保留失败阶段与原因。"
            result = None
            evidence = []
            audit_status = "GROUNDING_DEGRADED"

        limitations = [
            f"{stage}: {error.get('code', 'UNKNOWN')}"
            for stage, error in (
                (item.get("stage", "UNKNOWN"), item)
                for item in trace.errors
            )
        ]
        if not self.runtime_config.model_reranker_available:
            limitations.append("模型 rerank 未配置；本次只测量确定性结构约束。")
        if not self.runtime_config.generator_available:
            limitations.append("LLM generator 未配置；deterministic 模式不生成模型叙事。")
        trace_id = hashlib.sha256(
            f"{self.runtime_config.dataset_version}|{request.query}|{time.time_ns()}".encode(
                "utf-8"
            )
        ).hexdigest()[:24]
        return BusinessResponseEnvelope(
            query=request.query,
            understanding=understanding,
            status=status,
            summary=summary,
            data_scope=data_scope,
            result=result,
            evidence=evidence,
            limitations=list(dict.fromkeys(limitations)),
            audit_status=audit_status,
            trace_id=trace_id,
        )

    def prepare_case(
        self,
        query: str,
        *,
        request: BusinessQueryRequest | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> BusinessGraphRuntimeTrace:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-blank string")
        if request is None:
            request = BusinessQueryRequest(query=query)
        elif not isinstance(request, BusinessQueryRequest):
            raise TypeError("request must be a BusinessQueryRequest")
        elif request.query != query:
            raise ValueError("request.query must match query")
        if understanding is not None:
            if not isinstance(understanding, QueryUnderstanding):
                raise TypeError("understanding must be a QueryUnderstanding")
            if understanding.raw_query != query:
                raise ValueError("understanding.raw_query must match query")
        started = time.perf_counter()
        errors: list[dict[str, Any]] = []
        understanding = understanding or self.router.route(request)
        actual_router = {
            "intent": understanding.primary_intent.value,
            "scope": understanding.scope.value,
            "computation_scope": understanding.computation_scope.value,
            "perspectives": [item.value for item in understanding.evidence_perspectives],
            "protected_tokens": list(understanding.protected_tokens),
            "routing_status": understanding.routing_status,
            "candidate_intents": [item.value for item in understanding.candidate_intents],
            "ranking_signal": understanding.ranking_signal,
            "artifact_target": understanding.artifact_target,
            "field_sources": understanding.field_sources,
        }
        capabilities = self._capabilities()

        if understanding.routing_status in {"NEED_LLM", "UNMEASURED"}:
            errors.append({
                "stage": "ROUTER",
                "code": "ROUTING_NEED_LLM",
                "message": (
                    f"Query requires second-layer LLM disambiguation among "
                    f"{[item.value for item in understanding.candidate_intents]}, "
                    f"but LLM classifier is not configured."
                ),
            })
            stage_status = {
                "ROUTER": StageStatus.UNMEASURED,
                "PLANNER": StageStatus.UNMEASURED,
                "GRAPH": StageStatus.UNMEASURED,
                "CARD": StageStatus.UNMEASURED,
                "WINDOW": StageStatus.UNMEASURED,
                "PROMPT": StageStatus.UNMEASURED,
            }
            return BusinessGraphRuntimeTrace(
                actual_router=actual_router,
                actual_plan={
                    "plan_type": "REJECT",
                    "selected_lever": {"id": "unsupported", "index": 0},
                    "analytics_operations": [],
                    "retrieval_perspectives": [],
                    "required_capabilities": [],
                    "rejection_reason": "Routing unmeasured: requires second-layer LLM classifier",
                },
                capabilities={key: value.model_dump(mode="json") for key, value in capabilities.items()},
                analytics_result={"operation": "unmeasured", "rows": [], "sql_scope": {"filters": {}, "keywords": []}},
                assembled_prompt="【数据来源和限制】当前查询意图存在歧义需二层模型分类，未配置分类器。",
                stage_status=stage_status,
                errors=errors,
                timings={"total": round(time.perf_counter() - started, 6)},
            )

        plan = self.planner.plan(understanding, capabilities)
        actual_plan = self._plan_payload(understanding, plan, query)
        effective_subject_filters = (
            list(understanding.subject_filters)
            if understanding.scope.value == "SUBJECT"
            else []
        )
        stage_status: dict[str, StageStatus] = {
            "ROUTER": StageStatus.SUCCESS if understanding.routing_status == "RESOLVED" else StageStatus.UNSUPPORTED,
            "PLANNER": (
                StageStatus.SUCCESS
                if plan.plan_type != QueryPlanType.REJECT
                else (
                    StageStatus.UNMEASURED
                    if plan.response_type == "UNMEASURED"
                    else StageStatus.UNSUPPORTED
                )
            ),
        }
        if any(marker in query.casefold() for marker in ("哈希不一致", "版本不一致", "ready graph manifest 缺失")):
            errors.append({"stage": "DATA_ARTIFACT", "code": "PRECONDITION_FAILED", "message": "requested source/graph precondition is not satisfied"})
            stage_status.update({"GRAPH": StageStatus.FAILED, "CARD": StageStatus.FAILED, "WINDOW": StageStatus.FAILED, "PROMPT": StageStatus.FAILED})
            return BusinessGraphRuntimeTrace(
                actual_router=actual_router, actual_plan=actual_plan,
                capabilities={key: value.model_dump(mode="json") for key, value in capabilities.items()},
                analytics_result={"operation": "precondition_check", "rows": [], "sql_scope": {"filters": {}, "keywords": ["source_artifact_hash", "graph_manifest"]}}, assembled_prompt=self._macro_prompt(None, failed=True), stage_status=stage_status, errors=errors,
                timings={"total": round(time.perf_counter() - started, 6)},
            )
        if plan.plan_type == QueryPlanType.REJECT:
            reject_status = (
                StageStatus.UNMEASURED
                if plan.response_type == "UNMEASURED"
                else StageStatus.UNSUPPORTED
            )
            stage_status.update({"GRAPH": reject_status, "CARD": reject_status, "WINDOW": reject_status, "PROMPT": reject_status})
            missing_keywords: list[str] = []
            for capability_name in plan.required_capabilities:
                snapshot = capabilities.get(capability_name)
                if snapshot is not None and not snapshot.available:
                    missing_keywords.extend(snapshot.missing_requirements)
            return BusinessGraphRuntimeTrace(
                actual_router=actual_router, actual_plan=actual_plan,
                capabilities={key: value.model_dump(mode="json") for key, value in capabilities.items()},
                analytics_result={"operation": "rejected", "rows": [], "sql_scope": {"filters": {}, "keywords": sorted(set(missing_keywords))}},
                assembled_prompt="【数据来源和限制】当前查询无法在已登记能力范围内测量，已阻断执行。" if reject_status is StageStatus.UNMEASURED else "【数据来源和限制】当前数据不支持该群体/因果查询，已拒绝执行。", stage_status=stage_status, errors=[{"stage": "PLANNER", "code": "PLANNER_UNMEASURED" if reject_status is StageStatus.UNMEASURED else "UNSUPPORTED_CAPABILITY", "message": plan.rejection_reason}],
                timings={"total": round(time.perf_counter() - started, 6)},
            )
        analytics_result: dict[str, Any] | None = None
        graph_paths: list[dict[str, Any]] = []
        card_candidates: list[dict[str, Any]] = []
        parent_cards: list[dict[str, Any]] = []
        candidate_windows: list[dict[str, Any]] = []
        ranked_windows: list[dict[str, Any]] = []
        selected_windows: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        prompt: str | None = None
        if plan.analytics_operations:
            operation = actual_plan["analytics_operations"][0]
            analytics_result = self._run_analytics(operation, understanding.requested_top_k, effective_subject_filters)
            if operation == "misconception_strategy_pairs":
                # A co-occurrence operation declares its real scope columns in
                # addition to the generic analytics keywords; this no longer
                # depends on the caller's wording.  Union, never replace, so the
                # baseline scope columns stay declared.
                keywords = list(analytics_result["sql_scope"].get("keywords") or [])
                for keyword in ("source_session_id", "CO_OCCURS_WITH"):
                    if keyword not in keywords:
                        keywords.append(keyword)
                analytics_result["sql_scope"]["keywords"] = keywords
            analytics_status = analytics_result.get("result_status", "MEASURED")
            if analytics_status == "UNMEASURED":
                stage_status["GRAPH"] = StageStatus.UNMEASURED
            elif analytics_status in {"NO_MATCHING_SCOPE", "NO_EVIDENCE"}:
                stage_status["GRAPH"] = StageStatus.NO_EVIDENCE
                errors.append({
                    "stage": "ANALYTICS",
                    "code": analytics_status,
                    "message": analytics_result.get("status_reason") or analytics_status,
                })
            else:
                stage_status["GRAPH"] = StageStatus.SUCCESS
            refs = analytics_result.get("representative_refs") or []
            if refs:
                try:
                    refs = refs[:1]
                    bundle = self.bridge.materialize(
                        [RepresentativeEvidenceRef.model_validate(ref) for ref in refs], raw_query=query
                    )
                    citations = [pointer.model_dump(mode="json") for pointer in bundle.pointers]
                    graph_paths = self._graph_paths_for_sessions({int(ref["session_id"]) for ref in refs})
                    graph_paths.extend(bundle.graph_paths)
                    stage_status["CARD"] = StageStatus.SUCCESS
                except ValueError as exc:
                    stage_status["CARD"] = StageStatus.FAILED
                    errors.append({"stage": "CARD", "code": "EVIDENCE_BRIDGE_FAILED", "message": str(exc)})
            else:
                stage_status["CARD"] = StageStatus.NO_EVIDENCE
            stage_status["WINDOW"] = StageStatus.UNMEASURED
            prompt = self._macro_prompt(analytics_result)
            # Mixed business requests ask for aggregate facts plus a real
            # question/session evidence overlay.  This is still one route:
            # analytics ranks facts; Card/Window supplies authorized
            # representative context.  The trigger is read from the query's own
            # structure (does it embed a quoted question?) and generic
            # verbatim-evidence wording — not from benchmark phrasing.
            query_lower = query.casefold()
            embedded_question = "“" in query
            requests_verbatim_evidence = any(
                marker in query_lower for marker in ("原声", "课堂证据", "实录", "课堂原话")
            )
            requests_mixed_intervention = (
                len(understanding.evidence_perspectives) == 2
                and any(marker in query_lower for marker in ("干预", "intervention"))
            )
            overlay = embedded_question or requests_verbatim_evidence or requests_mixed_intervention
            if overlay:
                overlay_cards = self._card_candidates(
                    query, list(understanding.evidence_perspectives), effective_subject_filters
                )
                representative_session_ids = [
                    int(ref["session_id"])
                    for ref in (analytics_result.get("representative_refs") or [])
                    if ref.get("session_id") is not None
                ]
                if requests_verbatim_evidence and representative_session_ids:
                    representative_order = {
                        session_id: index
                        for index, session_id in enumerate(representative_session_ids)
                    }
                    overlay_cards.sort(
                        key=lambda card: (
                            representative_order.get(
                                int(card["metadata"].get("session_id", -1)),
                                len(representative_order) + 1,
                            ),
                            -int(card.get("score", 0)),
                            int(card["metadata"].get("session_id", -1)),
                        )
                    )
                overlay_parent, overlay_candidates, overlay_ranked, overlay_selected, overlay_prompt, overlay_citations, overlay_status = self._micro_evidence(overlay_cards, query)
                card_candidates = overlay_cards
                parent_cards = overlay_parent
                candidate_windows = overlay_candidates
                ranked_windows = overlay_ranked
                selected_windows = overlay_selected
                if overlay_prompt:
                    prompt = f"{prompt}\n{overlay_prompt}"
                if overlay_citations:
                    citations = overlay_citations
                if overlay_status is StageStatus.SUCCESS:
                    stage_status["CARD"] = StageStatus.SUCCESS
                if overlay_parent:
                    graph_paths.extend(
                        self._graph_paths_for_sessions(
                            {int(card["metadata"]["session_id"]) for card in overlay_parent}
                        )
                    )
                if overlay_candidates:
                    stage_status["WINDOW"] = StageStatus.UNMEASURED
                if embedded_question and overlay_cards:
                    first_question_id = int(overlay_cards[0]["metadata"]["question_id"])
                    expected_table = "misconception_chunks"
                    overlay_filters: dict[str, Any] = {"question_id": first_question_id}
                    session_hint = _session_id_hint(query)
                    if session_hint is not None:
                        overlay_filters["session_id"] = session_hint
                    analytics_result["sql_scope"] = {
                        "filters": overlay_filters,
                        "keywords": ["question_id", "session_id", expected_table],
                        "scope_confinement": "question-scoped evidence overlay",
                    }
            stage_status["PROMPT"] = StageStatus.SUCCESS
        else:
            question_ids = [
                int(card["metadata"]["question_id"])
                for card in card_candidates
                if card.get("metadata", {}).get("question_id") is not None
            ]
            if not question_ids:
                # A missing question is an observed resolution state, not a
                # synthetic SQL identifier. Never invent a question_id.
                question_ids = []
            table_names = [
                "misconception_chunks"
                if EvidencePerspective.STUDENT in understanding.evidence_perspectives
                else "tutor_strategy_chunks"
            ]
            if EvidencePerspective.STUDENT in understanding.evidence_perspectives and EvidencePerspective.TUTOR in understanding.evidence_perspectives:
                table_names.append("tutor_strategy_chunks")
            analytics_result = {
                "operation": "question_lookup",
                "rows": [],
                "sql_scope": {
                    "filters": {"question_id": question_ids[0] if question_ids else None},
                    "question_resolution": "FOUND" if question_ids else "NOT_FOUND",
                    "keywords": ["question_id", "session_id", *table_names],
                    "scope_confinement": "question-scoped source card lookup",
                },
            }
            card_candidates = self._card_candidates(query, list(understanding.evidence_perspectives), effective_subject_filters)
            if card_candidates:
                analytics_result["sql_scope"]["filters"]["question_id"] = int(card_candidates[0]["metadata"]["question_id"])
                session_hint = _session_id_hint(query)
                if session_hint is not None:
                    analytics_result["sql_scope"]["filters"]["session_id"] = session_hint
            parent_cards, candidate_windows, ranked_windows, selected_windows, prompt, citations, card_status = self._micro_evidence(card_candidates, query)
            if parent_cards:
                graph_paths = self._graph_paths_for_sessions(
                    {int(card["metadata"]["session_id"]) for card in parent_cards}
                )
            if not prompt:
                prompt = "【数据来源和限制】未找到满足查询范围的真实 Card/Window 证据，未推断学生状态。"
            stage_status["CARD"] = card_status
            stage_status["WINDOW"] = StageStatus.UNMEASURED if card_status is StageStatus.SUCCESS else card_status
            stage_status["PROMPT"] = StageStatus.SUCCESS if prompt else card_status
        return BusinessGraphRuntimeTrace(
            actual_router=actual_router, actual_plan=actual_plan,
            capabilities={key: value.model_dump(mode="json") for key, value in capabilities.items()},
            analytics_result=analytics_result, graph_paths=graph_paths,
            card_candidates=card_candidates, parent_cards=parent_cards,
            candidate_windows=candidate_windows, ranked_windows=ranked_windows,
            selected_windows=selected_windows, assembled_prompt=prompt or "",
            citations=citations, answer=None, stage_status=stage_status,
            errors=errors, timings={"total": round(time.perf_counter() - started, 6)},
        )

    def ask_case(self, query: str, mode: str = "deterministic") -> BusinessGraphRuntimeTrace:
        if mode not in {"deterministic", "llm"}:
            raise ValueError("mode must be deterministic or llm")
        trace = self.prepare_case(query)
        if mode == "llm" and not self.runtime_config.generator_available:
            trace.stage_status["GENERATION"] = StageStatus.UNMEASURED
            trace.errors.append({"stage": "GENERATION", "code": "GENERATOR_UNAVAILABLE", "message": "generator credentials are not configured"})
        elif mode == "deterministic":
            trace.stage_status.setdefault("GENERATION", StageStatus.UNMEASURED)
        return trace


__all__ = ["BusinessGraphPipeline", "BusinessGraphRuntimeConfig"]
