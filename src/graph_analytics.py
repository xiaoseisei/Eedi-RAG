"""图投影上的确定性业务宏观分析。"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import duckdb
from pydantic import Field

from src.business_models import StrictBusinessModel
from src.graph_store import GraphStore


class RepresentativeEvidenceRef(StrictBusinessModel):
    source_event_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    turn_ids: list[int] = Field(min_length=1)
    perspective: Literal["STUDENT", "TUTOR"]
    subject_path: str = Field(min_length=1)


class DataScope(StrictBusinessModel):
    dataset_version: str = Field(min_length=1)
    graph_version: str = Field(min_length=1)
    total_sessions_considered: int = Field(ge=0)
    matched_sessions: int = Field(ge=0)
    subject_filters: list[str] = Field(default_factory=list)
    unmeasured_metrics: list[str] = Field(default_factory=list)


class AnalyticsResult(StrictBusinessModel):
    operation: str = Field(min_length=1)
    rows: list[dict[str, Any]]
    data_scope: DataScope
    representative_refs: list[RepresentativeEvidenceRef] = Field(default_factory=list)
    derivation_version: str = Field(min_length=1)
    result_status: Literal["MEASURED", "NO_MATCHING_SCOPE", "NO_EVIDENCE", "UNMEASURED"] = "MEASURED"
    status_reason: str | None = None


class GraphRepository:
    """绑定只读 source DuckDB 与 READY graph sidecar。"""

    def __init__(self, *, graph_store: GraphStore, source_connection: duckdb.DuckDBPyConnection,
                 dataset_version: str = "source-v1") -> None:
        self.graph_store = graph_store
        self.source_connection = source_connection
        self.dataset_version = dataset_version


class GraphAnalytics:
    """只做 SQL/图数据聚合；不读取向量分数，也不调用模型。"""

    def __init__(self, repository: GraphRepository) -> None:
        self.repo = repository
        self.manifest = repository.graph_store.get_manifest()
        if self.manifest.build_status != "READY":
            raise ValueError("graph analytics requires a READY graph projection")

    @staticmethod
    def _filters(subject_filters: Sequence[str]) -> tuple[str, list[str]]:
        values = list(subject_filters or [])
        if not values:
            return "", []
        return " AND " + " AND ".join("s.subject_path LIKE ?" for _ in values), [f"%{v}%" for v in values]

    def _scope(self, subject_filters: Sequence[str]) -> tuple[DataScope, str, list[Any]]:
        suffix, params = self._filters(subject_filters)
        total = self.repo.source_connection.execute("SELECT COUNT(DISTINCT intervention_id) FROM tutoring_sessions").fetchone()[0]
        matched = self.repo.source_connection.execute(
            "SELECT COUNT(DISTINCT s.intervention_id) FROM tutoring_sessions s WHERE 1=1" + suffix, params
        ).fetchone()[0]
        scope = DataScope(dataset_version=self.repo.dataset_version, graph_version=self.manifest.graph_version,
                          total_sessions_considered=int(total), matched_sessions=int(matched),
                          subject_filters=list(subject_filters or []),
                          unmeasured_metrics=["UNRESOLVED_ENDING"])
        return scope, suffix, params

    def _session_ids(self, subject_filters: Sequence[str]) -> list[int]:
        suffix, params = self._filters(subject_filters)
        rows = self.repo.source_connection.execute(
            "SELECT intervention_id FROM tutoring_sessions s WHERE 1=1" + suffix,
            params,
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _aggregate(self, operation: str, label_type: str, perspective: Literal["STUDENT", "TUTOR"],
                   top_k: int, subject_filters: Sequence[str]) -> AnalyticsResult:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scope, suffix, params = self._scope(subject_filters)
        session_ids = self._session_ids(subject_filters)
        if not session_ids:
            if subject_filters:
                result_status = "NO_MATCHING_SCOPE"
                reason = "explicit subject filters matched zero sessions"
            else:
                result_status = "NO_EVIDENCE"
                reason = "no sessions available for analytics"
            return AnalyticsResult(operation=operation, rows=[], data_scope=scope,
                                   representative_refs=[], derivation_version=self.manifest.graph_version,
                                   result_status=result_status, status_reason=reason)
        session_placeholders = ",".join("?" for _ in session_ids)
        sql = f"""
            SELECT a.canonical_label AS canonical_name,
                   COUNT(*) AS event_count,
                   COUNT(DISTINCT a.source_session_id) AS distinct_session_count
            FROM graph_label_assignments a
            WHERE a.label_type = ? AND a.source_session_id IN ({session_placeholders})
            GROUP BY a.canonical_label
            ORDER BY distinct_session_count DESC, event_count DESC, canonical_name ASC
            LIMIT ?
        """
        rows = self.repo.graph_store._connection.execute(sql, [label_type, *session_ids, top_k]).fetchall()
        if scope.matched_sessions <= 0:
            raise RuntimeError(
                "analytics session denominator is zero; refusing to fabricate session_share"
            )
        denominator = scope.matched_sessions
        output = [{"rank": i, "canonical_name": r[0], "event_count": int(r[1]),
                   "distinct_session_count": int(r[2]), "session_share": r[2] / denominator}
                  for i, r in enumerate(rows, 1)]
        refs = self._representatives(label_type, perspective, subject_filters, output)
        result_status = "MEASURED" if output else "NO_EVIDENCE"
        return AnalyticsResult(operation=operation, rows=output, data_scope=scope,
                               representative_refs=refs, derivation_version=self.manifest.graph_version,
                               result_status=result_status,
                               status_reason=None if output else "sessions matched but no aggregate rows were produced")

    def _representatives(self, label_type: str, perspective: Literal["STUDENT", "TUTOR"],
                         subject_filters: Sequence[str], output: list[dict[str, Any]]) -> list[RepresentativeEvidenceRef]:
        if not output:
            return []
        names = [row["canonical_name"] for row in output]
        placeholders = ",".join("?" for _ in names)
        session_ids = self._session_ids(subject_filters)
        if not session_ids:
            return []
        session_placeholders = ",".join("?" for _ in session_ids)
        subject_by_session = {
            int(row[0]): row[1]
            for row in self.repo.source_connection.execute(
                f"SELECT intervention_id, subject_path FROM tutoring_sessions WHERE intervention_id IN ({session_placeholders})",
                session_ids,
            ).fetchall()
        }
        rows = self.repo.graph_store._connection.execute(f"""
            SELECT a.source_node_id, a.source_session_id, a.source_turn_ids
            FROM graph_label_assignments a
            WHERE a.label_type=? AND a.canonical_label IN ({placeholders})
              AND a.source_session_id IN ({session_placeholders})
            ORDER BY a.source_session_id, a.source_node_id
        """, [label_type, *names, *session_ids]).fetchall()
        seen: set[int] = set(); refs = []
        for node_id, sid, turns in rows:
            if sid in seen:
                continue
            seen.add(int(sid))
            subject_path = subject_by_session.get(int(sid))
            if not subject_path:
                raise ValueError(f"session {sid} has no authoritative subject_path")
            raw_turn_ids = [int(value) for value in (turns or [])]
            speaker = "tutor" if perspective == "TUTOR" else "student"
            authorized_turns = self.repo.source_connection.execute(
                "SELECT turn_id FROM session_dialogue_turns "
                "WHERE intervention_id=? AND turn_id IN (SELECT UNNEST(?)) "
                "AND lower(speaker)=? ORDER BY turn_id",
                [int(sid), raw_turn_ids, speaker],
            ).fetchall()
            authorized_turn_ids = [int(value[0]) for value in authorized_turns]
            if not authorized_turn_ids:
                continue
            refs.append(RepresentativeEvidenceRef(source_event_id=node_id, session_id=int(sid),
                                                   turn_ids=authorized_turn_ids, perspective=perspective,
                                                   subject_path=subject_path))
        return refs

    def frequent_student_questions(self, top_k: int = 3, subject_filters: Sequence[str] = ()) -> AnalyticsResult:
        return self._aggregate("frequent_student_questions", "QUESTION_FUNCTION", "STUDENT", top_k, subject_filters)

    def difficult_concepts(self, top_k: int = 3, subject_filters: Sequence[str] = ()) -> AnalyticsResult:
        return self._aggregate("difficult_concepts", "DIFFICULTY_SIGNAL", "STUDENT", top_k, subject_filters)

    def recurring_misconceptions(self, top_k: int = 3, subject_filters: Sequence[str] = ()) -> AnalyticsResult:
        return self._aggregate("recurring_misconceptions", "MISCONCEPTION", "STUDENT", top_k, subject_filters)

    def exam_error_patterns(self, top_k: int = 3, subject_filters: Sequence[str] = ()) -> AnalyticsResult:
        return self._aggregate("exam_error_patterns", "EXAM_ERROR", "STUDENT", top_k, subject_filters)

    def tutor_strategy_patterns(self, top_k: int = 3, subject_filters: Sequence[str] = ()) -> AnalyticsResult:
        return self._aggregate("tutor_strategy_patterns", "TUTOR_STRATEGY", "TUTOR", top_k, subject_filters)

    def misconception_strategy_pairs(self, top_k: int = 3, subject_filters: Sequence[str] = ()) -> AnalyticsResult:
        scope, suffix, params = self._scope(subject_filters)
        session_ids = self._session_ids(subject_filters)
        if not session_ids:
            result_status = "NO_MATCHING_SCOPE" if subject_filters else "NO_EVIDENCE"
            reason = (
                "explicit subject filters matched zero sessions"
                if subject_filters
                else "no sessions available for analytics"
            )
            return AnalyticsResult(operation="misconception_strategy_pairs", rows=[], data_scope=scope,
                                   derivation_version=self.manifest.graph_version,
                                   result_status=result_status,
                                   status_reason=reason)
        placeholders = ",".join("?" for _ in session_ids)
        rows = self.repo.graph_store._connection.execute(f"""
            SELECT m.canonical_label, t.canonical_label, COUNT(*) event_count,
                   COUNT(DISTINCT m.source_session_id) distinct_session_count
            FROM graph_label_assignments m JOIN graph_label_assignments t
              ON m.source_session_id=t.source_session_id
            WHERE m.label_type='MISCONCEPTION' AND t.label_type='TUTOR_STRATEGY'
              AND m.source_session_id IN ({placeholders})
            GROUP BY m.canonical_label,t.canonical_label
            ORDER BY distinct_session_count DESC,event_count DESC,m.canonical_label,t.canonical_label
            LIMIT ?
        """, session_ids + [top_k]).fetchall()
        if scope.matched_sessions <= 0:
            raise RuntimeError(
                "analytics session denominator is zero; refusing to fabricate session_share"
            )
        output = [{"rank": i, "misconception": r[0], "strategy": r[1], "relation": "CO_OCCURS_WITH",
                   "event_count": int(r[2]), "distinct_session_count": int(r[3]),
                   "session_share": r[3] / scope.matched_sessions} for i, r in enumerate(rows, 1)]
        refs = []
        pair_names = {(r[0], r[1]) for r in rows}
        scoped_placeholders = ",".join("?" for _ in session_ids)
        for misc_name, strat_name in sorted(pair_names):
            pair_rows = self.repo.graph_store._connection.execute(f"""
                SELECT m.source_node_id, m.source_session_id, m.source_turn_ids
                FROM graph_label_assignments m JOIN graph_label_assignments t
                  ON m.source_session_id=t.source_session_id
                WHERE m.label_type='MISCONCEPTION' AND t.label_type='TUTOR_STRATEGY'
                  AND m.canonical_label=? AND t.canonical_label=?
                  AND m.source_session_id IN ({scoped_placeholders})
                ORDER BY m.source_session_id
            """, [misc_name, strat_name, *session_ids]).fetchall()
            for node_id, sid, turns in pair_rows:
                subject = self.repo.source_connection.execute(
                    "SELECT subject_path FROM tutoring_sessions WHERE intervention_id=?", [sid]
                ).fetchone()
                if not subject or not subject[0]:
                    raise ValueError(f"session {sid} has no authoritative subject_path")
                refs.append(RepresentativeEvidenceRef(source_event_id=node_id, session_id=int(sid),
                                                       turn_ids=[int(x) for x in turns], perspective="STUDENT",
                                                       subject_path=subject[0]))
                break
        return AnalyticsResult(
            operation="misconception_strategy_pairs",
            rows=output,
            data_scope=scope,
            representative_refs=refs,
            derivation_version=self.manifest.graph_version,
            result_status="MEASURED" if output else "NO_EVIDENCE",
            status_reason=None if output else "sessions matched but no co-occurrence rows were produced",
        )

    def content_opportunities(self, top_k: int = 3, subject_filters: Sequence[str] = ()) -> AnalyticsResult:
        result = self.recurring_misconceptions(1000, subject_filters)
        content_sessions = self._session_ids(subject_filters)
        placeholders = ",".join("?" for _ in content_sessions)
        rows = []
        for row in result.rows:
            label_sessions = {
                int(item[0]) for item in self.repo.graph_store._connection.execute(
                    f"SELECT source_session_id FROM graph_label_assignments WHERE label_type='MISCONCEPTION' AND canonical_label=? AND source_session_id IN ({placeholders})",
                    [row["canonical_name"], *content_sessions],
                ).fetchall()
            } if content_sessions else set()
            missing_sessions = {
                int(item[0]) for item in self.repo.graph_store._connection.execute(
                    f"SELECT DISTINCT source_session_id FROM graph_label_assignments WHERE label_type='CONTENT_OPPORTUNITY' AND source_session_id IN ({placeholders})",
                    content_sessions,
                ).fetchall()
            } if content_sessions else set()
            missing = int(bool(label_sessions & missing_sessions))
            repeated = row["event_count"]
            rows.append({**row, "missing_content_flag": missing, "repeated_error_count": repeated,
                         "priority_score": repeated * (1 - row["session_share"]) + missing})
        rows.sort(key=lambda r: (-r["priority_score"], r["canonical_name"]))
        for i, row in enumerate(rows, 1): row["rank"] = i
        return result.model_copy(update={"operation": "content_opportunities", "rows": rows[:top_k]})


__all__ = ["AnalyticsResult", "DataScope", "GraphAnalytics", "GraphRepository", "RepresentativeEvidenceRef"]
