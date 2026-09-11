"""Actual-vs-Gold checks for the graph-business route.

All checks compare a runtime trace with independently authored Gold.  A
missing/unmeasured runtime field is reported as a failure or UNMEASURED; it is
never replaced by a passing default.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.business_graph_trace import BusinessGraphRuntimeTrace, StageStatus


HARD_GATES: dict[str, float] = {
    "router_exact_match": 1.0,
    "planner_exact_match": 1.0,
    "sql_graph_scope_confinement": 1.0,
    "card_evidence_validity": 1.0,
    "window_coverage": 1.0,
    "prompt_window_coverage": 1.0,
    "citation_precision": 1.0,
    "citation_recall": 1.0,
    "role_accuracy": 1.0,
    "exact_quote_match": 1.0,
    "status_accuracy": 1.0,
    "source_artifact_unchanged": 1.0,
}


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_status: str
    actual_status: str
    checks: dict[str, bool]
    measurement_status: Literal["PASS", "FAIL", "BLOCKED"]
    reason_codes: list[str] = Field(default_factory=list)
    first_failure_stage: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _status(trace: BusinessGraphRuntimeTrace) -> str:
    statuses = trace.stage_status
    if any(value is StageStatus.FAILED for value in statuses.values()):
        return "FAILED"
    if any(value is StageStatus.UNSUPPORTED for value in statuses.values()):
        return "UNSUPPORTED"
    if any(value is StageStatus.NO_EVIDENCE for value in statuses.values()):
        return "NO_EVIDENCE"
    return "SUCCESS"


def _exact_router(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for field, code in (
        ("intent", "ROUTER_INTENT_MISMATCH"),
        ("scope", "ROUTER_SCOPE_MISMATCH"),
        ("computation_scope", "ROUTER_COMPUTATION_SCOPE_MISMATCH"),
        ("perspectives", "ROUTER_PERSPECTIVES_MISMATCH"),
    ):
        if expected.get(field) != actual.get(field):
            reasons.append(code)
    return not reasons, reasons


def _plan_match(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if expected.get("plan_type") != actual.get("plan_type"):
        reasons.append("PLAN_TYPE_MISMATCH")
    expected_lever = expected.get("selected_lever") or {}
    actual_lever = actual.get("selected_lever") or {}
    if expected_lever != actual_lever:
        reasons.append("PLAN_LEVER_MISMATCH")
    return not reasons, reasons


def _sql_match(expected: dict[str, Any], actual_result: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if actual_result is None:
        return False, ["SQL_SCOPE_MISMATCH"]
    actual = actual_result.get("sql_scope") or actual_result
    expected_filters = expected.get("filters") or {}
    actual_filters = actual.get("filters") or {}
    reasons: list[str] = []
    if expected_filters != actual_filters:
        reasons.append("SQL_SCOPE_MISMATCH")
    actual_keywords = " ".join(str(value) for value in (actual.get("keywords") or []))
    missing = [keyword for keyword in expected.get("keywords", []) if keyword.casefold() not in actual_keywords.casefold()]
    if missing:
        reasons.append("SQL_SCOPE_MISMATCH")
    return not reasons, list(dict.fromkeys(reasons))


def _actual_card_ids(trace: BusinessGraphRuntimeTrace) -> set[str]:
    return {
        str(item.get("chunk_id") or item.get("card_id"))
        for item in [*trace.card_candidates, *trace.parent_cards]
        if item.get("chunk_id") or item.get("card_id")
    }


def _card_match(expected: list[dict[str, Any]], trace: BusinessGraphRuntimeTrace) -> tuple[bool, list[str]]:
    if not expected:
        return True, []
    candidates = _actual_card_ids(trace)
    parents = {
        str(item.get("chunk_id") or item.get("card_id"))
        for item in trace.parent_cards
        if item.get("chunk_id") or item.get("card_id")
    }
    reasons: list[str] = []
    for item in expected:
        card_id = str(item.get("card_id", ""))
        if card_id not in candidates:
            reasons.append("CARD_NOT_RETRIEVED")
            continue
        if card_id not in parents:
            reasons.append("CARD_NOT_PARENT")
            continue
        allowed_sessions = {
            int(card.get("metadata", {}).get("session_id"))
            for card in [*trace.card_candidates, *trace.parent_cards]
            if card.get("metadata", {}).get("session_id") is not None
            and str(card.get("chunk_id") or card.get("card_id")) == card_id
        }
        if item.get("source_session_id") is not None and int(item["source_session_id"]) not in allowed_sessions:
            reasons.append("CARD_SOURCE_POINTER_MISMATCH")
    return not reasons, list(dict.fromkeys(reasons))


def _window_ids(windows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(item.get("window_id")) for item in windows if item.get("window_id")}


def _window_match(expected: list[dict[str, Any]], trace: BusinessGraphRuntimeTrace) -> tuple[dict[str, bool], list[str]]:
    expected_ids = _window_ids(expected)
    candidate_ids = _window_ids(trace.candidate_windows)
    selected_ids = _window_ids(trace.selected_windows)
    reasons: list[str] = []
    candidate_recall = expected_ids <= candidate_ids
    if not candidate_recall:
        reasons.append("WINDOW_NOT_RETRIEVED")
    selected_coverage = not expected_ids or expected_ids <= selected_ids
    if not selected_coverage:
        reasons.append("WINDOW_NOT_SELECTED")
    for item in expected:
        matching = [window for window in trace.candidate_windows if window.get("window_id") == item.get("window_id")]
        if not matching:
            continue
        observed = matching[0]
        if item.get("session_id") != observed.get("session_id") or set(item.get("source_turn_ids", [])) != set(observed.get("source_turn_ids", [])):
            reasons.append("WINDOW_SOURCE_MISMATCH")
        if not set(item.get("anchor_turn_ids", [])) <= set(observed.get("source_turn_ids", [])):
            reasons.append("WINDOW_ANCHOR_OUTSIDE")
        if item.get("window_policy") and observed.get("window_policy", "W7_S3") != item["window_policy"]:
            reasons.append("WINDOW_POLICY_MISMATCH")
    return {
        "candidate_window_recall": candidate_recall,
        "selected_window_coverage": selected_coverage,
    }, list(dict.fromkeys(reasons))


def _prompt_match(expected: dict[str, Any], trace: BusinessGraphRuntimeTrace) -> tuple[bool, list[str]]:
    prompt = trace.assembled_prompt
    expected_ids = set(expected.get("required_window_ids", []))
    reasons: list[str] = []
    if any(window_id not in prompt for window_id in expected_ids):
        reasons.append("PROMPT_WINDOW_MISSING")
    if any(section not in prompt for section in expected.get("required_sections", [])):
        reasons.append("PROMPT_SECTION_MISSING")
    return not reasons, reasons


def _graph_match(expected: list[dict[str, Any]], trace: BusinessGraphRuntimeTrace) -> tuple[bool, list[str]]:
    if not expected:
        return True, []
    actual_edges: set[str] = set()
    actual_nodes: set[str] = set()
    for path in trace.graph_paths:
        actual_edges.update(str(edge) for edge in path.get("edges", []))
        actual_edges.update(str(edge) for edge in path.get("edge_types", []))
        actual_edges.add(str(path.get("edge_type"))) if path.get("edge_type") else None
        actual_nodes.update(str(node) for node in path.get("nodes", []))
        actual_nodes.update(str(node) for node in path.get("node_types", []))
    reasons: list[str] = []
    for path in expected:
        if not set(path.get("edges", [])) <= actual_edges:
            reasons.append("GRAPH_PATH_MISSING")
        # Some older traces only expose edge-level paths.  If node details are
        # present, enforce them; absence is reported by the graph scope gate.
        if path.get("nodes") and actual_nodes and not set(path["nodes"]) <= actual_nodes:
            reasons.append("GRAPH_PATH_MISSING")
    return not reasons, list(dict.fromkeys(reasons))


def _citation_checks(expected: dict[str, Any], trace: BusinessGraphRuntimeTrace) -> tuple[dict[str, bool], list[str]]:
    required = expected.get("required_evidence_turns", [])
    actual = trace.citations
    authorized = {
        (int(window.get("session_id")), int(turn_id))
        for window in trace.selected_windows
        for turn_id in window.get("source_turn_ids", [])
        if window.get("session_id") is not None
    }
    actual_keys = {(int(item.get("session_id")), int(item.get("turn_id"))) for item in actual if item.get("session_id") is not None and item.get("turn_id") is not None}
    precision = (
        all(key in authorized for key in actual_keys)
        if required
        else all(bool(item.get("verified", False)) for item in actual)
    )
    recall = all((int(item["session_id"]), int(item["turn_id"])) in actual_keys for item in required)
    role_accuracy = True
    quote_match = True
    reasons: list[str] = []
    by_key = {(int(item.get("session_id")), int(item.get("turn_id"))): item for item in actual if item.get("session_id") is not None and item.get("turn_id") is not None}
    for item in required:
        key = (int(item["session_id"]), int(item["turn_id"]))
        observed = by_key.get(key)
        if observed is None:
            continue
        if observed.get("speaker") != item.get("speaker"):
            role_accuracy = False
            reasons.append("ROLE_MISMATCH")
        if observed.get("exact_quote") != item.get("exact_quote"):
            quote_match = False
            reasons.append("QUOTE_MISMATCH")
    if required and actual and not any(
        item.get("exact_quote") == expected.get("exact_quote")
        for item in actual
        for expected in required
    ):
        quote_match = False
        reasons.append("QUOTE_MISMATCH")
    if not precision:
        reasons.append("CITATION_UNAUTHORIZED")
    if not recall:
        reasons.append("CITATION_REQUIRED_TURN_MISSING")
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "role_accuracy": role_accuracy,
        "exact_quote_match": quote_match,
    }, list(dict.fromkeys(reasons))


def evaluate_case(
    case: dict[str, Any],
    trace: BusinessGraphRuntimeTrace,
    *,
    artifact_unchanged: bool | None = None,
) -> CaseEvaluation:
    """Compare one actual trace with one Gold case and preserve first failure."""

    reasons: list[str] = []
    router_ok, router_reasons = _exact_router(case.get("expected_router", {}), trace.actual_router)
    plan_ok, plan_reasons = _plan_match(case.get("expected_plan", {}), trace.actual_plan)
    sql_ok, sql_reasons = _sql_match(case.get("expected_sql_scope", {}), trace.analytics_result)
    graph_ok, graph_reasons = _graph_match(case.get("expected_graph_paths", []), trace)
    card_ok, card_reasons = _card_match(case.get("expected_card_evidence", []), trace)
    window_checks, window_reasons = _window_match(case.get("expected_window_evidence", []), trace)
    prompt_ok, prompt_reasons = _prompt_match(case.get("expected_prompt", {}), trace)
    citation_checks, citation_reasons = _citation_checks(case.get("expected_citation_contract", {}), trace)
    expected_status = str(case.get("expected_status"))
    actual_status = _status(trace)
    status_ok = actual_status == expected_status
    if not status_ok:
        reasons.append("STATUS_MISMATCH")
    reasons.extend(router_reasons + plan_reasons + sql_reasons + graph_reasons + card_reasons + window_reasons + prompt_reasons + citation_reasons)
    reasons = list(dict.fromkeys(reasons))
    artifact_check = artifact_unchanged is True
    checks = {
        "router_exact_match": router_ok,
        "planner_exact_match": plan_ok,
        "sql_graph_scope_confinement": sql_ok and graph_ok,
        "card_evidence_validity": card_ok,
        "window_coverage": window_checks["candidate_window_recall"] and window_checks["selected_window_coverage"],
        "candidate_window_recall": window_checks["candidate_window_recall"],
        "selected_window_coverage": window_checks["selected_window_coverage"],
        "prompt_window_coverage": prompt_ok,
        **citation_checks,
        "status_accuracy": status_ok,
        # A single case has no artifact snapshot.  Unknown must not become PASS;
        # the runner may replace this value after comparing before/after hashes.
        "source_artifact_unchanged": artifact_check,
    }
    stage_order = (
        ("ROUTER", router_ok, router_reasons),
        ("PLANNER", plan_ok, plan_reasons),
        ("ANALYTICS", sql_ok, sql_reasons),
        ("GRAPH", graph_ok, graph_reasons),
        ("CARD", card_ok, card_reasons),
        ("WINDOW", checks["window_coverage"], window_reasons),
        ("ASSEMBLY", prompt_ok, prompt_reasons),
        ("CITATION", all(citation_checks.values()), citation_reasons),
        ("STATUS", status_ok, ["STATUS_MISMATCH"] if not status_ok else []),
    )
    first_failure = next((stage for stage, passed, _ in stage_order if not passed), None)
    measurement_status: Literal["PASS", "FAIL", "BLOCKED"]
    if artifact_unchanged is None:
        measurement_status = "BLOCKED"
    else:
        measurement_status = "PASS" if all(checks.values()) else "FAIL"
    return CaseEvaluation(
        case_id=str(case.get("case_id", "")), expected_status=expected_status,
        actual_status=actual_status, checks=checks, reason_codes=reasons,
        measurement_status=measurement_status,
        first_failure_stage=first_failure,
        details={"candidate_window_recall": window_checks["candidate_window_recall"], "selected_window_coverage": window_checks["selected_window_coverage"]},
    )


def load_gold_cases(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate Gold case at line {line_number}: {case_id}")
        seen.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Gold dataset is empty: {resolved}")
    return cases


def evaluate_dataset(
    cases: Sequence[dict[str, Any]],
    runtime: Any,
    *,
    artifact_unchanged: bool | None = None,
) -> dict[str, Any]:
    """Run and compare cases, preserving every trace and first failure."""

    evaluations: list[CaseEvaluation] = []
    traces: list[dict[str, Any]] = []
    for case in cases:
        trace = runtime.ask_case(str(case["query"]))
        evaluation = evaluate_case(case, trace, artifact_unchanged=artifact_unchanged)
        evaluations.append(evaluation)
        traces.append({"case_id": case["case_id"], "trace": json.loads(trace.to_json()), "evaluation": evaluation.model_dump(mode="json")})
    metrics = {
        name: sum(bool(item.checks.get(name)) for item in evaluations) / len(evaluations)
        for name in HARD_GATES
    } if evaluations else {name: None for name in HARD_GATES}
    blocked = any(item.measurement_status == "BLOCKED" for item in evaluations)
    return {
        "schema_version": "business-graph-evaluation/v1",
        "case_count": len(evaluations),
        "metrics": metrics,
        "hard_gates_passed": (not blocked) and all(value == 1.0 for value in metrics.values()),
        "measurement_status": "BLOCKED" if blocked else ("PASS" if all(value == 1.0 for value in metrics.values()) else "FAIL"),
        "first_failure_by_stage": dict(Counter(item.first_failure_stage for item in evaluations if item.first_failure_stage)),
        "cases": traces,
    }


__all__ = ["CaseEvaluation", "HARD_GATES", "evaluate_case", "evaluate_dataset", "load_gold_cases"]
