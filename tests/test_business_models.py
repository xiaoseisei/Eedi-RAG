"""Part 1 Task 1: strict business boundary contract tests."""

import pytest
from pydantic import ValidationError

from src.business_models import (
    BusinessQueryRequest,
    BusinessResponseEnvelope,
    BusinessResponseStatus,
    ComputationScope,
    DataCapabilitySnapshot,
    DataScope,
    DetailLevel,
    EvidencePerspective,
    QueryIntent,
    QueryPlan,
    QueryPlanType,
    QueryScope,
    QueryUnderstanding,
    RouteConfidence,
)


def _understanding() -> QueryUnderstanding:
    return QueryUnderstanding(
        raw_query="学生最常提出的问题是什么？",
        primary_intent=QueryIntent.STUDENT_FREQUENT_QUESTIONS,
        secondary_intents=[],
        scope=QueryScope.CORPUS,
        computation_scope=ComputationScope.HYBRID,
        evidence_perspectives=[EvidencePerspective.STUDENT],
        subject_filters=[],
        protected_tokens=[],
        requested_top_k=3,
        requires_aggregation=True,
        requires_case_retrieval=True,
        response_type="STUDENT_FAQ_REPORT",
        confidence=RouteConfidence.HIGH,
        route_source="deterministic",
    )


def test_business_request_is_strict_and_bounded() -> None:
    request = BusinessQueryRequest(
        query="列出 Top 10 高频问题",
        top_k=10,
        detail_level=DetailLevel.FULL,
    )
    assert request.top_k == 10
    with pytest.raises(ValidationError):
        BusinessQueryRequest(query="x", top_k=11)
    with pytest.raises(ValidationError):
        BusinessQueryRequest(query="x", unknown=True)


def test_success_requires_result_and_verified_audit() -> None:
    with pytest.raises(ValidationError, match="SUCCESS"):
        BusinessResponseEnvelope(
            query="学生最常提出的问题是什么？",
            understanding=_understanding(),
            status=BusinessResponseStatus.SUCCESS,
            summary="有结果",
            data_scope=DataScope(
                dataset_version="eedi-20260907",
                index_version="business-events-v1",
                total_sessions_considered=1576,
                matched_sessions=100,
            ),
            result=None,
            evidence=[],
            limitations=[],
            audit_status="NOT_APPLICABLE",
            trace_id="trace-1",
        )


def test_query_plan_rejects_vector_only_frequency_plan() -> None:
    understanding = _understanding()
    with pytest.raises(ValidationError, match="aggregation"):
        QueryPlan(
            understanding=understanding,
            plan_type=QueryPlanType.CASE_PLAYBOOK,
            analytics_operations=[],
            retrieval_perspectives=[EvidencePerspective.STUDENT],
            required_capabilities=[],
            response_type="STUDENT_FAQ_REPORT",
        )


def test_capability_snapshot_cannot_claim_available_with_missing_requirements() -> None:
    with pytest.raises(ValidationError, match="missing"):
        DataCapabilitySnapshot(
            capability="RETAKE_COHORT",
            available=True,
            coverage=1.0,
            missing_requirements=["attempt_number"],
            source_artifact="analytics.duckdb",
        )
