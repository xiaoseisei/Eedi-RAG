from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evals.contracts import (
    ComparisonOperator,
    MetricObservation,
    MetricStatus,
    L1ComponentSuiteSpec,
    RunContext,
    observation_from_measurement,
)


def _context() -> RunContext:
    return RunContext(
        run_id="run-contracts",
        dataset_version="component-fixture-v1",
        qrels_version="qrels-fixture-v1",
        system_config_hash="a" * 64,
    )


def test_observation_factory_derives_status_from_threshold() -> None:
    passed = observation_from_measurement(
        context=_context(),
        case_id="case-1",
        stage="retrieval",
        metric_name="mrr",
        value=0.91,
        threshold=0.85,
        operator=ComparisonOperator.GTE,
        hard_gate=True,
    )
    failed = observation_from_measurement(
        context=_context(),
        case_id="case-2",
        stage="storage",
        metric_name="orphan_rate",
        value=0.01,
        threshold=0.0,
        operator=ComparisonOperator.LTE,
        hard_gate=True,
    )

    assert passed.status is MetricStatus.SUCCESS
    assert failed.status is MetricStatus.FAILED


def test_observation_rejects_falsified_success_status() -> None:
    with pytest.raises(ValidationError, match="SUCCESS"):
        MetricObservation(
            run_id="run-contracts",
            case_id="case-1",
            layer="L1",
            stage="retrieval",
            scope="case",
            metric_name="mrr",
            value=0.2,
            threshold=0.85,
            operator="gte",
            status="SUCCESS",
            hard_gate=True,
            evaluator_type="deterministic",
            dataset_version="component-fixture-v1",
            qrels_version="qrels-fixture-v1",
            system_config_hash="a" * 64,
            created_at=datetime.now(UTC),
        )


def test_unmeasured_observation_requires_reason_and_forbids_value() -> None:
    with pytest.raises(ValidationError):
        MetricObservation(
            run_id="run-contracts",
            case_id="case-1",
            layer="L1",
            stage="generator",
            scope="case",
            metric_name="faithfulness",
            value=0.0,
            threshold=0.95,
            operator="gte",
            status="UNMEASURED",
            hard_gate=True,
            evaluator_type="deepeval",
            dataset_version="component-fixture-v1",
            qrels_version="qrels-fixture-v1",
            system_config_hash="a" * 64,
            error_message="DeepEval is unavailable",
        )


    with pytest.raises(ValidationError, match="reason"):
        MetricObservation(
            run_id="run-contracts",
            case_id="case-1",
            layer="L1",
            stage="generator",
            scope="case",
            metric_name="faithfulness",
            value=None,
            threshold=0.95,
            operator="gte",
            status="UNMEASURED",
            hard_gate=True,
            evaluator_type="deepeval",
            dataset_version="component-fixture-v1",
            qrels_version="qrels-fixture-v1",
            system_config_hash="a" * 64,
        )


def test_empty_component_suite_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no executable module"):
        L1ComponentSuiteSpec(context=_context())
