from __future__ import annotations

from collections.abc import Callable

from evals.contracts import (
    ComparisonOperator,
    MetricObservation,
    MetricStatus,
    RunContext,
    observation_from_measurement,
    unmeasured_observation,
)
from evals.metrics.retrieval import wilson_interval


def aggregate_case_observations(
    *,
    context: RunContext,
    case_observations: list[MetricObservation],
    source_metric_name: str,
    aggregate_metric_name: str | None,
    stage: str,
    threshold: float,
    operator: ComparisonOperator,
    hard_gate: bool,
    scope: str = "aggregate",
    slices: dict[str, str] | None = None,
    reducer: Callable[[list[float]], float] | None = None,
) -> MetricObservation:
    matching = [item for item in case_observations if item.metric_name == source_metric_name]
    if not matching:
        raise ValueError(f"no case observations found for {source_metric_name}")
    measured = [
        item for item in matching if item.status in {MetricStatus.SUCCESS, MetricStatus.FAILED}
    ]
    if not measured:
        return unmeasured_observation(
            context=context,
            case_id=_aggregate_case_id(scope, slices),
            stage=stage,
            metric_name=aggregate_metric_name or source_metric_name,
            threshold=threshold,
            operator=operator,
            hard_gate=hard_gate,
            reason=f"all {len(matching)} case observations are UNMEASURED or ERROR",
            scope=scope,  # type: ignore[arg-type]
            slices=slices,
            sample_size=len(matching),
        )

    values = [item.value for item in measured]
    assert all(value is not None for value in values)
    numeric_values = [float(value) for value in values if value is not None]
    aggregate_value = reducer(numeric_values) if reducer else sum(numeric_values) / len(numeric_values)
    passed = sum(item.status is MetricStatus.SUCCESS for item in measured)
    interval = wilson_interval(passed, len(measured))
    return observation_from_measurement(
        context=context,
        case_id=_aggregate_case_id(scope, slices),
        stage=stage,
        metric_name=aggregate_metric_name or source_metric_name,
        value=aggregate_value,
        threshold=threshold,
        operator=operator,
        hard_gate=hard_gate,
        scope=scope,  # type: ignore[arg-type]
        slices=slices,
        sample_size=len(matching),
        passed_cases=passed,
        measured_cases=len(measured),
        unmeasured_cases=len(matching) - len(measured),
        confidence_interval=interval,
    )


def _aggregate_case_id(scope: str, slices: dict[str, str] | None) -> str:
    if scope == "aggregate":
        return "__aggregate__"
    if scope == "slice" and slices:
        encoded = ",".join(f"{key}={value}" for key, value in sorted(slices.items()))
        return f"__slice__:{encoded}"
    raise ValueError(f"unsupported aggregation scope: {scope}")

