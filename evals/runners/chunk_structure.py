from __future__ import annotations

from evals.contracts import (
    ChunkStructuralEvalCase,
    ComparisonOperator,
    MetricObservation,
    RunContext,
    observation_from_measurement,
)
from evals.runners.common import aggregate_case_observations


class ChunkStructureRunner:
    stage = "chunk_structure"

    def run(self, context: RunContext, cases: list[ChunkStructuralEvalCase]) -> list[MetricObservation]:
        if not cases:
            raise ValueError("chunk structure runner requires at least one case")
        case_observations: list[MetricObservation] = []
        specs = (
            ("pointer_validity", ComparisonOperator.GTE, 1.0),
            ("verbatim_integrity", ComparisonOperator.GTE, 1.0),
            ("turn_coverage", ComparisonOperator.GTE, 1.0),
            ("boundary_integrity", ComparisonOperator.GTE, 1.0),
            ("orphan_count", ComparisonOperator.EQ, 0.0),
            ("out_of_bounds_turn_count", ComparisonOperator.EQ, 0.0),
            ("inflation_ratio", ComparisonOperator.LTE, 4.0),
        )
        for case in cases:
            for metric_name, operator, threshold in specs:
                case_observations.append(observation_from_measurement(
                    context=context,
                    case_id=case.case_id,
                    stage=self.stage,
                    metric_name=metric_name,
                    value=float(getattr(case, metric_name)),
                    threshold=threshold,
                    operator=operator,
                    hard_gate=metric_name != "inflation_ratio",
                    slices=case.slices,
                ))
        observations = list(case_observations)
        for metric_name, operator, threshold in specs:
            observations.append(aggregate_case_observations(
                context=context,
                case_observations=case_observations,
                source_metric_name=metric_name,
                aggregate_metric_name=None,
                stage=self.stage,
                threshold=threshold,
                operator=operator,
                hard_gate=metric_name != "inflation_ratio",
            ))
        return observations
