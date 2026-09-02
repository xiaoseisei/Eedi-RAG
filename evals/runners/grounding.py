from __future__ import annotations

from dataclasses import dataclass

from evals.contracts import (
    ComparisonOperator,
    GroundingEvalCase,
    MetricObservation,
    RunContext,
    observation_from_measurement,
    unmeasured_observation,
)
from evals.metrics.citation import audit_grounding
from evals.runners.common import aggregate_case_observations


@dataclass(frozen=True)
class GroundingThresholds:
    citation_precision: float = 1.0
    citation_recall: float = 1.0
    role_accuracy: float = 1.0
    quote_grounding_precision: float = 1.0


class GroundingRunner:
    stage = "grounding"

    def __init__(self, thresholds: GroundingThresholds | None = None) -> None:
        self.thresholds = thresholds or GroundingThresholds()

    def run(self, context: RunContext, cases: list[GroundingEvalCase]) -> list[MetricObservation]:
        if not cases:
            raise ValueError("grounding runner requires at least one case")
        observations: list[MetricObservation] = []
        for case in cases:
            audit = audit_grounding(
                output_citations=case.output_citations,
                authoritative_turns=case.authoritative_turns,
                required_evidence=case.required_evidence,
                expected_speaker=case.expected_speaker,
                quote_match_mode=case.quote_match_mode,
            )
            evidence = [f"invalid_output_citation_index:{item}" for item in audit.invalid_citation_indexes]
            for metric_name, value, threshold in (
                ("citation_precision", audit.citation_precision, self.thresholds.citation_precision),
                ("citation_recall", audit.citation_recall, self.thresholds.citation_recall),
                ("role_accuracy", audit.role_accuracy, self.thresholds.role_accuracy),
                (
                    "quote_grounding_precision",
                    audit.quote_grounding_precision,
                    self.thresholds.quote_grounding_precision,
                ),
            ):
                if value is None:
                    observations.append(
                        unmeasured_observation(
                            context=context,
                            case_id=case.case_id,
                            stage=self.stage,
                            metric_name=metric_name,
                            threshold=threshold,
                            operator=ComparisonOperator.GTE,
                            hard_gate=True,
                            reason="output citation list is empty; metric denominator is zero",
                            slices=case.slices,
                        )
                    )
                else:
                    observations.append(
                        observation_from_measurement(
                            context=context,
                            case_id=case.case_id,
                            stage=self.stage,
                            metric_name=metric_name,
                            value=value,
                            threshold=threshold,
                            operator=ComparisonOperator.GTE,
                            hard_gate=True,
                            slices=case.slices,
                            evidence_refs=evidence,
                        )
                    )

        case_observations = list(observations)
        for metric_name, threshold in (
            ("citation_precision", self.thresholds.citation_precision),
            ("citation_recall", self.thresholds.citation_recall),
            ("role_accuracy", self.thresholds.role_accuracy),
            ("quote_grounding_precision", self.thresholds.quote_grounding_precision),
        ):
            observations.append(
                aggregate_case_observations(
                    context=context,
                    case_observations=case_observations,
                    source_metric_name=metric_name,
                    aggregate_metric_name=None,
                    stage=self.stage,
                    threshold=threshold,
                    operator=ComparisonOperator.GTE,
                    hard_gate=True,
                )
            )
        return observations

