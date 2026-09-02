from __future__ import annotations

from dataclasses import dataclass

from evals.contracts import ComparisonOperator, MetricObservation, RetrievalEvalCase, RunContext, observation_from_measurement
from evals.metrics.retrieval import (
    candidate_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)
from evals.runners.common import aggregate_case_observations


@dataclass(frozen=True)
class RetrievalThresholds:
    candidate_recall_at_20: float = 0.98
    recall_at_1: float = 0.0
    recall_at_3: float = 0.90
    recall_at_5: float = 0.95
    recall_at_20: float = 0.98
    precision_at_5: float = 0.0
    noise_ratio_at_5: float = 0.05
    mrr: float = 0.85
    ndcg_at_5: float = 0.90
    latency_p95_ms: float = 150.0


class RetrievalRunner:
    stage = "retrieval"

    def __init__(self, thresholds: RetrievalThresholds | None = None) -> None:
        self.thresholds = thresholds or RetrievalThresholds()

    def run(self, context: RunContext, cases: list[RetrievalEvalCase]) -> list[MetricObservation]:
        if not cases:
            raise ValueError("retrieval runner requires at least one case")
        case_observations: list[MetricObservation] = []
        for case in cases:
            values = self._case_values(case)
            for metric_name, value in values.items():
                threshold, operator, hard_gate = self._case_gate(metric_name)
                case_observations.append(
                    observation_from_measurement(
                        context=context,
                        case_id=case.case_id,
                        stage=self.stage,
                        metric_name=metric_name,
                        value=value,
                        threshold=threshold,
                        operator=operator,
                        hard_gate=hard_gate,
                        slices=case.slices,
                    )
                )

        observations = list(case_observations)
        observations.extend(self._aggregate(context, case_observations, cases, scope="aggregate", slices=None))

        slice_groups: dict[tuple[str, str], set[str]] = {}
        for case in cases:
            for key, value in case.slices.items():
                slice_groups.setdefault((key, value), set()).add(case.case_id)
        for (key, value), case_ids in sorted(slice_groups.items()):
            selected_observations = [item for item in case_observations if item.case_id in case_ids]
            selected_cases = [item for item in cases if item.case_id in case_ids]
            observations.extend(
                self._aggregate(
                    context,
                    selected_observations,
                    selected_cases,
                    scope="slice",
                    slices={key: value},
                )
            )
        return observations

    def _case_values(self, case: RetrievalEvalCase) -> dict[str, float]:
        return {
            "candidate_recall_at_20": candidate_recall_at_k(case.ranked_ids, case.qrels, 20),
            "recall_at_1": recall_at_k(case.ranked_ids, case.qrels, 1),
            "recall_at_3": recall_at_k(case.ranked_ids, case.qrels, 3),
            "recall_at_5": recall_at_k(case.ranked_ids, case.qrels, 5),
            "recall_at_20": recall_at_k(case.ranked_ids, case.qrels, 20),
            "precision_at_5": precision_at_k(case.ranked_ids, case.qrels, 5),
            "noise_ratio_at_5": 1.0 - precision_at_k(case.ranked_ids, case.qrels, 5),
            "mrr": mean_reciprocal_rank(case.ranked_ids, case.qrels),
            "ndcg_at_5": ndcg_at_k(case.ranked_ids, case.qrels, 5),
            "latency_ms": case.latency_ms,
        }

    def _case_gate(self, metric_name: str) -> tuple[float, ComparisonOperator, bool]:
        if metric_name == "latency_ms":
            return self.thresholds.latency_p95_ms, ComparisonOperator.LTE, False
        aggregate_threshold = getattr(self.thresholds, metric_name)
        if metric_name == "noise_ratio_at_5":
            return aggregate_threshold, ComparisonOperator.LTE, True
        if metric_name == "precision_at_5":
            return aggregate_threshold, ComparisonOperator.GTE, False
        case_threshold = 1.0 if metric_name.startswith(("candidate_recall", "recall_at_")) else aggregate_threshold
        hard_gate = metric_name != "recall_at_1"
        return case_threshold, ComparisonOperator.GTE, hard_gate

    def _aggregate(
        self,
        context: RunContext,
        case_observations: list[MetricObservation],
        cases: list[RetrievalEvalCase],
        *,
        scope: str,
        slices: dict[str, str] | None,
    ) -> list[MetricObservation]:
        result: list[MetricObservation] = []
        for metric_name in (
            "candidate_recall_at_20",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_20",
            "precision_at_5",
            "noise_ratio_at_5",
            "mrr",
            "ndcg_at_5",
        ):
            threshold = getattr(self.thresholds, metric_name)
            operator = ComparisonOperator.LTE if metric_name == "noise_ratio_at_5" else ComparisonOperator.GTE
            hard_gate = metric_name not in {"recall_at_1", "precision_at_5"}
            result.append(
                aggregate_case_observations(
                    context=context,
                    case_observations=case_observations,
                    source_metric_name=metric_name,
                    aggregate_metric_name=None,
                    stage=self.stage,
                    threshold=threshold,
                    operator=operator,
                    hard_gate=hard_gate,
                    scope=scope,
                    slices=slices,
                )
            )
        result.append(
            aggregate_case_observations(
                context=context,
                case_observations=case_observations,
                source_metric_name="latency_ms",
                aggregate_metric_name="latency_p95_ms",
                stage=self.stage,
                threshold=self.thresholds.latency_p95_ms,
                operator=ComparisonOperator.LTE,
                hard_gate=True,
                scope=scope,
                slices=slices,
                reducer=lambda values: percentile(values, 0.95),
            )
        )
        return result
