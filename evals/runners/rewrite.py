from __future__ import annotations

from dataclasses import dataclass

from evals.contracts import (
    ComparisonOperator,
    MetricObservation,
    RewriteEvalCase,
    FusionEvalCase,
    RunContext,
    observation_from_measurement,
    unmeasured_observation,
)
from evals.metrics.retrieval import mean_reciprocal_rank, percentile, recall_at_k
from evals.runners.common import aggregate_case_observations


@dataclass(frozen=True)
class RewriteThresholds:
    protected_token_preservation: float = 1.0
    entity_preservation: float = 0.99
    intent_preservation: float = 0.95
    domain_injection_coverage: float = 1.0
    raw_mrr: float = 0.0
    rewritten_mrr: float = 0.0
    retrieval_mrr_lift: float = 0.0
    raw_recall_at_5: float = 0.0
    rewritten_recall_at_5: float = 0.0
    retrieval_recall_at_5_lift: float = 0.0
    latency_p95_ms: float = 100.0


class RewriteRunner:
    stage = "rewrite"

    def __init__(self, thresholds: RewriteThresholds | None = None) -> None:
        self.thresholds = thresholds or RewriteThresholds()

    def run(self, context: RunContext, cases: list[RewriteEvalCase | FusionEvalCase]) -> list[MetricObservation]:
        if not cases:
            raise ValueError("rewrite runner requires at least one case")
        case_observations: list[MetricObservation] = []
        for case in cases:
            if isinstance(case, FusionEvalCase):
                case_observations.extend(self._evaluate_fusion_case(context, case))
            else:
                case_observations.extend(self._evaluate_case(context, case))

        observations = list(case_observations)
        if any(isinstance(case, RewriteEvalCase) for case in cases):
            observations.extend(self._aggregate(context, case_observations, scope="aggregate", slices=None))
        if any(isinstance(case, FusionEvalCase) for case in cases):
            observations.extend(self._aggregate_fusion(context, case_observations))
        if any(isinstance(case, RewriteEvalCase) for case in cases):
            slice_groups: dict[tuple[str, str], set[str]] = {}
            for case in cases:
                if isinstance(case, RewriteEvalCase):
                    for key, value in case.slices.items():
                        slice_groups.setdefault((key, value), set()).add(case.case_id)
            for (key, value), case_ids in sorted(slice_groups.items()):
                selected = [item for item in case_observations if item.case_id in case_ids]
                observations.extend(self._aggregate(context, selected, scope="slice", slices={key: value}))
        return observations

    def _aggregate_fusion(self, context: RunContext, case_observations: list[MetricObservation]) -> list[MetricObservation]:
        result = []
        for metric_name in (
            "raw_rrf_mrr",
            "rewrite_only_rrf_mrr",
            "raw_inclusive_rrf_mrr",
            "rewrite_only_rrf_mrr_lift",
            "raw_inclusive_rrf_mrr_lift",
            "raw_inclusive_rrf_recall_at_5_lift",
        ):
            matching = [item for item in case_observations if item.metric_name == metric_name]
            if not matching:
                continue
            result.append(aggregate_case_observations(
                context=context,
                case_observations=matching,
                source_metric_name=metric_name,
                aggregate_metric_name=None,
                stage="fusion",
                threshold=0.0,
                operator=ComparisonOperator.GTE,
                hard_gate=metric_name.startswith("raw_inclusive"),
            ))
        return result

    def _evaluate_fusion_case(self, context: RunContext, case: FusionEvalCase) -> list[MetricObservation]:
        raw = case.lane_ranked_ids["raw"]
        raw_mrr = mean_reciprocal_rank(raw, case.qrels)
        rewrite_mrr = mean_reciprocal_rank(case.rewrite_only_ranked_ids, case.qrels)
        inclusive_mrr = mean_reciprocal_rank(case.raw_inclusive_ranked_ids, case.qrels)
        raw_recall = recall_at_k(raw, case.qrels, 5)
        inclusive_recall = recall_at_k(case.raw_inclusive_ranked_ids, case.qrels, 5)
        values = {
            "raw_rrf_mrr": raw_mrr,
            "rewrite_only_rrf_mrr": rewrite_mrr,
            "raw_inclusive_rrf_mrr": inclusive_mrr,
            "rewrite_only_rrf_mrr_lift": rewrite_mrr - raw_mrr,
            "raw_inclusive_rrf_mrr_lift": inclusive_mrr - raw_mrr,
            "raw_inclusive_rrf_recall_at_5_lift": inclusive_recall - raw_recall,
            "latency_ms": case.latency_ms,
        }
        result: list[MetricObservation] = []
        for metric_name, value in values.items():
            threshold = self.thresholds.latency_p95_ms if metric_name == "latency_ms" else 0.0
            operator = ComparisonOperator.LTE if metric_name == "latency_ms" else ComparisonOperator.GTE
            result.append(observation_from_measurement(
                context=context,
                case_id=case.case_id,
                stage="fusion",
                metric_name=metric_name,
                value=value,
                threshold=threshold,
                operator=operator,
                hard_gate=False,
                slices={**case.slices, "intent": case.intent},
            ))
        return result

    def _evaluate_case(self, context: RunContext, case: RewriteEvalCase) -> list[MetricObservation]:
        observations: list[MetricObservation] = []
        rewritten = case.rewritten_query.casefold()
        optional_values = (
            (
                "protected_token_preservation",
                _coverage(case.protected_tokens, rewritten),
                1.0,
                True,
                "query has no explicitly protected numeric/formula/option tokens",
            ),
            (
                "entity_preservation",
                _coverage(case.critical_entities, rewritten),
                1.0,
                True,
                "query has no labeled critical entities",
            ),
            (
                "intent_preservation",
                float(case.intent_preserved) if case.intent_preserved is not None else None,
                1.0,
                True,
                "intent preservation has no human or DeepEval label",
            ),
            (
                "domain_injection_coverage",
                float(bool(case.injected_domain_context.strip())) if case.domain_injection_expected else None,
                1.0,
                False,
                "domain injection is not applicable to this case",
            ),
        )
        for metric_name, value, threshold, hard_gate, reason in optional_values:
            if value is None:
                observations.append(
                    unmeasured_observation(
                        context=context,
                        case_id=case.case_id,
                        stage=self.stage,
                        metric_name=metric_name,
                        threshold=threshold,
                        operator=ComparisonOperator.GTE,
                        hard_gate=hard_gate if metric_name == "intent_preservation" else False,
                        reason=reason,
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
                        hard_gate=hard_gate,
                        slices=case.slices,
                    )
                )

        raw_mrr = mean_reciprocal_rank(case.raw_ranked_ids, case.qrels)
        rewritten_mrr = mean_reciprocal_rank(case.rewritten_ranked_ids, case.qrels)
        raw_recall = recall_at_k(case.raw_ranked_ids, case.qrels, 5)
        rewritten_recall = recall_at_k(case.rewritten_ranked_ids, case.qrels, 5)
        measured_values = {
            "raw_mrr": raw_mrr,
            "rewritten_mrr": rewritten_mrr,
            "retrieval_mrr_lift": rewritten_mrr - raw_mrr,
            "raw_recall_at_5": raw_recall,
            "rewritten_recall_at_5": rewritten_recall,
            "retrieval_recall_at_5_lift": rewritten_recall - raw_recall,
            "latency_ms": case.latency_ms,
        }
        for metric_name, value in measured_values.items():
            threshold, operator, hard_gate = self._gate(metric_name, aggregate=False)
            observations.append(
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
        return observations

    def _gate(self, metric_name: str, *, aggregate: bool) -> tuple[float, ComparisonOperator, bool]:
        if metric_name == "latency_ms":
            return self.thresholds.latency_p95_ms, ComparisonOperator.LTE, False
        threshold = getattr(self.thresholds, metric_name)
        hard_gate = aggregate and metric_name in {"retrieval_mrr_lift", "retrieval_recall_at_5_lift"}
        return threshold, ComparisonOperator.GTE, hard_gate

    def _aggregate(
        self,
        context: RunContext,
        case_observations: list[MetricObservation],
        *,
        scope: str,
        slices: dict[str, str] | None,
    ) -> list[MetricObservation]:
        result = []
        metric_names = (
            "protected_token_preservation",
            "entity_preservation",
            "intent_preservation",
            "domain_injection_coverage",
            "raw_mrr",
            "rewritten_mrr",
            "retrieval_mrr_lift",
            "raw_recall_at_5",
            "rewritten_recall_at_5",
            "retrieval_recall_at_5_lift",
            "latency_ms",
        )
        for metric_name in metric_names:
            aggregate_name = "latency_p95_ms" if metric_name == "latency_ms" else None
            threshold_name = aggregate_name or metric_name
            if metric_name in {
                "protected_token_preservation",
                "entity_preservation",
                "intent_preservation",
                "domain_injection_coverage",
            }:
                operator = ComparisonOperator.GTE
                hard_gate = metric_name != "domain_injection_coverage"
                threshold = getattr(self.thresholds, metric_name)
            else:
                threshold, operator, hard_gate = self._gate(metric_name, aggregate=True)
                if aggregate_name:
                    threshold = getattr(self.thresholds, threshold_name)
            result.append(
                aggregate_case_observations(
                    context=context,
                    case_observations=case_observations,
                    source_metric_name=metric_name,
                    aggregate_metric_name=aggregate_name,
                    stage=self.stage,
                    threshold=threshold,
                    operator=operator,
                    hard_gate=hard_gate,
                    scope=scope,
                    slices=slices,
                    reducer=(lambda values: percentile(values, 0.95)) if metric_name == "latency_ms" else None,
                )
            )
        return result


def _coverage(expected: set[str], rewritten_casefold: str) -> float | None:
    if not expected:
        return None
    hits = sum(item.casefold() in rewritten_casefold for item in expected)
    return hits / len(expected)
