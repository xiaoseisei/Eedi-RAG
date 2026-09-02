from __future__ import annotations

from dataclasses import dataclass

from evals.contracts import (
    AssemblerEvalCase,
    ComparisonOperator,
    MetricObservation,
    RunContext,
    observation_from_measurement,
    unmeasured_observation,
)
from evals.metrics.retrieval import ndcg_at_k, percentile
from evals.runners.common import aggregate_case_observations


@dataclass(frozen=True)
class AssemblerThresholds:
    candidate_recall_retention: float = 0.98
    # Diagnostic-only rates; thresholds are informational and do not block
    # release until qrels and slot semantics are formally calibrated.
    retriever_miss_rate: float = 1.0
    assembler_drop_rate: float = 1.0
    final_evidence_recall: float = 0.95
    selection_precision: float = 0.95
    noise_ratio: float = 0.05
    token_budget_violation_rate: float = 0.0
    evidence_empty_rate: float = 0.0
    compression_ratio: float = 0.0
    rerank_ndcg_gain: float = 0.0
    latency_p95_ms: float = 300.0


class AssemblerRunner:
    stage = "assembler"

    def __init__(self, thresholds: AssemblerThresholds | None = None) -> None:
        self.thresholds = thresholds or AssemblerThresholds()

    def run(self, context: RunContext, cases: list[AssemblerEvalCase]) -> list[MetricObservation]:
        if not cases:
            raise ValueError("assembler runner requires at least one case")
        case_observations: list[MetricObservation] = []
        for case in cases:
            case_observations.extend(self._evaluate_case(context, case))

        observations = list(case_observations)
        aggregate_specs = (
            ("candidate_recall_retention", ComparisonOperator.GTE, True, None),
            ("retriever_miss_rate", ComparisonOperator.LTE, False, None),
            ("assembler_drop_rate", ComparisonOperator.LTE, False, None),
            ("final_evidence_recall", ComparisonOperator.GTE, True, None),
            ("selection_precision", ComparisonOperator.GTE, True, None),
            ("noise_ratio", ComparisonOperator.LTE, True, None),
            ("token_budget_violation_rate", ComparisonOperator.LTE, True, None),
            ("evidence_empty_rate", ComparisonOperator.LTE, True, None),
            ("compression_ratio", ComparisonOperator.GTE, False, None),
            ("rerank_ndcg_gain", ComparisonOperator.GTE, False, None),
            ("latency_ms", ComparisonOperator.LTE, True, "latency_p95_ms"),
        )
        for source_name, operator, hard_gate, aggregate_name in aggregate_specs:
            threshold_name = aggregate_name or source_name
            observations.append(
                aggregate_case_observations(
                    context=context,
                    case_observations=case_observations,
                    source_metric_name=source_name,
                    aggregate_metric_name=aggregate_name,
                    stage=self.stage,
                    threshold=getattr(self.thresholds, threshold_name),
                    operator=operator,
                    hard_gate=hard_gate,
                    reducer=(lambda values: percentile(values, 0.95)) if source_name == "latency_ms" else None,
                )
            )
        return observations

    def _evaluate_case(self, context: RunContext, case: AssemblerEvalCase) -> list[MetricObservation]:
        observations: list[MetricObservation] = []
        positive = {document_id for document_id, relevance in case.qrels.items() if relevance > 0}
        input_positive = positive.intersection(case.input_ranked_ids)
        final_positive = positive.intersection(case.final_context_ids)

        # Keep upstream retrieval misses separate from assembler drops.  A
        # missing positive candidate is measured as a retriever miss; drop
        # rate is only defined when at least one positive candidate arrived.
        observations.append(observation_from_measurement(
            context=context,
            case_id=case.case_id,
            stage=self.stage,
            metric_name="retriever_miss_rate",
            value=float(not input_positive),
            threshold=1.0,
            operator=ComparisonOperator.LTE,
            hard_gate=False,
            slices=case.slices,
        ))
        if input_positive:
            retention = len(final_positive.intersection(input_positive)) / len(input_positive)
            observations.append(observation_from_measurement(
                context=context,
                case_id=case.case_id,
                stage=self.stage,
                metric_name="assembler_drop_rate",
                value=1.0 - retention,
                threshold=1.0,
                operator=ComparisonOperator.LTE,
                hard_gate=False,
                slices=case.slices,
            ))
        else:
            observations.append(unmeasured_observation(
                context=context,
                case_id=case.case_id,
                stage=self.stage,
                metric_name="assembler_drop_rate",
                threshold=1.0,
                operator=ComparisonOperator.LTE,
                hard_gate=False,
                reason="retriever supplied no positive candidate; assembler drop is undefined",
                slices=case.slices,
            ))

        # v2 evaluates each semantic slot independently.  This prevents a
        # strategy card from masking a missing misconception card (and vice
        # versa) when the assembler emits multiple fixed slots.
        for slot in ("misconception", "strategy", "fallback_window"):
            slot_positive = {
                doc_id for doc_id, relevance in case.qrels.items()
                if relevance > 0 and case.slot_by_id.get(doc_id) == slot
            }
            slot_final = {
                doc_id for doc_id in case.final_context_ids
                if case.slot_by_id.get(doc_id) == slot
            }
            if not slot_positive:
                for metric_name in (f"{slot}_slot_precision", f"{slot}_slot_recall"):
                    observations.append(unmeasured_observation(
                        context=context,
                        case_id=case.case_id,
                        stage=self.stage,
                        metric_name=metric_name,
                        threshold=1.0,
                        operator=ComparisonOperator.GTE,
                        hard_gate=True,
                        reason=f"no positive qrels for {slot} slot",
                        slices=case.slices,
                    ))
                continue
            precision = len(slot_final.intersection(slot_positive)) / len(slot_final) if slot_final else 0.0
            recall = len(slot_final.intersection(slot_positive)) / len(slot_positive)
            observations.extend([
                observation_from_measurement(
                    context=context,
                    case_id=case.case_id,
                    stage=self.stage,
                    metric_name=f"{slot}_slot_precision",
                    value=precision,
                    threshold=1.0,
                    operator=ComparisonOperator.GTE,
                    hard_gate=True,
                    slices=case.slices,
                ),
                observation_from_measurement(
                    context=context,
                    case_id=case.case_id,
                    stage=self.stage,
                    metric_name=f"{slot}_slot_recall",
                    value=recall,
                    threshold=1.0,
                    operator=ComparisonOperator.GTE,
                    hard_gate=True,
                    slices=case.slices,
                ),
            ])

        if not input_positive:
            observations.append(
                unmeasured_observation(
                    context=context,
                    case_id=case.case_id,
                    stage=self.stage,
                    metric_name="candidate_recall_retention",
                    threshold=1.0,
                    operator=ComparisonOperator.GTE,
                    hard_gate=True,
                    reason="retriever supplied no relevant candidate; assembler retention is undefined",
                    slices=case.slices,
                )
            )
        else:
            observations.append(
                observation_from_measurement(
                    context=context,
                    case_id=case.case_id,
                    stage=self.stage,
                    metric_name="candidate_recall_retention",
                    value=len(final_positive.intersection(input_positive)) / len(input_positive),
                    threshold=1.0,
                    operator=ComparisonOperator.GTE,
                    hard_gate=True,
                    slices=case.slices,
                )
            )

        required_hits = case.required_evidence_ids.intersection(case.final_context_ids)
        values: list[tuple[str, float, float, ComparisonOperator, bool]] = [
            (
                "final_evidence_recall",
                len(required_hits) / len(case.required_evidence_ids),
                1.0,
                ComparisonOperator.GTE,
                True,
            ),
            (
                "token_budget_violation_rate",
                float(case.estimated_tokens > case.max_prompt_tokens),
                self.thresholds.token_budget_violation_rate,
                ComparisonOperator.LTE,
                True,
            ),
            (
                "evidence_empty_rate",
                float(case.evidence_count == 0),
                self.thresholds.evidence_empty_rate,
                ComparisonOperator.LTE,
                True,
            ),
            (
                "latency_ms",
                case.latency_ms,
                self.thresholds.latency_p95_ms,
                ComparisonOperator.LTE,
                False,
            ),
        ]
        if case.final_context_ids:
            values.extend(
                [
                    (
                        "selection_precision",
                        len(final_positive) / len(case.final_context_ids),
                        1.0,
                        ComparisonOperator.GTE,
                        True,
                    ),
                    (
                        "noise_ratio",
                        (len(case.final_context_ids) - len(final_positive)) / len(case.final_context_ids),
                        self.thresholds.noise_ratio,
                        ComparisonOperator.LTE,
                        True,
                    ),
                ]
            )
        else:
            for metric_name, operator, threshold in (
                ("selection_precision", ComparisonOperator.GTE, 1.0),
                ("noise_ratio", ComparisonOperator.LTE, self.thresholds.noise_ratio),
            ):
                observations.append(
                    unmeasured_observation(
                        context=context,
                        case_id=case.case_id,
                        stage=self.stage,
                        metric_name=metric_name,
                        threshold=threshold,
                        operator=operator,
                        hard_gate=True,
                        reason="final context selection is empty; metric denominator is zero",
                        slices=case.slices,
                    )
                )

        if case.candidate_char_count:
            values.append(
                (
                    "compression_ratio",
                    1.0 - case.assembled_char_count / case.candidate_char_count,
                    self.thresholds.compression_ratio,
                    ComparisonOperator.GTE,
                    False,
                )
            )
        else:
            observations.append(
                unmeasured_observation(
                    context=context,
                    case_id=case.case_id,
                    stage=self.stage,
                    metric_name="compression_ratio",
                    threshold=self.thresholds.compression_ratio,
                    operator=ComparisonOperator.GTE,
                    hard_gate=False,
                    reason="candidate character count is zero; compression ratio is undefined",
                    slices=case.slices,
                )
            )

        k = max(1, min(5, len(case.input_ranked_ids)))
        before_ndcg = ndcg_at_k(case.input_ranked_ids, case.qrels, k)
        after_ndcg = ndcg_at_k(case.final_context_ids, case.qrels, k)
        values.append(
            (
                "rerank_ndcg_gain",
                after_ndcg - before_ndcg,
                self.thresholds.rerank_ndcg_gain,
                ComparisonOperator.GTE,
                False,
            )
        )

        for metric_name, value, threshold, operator, hard_gate in values:
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
