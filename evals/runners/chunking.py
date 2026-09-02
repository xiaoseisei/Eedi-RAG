from __future__ import annotations

from dataclasses import dataclass

from evals.contracts import ChunkingEvalCase, ComparisonOperator, MetricObservation, RunContext, observation_from_measurement
from evals.metrics.retrieval import percentile
from evals.runners.common import aggregate_case_observations


@dataclass(frozen=True)
class ChunkingThresholds:
    session_recall_at_1: float = 0.0
    session_recall_at_3: float = 0.95
    turn_recall_at_1: float = 0.0
    turn_recall_at_3: float = 0.95
    turn_mrr: float = 0.90
    inflation_ratio: float = 4.0
    latency_p95_ms: float = 150.0


class ChunkingRunner:
    stage = "chunking"

    def __init__(self, thresholds: ChunkingThresholds | None = None) -> None:
        self.thresholds = thresholds or ChunkingThresholds()

    def run(self, context: RunContext, cases: list[ChunkingEvalCase]) -> list[MetricObservation]:
        if not cases:
            raise ValueError("chunking runner requires at least one case")
        case_observations: list[MetricObservation] = []
        for case in cases:
            for metric_name, value in self._case_values(case).items():
                if metric_name == "inflation_ratio" and not case.inflation_applicable:
                    from evals.contracts import unmeasured_observation

                    case_observations.append(unmeasured_observation(
                        context=context,
                        case_id=case.case_id,
                        stage=self.stage,
                        metric_name=metric_name,
                        threshold=self.thresholds.inflation_ratio,
                        operator=ComparisonOperator.LTE,
                        hard_gate=False,
                        reason="retrieval candidate expansion is not chunk-generation inflation",
                        slices=case.slices,
                    ))
                    continue
                threshold, operator, hard_gate = self._gate(metric_name, per_case=True)
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
        observations.extend(self._aggregate(context, case_observations, scope="aggregate", slices=None))
        slice_groups: dict[tuple[str, str], set[str]] = {}
        for case in cases:
            for key, value in case.slices.items():
                slice_groups.setdefault((key, value), set()).add(case.case_id)
        for (key, value), case_ids in sorted(slice_groups.items()):
            selected = [item for item in case_observations if item.case_id in case_ids]
            observations.extend(
                self._aggregate(context, selected, scope="slice", slices={key: value})
            )
        return observations

    @staticmethod
    def _case_values(case: ChunkingEvalCase) -> dict[str, float]:
        target_turns = {(item.session_id, item.turn_id) for item in case.target_turns}
        target_sessions = {session_id for session_id, _ in target_turns}

        def session_recall(k: int) -> float:
            sessions = {item.session_id for item in case.ranked_chunks[:k]}
            return len(sessions.intersection(target_sessions)) / len(target_sessions)

        def turn_recall(k: int) -> float:
            covered = {
                (chunk.session_id, turn_id)
                for chunk in case.ranked_chunks[:k]
                for turn_id in chunk.source_turn_ids
            }
            return len(covered.intersection(target_turns)) / len(target_turns)

        first_turn_rank = next(
            (
                rank
                for rank, chunk in enumerate(case.ranked_chunks, start=1)
                if any((chunk.session_id, turn_id) in target_turns for turn_id in chunk.source_turn_ids)
            ),
            None,
        )
        return {
            "session_recall_at_1": session_recall(1),
            "session_recall_at_3": session_recall(3),
            "turn_recall_at_1": turn_recall(1),
            "turn_recall_at_3": turn_recall(3),
            "turn_mrr": 1.0 / first_turn_rank if first_turn_rank else 0.0,
            "inflation_ratio": case.emitted_chunk_token_count / case.original_token_count,
            "latency_ms": case.latency_ms,
        }

    def _gate(self, metric_name: str, *, per_case: bool) -> tuple[float, ComparisonOperator, bool]:
        if metric_name == "latency_ms":
            return self.thresholds.latency_p95_ms, ComparisonOperator.LTE, False
        if metric_name == "inflation_ratio":
            return self.thresholds.inflation_ratio, ComparisonOperator.LTE, False
        threshold = getattr(self.thresholds, metric_name)
        if per_case and "recall" in metric_name:
            threshold = 1.0
        hard_gate = metric_name not in {"session_recall_at_1", "turn_recall_at_1"}
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
        for metric_name in (
            "session_recall_at_1",
            "session_recall_at_3",
            "turn_recall_at_1",
            "turn_recall_at_3",
            "turn_mrr",
            "inflation_ratio",
            "latency_ms",
        ):
            threshold, operator, hard_gate = self._gate(metric_name, per_case=False)
            aggregate_name = "latency_p95_ms" if metric_name == "latency_ms" else None
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
