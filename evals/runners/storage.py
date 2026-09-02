from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from evals.contracts import (
    ComparisonOperator,
    MetricObservation,
    RunContext,
    StorageEvalSnapshot,
    observation_from_measurement,
    unmeasured_observation,
)


@dataclass(frozen=True)
class StorageThresholds:
    id_parity: float = 1.0
    metadata_contract_rate: float = 1.0
    orphan_rate: float = 0.0


class StorageRunner:
    """Evaluate relational/vector parity without mutating either store."""

    stage = "storage"

    def __init__(self, thresholds: StorageThresholds | None = None) -> None:
        self.thresholds = thresholds or StorageThresholds()

    def run(self, context: RunContext, snapshot: StorageEvalSnapshot) -> list[MetricObservation]:
        observations: list[MetricObservation] = []
        all_records = list(snapshot.additional_relational_records)

        for collection in snapshot.collections:
            all_records.extend(collection.relational_records)
            all_records.extend(collection.index_records)
            case_id = f"{snapshot.case_id}:{collection.name}"
            expected = set(collection.expected_ids)
            relational = {item.record_id for item in collection.relational_records}
            indexed = {item.record_id for item in collection.index_records}

            if not expected:
                observations.append(
                    unmeasured_observation(
                        context=context,
                        case_id=case_id,
                        stage=self.stage,
                        metric_name="id_parity",
                        threshold=self.thresholds.id_parity,
                        operator=ComparisonOperator.GTE,
                        hard_gate=True,
                        reason="expected ID set is empty; parity denominator is zero",
                        evidence_refs=[f"collection:{collection.name}"],
                    )
                )
            else:
                mismatched = (relational ^ indexed) | (expected ^ relational) | (expected ^ indexed)
                parity = max(0.0, 1.0 - len(mismatched) / len(expected))
                observations.append(
                    observation_from_measurement(
                        context=context,
                        case_id=case_id,
                        stage=self.stage,
                        metric_name="id_parity",
                        value=parity,
                        threshold=self.thresholds.id_parity,
                        operator=ComparisonOperator.GTE,
                        hard_gate=True,
                        evidence_refs=[f"mismatched_id:{item}" for item in sorted(mismatched)],
                    )
                )

            if not collection.index_records:
                observations.append(
                    unmeasured_observation(
                        context=context,
                        case_id=case_id,
                        stage=self.stage,
                        metric_name="metadata_contract_rate",
                        threshold=self.thresholds.metadata_contract_rate,
                        operator=ComparisonOperator.GTE,
                        hard_gate=True,
                        reason="index collection is empty; metadata coverage denominator is zero",
                    )
                )
            else:
                incomplete_ids = [
                    record.record_id
                    for record in collection.index_records
                    if not all(_is_meaningful(record.metadata.get(field)) for field in collection.required_metadata_fields)
                ]
                complete = len(collection.index_records) - len(incomplete_ids)
                observations.append(
                    observation_from_measurement(
                        context=context,
                        case_id=case_id,
                        stage=self.stage,
                        metric_name="metadata_contract_rate",
                        value=complete / len(collection.index_records),
                        threshold=self.thresholds.metadata_contract_rate,
                        operator=ComparisonOperator.GTE,
                        hard_gate=True,
                        evidence_refs=[f"incomplete_metadata:{item}" for item in incomplete_ids],
                    )
                )

        if not all_records:
            observations.append(
                unmeasured_observation(
                    context=context,
                    case_id=snapshot.case_id,
                    stage=self.stage,
                    metric_name="orphan_rate",
                    threshold=self.thresholds.orphan_rate,
                    operator=ComparisonOperator.LTE,
                    hard_gate=True,
                    reason="storage snapshot contains no child records",
                )
            )
        else:
            orphan_ids = [
                item.record_id
                for item in all_records
                if item.parent_id is None or item.parent_id not in snapshot.valid_parent_ids
            ]
            observations.append(
                observation_from_measurement(
                    context=context,
                    case_id=snapshot.case_id,
                    stage=self.stage,
                    metric_name="orphan_rate",
                    value=len(orphan_ids) / len(all_records),
                    threshold=self.thresholds.orphan_rate,
                    operator=ComparisonOperator.LTE,
                    hard_gate=True,
                    evidence_refs=[f"orphan:{item}" for item in orphan_ids],
                )
            )
        return observations


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        if stripped[:1] in {"[", "{"}:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return False
            return _is_meaningful(decoded)
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True

