from __future__ import annotations

import pytest

from evals.contracts import (
    AssemblerEvalCase,
    EvidenceRef,
    GroundingEvalCase,
    IndexedRecord,
    RetrievalEvalCase,
    RunContext,
    StorageCollectionSnapshot,
    StorageEvalSnapshot,
)
from evals.runners.assembler import AssemblerRunner
from evals.runners.grounding import GroundingRunner
from evals.runners.retrieval import RetrievalRunner
from evals.runners.storage import StorageRunner


@pytest.fixture
def context() -> RunContext:
    return RunContext(
        run_id="run-runners",
        dataset_version="component-fixture-v1",
        qrels_version="qrels-fixture-v1",
        system_config_hash="b" * 64,
    )


def _observation(observations, metric_name: str, *, case_id: str = "__aggregate__"):
    return next(
        item
        for item in observations
        if item.metric_name == metric_name and item.case_id == case_id
    )


def test_storage_runner_measures_id_parity_orphans_and_metadata(context: RunContext) -> None:
    snapshot = StorageEvalSnapshot(
        case_id="storage-current",
        valid_parent_ids={10, 20},
        collections=[
            StorageCollectionSnapshot(
                name="student_misconceptions",
                expected_ids={"m10", "m20"},
                relational_records=[
                    IndexedRecord(record_id="m10", parent_id=10, metadata={}),
                    IndexedRecord(record_id="m20", parent_id=20, metadata={}),
                ],
                index_records=[
                    IndexedRecord(
                        record_id="m10",
                        parent_id=10,
                        metadata={"misconception_name": "place value", "deep_mechanism": "confusion"},
                    ),
                    IndexedRecord(
                        record_id="m20",
                        parent_id=20,
                        metadata={"misconception_name": "fractions", "deep_mechanism": ""},
                    ),
                ],
                required_metadata_fields=["misconception_name", "deep_mechanism"],
            )
        ],
        additional_relational_records=[
            IndexedRecord(record_id="turn-10-1", parent_id=10, metadata={}),
            IndexedRecord(record_id="turn-orphan", parent_id=999, metadata={}),
        ],
    )

    observations = StorageRunner().run(context, snapshot)

    assert _observation(observations, "id_parity", case_id="storage-current:student_misconceptions").value == 1.0
    metadata = _observation(
        observations,
        "metadata_contract_rate",
        case_id="storage-current:student_misconceptions",
    )
    assert metadata.value == 0.5
    assert metadata.status.value == "FAILED"
    orphan = _observation(observations, "orphan_rate", case_id="storage-current")
    assert orphan.value == pytest.approx(1 / 6)
    assert orphan.status.value == "FAILED"


def test_grounding_runner_reports_unmeasured_precision_for_empty_output(context: RunContext) -> None:
    source = EvidenceRef(session_id=10, turn_id=1, speaker="student", quote_text="5.45 maybe")
    observations = GroundingRunner().run(
        context,
        [
            GroundingEvalCase(
                case_id="grounding-1",
                output_citations=[],
                authoritative_turns=[source],
                required_evidence=[source],
                expected_speaker="student",
            )
        ],
    )

    precision = _observation(observations, "citation_precision", case_id="grounding-1")
    recall = _observation(observations, "citation_recall", case_id="grounding-1")
    assert precision.status.value == "UNMEASURED"
    assert precision.value is None
    assert recall.value == 0.0
    assert recall.status.value == "FAILED"


def test_retrieval_runner_emits_case_aggregate_slice_and_latency_metrics(context: RunContext) -> None:
    cases = [
        RetrievalEvalCase(
            case_id="r1",
            query="misconception query",
            ranked_ids=["noise", "a"],
            qrels={"a": 3},
            latency_ms=10,
            slices={"card_type": "misconception"},
        ),
        RetrievalEvalCase(
            case_id="r2",
            query="strategy query",
            ranked_ids=["b", "noise"],
            qrels={"b": 3},
            latency_ms=30,
            slices={"card_type": "strategy"},
        ),
    ]

    observations = RetrievalRunner().run(context, cases)

    assert _observation(observations, "mrr").value == 0.75
    assert _observation(observations, "candidate_recall_at_20").value == 1.0
    assert _observation(observations, "recall_at_1").value == 0.5
    assert _observation(observations, "noise_ratio_at_5").value == 0.8
    assert _observation(observations, "noise_ratio_at_5").status.value == "FAILED"
    assert _observation(observations, "recall_at_1").hard_gate is False
    assert _observation(observations, "precision_at_5").hard_gate is False
    assert _observation(observations, "latency_p95_ms").value == pytest.approx(29.0)
    slice_obs = next(
        item
        for item in observations
        if item.scope == "slice"
        and item.metric_name == "mrr"
        and item.slices == {"card_type": "strategy"}
    )
    assert slice_obs.value == 1.0


def test_assembler_runner_measures_retention_precision_budget_and_compression(context: RunContext) -> None:
    case = AssemblerEvalCase(
        case_id="assembler-1",
        input_ranked_ids=["supporting", "direct", "noise"],
        final_context_ids=["direct"],
        qrels={"direct": 3, "supporting": 1, "noise": 0},
        required_evidence_ids={"direct"},
        candidate_char_count=1000,
        assembled_char_count=250,
        estimated_tokens=100,
        max_prompt_tokens=1500,
        evidence_count=1,
        latency_ms=5,
    )

    observations = AssemblerRunner().run(context, [case])

    assert _observation(observations, "candidate_recall_retention").value == 1.0
    assert _observation(observations, "final_evidence_recall").value == 1.0
    assert _observation(observations, "selection_precision").value == 1.0
    assert _observation(observations, "noise_ratio").value == 0.0
    assert _observation(observations, "token_budget_violation_rate").value == 0.0
    assert _observation(observations, "evidence_empty_rate").value == 0.0
    assert _observation(observations, "compression_ratio").value == 0.75
