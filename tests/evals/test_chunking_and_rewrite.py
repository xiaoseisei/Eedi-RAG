from __future__ import annotations

from evals.contracts import (
    ChunkRankedRecord,
    ChunkingEvalCase,
    RewriteEvalCase,
    RunContext,
    TurnKey,
)
from evals.runners.chunking import ChunkingRunner
from evals.runners.rewrite import RewriteRunner


def _context() -> RunContext:
    return RunContext(
        run_id="run-chunk-rewrite",
        dataset_version="component-fixture-v1",
        qrels_version="qrels-fixture-v1",
        system_config_hash="e" * 64,
    )


def _aggregate(observations, metric_name: str):
    return next(
        item
        for item in observations
        if item.case_id == "__aggregate__" and item.metric_name == metric_name
    )


def test_chunking_runner_measures_turn_recall_mrr_inflation_and_latency() -> None:
    case = ChunkingEvalCase(
        case_id="chunk-1",
        query="rounding mistake",
        target_turns=[TurnKey(session_id=10, turn_id=2)],
        ranked_chunks=[
            ChunkRankedRecord(
                chunk_id="noise",
                session_id=20,
                source_turn_ids={1, 2},
                token_count=50,
            ),
            ChunkRankedRecord(
                chunk_id="target",
                session_id=10,
                source_turn_ids={1, 2, 3},
                token_count=100,
            ),
        ],
        original_token_count=100,
        emitted_chunk_token_count=200,
        latency_ms=12,
        slices={"strategy": "sliding-window"},
    )

    observations = ChunkingRunner().run(_context(), [case])

    assert _aggregate(observations, "session_recall_at_1").value == 0.0
    assert _aggregate(observations, "turn_recall_at_3").value == 1.0
    assert _aggregate(observations, "turn_mrr").value == 0.5
    assert _aggregate(observations, "inflation_ratio").value == 2.0
    assert _aggregate(observations, "latency_p95_ms").value == 12.0
    assert any(item.scope == "slice" and item.slices == {"strategy": "sliding-window"} for item in observations)


def test_rewrite_runner_measures_preservation_intent_injection_and_retrieval_lift() -> None:
    case = RewriteEvalCase(
        case_id="rewrite-1",
        raw_query="Why does -2m > 12 require flipping the inequality?",
        rewritten_query="Diagnose why -2m > 12 requires flipping the inequality in algebra.",
        protected_tokens={"-2", "12", ">"},
        critical_entities={"inequality"},
        intent_preserved=True,
        intent_label_source="human",
        domain_injection_expected=True,
        injected_domain_context="Algebra > Inequalities",
        raw_ranked_ids=["noise", "target"],
        rewritten_ranked_ids=["target", "noise"],
        qrels={"target": 3},
        latency_ms=2,
        slices={"mode": "deterministic"},
    )

    observations = RewriteRunner().run(_context(), [case])

    assert _aggregate(observations, "protected_token_preservation").value == 1.0
    assert _aggregate(observations, "entity_preservation").value == 1.0
    assert _aggregate(observations, "intent_preservation").value == 1.0
    assert _aggregate(observations, "domain_injection_coverage").value == 1.0
    assert _aggregate(observations, "retrieval_mrr_lift").value == 0.5
    assert _aggregate(observations, "retrieval_recall_at_5_lift").value == 0.0


def test_rewrite_runner_marks_non_applicable_semantic_metrics_unmeasured() -> None:
    case = RewriteEvalCase(
        case_id="rewrite-unmeasured",
        raw_query="rounding help",
        rewritten_query="rounding misconception help",
        protected_tokens=set(),
        critical_entities=set(),
        intent_preserved=None,
        intent_label_source=None,
        domain_injection_expected=False,
        injected_domain_context="",
        raw_ranked_ids=[],
        rewritten_ranked_ids=[],
        qrels={"target": 3},
        latency_ms=1,
    )

    observations = RewriteRunner().run(_context(), [case])
    case_metrics = {item.metric_name: item for item in observations if item.case_id == case.case_id}

    assert case_metrics["protected_token_preservation"].status.value == "UNMEASURED"
    assert case_metrics["entity_preservation"].status.value == "UNMEASURED"
    assert case_metrics["intent_preservation"].status.value == "UNMEASURED"
    assert case_metrics["domain_injection_coverage"].status.value == "UNMEASURED"

