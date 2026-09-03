from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evals.contracts import (
    AssemblerEvalCase,
    EvidenceRef,
    GroundingEvalCase,
    RunContext,
    TurnKey,
    ChunkRankedRecord,
    ChunkingEvalCase,
    RewriteEvalCase,
)
from evals.metrics.citation import audit_grounding
from evals.runners.assembler import AssemblerRunner
from evals.runners.rewrite import RewriteRunner


def _context() -> RunContext:
    return RunContext(
        run_id="v2-contracts",
        dataset_version="l1-v2-fixture",
        qrels_version="qrels-v2-fixture",
        system_config_hash="a" * 64,
    )


def test_grounding_rejects_evidence_outside_retrieved_candidates() -> None:
    source = EvidenceRef(session_id=1, turn_id=1, speaker="student", quote_text="source")
    outside = EvidenceRef(session_id=1, turn_id=2, speaker="student", quote_text="outside")
    audit = audit_grounding(
        output_citations=[outside],
        authoritative_turns=[source, outside],
        required_evidence=[source],
        retrieved_evidence=[source],
    )
    assert audit.candidate_authorization_rate == 0.0
    assert audit.citation_precision == 0.0


def test_grounding_adapter_captures_real_response_citations_and_trace() -> None:
    from evals.adapters.grounding import capture_system_grounding_trace

    citation = {"session_id": 1, "turn_id": 2, "speaker": "student", "quote_text": "source"}
    response = SimpleNamespace(
        dialogue_citations=[citation],
        retrieved_sources_debug=[{"evidence_turns": [{**citation, "text": "source"}]}],
    )
    pipeline = SimpleNamespace(ask=lambda **kwargs: response)
    captured = capture_system_grounding_trace(pipeline, "q")
    assert captured["output_citations"] == [citation]
    assert captured["retrieved_evidence"][0]["turn_id"] == 2


def test_chunk_adapter_uses_production_tail_window() -> None:
    from evals.adapters.chunking import capture_production_chunks
    from src.models import CleanedSession

    sample_path = __import__("pathlib").Path("data/sample/cleaned_sessions_sample.jsonl")
    sample_session = CleanedSession.model_validate(json.loads(sample_path.read_text(encoding="utf-8").splitlines()[0]))

    chunks = capture_production_chunks(sample_session)
    assert chunks
    assert max(chunks[-1].source_turn_ids) == sample_session.turns[-1].turn_id
    assert chunks[-1].chunk_id.endswith(
        f"_{sample_session.turns[-1].turn_id}"
    )


def test_chunk_structure_audit_is_independent_of_retrieval_ranking() -> None:
    from evals.adapters.chunking import audit_production_chunk_structure
    from src.models import CleanedSession

    session = CleanedSession.model_validate(json.loads(__import__("pathlib").Path("data/sample/cleaned_sessions_sample.jsonl").read_text(encoding="utf-8").splitlines()[0]))
    audit = audit_production_chunk_structure(session)
    assert audit.pointer_validity == 1.0
    assert audit.verbatim_integrity == 1.0
    assert audit.boundary_integrity == 1.0


def test_chunk_retrieval_does_not_oracle_sort_hits() -> None:
    from scripts.build_l1_component_datasets import order_ranked_chunks

    records = [
        ChunkRankedRecord(chunk_id="miss", session_id=1, source_turn_ids={1}, token_count=1),
        ChunkRankedRecord(chunk_id="hit", session_id=1, source_turn_ids={2}, token_count=1),
    ]
    assert [item.chunk_id for item in order_ranked_chunks(records)] == ["miss", "hit"]


def test_strategy_query_uses_strategy_lane() -> None:
    from scripts.build_l1_component_datasets import query_lane_for_intent

    assert query_lane_for_intent("TUTOR_INTERVENTION") == "strategy"
    assert query_lane_for_intent("STUDENT_INSIGHT") == "misconception"


def test_unlabeled_intent_is_unmeasured() -> None:
    case = RewriteEvalCase(
        case_id="unlabeled",
        raw_query="why 2 + 2?",
        rewritten_query="explain 2 + 2",
        protected_tokens={"2"},
        critical_entities=set(),
        intent_preserved=None,
        intent_label_source=None,
        domain_injection_expected=False,
        raw_ranked_ids=["x"],
        rewritten_ranked_ids=["x"],
        qrels={"x": 3},
        latency_ms=1,
    )
    obs = RewriteRunner().run(_context(), [case])
    intent = next(item for item in obs if item.case_id == "unlabeled" and item.metric_name == "intent_preservation")
    assert intent.status.value == "UNMEASURED"


def test_fusion_compares_raw_with_full_rrf() -> None:
    from evals.contracts import FusionEvalCase

    case = FusionEvalCase(
        case_id="fusion-1",
        raw_query="how to help",
        intent="TUTOR_INTERVENTION",
        lane_ranked_ids={
            "raw": ["noise", "target"],
            "strategy": ["target", "noise"],
            "curriculum": ["noise", "target"],
        },
        rewrite_only_ranked_ids=["target", "noise"],
        raw_inclusive_ranked_ids=["target", "noise"],
        selected_ranked_ids=["noise", "target"],
        selected_strategy="raw_first",
        qrels={"target": 3},
        latency_ms=1,
    )
    obs = RewriteRunner().run(_context(), [case])
    names = {item.metric_name for item in obs}
    assert "raw_inclusive_rrf_mrr_lift" in names
    selected_lift = next(
        item for item in obs
        if item.metric_name == "selected_fusion_mrr_lift" and item.scope == "aggregate"
    )
    assert selected_lift.value == 0.0
    assert selected_lift.hard_gate is True


def test_assembler_precision_is_slot_scoped() -> None:
    case = AssemblerEvalCase(
        case_id="slot-1",
        input_ranked_ids=["m1", "s1"],
        final_context_ids=["m1", "s1"],
        qrels={"m1": 3, "s1": 3},
        required_evidence_ids={"m1", "s1"},
        slot_by_id={"m1": "misconception", "s1": "strategy"},
        candidate_char_count=100,
        assembled_char_count=80,
        estimated_tokens=20,
        max_prompt_tokens=100,
        evidence_count=2,
        latency_ms=1,
    )
    obs = AssemblerRunner().run(_context(), [case])
    assert any(item.metric_name == "misconception_slot_precision" for item in obs)
    assert any(item.metric_name == "strategy_slot_precision" for item in obs)


def test_assembler_trace_contains_real_evidence() -> None:
    from evals.adapters.assembler import assemble_with_trace

    evidence = {"session_id": 1, "turn_id": 1, "speaker": "student", "text": "source"}
    assembler = SimpleNamespace(
        assemble=lambda raw_query, retrieval_results: SimpleNamespace(
            selected_misconception=retrieval_results["misconceptions"][0],
            selected_strategy=None,
            evidence_turns=[evidence],
            prompt_context_markdown="source",
            estimated_token_count=2,
        )
    )
    result = assemble_with_trace("q", {"misconceptions": [{"chunk_id": "m1", "document": "source", "metadata": {}}]}, assembler)
    assert result["final_evidence_turns"] == [evidence]
    assert result["actual_prompt_context"] == "source"


def test_assembler_trace_is_json_serializable_when_backend_returns_arrays() -> None:
    import numpy as np

    from evals.adapters.assembler import assemble_with_trace

    assembler = SimpleNamespace(
        assemble=lambda raw_query, retrieval_results: SimpleNamespace(
            selected_misconception=None,
            selected_strategy=None,
            evidence_turns=[{
                "session_id": np.int64(1),
                "turn_id": np.int64(2),
                "speaker": "student",
                "text": "source",
                "talk_moves": np.array(["<Press for Accuracy>"]),
            }],
            prompt_context_markdown="source",
            estimated_token_count=np.int64(2),
        )
    )
    trace = assemble_with_trace("q", {}, assembler)
    assert json.loads(json.dumps(trace))["final_evidence_turns"][0]["turn_id"] == 2


def test_v2_builder_requires_explicit_artifact_paths(tmp_path) -> None:
    from scripts.build_l1_component_datasets import parse_args

    db = tmp_path / "db.duckdb"
    db.write_bytes(b"fixture")
    chroma = tmp_path / "chroma"
    chroma.mkdir()

    args = parse_args([
        "--db", str(db),
        "--chroma", str(chroma),
    ])
    assert args.db.name == "db.duckdb"
    assert args.chroma.name == "chroma"


def test_chunk_retrieval_does_not_measure_generation_inflation() -> None:
    from evals.runners.chunking import ChunkingRunner

    case = ChunkingEvalCase(
        case_id="retrieval-window",
        query="q",
        target_turns=[TurnKey(session_id=1, turn_id=1)],
        ranked_chunks=[ChunkRankedRecord(chunk_id="w", session_id=1, source_turn_ids={1}, token_count=100)],
        original_token_count=1,
        emitted_chunk_token_count=100,
        latency_ms=1,
        slices={"source": "fallback_windows_actual_ranking"},
        inflation_applicable=False,
    )
    obs = ChunkingRunner().run(_context(), [case])
    inflation = next(item for item in obs if item.metric_name == "inflation_ratio" and item.scope == "case")
    assert inflation.status.value == "UNMEASURED"
    assert inflation.value is None


def test_document_qrels_deduplicate_quote_evidence_and_mark_derived() -> None:
    from scripts.build_l1_component_datasets import build_document_qrels

    rows = build_document_qrels({
        "category": "STUDENT_INSIGHT",
        "verbatim_grounding_quotes": [
            {"session_id": 1, "turn_id": 2},
            {"session_id": 1, "turn_id": 3},
            {"session_id": 1, "turn_id": 2},
        ],
    }, query_id="q1")
    assert len(rows) == 1
    assert rows[0]["document_id"] == "session_1_misconception"
    assert rows[0]["label_source"] == "derived_from_human_quote"
    assert rows[0]["evidence_turns"] == [{"session_id": 1, "turn_id": 2}, {"session_id": 1, "turn_id": 3}]


def test_assembler_qrels_do_not_invent_unlabeled_complementary_slot() -> None:
    from scripts.build_l1_component_datasets import build_assembler_qrels

    rows = build_assembler_qrels({
        "category": "STUDENT_INSIGHT",
        "verbatim_grounding_quotes": [{"session_id": 10, "turn_id": 1}],
    }, query_id="q")

    assert [row["slot"] for row in rows] == ["misconception"]


def test_assembler_runner_separates_retriever_miss_and_assembler_drop() -> None:
    miss = AssemblerEvalCase(
        case_id="miss",
        input_ranked_ids=["noise"],
        final_context_ids=["noise"],
        qrels={"target": 3},
        required_evidence_ids={"target"},
        slot_by_id={"target": "misconception", "noise": "misconception"},
        candidate_char_count=10,
        assembled_char_count=5,
        estimated_tokens=2,
        max_prompt_tokens=100,
        evidence_count=1,
        latency_ms=1,
    )
    drop = miss.model_copy(update={"case_id": "drop", "input_ranked_ids": ["target", "noise"], "final_context_ids": ["noise"], "slot_by_id": {"target": "misconception", "noise": "misconception"}})
    obs = AssemblerRunner().run(_context(), [miss, drop])
    miss_rate = next(item for item in obs if item.metric_name == "retriever_miss_rate" and item.case_id == "miss")
    drop_rate = next(item for item in obs if item.metric_name == "assembler_drop_rate" and item.case_id == "drop")
    assert miss_rate.value == 1.0
    assert drop_rate.value == 1.0


def test_manifest_provenance_contains_distinct_counts_and_hashes(tmp_path) -> None:
    from scripts.build_l1_component_datasets import build_manifest_provenance

    db = tmp_path / "db.duckdb"
    chroma = tmp_path / "chroma"
    db.write_bytes(b"db")
    chroma.mkdir()
    (chroma / "index.bin").write_bytes(b"index")
    result = build_manifest_provenance(
        db_path=db,
        chroma_path=chroma,
        source_session_count=1576,
        indexed_session_count=100,
        golden_case_count=30,
        system_config={"embedding_backend": "deterministic"},
    )
    assert result["source_session_count"] == 1576
    assert result["indexed_session_count"] == 100
    assert result["golden_case_count"] == 30
    assert len(result["artifact_hashes"]["duckdb_sha256"]) == 64
    assert len(result["artifact_hashes"]["chroma_sha256"]) == 64
    assert len(result["system_config_hash"]) == 64


def test_split_manifest_keeps_connected_source_sessions_in_one_split() -> None:
    from scripts.build_l1_component_datasets import build_split_manifest

    cases = [
        {"question": "q1", "verbatim_grounding_quotes": [{"session_id": 10}]},
        {"question": "q2", "verbatim_grounding_quotes": [{"session_id": 10}, {"session_id": 20}]},
        {"question": "q3", "verbatim_grounding_quotes": [{"session_id": 30}]},
        {"question": "q4", "verbatim_grounding_quotes": [{"session_id": 40}]},
        {"question": "q5", "verbatim_grounding_quotes": [{"session_id": 50}]},
    ]

    manifest = build_split_manifest(cases, holdout_fraction=0.4, salt="unit-test")
    assignments = manifest["query_assignments"]

    assert assignments["eedi-l1-0001"] == assignments["eedi-l1-0002"]
    assert manifest["session_leakage_count"] == 0
    assert set(assignments.values()) == {"dev", "holdout"}
    assert manifest["split_counts"]["dev"] + manifest["split_counts"]["holdout"] == 5
