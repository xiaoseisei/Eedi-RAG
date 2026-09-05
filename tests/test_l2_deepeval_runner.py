from __future__ import annotations

from scripts.run_l2_deepeval import (
    _append_checkpoint,
    _load_checkpoint,
    _resolve_expected_artifact_hash,
    _resolve_l2_route,
    _select_l2_cases,
)
from scripts.merge_l2_deepeval_reports import merge_reports
from scripts.run_l2_deepeval import (
    _build_gold_context_v2,
    audit_gold_alignment,
    _metric_eval_payload,
    _rubric_config,
)


def test_l2_trace_claims_field_is_model_claim_ledger() -> None:
    from types import SimpleNamespace
    from scripts.run_l2_deepeval import _safe

    response = SimpleNamespace(
        generator_claims=[
            {
                "claim_id": "C1",
                "claim_type": "fact",
                "evidence_ids": ["E001"],
            }
        ]
    )

    claims = _safe(getattr(response, "__dict__", {}).get("generator_claims", []))

    assert claims[0]["claim_id"] == "C1"
    assert claims[0]["evidence_ids"] == ["E001"]


def test_l2_default_route_matches_production_anchored_parent3_w5() -> None:
    route = _resolve_l2_route(
        reranker_backend="siliconflow",
        rerank_unit=None,
        reranker_pool_size=None,
        parent_card_count=3,
        evidence_selection_count=5,
    )

    assert route == {
        "reranker_backend": "siliconflow",
        "rerank_unit": "anchored_logical_window",
        "reranker_pool_size": 20,
        "parent_card_count": 3,
        "evidence_selection_count": 5,
    }


def test_l2_explicit_legacy_route_remains_available() -> None:
    route = _resolve_l2_route(
        reranker_backend="none",
        rerank_unit="card",
        reranker_pool_size=15,
        parent_card_count=3,
        evidence_selection_count=5,
    )

    assert route["rerank_unit"] == "card"
    assert route["reranker_pool_size"] == 15


def test_l2_artifact_override_is_explicit_and_does_not_mutate_manifest() -> None:
    manifest = {"artifact_hashes": {"qwen_combined_sha256": "a" * 64}}

    expected, matches_frozen = _resolve_expected_artifact_hash(manifest, "b" * 64)

    assert expected == "b" * 64
    assert matches_frozen is False
    assert manifest["artifact_hashes"]["qwen_combined_sha256"] == "a" * 64


def test_l2_case_shards_preserve_original_case_numbers() -> None:
    golden = [{"question": f"q-{index}"} for index in range(1, 6)]

    selected = _select_l2_cases(golden, case_start=3, max_cases=2)

    assert [(index, case["question"]) for index, case in selected] == [
        (3, "q-3"),
        (4, "q-4"),
    ]


def test_l2_checkpoint_marks_only_complete_gold_real_pairs(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _append_checkpoint(
        checkpoint,
        [
            {"case_id": "eedi-l2-0001", "track": "gold"},
            {"case_id": "eedi-l2-0001", "track": "real"},
        ],
        [{"case_id": "eedi-l2-0001", "track": "gold", "metric": "x"}],
    )
    _append_checkpoint(
        checkpoint,
        [{"case_id": "eedi-l2-0002", "track": "gold"}],
        [],
    )

    traces, metrics, completed = _load_checkpoint(checkpoint)

    assert len(traces) == 2
    assert len(metrics) == 1
    assert completed == {1}


def test_l2_merge_rejects_overlapping_shards(tmp_path) -> None:
    def write_shard(path, case_index):
        path.mkdir()
        report = {
            "artifact_hash_before": "a" * 64,
            "l1_precondition": "BLOCKED",
            "runtime_route": {"rerank_unit": "anchored_logical_window"},
            "judge_readiness": {"status": "SUCCESS"},
        }
        (path / "report.json").write_text(__import__("json").dumps(report), encoding="utf-8")
        (path / "traces.jsonl").write_text(
            __import__("json").dumps({"case_id": f"eedi-l2-{case_index:04d}", "track": "gold"})
            + "\n"
            + __import__("json").dumps({"case_id": f"eedi-l2-{case_index:04d}", "track": "real"})
            + "\n",
            encoding="utf-8",
        )
        (path / "metrics.jsonl").write_text("", encoding="utf-8")

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_shard(first, 1)
    write_shard(second, 1)

    import pytest

    with pytest.raises(ValueError, match="overlapping"):
        merge_reports([first, second], tmp_path / "merged")


def test_gold_context_v2_has_card_facts_complete_windows_and_boundaries() -> None:
    import json
    from pathlib import Path
    from scripts.run_l2_deepeval import _load_sessions

    root = Path(__file__).resolve().parents[1]
    golden = json.loads((root / "data/golden_test_set.json").read_text(encoding="utf-8"))
    case = golden[0]
    sessions = _load_sessions(root / "data/cleaned_sessions.jsonl")

    class Storage:
        def query_misconceptions_sql(self):
            return [{
                "session_id": 10,
                "misconception_name": "place-value confusion",
                "deep_mechanism": "wrong target decimal place",
                "error_choice": "D",
                "confusion_triggers": ["1dp", "2dp"],
            }]

        def query_strategies_sql(self):
            return [{
                "session_id": 10,
                "pedagogical_goal": "identify the deciding digit",
                "strategy_category": "Scaffolding",
                "key_aha_question": "Which digit decides?",
                "scaffolding_steps": ["Find the next digit"],
            }]

    context = _build_gold_context_v2(case, sessions, Storage())

    assert "[CARD_FACT]" in context.prompt_context_markdown
    assert "[AUTHORITATIVE_TURN]" in context.prompt_context_markdown
    assert "[ALLOWED_INFERENCE]" in context.prompt_context_markdown
    assert "place-value confusion" in context.prompt_context_markdown
    assert "[Turn 10]" in context.prompt_context_markdown
    assert case["ground_truth"] not in context.prompt_context_markdown
    assert "required_evidence_quads" not in context.prompt_context_markdown


def test_gold_context_v2_exposes_deduplicated_required_evidence_nodes() -> None:
    import json
    from pathlib import Path
    from scripts.run_l2_deepeval import _load_sessions

    root = Path(__file__).resolve().parents[1]
    golden = json.loads((root / "data/golden_test_set.json").read_text(encoding="utf-8"))
    case = golden[0]
    sessions = _load_sessions(root / "data/cleaned_sessions.jsonl")

    class Storage:
        def query_misconceptions_sql(self):
            return [{"session_id": 10, "misconception_name": "place-value confusion"}]

        def query_strategies_sql(self):
            return [{"session_id": 10, "key_aha_question": "Which digit decides?"}]

    context = _build_gold_context_v2(case, sessions, Storage())

    assert len(context.deepeval_context_nodes) >= 3
    assert all(node.strip() for node in context.deepeval_context_nodes)
    assert context.deepeval_context_nodes[-1].startswith("## [ALLOWED_INFERENCE]")
    authoritative = [
        node for node in context.deepeval_context_nodes
        if node.startswith("## [AUTHORITATIVE_TURN]")
    ]
    assert authoritative
    combined = "\n".join(authoritative)
    for required in case["verbatim_grounding_quotes"]:
        if int(required["session_id"]) == 10:
            assert f"[Turn {required['turn_id']}]" in combined
    assert combined.count("[Turn 10]") == 1
    assert "[Turn 1]" not in combined
    for node in context.deepeval_context_nodes:
        assert node in context.prompt_context_markdown


def test_l2_rubric_v2_allows_qualified_inference_and_requires_actionable_pedagogy() -> None:
    rubric = _rubric_config("v2")

    assert "qualified" in rubric["faithfulness"].casefold()
    assert "unsupported" in rubric["faithfulness"].casefold()
    assert "actionable" in rubric["pedagogical_geval"].casefold()
    assert "misconception" in rubric["pedagogical_geval"].casefold()
    assert _rubric_config("v1") != rubric


def test_deepeval_payload_uses_core_answer_only_for_relevancy() -> None:
    core = "核心答案"
    structured = "核心答案\n诊断：详细诊断\n教学干预：步骤"
    nodes = ["card node", "turn node"]

    answer, context = _metric_eval_payload(
        "answer_relevancy",
        structured,
        actual_output_by_metric={"answer_relevancy": core},
        retrieval_context=nodes,
    )

    assert answer == core
    assert context == nodes


def test_deepeval_payload_keeps_structured_output_for_grounding_metrics() -> None:
    structured = "核心答案\n诊断：详细诊断\n教学干预：步骤"

    answer, context = _metric_eval_payload(
        "faithfulness",
        structured,
        actual_output_by_metric={"answer_relevancy": "核心答案"},
        retrieval_context=["node-1", "node-2"],
    )

    assert answer == structured
    assert context == ["node-1", "node-2"]


def test_audit_gold_alignment_reports_explicit_student_choice_conflict() -> None:
    warnings = audit_gold_alignment(
        {
            "case_id": "eedi-l2-0001",
            "ground_truth": "学生选了 C，但正确答案是 D。",
            "verbatim_grounding_quotes": [{"session_id": 10}],
        },
        [{"session_id": 10, "error_choice": "B"}],
    )

    assert warnings[0]["field"] == "error_choice"
    assert warnings[0]["card_value"] == "B"
    assert warnings[0]["gold_value"] == "C"
