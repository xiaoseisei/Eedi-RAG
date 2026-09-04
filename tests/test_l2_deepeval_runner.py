from __future__ import annotations

from scripts.run_l2_deepeval import (
    _append_checkpoint,
    _load_checkpoint,
    _resolve_expected_artifact_hash,
    _resolve_l2_route,
    _select_l2_cases,
)
from scripts.merge_l2_deepeval_reports import merge_reports
from scripts.run_l2_deepeval import _build_gold_context_v2, _rubric_config


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


def test_l2_rubric_v2_allows_qualified_inference_and_requires_actionable_pedagogy() -> None:
    rubric = _rubric_config("v2")

    assert "qualified" in rubric["faithfulness"].casefold()
    assert "unsupported" in rubric["faithfulness"].casefold()
    assert "actionable" in rubric["pedagogical_geval"].casefold()
    assert "misconception" in rubric["pedagogical_geval"].casefold()
    assert _rubric_config("v1") != rubric
