"""
================================================================================
测试模块: tests/test_reranker.py
测试定位: 验证 Step 4 轻量级 MMR 压缩与教研多样性黄金装配器 (src/reranker.py)
测试覆盖:
  1. Jaccard 文本相似度与内容冗余惩罚计算
  2. 业务置信度与原声证据加权 (_calculate_boosted_score)
  3. MMR 贪心多样性选卡算法 (_select_mmr_best)
  4. 教研四槽位黄金装配输出规范与 Markdown 渲染
  5. Token 预算压缩率与防迷失校验
================================================================================
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.reranker import (
    PedagogicalGoldAssembler,
    GoldAssembledContext,
    compute_text_jaccard_similarity
)


def test_jaccard_similarity():
    """测试 Jaccard 文本相似度计算。"""
    text1 = "rounding 5.4598 to 1 decimal place"
    text2 = "rounding 5.4598 to one decimal place"
    text3 = "prime composite factors without numbers"
    
    sim12 = compute_text_jaccard_similarity(text1, text2)
    sim13 = compute_text_jaccard_similarity(text1, text3)
    
    assert sim12 > 0.5
    assert sim13 == 0.0


def test_boosted_scoring():
    """测试证据覆盖、破局一问与关键词覆盖加权。"""
    assembler = PedagogicalGoldAssembler()
    
    cand_with_evidence = {
        "rrf_score": 0.02,
        "document": "5.4598 rounding to 1 decimal place",
        "evidence_turns": [{"turn_id": 1, "speaker": "student", "text": "5.45"}],
        "metadata": {"key_aha_question": "Can you round 5.45 to one decimal place?"}
    }
    
    cand_plain = {
        "rrf_score": 0.02,
        "document": "general rounding concept",
        "evidence_turns": [],
        "metadata": {}
    }
    
    score_boosted = assembler._calculate_boosted_score(cand_with_evidence, "5.4598 rounding", ["5.4598", "rounding"])
    score_plain = assembler._calculate_boosted_score(cand_plain, "5.4598 rounding", ["5.4598", "rounding"])
    
    assert score_boosted > score_plain
    assert score_boosted >= (score_plain + 0.15)


def test_gold_assembler_end_to_end():
    """测试 MMR 黄金装配完整闭环。"""
    assembler = PedagogicalGoldAssembler(lambda_diversity=0.7)
    
    query = "四舍五入 5.4598 到 1 位小数时，学生为什么会误选 5.45？名师如何引导？"
    evidence = [{"turn_id": 1, "speaker": "student", "text": "5.45"}]
    ret_res = {
        "misconceptions": [{
            "hybrid_score": 1.0,
            "document": "5.4598 rounding misconception",
            "metadata": {
                "session_id": 10,
                "subject_path": "Number > Rounding",
                "misconception_name": "Place-value confusion",
            },
            "evidence_turns": evidence,
        }],
        "strategies": [{
            "hybrid_score": 1.0,
            "document": "rounding teaching strategy",
            "metadata": {
                "session_id": 10,
                "subject_path": "Number > Rounding",
                "key_aha_question": "Which digit decides?",
            },
            "evidence_turns": evidence,
        }],
        "rewritten_queries": {"extracted_keywords": ["5.4598", "rounding"]},
    }
    
    gold_ctx = assembler.assemble(query, ret_res)
    
    assert isinstance(gold_ctx, GoldAssembledContext)
    assert gold_ctx.raw_query == query
    assert gold_ctx.selected_misconception is not None
    assert gold_ctx.selected_strategy is not None
    assert len(gold_ctx.evidence_turns) > 0
    assert gold_ctx.estimated_token_count > 0
    assert gold_ctx.estimated_token_count < 2000
    assert 0.0 <= gold_ctx.compression_ratio <= 1.0
    
    # 验证 Markdown 格式包含四大教研槽位
    md = gold_ctx.prompt_context_markdown
    assert "## 一、 学情认知误区诊断" in md
    assert "## 二、 名师破局启发策略" in md
    assert "## 三、 真实师生对白实录证据" in md
    assert "## 四、 考纲考点与原题锚点" in md
    assert "[Turn" in md


def test_evidence_deduplication_keeps_same_turn_from_different_sessions():
    assembler = PedagogicalGoldAssembler()
    result = assembler.assemble("rounding", {
        "misconceptions": [{
            "hybrid_score": 1.0,
            "document": "misconception",
            "metadata": {"session_id": 10, "subject_path": "Number"},
            "evidence_turns": [{"turn_id": 1, "speaker": "student", "text": "5.45"}],
        }],
        "strategies": [{
            "hybrid_score": 1.0,
            "document": "strategy",
            "metadata": {"session_id": 33, "subject_path": "Number"},
            "evidence_turns": [{"turn_id": 1, "speaker": "tutor", "text": "Which digit decides?"}],
        }],
    })

    assert [(t["session_id"], t["turn_id"]) for t in result.evidence_turns] == [(10, 1), (33, 1)]


def test_strategy_selection_is_independent_from_misconception_redundancy():
    """Complementary slots must select their own best candidate independently."""
    assembler = PedagogicalGoldAssembler(lambda_diversity=0.7)
    shared_evidence = [{"turn_id": 1, "speaker": "tutor", "text": "Use the same place value."}]
    result = assembler.assemble("rounding place value", {
        "misconceptions": [{
            "rrf_score": 0.03,
            "document": "rounding place value shared terms",
            "metadata": {"session_id": 10, "subject_path": "Number"},
            "evidence_turns": [{"turn_id": 2, "speaker": "student", "text": "I used the wrong place."}],
        }],
        "strategies": [
            {
                "rrf_score": 0.03,
                "document": "rounding place value shared terms",
                "metadata": {"session_id": 10, "subject_path": "Number", "key_aha_question": "Which place?"},
                "evidence_turns": shared_evidence,
            },
            {
                "rrf_score": 0.02,
                "document": "unrelated distinct intervention",
                "metadata": {"session_id": 33, "subject_path": "Number", "key_aha_question": "Try another way?"},
                "evidence_turns": shared_evidence,
            },
        ],
        "rewritten_queries": {"extracted_keywords": ["rounding", "place", "value"]},
    })

    assert result.selected_strategy["metadata"]["session_id"] == 10


def test_raw_query_rank_breaks_rrf_score_saturation_for_slot_selection():
    assembler = PedagogicalGoldAssembler()
    candidates = [
        {
            "rrf_score": 0.02,
            "document": "direct answer",
            "metadata": {},
            "perspective_hits": ["raw_query_perspective(#Rank1)"],
            "evidence_turns": [{"turn_id": 1, "speaker": "student", "text": "direct"}],
        },
        {
            "rrf_score": 0.03,
            "document": "glossary rounding place value",
            "metadata": {},
            "perspective_hits": ["raw_query_perspective(#Rank10)"],
            "evidence_turns": [{"turn_id": 2, "speaker": "student", "text": "incidental"}],
        },
    ]

    selected = assembler._select_mmr_best(candidates, "direct answer", ["rounding", "place", "value"], [])

    assert selected is candidates[0]


def test_model_reranker_is_applied_before_mmr_selection():
    class FakeReranker:
        def name(self):
            return "fake-reranker"

        def rerank(self, query, documents, *, top_n):
            from src.reranker_provider import RerankResult

            return [RerankResult(index=1, relevance_score=0.95, document=documents[1]), RerankResult(index=0, relevance_score=0.1, document=documents[0])]

    assembler = PedagogicalGoldAssembler(model_reranker=FakeReranker())
    result = assembler.assemble("q", {
        "misconceptions": [
            {"hybrid_score": 1.0, "document": "first", "metadata": {"session_id": 1, "subject_path": "Number"}, "evidence_turns": []},
            {"hybrid_score": 0.1, "document": "second", "metadata": {"session_id": 2, "subject_path": "Number"}, "evidence_turns": []},
        ],
        "strategies": [],
    })
    assert result.selected_misconception["metadata"]["session_id"] == 2
    assert result.selected_misconception["reranking_applied"] is True


def test_logical_evidence_rerank_preserves_ranked_units_as_context():
    class FakeReranker:
        def name(self):
            return "fake-evidence-reranker"

        def rerank(self, query, documents, *, top_n):
            from src.reranker_provider import RerankResult

            return [
                RerankResult(index=1, relevance_score=0.95, document=documents[1]),
                RerankResult(index=0, relevance_score=0.10, document=documents[0]),
            ]

    assembler = PedagogicalGoldAssembler(
        model_reranker=FakeReranker(),
        rerank_unit="logical_evidence",
        evidence_selection_count=1,
    )
    result = assembler.assemble("q", {
        "chunk_strategy": "card",
        "misconceptions": [{
            "hybrid_score": 1.0,
            "document": "card anchor",
            "metadata": {"session_id": 1, "subject_path": "Number", "misconception_name": "anchor"},
            "evidence_turns": [],
        }],
        "strategies": [],
        "evidence_units": [
            {
                "chunk_id": "evidence-low",
                "document": "low-ranked chain",
                "metadata": {"session_id": 1, "window_start_turn": 1, "window_end_turn": 2},
                "evidence_turns": [{"turn_id": 1, "speaker": "student", "text": "low"}],
            },
            {
                "chunk_id": "evidence-high",
                "document": "high-ranked chain",
                "metadata": {"session_id": 1, "window_start_turn": 3, "window_end_turn": 4},
                "evidence_turns": [{"turn_id": 3, "speaker": "tutor", "text": "high"}],
            },
        ],
    })

    assert result.rerank_unit == "logical_evidence"
    assert result.evidence_selection_count == 1
    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["evidence-high"]
    assert [(item["turn_id"], item["text"]) for item in result.evidence_turns] == [(3, "high")]
    assert "high-ranked chain" in result.prompt_context_markdown
    assert "可引用的真实 Turn 证据" in result.prompt_context_markdown


def _overlapping_window(window_id: str, turns: list[int]) -> dict:
    evidence = [
        {"session_id": 10, "turn_id": turn_id, "speaker": "student", "text": f"Turn {turn_id} " + "x" * 60}
        for turn_id in turns
    ]
    document = "\n".join(
        f"[Turn {item['turn_id']}] [{item['speaker']}] {item['text']}"
        for item in evidence
    )
    return {
        "chunk_id": window_id,
        "document": document,
        "metadata": {
            "session_id": 10,
            "window_start_turn": turns[0],
            "window_end_turn": turns[-1],
        },
        "evidence_turns": evidence,
    }


def test_logical_evidence_assembler_packs_overlapping_windows_by_unique_turns():
    class IdentityReranker:
        def name(self):
            return "identity"

        def rerank(self, query, documents, *, top_n):
            from src.reranker_provider import RerankResult

            return [
                RerankResult(index=index, relevance_score=1.0 - index * 0.01, document=document)
                for index, document in enumerate(documents)
            ]

    windows = [
        _overlapping_window("w1", [1, 2, 3, 4, 5, 6]),
        _overlapping_window("w2", [4, 5, 6, 7, 8, 9]),
        _overlapping_window("w3", [7, 8, 9, 10, 11, 12]),
        _overlapping_window("w4", [10, 11, 12, 13, 14, 15]),
        _overlapping_window("w5", [13, 14, 15, 16, 17, 18]),
    ]
    assembler = PedagogicalGoldAssembler(
        model_reranker=IdentityReranker(),
        rerank_unit="logical_evidence",
        evidence_selection_count=5,
        max_prompt_tokens=1500,
    )

    result = assembler.assemble(
        "q",
        {"chunk_strategy": "card", "misconceptions": [], "strategies": [], "evidence_units": windows},
    )

    assert len(result.selected_evidence_units) == 5
    assert [item["reranker_rank"] for item in result.selected_evidence_units] == [1, 2, 3, 4, 5]
    assert len(result.evidence_turns) == 18
    assert result.prompt_context_markdown.count("[Turn 4]") == 1
    assert result.prompt_context_markdown.count("[Turn 18]") == 1
    assert result.estimated_token_count <= 1500


def test_anchored_assembler_uses_same_overlap_aware_packing():
    windows = [
        _overlapping_window("w1", [1, 2, 3, 4, 5, 6]),
        _overlapping_window("w2", [4, 5, 6, 7, 8, 9]),
        _overlapping_window("w3", [7, 8, 9, 10, 11, 12]),
        _overlapping_window("w4", [10, 11, 12, 13, 14, 15]),
        _overlapping_window("w5", [13, 14, 15, 16, 17, 18]),
    ]
    for rank, window in enumerate(windows, start=1):
        window["reranker_rank"] = rank
        window["reranker_score"] = 1.0 - rank * 0.01
    assembler = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=5,
        max_prompt_tokens=2000,
    )

    result = assembler.assemble(
        "q",
        {"chunk_strategy": "card", "misconceptions": [], "strategies": [], "evidence_units": windows},
    )

    assert len(result.selected_evidence_units) == 3
    assert [item["reranker_rank"] for item in result.selected_evidence_units] == [1, 3, 5]
    assert len(result.evidence_turns) == 18
    assert result.prompt_context_markdown.count("[Turn 4]") == 1
    assert result.estimated_token_count <= 2000


def test_anchored_assembler_injects_source_linked_parent_facts():
    window = _overlapping_window("w1", [1, 2])
    window["reranker_rank"] = 1
    window["reranker_score"] = 0.99
    parent = {
        "chunk_id": "parent-1",
        "document": "parent card document",
        "metadata": {
            "session_id": 10,
            "source_turn_ids": [1],
            "misconception_name": "parent-only title",
            "deep_mechanism": "parent-only summary",
        },
        "evidence_turns": [],
    }
    assembler = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=1,
        max_prompt_tokens=1500,
    )

    result = assembler.assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [parent],
            "strategies": [],
            "evidence_units": [window],
        },
    )

    assert "parent-only title" in result.prompt_context_markdown
    assert "parent-only summary" in result.prompt_context_markdown
    assert any(node.startswith("## [DERIVED_FACT]") for node in result.deepeval_context_nodes)
    assert all(node in result.prompt_context_markdown for node in result.deepeval_context_nodes)
    assert "[Turn 1]" in result.prompt_context_markdown


def test_anchored_assembler_selects_conversation_endpoints_and_roles():
    def window(window_id, start, speaker):
        turns = list(range(start, start + 3))
        evidence = [
            {"session_id": 10, "turn_id": turn_id, "speaker": speaker, "text": f"{speaker} {turn_id}"}
            for turn_id in turns
        ]
        return {
            "chunk_id": window_id,
            "document": "\n".join(
                f"[Turn {item['turn_id']}] [{item['speaker']}] {item['text']}"
                for item in evidence
            ),
            "metadata": {
                "session_id": 10,
                "window_start_turn": turns[0],
                "window_end_turn": turns[-1],
                "source_turn_ids": turns,
            },
            "evidence_turns": evidence,
        }

    windows = [window("middle", 4, "student"), window("start", 1, "student"), window("end", 7, "tutor")]
    for rank, item in enumerate(windows, start=1):
        item["reranker_rank"] = rank
        item["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=2,
        max_prompt_tokens=2000,
    ).assemble(
        "why?",
        {
            "chunk_strategy": "card",
            "misconceptions": [],
            "strategies": [],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["start", "end"]
    assert {item["speaker"] for item in result.evidence_turns} == {"student", "tutor"}


def test_anchored_assembler_prioritizes_parent_coverage_over_endpoint_bonus():
    def window(window_id, turns):
        evidence = [
            {"session_id": 10, "turn_id": turn_id, "speaker": "student", "text": f"turn {turn_id}"}
            for turn_id in turns
        ]
        return {
            "chunk_id": window_id,
            "document": "\n".join(
                f"[Turn {item['turn_id']}] [{item['speaker']}] {item['text']}"
                for item in evidence
            ),
            "metadata": {
                "session_id": 10,
                "window_start_turn": turns[0],
                "window_end_turn": turns[-1],
                "source_turn_ids": turns,
            },
            "evidence_turns": evidence,
        }

    windows = [window("head", [1, 2]), window("anchored", [3, 4]), window("tail", [5, 6])]
    for rank, item in enumerate(windows, start=1):
        item["reranker_rank"] = rank
        item["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=1,
        max_prompt_tokens=2000,
    ).assemble(
        "why?",
        {
            "chunk_strategy": "card",
            "misconceptions": [{
                "chunk_id": "parent",
                "metadata": {"session_id": 10, "source_turn_ids": [3]},
            }],
            "strategies": [],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["anchored"]


def test_assembler_default_budget_is_two_thousand_tokens():
    assert PedagogicalGoldAssembler().max_prompt_tokens == 2000


def test_anchored_assembler_preserves_parent_anchor_coverage_when_budget_is_limited():
    windows = [
        _overlapping_window("noise", [1, 2, 3, 4, 5, 6]),
        _overlapping_window("anchor", [7, 8, 9, 10, 11, 12]),
    ]
    for rank, window in enumerate(windows, start=1):
        window["reranker_rank"] = rank
        window["reranker_score"] = 1.0 - rank * 0.01
    assembler = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=1,
        max_prompt_tokens=1500,
    )

    windows[0]["metadata"]["parent_contexts"] = [{"card_id": "parent-misconception"}]
    windows[1]["metadata"]["parent_contexts"] = [{"card_id": "parent-misconception"}]
    result = assembler.assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [{
                "chunk_id": "parent-misconception",
                "metadata": {"session_id": 10, "source_turn_ids": [10]},
                "evidence_turns": [],
            }],
            "strategies": [],
            "evidence_units": windows,
        },
    )

    assert result.evidence_turns
    assert {item["chunk_id"] for item in result.selected_evidence_units} == {"anchor"}
    assert any(item["turn_id"] == 10 for item in result.evidence_turns)
