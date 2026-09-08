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


def _overlapping_window(window_id: str, turns: list[int], session_id: int = 10) -> dict:
    evidence = [
        {"session_id": session_id, "turn_id": turn_id, "speaker": "student", "text": f"Turn {turn_id} " + "x" * 60}
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
            "session_id": session_id,
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

    assert len(result.selected_evidence_units) <= 5
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


def test_anchored_assembler_budget_includes_compact_evidence_catalog():
    window = _overlapping_window("w1", [1, 2])
    window["reranker_rank"] = 1
    window["reranker_score"] = 0.99
    parent = {
        "chunk_id": "parent-1",
        "metadata": {
            "session_id": 10,
            "source_turn_ids": [1],
            "misconception_name": "rounding",
            "deep_mechanism": "The target place is confused.",
        },
    }

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=1,
        max_prompt_tokens=2000,
    ).assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [parent],
            "strategies": [],
            "evidence_units": [window],
        },
    )

    from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
    from src.reranker import estimate_text_tokens

    generator_context = EndToEndPedagogicalRAGPipeline(
        generator_contract_version="v2",
    )._generator_context_text(result)

    assert estimate_text_tokens(generator_context) <= 2000
    assert "[E001] [Turn 1]" in generator_context
    assert "quote_text=" not in generator_context
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
        anchored_assembly_strategy="heuristic_combinatorial",
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


def test_anchored_assembler_requires_primary_session_endpoints_when_budget_allows():
    windows = [
        _overlapping_window("middle", [7, 8, 9, 10, 11, 12]),
        _overlapping_window("start", [1, 2, 3, 4, 5, 6]),
        _overlapping_window("end", [13, 14, 15, 16, 17, 18]),
    ]
    for rank, window in enumerate(windows, start=1):
        window["reranker_rank"] = rank
        window["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=2,
        max_prompt_tokens=2000,
        anchored_assembly_strategy="heuristic_combinatorial",
    ).assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [{
                "chunk_id": "primary-parent",
                "metadata": {"session_id": 10, "source_turn_ids": [7, 8, 9, 10, 11, 12]},
            }],
            "strategies": [],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["start", "end"]


def test_anchored_assembler_maximizes_unique_turns_after_endpoint_constraint():
    windows = [
        _overlapping_window("parent-middle", [10, 11, 12]),
        _overlapping_window("start", [1, 2, 3]),
        _overlapping_window("bridge", [3, 4, 5, 6, 7, 8]),
        _overlapping_window("end", [13, 14, 15]),
    ]
    for rank, window in enumerate(windows, start=1):
        window["reranker_rank"] = rank
        window["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=3,
        max_prompt_tokens=2000,
        anchored_assembly_strategy="heuristic_combinatorial",
    ).assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [],
            "strategies": [],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == [
        "start",
        "bridge",
        "end",
    ]


def test_anchored_assembler_prefers_primary_session_prefix_after_unique_tie():
    windows = [
        _overlapping_window("late", list(range(10, 19))),
        _overlapping_window("start", [1, 2]),
        _overlapping_window("prefix", list(range(2, 12))),
        _overlapping_window("end", list(range(19, 24))),
    ]
    for rank, window in enumerate(windows, start=1):
        window["reranker_rank"] = rank
        window["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=3,
        max_prompt_tokens=2000,
        anchored_assembly_strategy="heuristic_combinatorial",
    ).assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [],
            "strategies": [],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == [
        "start",
        "prefix",
        "end",
    ]


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
        anchored_assembly_strategy="heuristic_combinatorial",
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


def test_anchored_assembler_prefers_primary_parent_coverage_over_distractor_sessions():
    def window(window_id, session_id, turns):
        evidence = [
            {"session_id": session_id, "turn_id": turn_id, "speaker": "student", "text": f"turn {turn_id}"}
            for turn_id in turns
        ]
        return {
            "chunk_id": window_id,
            "document": "\n".join(
                f"[Turn {item['turn_id']}] [{item['speaker']}] {item['text']}"
                for item in evidence
            ),
            "metadata": {
                "session_id": session_id,
                "window_start_turn": turns[0],
                "window_end_turn": turns[-1],
                "source_turn_ids": turns,
            },
            "evidence_turns": evidence,
        }

    primary = {
        "chunk_id": "primary",
        "metadata": {"session_id": 10, "source_turn_ids": [3, 4, 5]},
    }
    distractor = {
        "chunk_id": "distractor",
        "metadata": {"session_id": 99, "source_turn_ids": [1, 2, 3]},
    }
    windows = [
        window("distractor-window", 99, [1, 2, 3]),
        window("primary-window", 10, [3, 4, 5]),
    ]
    for rank, item in enumerate(windows, start=1):
        item["reranker_rank"] = rank
        item["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=1,
        max_prompt_tokens=2000,
        anchored_assembly_strategy="heuristic_combinatorial",
    ).assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [primary],
            "strategies": [],
            "anchored_parent_cards": [primary, distractor],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["primary-window"]
    assert {turn["session_id"] for turn in result.evidence_turns} == {10}


def test_anchored_assembler_prefers_session_with_both_parent_card_kinds():
    def card(card_id, session_id, kind):
        metadata = {"session_id": session_id, "source_turn_ids": [1]}
        if kind == "strategy":
            metadata["key_aha_question"] = "Which step?"
        else:
            metadata["misconception_name"] = "Wrong step"
        return {"chunk_id": card_id, "metadata": metadata}

    def window(window_id, session_id, start):
        evidence = [
            {"session_id": session_id, "turn_id": turn_id, "speaker": "student", "text": f"turn {turn_id}"}
            for turn_id in range(start, start + 3)
        ]
        return {
            "chunk_id": window_id,
            "document": "\n".join(
                f"[Turn {item['turn_id']}] [{item['speaker']}] {item['text']}"
                for item in evidence
            ),
            "metadata": {
                "session_id": session_id,
                "window_start_turn": start,
                "window_end_turn": start + 2,
                "source_turn_ids": [item["turn_id"] for item in evidence],
            },
            "evidence_turns": evidence,
        }

    primary_misconception = card("primary-m", 10, "misconception")
    primary_strategy = card("primary-s", 10, "strategy")
    distractor_strategy = card("distractor-s", 99, "strategy")
    windows = [window("distractor-window", 99, 1), window("primary-window", 10, 1)]
    for rank, item in enumerate(windows, start=1):
        item["reranker_rank"] = rank
        item["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=1,
        max_prompt_tokens=2000,
        anchored_assembly_strategy="heuristic_combinatorial",
    ).assemble(
        "学生和导师如何处理这个问题？",
        {
            "chunk_strategy": "card",
            "misconceptions": [primary_misconception],
            "strategies": [primary_strategy, distractor_strategy],
            "anchored_parent_cards": [primary_misconception, primary_strategy, distractor_strategy],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["primary-window"]


def test_anchored_assembler_uses_upstream_parent_cards_for_turn_coverage():
    def card(card_id, session_id, turns, *, strategy=False):
        metadata = {"session_id": session_id, "source_turn_ids": turns}
        if strategy:
            metadata.update({"key_aha_question": "Use the deciding step"})
        else:
            metadata.update({"misconception_name": "Primary misconception"})
        return {"chunk_id": card_id, "metadata": metadata}

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

    upstream_primary = card("upstream-primary", 10, [5], strategy=True)
    upstream_distractor = card("upstream-distractor", 99, [1])
    mmr_only_card = card("mmr-only", 10, [1], strategy=True)
    windows = [window("primary-window", [4, 5, 6]), window("distractor-window", [1, 2, 3])]
    for rank, item in enumerate(windows, start=1):
        item["reranker_rank"] = rank
        item["reranker_score"] = 1.0 - rank * 0.01

    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=1,
        max_prompt_tokens=2000,
    ).assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [],
            "strategies": [mmr_only_card],
            "anchored_parent_cards": [upstream_primary, upstream_distractor],
            "evidence_units": windows,
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["primary-window"]


def test_anchored_assembler_prefers_window_that_fills_temporal_gap():
    def window(window_id, start, end, rank):
        turns = list(range(start, end + 1))
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
                "window_start_turn": start,
                "window_end_turn": end,
                "source_turn_ids": turns,
            },
            "evidence_turns": evidence,
            "reranker_rank": rank,
            "reranker_score": 1.0 - rank * 0.01,
        }

    parent = {
        "chunk_id": "primary",
        "metadata": {"session_id": 10, "source_turn_ids": [4]},
    }
    result = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=3,
        max_prompt_tokens=2000,
        anchored_assembly_strategy="heuristic_combinatorial",
    ).assemble(
        "q",
        {
            "chunk_strategy": "card",
            "misconceptions": [parent],
            "strategies": [],
            "evidence_units": [
                window("left", 1, 6, 1),
                window("bridge", 5, 10, 2),
                window("right", 11, 16, 3),
                window("tail", 17, 22, 4),
            ],
        },
    )

    assert [item["chunk_id"] for item in result.selected_evidence_units] == ["left", "right", "tail"]


def test_assembler_default_budget_is_four_thousand_tokens():
    assembler = PedagogicalGoldAssembler()
    assert assembler.max_prompt_tokens == 4000
    assert assembler.evidence_selection_count == 7
    assert assembler.parent_card_count == 5
    assert assembler.anchored_assembly_strategy == "greedy_budget"


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
        anchored_assembly_strategy="heuristic_combinatorial",
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


def test_anchored_assembler_greedy_fallback_on_combinatorial_explosion():
    # 28 candidate windows across sessions with max_selection=7 yields > 1.6M combinations.
    # The assembler must safely fallback to greedy linear budget packing in milliseconds.
    windows = []
    for i in range(28):
        window = _overlapping_window(f"win_{i}", [i * 2 + 1, i * 2 + 2], session_id=100)
        window["reranker_rank"] = i + 1
        window["reranker_score"] = 1.0 - i * 0.01
        windows.append(window)

    assembler = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=7,
        max_prompt_tokens=4000,
        anchored_assembly_strategy="heuristic_combinatorial",
    )
    result = assembler.assemble(
        "test query",
        {
            "chunk_strategy": "card",
            "misconceptions": [{
                "chunk_id": "m1",
                "metadata": {"session_id": 100, "source_turn_ids": [1]},
                "evidence_turns": [],
            }],
            "strategies": [],
            "evidence_units": windows,
        },
    )

    assert len(result.selected_evidence_units) == 7
    assert result.estimated_token_count <= 4000
    assert not result.budget_violation


def test_anchored_assembler_greedy_budget_packs_by_reranker_rank_and_deduplicates():
    w1 = _overlapping_window("w1", [1, 2, 3])
    w1["reranker_rank"] = 1
    w1["reranker_score"] = 0.95
    # w2 is a strict subset of w1's turns [2, 3] -> must be skipped by greedy packing
    w2 = _overlapping_window("w2", [2, 3])
    w2["reranker_rank"] = 2
    w2["reranker_score"] = 0.90
    # w3 brings fresh turns [4, 5] -> must be included
    w3 = _overlapping_window("w3", [4, 5])
    w3["reranker_rank"] = 3
    w3["reranker_score"] = 0.85

    assembler = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=2,
        max_prompt_tokens=4000,
        anchored_assembly_strategy="greedy_budget",
    )
    result = assembler.assemble(
        "test query",
        {
            "chunk_strategy": "card",
            "misconceptions": [],
            "strategies": [],
            "evidence_units": [w1, w2, w3],
        },
    )

    selected_ids = [unit["chunk_id"] for unit in result.selected_evidence_units]
    assert selected_ids == ["w1", "w3"]
    assert len(result.selected_evidence_units) == 2


def test_anchored_assembler_greedy_budget_enforces_token_limit():
    import pytest

    w1 = _overlapping_window("w1", list(range(1, 10)))
    w1["reranker_rank"] = 1
    w1["reranker_score"] = 0.95
    w2 = _overlapping_window("w2", list(range(10, 30)))
    w2["reranker_rank"] = 2
    w2["reranker_score"] = 0.85

    # Budget 800: fits w1 (approx 640 tokens) but rejects w2 (which would exceed 800)
    assembler_800 = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=2,
        max_prompt_tokens=800,
        anchored_assembly_strategy="greedy_budget",
    )
    result = assembler_800.assemble(
        "test query",
        {
            "chunk_strategy": "card",
            "misconceptions": [],
            "strategies": [],
            "evidence_units": [w1, w2],
        },
    )

    assert result.estimated_token_count <= 800
    assert [u["chunk_id"] for u in result.selected_evidence_units] == ["w1"]

    # Budget 200: too small even for w1 alone -> raises ValueError
    assembler_200 = PedagogicalGoldAssembler(
        rerank_unit="anchored_logical_window",
        evidence_selection_count=2,
        max_prompt_tokens=200,
        anchored_assembly_strategy="greedy_budget",
    )
    with pytest.raises(ValueError, match="within the token budget"):
        assembler_200.assemble(
            "test query",
            {
                "chunk_strategy": "card",
                "misconceptions": [],
                "strategies": [],
                "evidence_units": [w1, w2],
            },
        )

