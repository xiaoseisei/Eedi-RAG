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
    assert gold_ctx.estimated_token_count < 1500
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
