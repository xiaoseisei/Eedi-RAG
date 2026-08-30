# -*- coding: utf-8 -*-
"""
tests/chunk/test_chunkers.py
----------------------------
分块策略与基准评测体系单元测试套件 (TDD Unit Tests)。
覆盖所有候选切块器的切分契约、边界条件与评测运行器逻辑。
"""

import pytest
import json
from pathlib import Path
from tests.chunk.chunkers import (
    TurnSlidingWindowChunker,
    HeaderInjectedChunker,
    ParentChildChunker,
    KnowledgeDistilledChunker,
    Chunk,
)
from tests.chunk.benchmark_eval import (
    load_cleaned_sessions,
    load_ground_truth_queries,
    evaluate_strategy,
    LightweightRetriever,
)


@pytest.fixture
def sample_session_dict():
    """提供单场标准测试会话样本"""
    return {
        "intervention_id": 999,
        "question_id": 888,
        "tutor_id": 777,
        "subjects": {
            "paths": ["Mathematics > Algebra > Linear Equations"],
            "subjects": ["Mathematics"],
            "topics": ["Algebra"],
            "subtopics": ["Linear Equations"],
        },
        "question": {
            "question_id": 888,
            "question_text": "Solve 2x + 4 = 10",
            "options": {"A": "2", "B": "3", "C": "4", "D": "5"},
            "images": {},
        },
        "turns": [
            {"turn_id": 1, "speaker": "tutor", "is_tutor": True, "text": "Hello, how can I help?", "talk_moves": [], "is_greeting_or_noise": True},
            {"turn_id": 2, "speaker": "student", "is_tutor": False, "text": "I think x is 4", "talk_moves": [], "is_greeting_or_noise": False},
            {"turn_id": 3, "speaker": "tutor", "is_tutor": True, "text": "What is 2 times 4 plus 4?", "talk_moves": ["<Press for Accuracy>"], "is_greeting_or_noise": False},
            {"turn_id": 4, "speaker": "student", "is_tutor": False, "text": "It is 12, oh wait!", "talk_moves": [], "is_greeting_or_noise": False},
            {"turn_id": 5, "speaker": "tutor", "is_tutor": True, "text": "Great, so what should x be?", "talk_moves": ["<Keep Together>"], "is_greeting_or_noise": False},
            {"turn_id": 6, "speaker": "student", "is_tutor": False, "text": "x is 3!", "talk_moves": [], "is_greeting_or_noise": False},
        ],
        "total_turns": 6,
        "student_turn_count": 3,
        "tutor_turn_count": 3,
        "has_valid_tutoring": True,
    }


def test_sliding_window_chunker(sample_session_dict):
    """测试轮次滑动窗口切块器"""
    chunker = TurnSlidingWindowChunker(window_size=4, step_size=2)
    chunks = chunker.chunk_session(sample_session_dict)
    
    assert len(chunks) == 2
    assert chunks[0].turn_ids == [1, 2, 3, 4]
    assert chunks[1].turn_ids == [3, 4, 5, 6]
    assert "[Turn 1 - tutor]" in chunks[0].text
    assert chunks[0].intervention_id == 999
    assert chunks[0].token_count_approx > 0


def test_header_injected_chunker(sample_session_dict):
    """测试 Header 注入切块器"""
    chunker = HeaderInjectedChunker(pair_size=2, step_size=2)
    chunks = chunker.chunk_session(sample_session_dict)
    
    assert len(chunks) == 3
    for c in chunks:
        assert "[考纲考点]: Mathematics > Algebra > Linear Equations" in c.text
        assert "[题目题干]: Solve 2x + 4 = 10" in c.text
        assert "[选项分布]: A: 2, B: 3, C: 4, D: 5" in c.text
        assert c.metadata["has_header_injected"] is True


def test_parent_child_chunker(sample_session_dict):
    """测试层次化父子索引切块器"""
    chunker = ParentChildChunker(child_turn_size=2, child_step_size=2)
    chunks = chunker.chunk_session(sample_session_dict)
    
    assert len(chunks) == 3
    for c in chunks:
        assert c.metadata["is_child_chunk"] is True
        assert "PARENT SESSION (999)" in c.metadata["parent_context"]
        assert "Solve 2x + 4 = 10" in c.metadata["parent_context"]


def test_knowledge_distilled_chunker(sample_session_dict):
    """测试教学知识卡片蒸馏切块器"""
    chunker = KnowledgeDistilledChunker()
    chunks = chunker.chunk_session(sample_session_dict)
    
    assert len(chunks) == 2
    types = [c.metadata["card_type"] for c in chunks]
    assert "misconception" in types
    assert "tutor_strategy" in types
    
    misconception_chunk = next(c for c in chunks if c.metadata["card_type"] == "misconception")
    assert "【学生认知错因卡】" in misconception_chunk.text
    assert "I think x is 4" in misconception_chunk.text


def test_retriever_and_benchmark_runner():
    """测试轻量检索器与评测指标计算"""
    project_root = Path(__file__).resolve().parent.parent.parent
    sample_jsonl = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    gt_json = project_root / "tests" / "chunk" / "ground_truth_queries.json"

    sessions = load_cleaned_sessions(sample_jsonl)
    queries = load_ground_truth_queries(gt_json)
    
    assert len(sessions) == 10
    assert len(queries) == 10

    chunker = HeaderInjectedChunker(pair_size=2, step_size=2)
    res = evaluate_strategy(chunker, sessions, queries)
    
    assert res["num_chunks"] > 0
    assert res["session_mrr"] > 0.0
    assert res["turn_mrr"] > 0.0
    assert "avg_latency_ms" in res
