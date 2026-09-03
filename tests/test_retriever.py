"""
================================================================================
测试模块: tests/test_retriever.py
测试定位: 验证 Step 4 双度量向量混合检索器 (src/retriever.py)
测试覆盖:
  1. 余弦相似度、欧氏距离与欧氏相似度数学计算的精确度
  2. 双度量综合加权评分 (compute_hybrid_score) 的边界与单调性
  3. 原始提问向量化后在 student_misconceptions 中的双度量检索
  4. 原始提问向量化后在 tutor_strategies 中的双度量检索
  5. 证据回溯 (evidence_turns) 是否从 DuckDB 中 100% 正确抓取
  6. 零信任模式下 fallback_windows 的直接召回
  7. retrieve_hybrid 双路并行召回与结果包装契约
================================================================================
"""

import os
import sys
import json
import pytest
import math
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import CleanedSession, ExtractedPIU, Chunk
from src.storage_manager import DualEngineStorageManager
from src.retriever import (
    DualMetricRetriever,
    compute_cosine_similarity,
    compute_euclidean_distance,
    compute_euclidean_similarity,
    compute_hybrid_score,
    numeric_formula_exact_score,
)


@pytest.fixture
def in_memory_storage():
    """构建已填充真实样本的内存存储底座。"""
    sample_cleaned = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    sample_extracted = project_root / "data" / "sample" / "extracted_pius_sample.jsonl"
    sample_chunks = project_root / "data" / "sample" / "chunks_sample.jsonl"
    
    sessions = [CleanedSession.model_validate(json.loads(line)) for line in open(sample_cleaned, encoding="utf-8") if line.strip()]
    extracted = [ExtractedPIU.model_validate(json.loads(line)) for line in open(sample_extracted, encoding="utf-8") if line.strip()]
    chunks = [Chunk.model_validate(json.loads(line)) for line in open(sample_chunks, encoding="utf-8") if line.strip()]
    
    manager = DualEngineStorageManager(
        db_path=":memory:",
        in_memory=True
    )
    manager.ingest_all(sessions, extracted, chunks)
    yield manager
    manager.close()


def test_math_metric_functions():
    """测试余弦相似度与欧氏距离的数学计算准确性。"""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_c = [1.0, 0.0, 0.0]
    
    # 正交向量: Cosine = 0, L2 = sqrt(2)
    assert compute_cosine_similarity(vec_a, vec_b) == 0.0
    assert abs(compute_euclidean_distance(vec_a, vec_b) - math.sqrt(2)) < 1e-6
    
    # 同向相同向量: Cosine = 1.0, L2 = 0.0, L2_Sim = 1.0
    assert compute_cosine_similarity(vec_a, vec_c) == 1.0
    assert compute_euclidean_distance(vec_a, vec_c) == 0.0
    assert compute_euclidean_similarity(vec_a, vec_c) == 1.0
    
    # 综合打分 (alpha=0.5)
    hybrid_score, norm_cos, l2_sim = compute_hybrid_score(vec_a, vec_c, alpha=0.5)
    assert norm_cos == 1.0
    assert l2_sim == 1.0
    assert hybrid_score == 1.0


def test_retriever_misconception_retrieval(in_memory_storage):
    """测试原始问题直接检索错因卡片。"""
    retriever = DualMetricRetriever(storage_manager=in_memory_storage, query_rewrite_mode="deterministic", alpha=0.5)
    
    raw_query = "四舍五入 5.4598 到 1 位小数学生容易犯什么错？"
    hits = retriever.retrieve_misconceptions(raw_query, top_k=2, fetch_evidence=True)
    
    assert len(hits) > 0
    top_hit = hits[0]
    assert "hybrid_score" in top_hit
    assert "cosine_score" in top_hit
    assert "euclidean_score" in top_hit
    assert "bm25_score" in top_hit
    assert "dense_hybrid_score" in top_hit
    assert "retrieval_channels" in top_hit
    assert top_hit["hybrid_score"] > 0.0
    assert top_hit["metadata"]["session_id"] is not None
    assert "evidence_turns" in top_hit
    assert len(top_hit["evidence_turns"]) > 0


def test_retriever_strategy_retrieval(in_memory_storage):
    """测试原始问题直接检索名师策略卡片。"""
    retriever = DualMetricRetriever(storage_manager=in_memory_storage, query_rewrite_mode="deterministic", alpha=0.6)
    
    raw_query = "名师如何提问引导四舍五入保留一位小数？"
    hits = retriever.retrieve_strategies(raw_query, top_k=2, fetch_evidence=True)
    
    assert len(hits) > 0
    top_hit = hits[0]
    assert "key_aha_question" in top_hit["metadata"]
    assert "evidence_turns" in top_hit
    assert len(top_hit["evidence_turns"]) > 0
    assert top_hit["evidence_turns"][0]["speaker"] in ["student", "tutor"]


def test_retriever_fallback_windows(in_memory_storage):
    """测试纯原文滑动窗口直接检索。"""
    retriever = DualMetricRetriever(storage_manager=in_memory_storage, query_rewrite_mode="deterministic")
    
    raw_query = "Alex and Sophie are arguing about rounding"
    hits = retriever.retrieve_fallback_windows(raw_query, top_k=2)
    
    assert len(hits) > 0
    assert hits[0]["collection"] == "fallback_windows"


def test_retriever_hybrid_full_contract(in_memory_storage):
    """测试 retrieve_hybrid 主入口输出契约与多路召回。"""
    retriever = DualMetricRetriever(storage_manager=in_memory_storage, query_rewrite_mode="deterministic", alpha=0.5)
    
    raw_query = "学生把 5.4598 算成 5.45 怎么引导？"
    res = retriever.retrieve_hybrid(raw_query, top_k_each=2, fetch_evidence=True)
    
    assert res["query"] == raw_query
    assert "metric_config" in res
    assert res["metric_config"]["alpha"] == 0.5
    assert len(res["misconceptions"]) > 0
    assert len(res["strategies"]) > 0
    assert res["total_retrieved"] == len(res["misconceptions"]) + len(res["strategies"])


def test_multi_lane_retrieval_embeds_unique_queries_in_one_batch(in_memory_storage):
    class RecordingEmbedding:
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = []

        def __call__(self, texts):
            self.calls.append(list(texts))
            return self.delegate(texts)

    retriever = DualMetricRetriever(
        storage_manager=in_memory_storage,
        query_rewrite_mode="deterministic",
    )
    recording = RecordingEmbedding(in_memory_storage.embedding_function)
    retriever.embedding_fn = recording

    result = retriever.retrieve_multi_perspective_rrf(
        "学生把 5.4598 算成 5.45 怎么引导？",
        top_k_each=2,
        fetch_evidence=False,
    )

    assert len(recording.calls) == 1
    assert len(recording.calls[0]) == len(set(recording.calls[0]))
    assert len(recording.calls[0]) <= 4
    assert result["trace"]["embedding"]["unique_query_count"] == len(recording.calls[0])


def test_numeric_formula_exact_score_prefers_dialogue_match_over_question_anchor():
    query = "学生为什么把 5.45 当成一位小数？"
    question_only = "[考题原题]: Round 5.45 to one decimal place\n[对话片段 (Turn 1 ~ 2)]:\nI do not know"
    dialogue_match = "[考题原题]: Round 5.45 to one decimal place\n[对话片段 (Turn 3 ~ 4)]:\nStudent: I chose 5.45"

    assert numeric_formula_exact_score(query, dialogue_match) > numeric_formula_exact_score(query, question_only)


def test_raw_first_fusion_cannot_demote_raw_top_results(in_memory_storage):
    query = "学生把 5.4598 算成 5.45 怎么引导？"
    retriever = DualMetricRetriever(
        storage_manager=in_memory_storage,
        query_rewrite_mode="deterministic",
        fusion_strategy="raw_first",
    )
    raw_misc = retriever.retrieve_misconceptions(query, top_k=3, fetch_evidence=False)
    raw_strategy = retriever.retrieve_strategies(query, top_k=3, fetch_evidence=False)

    result = retriever.retrieve_multi_perspective_rrf(query, top_k_each=3, fetch_evidence=False)

    assert [item["chunk_id"] for item in result["misconceptions"]] == [item["chunk_id"] for item in raw_misc]
    assert [item["chunk_id"] for item in result["strategies"]] == [item["chunk_id"] for item in raw_strategy]
    assert result["trace"]["fusion"] == "raw_first"


def test_dense_mode_remains_available_for_regression(in_memory_storage):
    retriever = DualMetricRetriever(
        storage_manager=in_memory_storage,
        query_rewrite_mode="deterministic",
        retrieval_mode="dense",
    )
    hits = retriever.retrieve_misconceptions("四舍五入 5.4598", top_k=1, fetch_evidence=False)
    assert hits and hits[0]["retrieval_channels"] == ["dense"]


def test_bm25_dense_mode_exposes_both_channels(in_memory_storage):
    retriever = DualMetricRetriever(
        storage_manager=in_memory_storage,
        query_rewrite_mode="deterministic",
        retrieval_mode="bm25_dense",
        bm25_weight=0.35,
    )
    hits = retriever.retrieve_misconceptions("四舍五入 5.4598", top_k=3, fetch_evidence=False)
    assert hits
    assert all("bm25_score" in hit for hit in hits)
    assert retriever.retrieval_mode == "bm25_dense"
