"""
================================================================================
测试模块: tests/test_query_rewriter.py
测试定位: 验证 Step 4 多视角子查询改写器与领域知识注入 (src/query_rewriter.py)
测试覆盖:
  1. DomainKnowledgeInjector 考纲拓扑与专有名词词表匹配
  2. MultiPerspectiveQueryRewriter 离线规则增强改写契约 (MultiPerspectiveQueries)
  3. 跨语种术语对齐注入 (最小公倍数 -> LCM, 极差 -> Range)
  4. DualMetricRetriever.retrieve_multi_perspective_rrf 多视角 RRF 晚期融合
================================================================================
"""

import os
import sys
import json
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import CleanedSession, ExtractedPIU, Chunk, MultiPerspectiveQueries
from src.storage_manager import DualEngineStorageManager
from src.query_rewriter import DomainKnowledgeInjector, MultiPerspectiveQueryRewriter
from src.retriever import DualMetricRetriever


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


def test_domain_knowledge_injector():
    """测试考纲知识与专有名词动态注入器。"""
    injector = DomainKnowledgeInjector()
    
    # 案例 1: 最小公倍数 (LCM)
    ctx1, terms1 = injector.match_and_inject("两数相乘为什么不一定是最小公倍数？")
    assert "LCM" in " ".join(terms1)
    assert "Multiples and Factors" in ctx1
    
    # 案例 2: 极差与频数表 (Range)
    ctx2, terms2 = injector.match_and_inject("看电视统计表里极差怎么算？")
    assert "Range" in " ".join(terms2)
    assert "Frequency Tables" in ctx2
    
    # 案例 3: 四舍五入 (Rounding)
    ctx3, terms3 = injector.match_and_inject("5.4598 保留一位小数怎么教？")
    assert "Rounding" in " ".join(terms3)
    assert "Rounding to Decimal Places" in ctx3


def test_query_rewriter_offline_mode():
    """测试改写器离线增强模式及 Pydantic 契约校验。"""
    rewriter = MultiPerspectiveQueryRewriter()
    
    raw = "学生解不等式两边除以负数老是忘翻转符号"
    res = rewriter.rewrite(raw)
    
    assert isinstance(res, MultiPerspectiveQueries)
    assert res.raw_query == raw
    assert len(res.misconception_query) > 0
    assert len(res.strategy_query) > 0
    assert len(res.curriculum_query) > 0
    assert "Inequalities" in res.injected_domain_context


def test_multi_perspective_rrf_retrieval(in_memory_storage):
    """测试 Multi-Query + RRF 融合检索完整闭环。"""
    retriever = DualMetricRetriever(storage_manager=in_memory_storage)
    
    # 之前单纯靠原始向量漏召回的 LCM 问题 (#05)
    query = "学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？"
    res = retriever.retrieve_multi_perspective_rrf(query, top_k_each=2, fetch_evidence=True)
    
    assert res["raw_query"] == query
    assert len(res["misconceptions"]) > 0
    top_m = res["misconceptions"][0]
    
    # 验证 Session #23 (LCM 题) 是否精准被 RRF 顶升至首位！
    assert top_m["metadata"]["session_id"] == 23
    assert "rrf_score" in top_m
    assert top_m["rrf_score"] > 0.0
    assert len(top_m.get("evidence_turns", [])) > 0
