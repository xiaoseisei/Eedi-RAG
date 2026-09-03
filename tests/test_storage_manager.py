"""
================================================================================
测试模块: tests/test_storage_manager.py
测试定位: 验证 Step 3 双引擎存储持久化管理器 (src/storage_manager.py)
测试覆盖:
  1. 纯内存与文件级 DuckDB + ChromaDB 双引擎初始化
  2. 会话事实表、对话明细表、错因表、策略表与滑动窗口表的批量摄取
  3. 确定性 SQL 过滤查询 (考点过滤、关键字匹配、教学分类过滤)
  4. DuckDB 学情看板 GROUP BY 聚合统计功能
  5. 核心证据反查接口 (get_dialogue_turns) 的精确字面量与行号回溯
  6. ChromaDB student_misconceptions 向量模糊语义检索
  7. ChromaDB tutor_strategies 向量模糊语义检索
  8. ChromaDB fallback_windows 纯原文直取检索 (零信任模式)
  9. 并行多集合检索与晚期融合 (search_parallel_and_fuse) 及原声证据自动附挂
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

from src.models import CleanedSession, ExtractedPIU, Chunk
from src.storage_manager import (
    DualEngineStorageManager,
    FastDeterministicEmbeddingFunction,
    build_misconception_embedding_doc,
    build_tutor_strategy_embedding_doc
)
from src.embedding_provider import EmbeddingIndexContract, EmbeddingProviderError


@pytest.fixture
def sample_sessions() -> list[CleanedSession]:
    """载入真实 CleanedSession 样本。"""
    sample_file = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    sessions = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sessions.append(CleanedSession.model_validate(json.loads(line)))
    return sessions


@pytest.fixture
def sample_extracted() -> list[ExtractedPIU]:
    """载入真实 ExtractedPIU 样本。"""
    sample_file = project_root / "data" / "sample" / "extracted_pius_sample.jsonl"
    extracted = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                extracted.append(ExtractedPIU.model_validate(json.loads(line)))
    return extracted


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """载入真实 Chunks 样本。"""
    sample_file = project_root / "data" / "sample" / "chunks_sample.jsonl"
    chunks = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


@pytest.fixture
def in_memory_manager(sample_sessions, sample_extracted, sample_chunks):
    """构建已灌入真实测试数据的纯内存双引擎管理器 fixture。"""
    manager = DualEngineStorageManager(
        db_path=":memory:",
        in_memory=True
    )
    manager.ingest_all(sample_sessions, sample_extracted, sample_chunks)
    yield manager
    manager.close()


def test_build_embedding_docs(sample_extracted):
    """测试语义增强模板构建，确保考纲路径、原题与机理等分类学特征被正确拼装。"""
    card = sample_extracted[0].misconception
    doc = build_misconception_embedding_doc(card, question_text="Test Question 104614")
    
    assert "【知识卡片类型】: 学生认知误区卡" in doc
    assert "【学科考纲路径】" in doc
    assert "【关联考题原题】: Test Question 104614" in doc
    assert card.misconception_name in doc
    assert card.deep_mechanism in doc
    
    strat_card = sample_extracted[0].tutor_strategy
    strat_doc = build_tutor_strategy_embedding_doc(strat_card, question_text="Test Question 104614", subject_path="Number > Test")
    assert "【知识卡片类型】: 名师启发式策略卡" in doc or "【知识卡片类型】: 名师启发式策略卡" in strat_doc
    assert strat_card.key_aha_question in strat_doc


def test_duckdb_tables_ingestion(in_memory_manager: DualEngineStorageManager, sample_sessions, sample_extracted):
    """测试 DuckDB 各关系底表的摄取完整性与主键约束。"""
    # 1. 验证 tutoring_sessions
    sess_count = in_memory_manager.duck_conn.execute("SELECT COUNT(*) FROM tutoring_sessions").fetchone()[0]
    assert sess_count == len(sample_sessions)
    
    # 2. 验证 session_dialogue_turns
    turn_count = in_memory_manager.duck_conn.execute("SELECT COUNT(*) FROM session_dialogue_turns").fetchone()[0]
    expected_turns = sum(s.total_turns for s in sample_sessions)
    assert turn_count == expected_turns
    
    # 3. 验证 misconception_chunks
    misc_count = in_memory_manager.duck_conn.execute("SELECT COUNT(*) FROM misconception_chunks").fetchone()[0]
    assert misc_count == len(sample_extracted)
    
    # 4. 验证 tutor_strategy_chunks
    strat_count = in_memory_manager.duck_conn.execute("SELECT COUNT(*) FROM tutor_strategy_chunks").fetchone()[0]
    assert strat_count == len(sample_extracted)


def test_get_dialogue_turns_exact_evidence_lookup(in_memory_manager: DualEngineStorageManager):
    """测试核心证据反查接口：根据 (session_id, turn_ids) 秒级从 DuckDB 精确反查原始对话。"""
    # 查询 Session 10 的第 10、12、18 轮对话
    turns = in_memory_manager.get_dialogue_turns(session_id=10, turn_ids=[10, 12, 18])
    
    assert len(turns) == 3
    assert turns[0]["turn_id"] == 10
    assert turns[1]["turn_id"] == 12
    assert turns[2]["turn_id"] == 18
    assert "5.45" in turns[0]["text"] or "clue" in turns[1]["text"] or "54.5" in turns[2]["text"]
    
    # 查询全量轮次
    all_turns = in_memory_manager.get_dialogue_turns(session_id=10)
    assert len(all_turns) == 26


def test_sql_analytics_and_group_by(in_memory_manager: DualEngineStorageManager):
    """测试 DuckDB 确定性 SQL 过滤与 GROUP BY 聚合统计。"""
    # 1. 关键字过滤
    results = in_memory_manager.query_misconceptions_sql(keyword="四舍五入")
    assert len(results) >= 1
    assert any("四舍五入" in r["misconception_name"] for r in results)
    
    # 2. 策略分类过滤
    scaffolding_strats = in_memory_manager.query_strategies_sql(strategy_category="Scaffolding")
    assert len(scaffolding_strats) >= 1
    assert all(r["strategy_category"] == "Scaffolding" for r in scaffolding_strats)
    
    # 3. 学情看板 GROUP BY 聚合
    stats = in_memory_manager.get_misconception_stats_by_subject()
    assert len(stats) >= 1
    assert "subject_path" in stats[0]
    assert "total_misconceptions" in stats[0]
    assert stats[0]["total_misconceptions"] >= 1


def test_chromadb_vector_searches(in_memory_manager: DualEngineStorageManager):
    """测试 ChromaDB 各独立向量集合的语义召回能力。"""
    # 1. 错因向量检索
    misc_hits = in_memory_manager.search_misconceptions_vector("学生把一位小数算成了两位数", top_k=2)
    assert len(misc_hits) > 0
    assert "session_id" in misc_hits[0]["metadata"]
    assert misc_hits[0]["chunk_type"] == "misconception"
    
    # 2. 策略向量检索
    strat_hits = in_memory_manager.search_strategies_vector("如何引导学生理解一位小数的四舍五入？", top_k=2)
    assert len(strat_hits) > 0
    assert "strategy_category" in strat_hits[0]["metadata"]
    assert strat_hits[0]["chunk_type"] == "tutor_strategy"
    
    # 3. 纯原文滑动窗口直取检索
    win_hits = in_memory_manager.search_fallback_windows_vector("Alex says 5.4598 rounded to 1 decimal place", top_k=2)
    assert len(win_hits) > 0
    assert win_hits[0]["chunk_type"] == "sliding_window"


def test_search_parallel_and_fuse_with_evidence(in_memory_manager: DualEngineStorageManager):
    """测试【反硬路由黄金接口】：双路并行召回 + 原声实录证据自动附挂全链路。"""
    query = "四舍五入 5.4598 到 1 位小数时，学生容易犯什么错？名师如何提问引导？"
    res = in_memory_manager.search_parallel_and_fuse(query, top_k_each=2, fetch_evidence_turns=True)
    
    assert res["query"] == query
    assert len(res["misconceptions"]) > 0
    assert len(res["strategies"]) > 0
    assert res["total_retrieved"] == len(res["misconceptions"]) + len(res["strategies"])
    
    # 核心断言：验证命中的卡片是否成功自动附带了来自 DuckDB 的原始对话证据实录
    top_misc = res["misconceptions"][0]
    assert "evidence_turns" in top_misc
    assert isinstance(top_misc["evidence_turns"], list)
    assert len(top_misc["evidence_turns"]) > 0
    assert "turn_id" in top_misc["evidence_turns"][0]
    assert "text" in top_misc["evidence_turns"][0]
    
    top_strat = res["strategies"][0]
    assert "evidence_turns" in top_strat
    assert isinstance(top_strat["evidence_turns"], list)
    assert len(top_strat["evidence_turns"]) > 0


def test_ingest_all_rejects_non_success_before_any_write(sample_sessions, sample_extracted):
    manager = DualEngineStorageManager(db_path=":memory:", in_memory=True)
    invalid = sample_extracted[0].model_copy(
        update={"extraction_status": "failed", "misconception": None, "tutor_strategy": None}
    )

    with pytest.raises(ValueError, match="拒绝写入"):
        manager.ingest_all(sample_sessions, [invalid])

    assert manager.duck_conn.execute("SELECT COUNT(*) FROM tutoring_sessions").fetchone()[0] == 0
    assert manager.duck_conn.execute("SELECT COUNT(*) FROM misconception_chunks").fetchone()[0] == 0
    manager.close()


def test_direct_card_ingest_enforces_evidence_roles(sample_sessions, sample_extracted):
    manager = DualEngineStorageManager(db_path=":memory:", in_memory=True)
    initial_vector_count = manager.coll_misconceptions.count()
    session = next(s for s in sample_sessions if s.intervention_id == sample_extracted[0].session_id)
    tutor_turn = next(turn.turn_id for turn in session.turns if turn.is_tutor)
    bad_misc = sample_extracted[0].misconception.model_copy(update={"source_turn_ids": [tutor_turn]})
    invalid = sample_extracted[0].model_copy(update={"misconception": bad_misc})

    with pytest.raises(ValueError, match="学生证据"):
        manager.ingest_extracted_pius([invalid], sessions_map={session.intervention_id: session})

    assert manager.duck_conn.execute("SELECT COUNT(*) FROM misconception_chunks").fetchone()[0] == 0
    assert manager.coll_misconceptions.count() == initial_vector_count
    manager.close()


def test_direct_card_ingest_requires_sessions_for_grounding(sample_extracted):
    manager = DualEngineStorageManager(db_path=":memory:", in_memory=True)
    with pytest.raises(ValueError, match="sessions_map"):
        manager.ingest_extracted_pius([sample_extracted[0]])
    manager.close()


def test_storage_exposes_embedding_backend():
    manager = DualEngineStorageManager(db_path=":memory:", in_memory=True)
    assert manager.embedding_backend == "fast_deterministic"
    manager.close()


def test_persistent_storage_rejects_index_with_mismatched_embedding_contract(tmp_path):
    class ContractEmbedding(FastDeterministicEmbeddingFunction):
        def contract(self, *, index_version=None):
            return EmbeddingIndexContract(self.name(), "test-model", self.dim, index_version or "test-v1")

    db_path = tmp_path / "test.duckdb"
    chroma_path = tmp_path / "chroma"
    first = DualEngineStorageManager(db_path=db_path, chroma_dir=chroma_path, embedding_function=ContractEmbedding())
    first.close()
    with pytest.raises(EmbeddingProviderError, match="contract mismatch"):
        DualEngineStorageManager(
            db_path=db_path,
            chroma_dir=chroma_path,
            embedding_function=ContractEmbedding(),
            embedding_index_version="test-v1",
        )
