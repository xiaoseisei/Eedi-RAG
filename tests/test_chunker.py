"""
================================================================================
测试模块: tests/test_chunker.py
测试定位: 验证 Step 2 生产级双基座切块流水线引擎 (src/chunker.py)
测试覆盖:
  1. 主线知识卡片切块器在缺少有效抽取时的 Fail-Fast 异常拦截 (KnowledgeDistilledChunker)
  2. 主线知识卡片切块器在合规输入下的双卡片切块生成
  3. 兜底 6 轮滑动窗口切块器 (SlidingWindowChunker, W6_S3)
  4. 统一调度门面类 (SessionChunker) 的策略路由与透明降级契约 (primary, fallback, hybrid)
  5. 真实清洗会话样本的端到端批量切块与 JSONL 序列化
================================================================================
"""

import sys
import json
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    CleanedSession,
    Chunk,
    StudentMisconceptionProfile,
    TutorStrategyProfile,
    ExtractedPIU
)
from src.chunker import (
    KnowledgeDistilledChunker,
    SlidingWindowChunker,
    SessionChunker
)


@pytest.fixture
def sample_session() -> CleanedSession:
    """载入一个真实清洗后的 CleanedSession 样本用于测试。"""
    sample_file = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    with open(sample_file, "r", encoding="utf-8") as f:
        line = f.readline()
        data = json.loads(line)
        return CleanedSession.model_validate(data)


@pytest.fixture
def valid_extracted_piu(sample_session: CleanedSession) -> ExtractedPIU:
    """提供一个合法的 ExtractedPIU 真实测试夹具。"""
    misc = StudentMisconceptionProfile(
        session_id=sample_session.intervention_id,
        question_id=sample_session.question_id,
        subject_path="Mathematics > Algebra",
        misconception_name="四舍五入数位保留混淆",
        error_choice="Sophie",
        deep_mechanism="学生误将保留 1 位小数理解为截断取 2 位。",
        confusion_triggers=["5.45", "1dp"],
        verbatim_student_quotes=["5.45"],
        source_turn_ids=[17]
    )
    strat = TutorStrategyProfile(
        session_id=sample_session.intervention_id,
        question_id=sample_session.question_id,
        pedagogical_goal="引导学生理解 1 位小数的进位规则",
        strategy_category="Socratic_Questioning",
        key_aha_question="what do you think 5.4598 rounds to to 1dp?",
        scaffolding_steps=["Step 1: 圈出第一位小数", "Step 2: 观察第二位进位"],
        analogy_or_metaphor=None,
        talk_moves=["<Press for Accuracy>"],
        resolution_outcome="学生自主推导出正确结果",
        source_turn_ids=[16, 18]
    )
    return ExtractedPIU(
        session_id=sample_session.intervention_id,
        question_id=sample_session.question_id,
        misconception=misc,
        tutor_strategy=strat,
        extraction_status="success"
    )


def test_knowledge_distilled_chunker_fails_fast_on_none(sample_session: CleanedSession):
    """测试当 extracted 为 None 时，主线切块器坚决拒绝伪造执行并抛出 ValueError。"""
    chunker = KnowledgeDistilledChunker()
    with pytest.raises(ValueError) as exc_info:
        chunker.chunk(sample_session, extracted=None)
    assert "需要有效的 ExtractedPIU 输入" in str(exc_info.value)
    assert "严禁在无大模型真实抽取时静默生成假卡片" in str(exc_info.value)


def test_knowledge_distilled_chunker_with_valid_extracted(sample_session: CleanedSession, valid_extracted_piu: ExtractedPIU):
    """测试主线知识卡片切块器生成两大独立 Chunk。"""
    chunker = KnowledgeDistilledChunker()
    chunks = chunker.chunk(sample_session, valid_extracted_piu)
    
    assert len(chunks) == 2
    assert chunks[0].chunk_type == "misconception"
    assert chunks[1].chunk_type == "tutor_strategy"
    
    # 验证错因卡
    misc_chunk = chunks[0]
    assert misc_chunk.session_id == sample_session.intervention_id
    assert misc_chunk.question_id == sample_session.question_id
    assert "学生认知误区卡" in misc_chunk.content
    assert "深层认知机理" in misc_chunk.content
    assert misc_chunk.token_count > 0
    assert len(misc_chunk.source_turn_ids) > 0
    
    # 验证策略卡
    strat_chunk = chunks[1]
    assert strat_chunk.session_id == sample_session.intervention_id
    assert "名师启发式策略卡" in strat_chunk.content
    assert "名师破局一问" in strat_chunk.content
    assert strat_chunk.token_count > 0


def test_sliding_window_chunker(sample_session: CleanedSession):
    """测试 6 轮窗口、步长 3 的兜底滑动窗口切块器。"""
    chunker = SlidingWindowChunker(window_size=6, step=3)
    chunks = chunker.chunk(sample_session)
    
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert first_chunk.chunk_type == "sliding_window"
    assert "学科考纲" in first_chunk.content
    assert "考题原题" in first_chunk.content
    assert "对话片段" in first_chunk.content
    assert first_chunk.metadata["window_size"] == 6
    assert first_chunk.metadata["step"] == 3
    assert len(first_chunk.source_turn_ids) <= 6


def test_session_chunker_modes_and_degradation(sample_session: CleanedSession, valid_extracted_piu: ExtractedPIU):
    """测试统一门面类 SessionChunker 在各模式下的边界与降级行为。"""
    session_chunker = SessionChunker()
    
    # 1. Primary 模式下无 extracted ➔ 拒绝执行并抛错
    with pytest.raises(ValueError) as exc_info:
        session_chunker.chunk_session(sample_session, extracted=None, strategy="primary")
    assert "请求了 'primary' 切块策略，但未提供有效的 ExtractedPIU" in str(exc_info.value)
    
    # 2. Primary 模式下有 extracted ➔ 正常生成 2 个卡片
    primary_chunks = session_chunker.chunk_session(sample_session, extracted=valid_extracted_piu, strategy="primary")
    assert len(primary_chunks) == 2
    assert primary_chunks[0].chunk_type == "misconception"
    
    # 3. Fallback 模式 ➔ 纯规则滑动窗口切块
    fallback_chunks = session_chunker.chunk_session(sample_session, extracted=None, strategy="fallback")
    assert len(fallback_chunks) > 2
    assert all(c.chunk_type == "sliding_window" for c in fallback_chunks)
    
    # 4. Hybrid 模式下有 extracted ➔ 主线卡片 + 兜底窗口全量合并
    hybrid_chunks_full = session_chunker.chunk_session(sample_session, extracted=valid_extracted_piu, strategy="hybrid")
    assert len(hybrid_chunks_full) == len(primary_chunks) + len(fallback_chunks)
    
    # 5. Hybrid 模式下无 extracted ➔ 优雅降级为纯滑动窗口，绝不编造假卡片
    hybrid_chunks_degraded = session_chunker.chunk_session(sample_session, extracted=None, strategy="hybrid")
    assert len(hybrid_chunks_degraded) == len(fallback_chunks)
    assert all(c.chunk_type == "sliding_window" for c in hybrid_chunks_degraded)


def test_batch_chunking_and_serialization(tmp_path: Path, valid_extracted_piu: ExtractedPIU):
    """测试批量切块与保存为 JSONL 文件。"""
    sample_file = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    output_file = tmp_path / "chunks_test.jsonl"
    
    sessions = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sessions.append(CleanedSession.model_validate(json.loads(line)))
                if len(sessions) >= 3:
                    break
                    
    extracted_map = {sessions[0].intervention_id: valid_extracted_piu}
    
    session_chunker = SessionChunker()
    # 使用 fallback 模式批量切块
    all_chunks = session_chunker.batch_chunk_sessions(sessions, extracted_map=extracted_map, strategy="fallback")
    assert len(all_chunks) > 0
    assert all(c.chunk_type == "sliding_window" for c in all_chunks)
    
    session_chunker.save_chunks_jsonl(all_chunks, output_file)
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == len(all_chunks)
        data = json.loads(lines[0])
        assert "chunk_id" in data
        assert "content" in data
        assert "metadata" in data
