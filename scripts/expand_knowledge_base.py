"""
================================================================================
脚本名称: scripts/expand_knowledge_base.py
业务定位: Step 6 - 工业级知识库分层抽样扩容引擎 (Knowledge Base Expansion Engine)
核心功能:
  1. 从 1,576 场清洗会话 (cleaned_sessions.jsonl) 中分层抽样 100 场涵盖全学科考纲的真实辅导会话:
     - Number (数与运算): ~40 场 (四舍五入、分数、小数、质数与倍数、速算)
     - Algebra (代数与方程): ~25 场 (一元一次不等式、代数式化简、指数与负数幂)
     - Geometry and Measure (几何与度量): ~20 场 (三棱柱/长方体体积、表面积、时钟读数)
     - Data and Statistics (数据与统计): ~15 场 (频数分布表、极差、平均数、概率)
  2. 提取高保真原子知识资产 (ExtractedPIU) 与 6 轮滑动窗口冷备 (SlidingWindowChunk)。
  3. 双引擎批量持久化至:
     - DuckDB: data/db/tutoring_knowledge.duckdb (100 sessions, ~1,800+ turns, 100 错因卡, 100 策略卡, 600+ 滑动窗口)
     - ChromaDB: data/chroma (student_misconceptions, tutor_strategies, fallback_windows)
  4. 输出全量白盒审计报告，为用户设计 30 题 Golden Benchmark 提供坚实、无污染的底层知识底座。
================================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    CleanedSession,
    ExtractedPIU,
    Chunk
)
from src.chunker import SessionChunker
from src.evidence_index import bind_card_to_evidence_index, build_evidence_index
from src.storage_manager import DualEngineStorageManager
from src.extract_knowledge import (
    EXTRACTION_PROMPT_VERSION,
    LLMExtractionError,
    LLMUnavailableError,
    extract_knowledge_from_session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s"
)
logger = logging.getLogger(__name__)


def _ground_quotes_to_actual_dialogue(session: CleanedSession, piu: ExtractedPIU) -> ExtractedPIU:
    """
    用真实对话原文替换 LLM 转录的引用，确保逐字精确。
    LLM 不擅长精确复制原文（会截断、省略 emoji、改写），这一步保证 ground truth 可信。
    """
    # 建立 turn_id -> 实际文本的映射
    turn_map = {turn.turn_id: turn.text for turn in session.turns}

    if piu.misconception:
        # 替换学生原声引用为实际文本
        grounded_quotes = []
        grounded_turn_ids = []
        for turn_id in piu.misconception.source_turn_ids:
            actual_text = turn_map.get(turn_id)
            if actual_text:
                grounded_quotes.append(actual_text)
                grounded_turn_ids.append(turn_id)
        piu.misconception.verbatim_student_quotes = grounded_quotes
        piu.misconception.source_turn_ids = grounded_turn_ids

    if piu.tutor_strategy:
        # 替换破局一问为实际文本
        if piu.tutor_strategy.source_turn_ids:
            # 从导师轮次中找到第一个问句作为破局一问
            for turn_id in piu.tutor_strategy.source_turn_ids:
                actual_text = turn_map.get(turn_id)
                if actual_text and '?' in actual_text:
                    piu.tutor_strategy.key_aha_question = actual_text
                    break

        # 替换导师证据轮次的逐字引用
        grounded_turn_ids = []
        for turn_id in piu.tutor_strategy.source_turn_ids:
            if turn_id in turn_map:
                grounded_turn_ids.append(turn_id)
        piu.tutor_strategy.source_turn_ids = grounded_turn_ids

    return piu


def extract_piu_with_llm(
    session: CleanedSession,
    llm_client: "openai.OpenAI",
    model_name: str,
    max_retries: int = 2
) -> ExtractedPIU:
    """使用 evidence-completeness-v2 与严格 schema/角色 grounding 门禁。"""
    if not model_name or not model_name.strip():
        raise LLMUnavailableError("知识库扩容必须显式配置非空 LLM_MODEL")
    try:
        piu = extract_knowledge_from_session(
            session,
            llm_client=llm_client,
            model=model_name,
            max_retries=max_retries,
            # 第二遍 evidence audit 尚未通过真实 provider 评测，保持显式关闭。
            evidence_audit=False,
            semantic_only=True,
        )
        if piu.misconception is None or piu.tutor_strategy is None:
            raise LLMExtractionError("semantic-only 抽取没有生成完整双卡片")
        evidence_index = build_evidence_index(session, window_size=6, step=3)
        return piu.model_copy(update={
            "misconception": bind_card_to_evidence_index(piu.misconception, evidence_index),
            "tutor_strategy": bind_card_to_evidence_index(piu.tutor_strategy, evidence_index),
        })
    except (LLMExtractionError, LLMUnavailableError):
        logger.error("Session #%s LLM 抽取失败，扩容任务立即终止", session.intervention_id)
        raise
    except Exception as exc:
        logger.error("Session #%s LLM 抽取发生未预期异常，扩容任务立即终止: %s", session.intervention_id, exc)
        raise LLMExtractionError(
            f"Session {session.intervention_id} 扩容抽取失败，拒绝继续部分写入"
        ) from exc


def extract_piu_deterministic_high_fidelity(session: CleanedSession) -> ExtractedPIU:
    """已禁用：规则可以切块真实原文，但不能冒充语义知识抽取。"""
    raise LLMUnavailableError(
        "确定性模板知识卡抽取已禁用；请使用真实 LLM 严格抽取或人工审核卡片。"
    )


def select_stratified_100_sessions(cleaned_sessions_path: str = "data/cleaned_sessions.jsonl") -> List[CleanedSession]:
    """分层抽样 100 场各考纲分布均衡的高质量辅导会话。"""
    sessions_by_subject = defaultdict(list)
    
    with open(cleaned_sessions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            sess = CleanedSession.model_validate(data)
            if sess.has_valid_tutoring and sess.total_turns >= 6:
                main_subj = sess.subjects.paths[0].split(" > ")[0] if sess.subjects.paths else "General"
                sessions_by_subject[main_subj].append(sess)
                
    logger.info(f"📊 全量可用清洗会话分布: { {k: len(v) for k, v in sessions_by_subject.items()} }")
    
    # 目标配比: Number 40, Algebra 25, Geometry 20, Data/Statistics 15
    quota = {
        "Number": 40,
        "Algebra": 25,
        "Geometry and Measure": 20,
        "Data and Statistics": 15
    }
    
    selected_sessions: List[CleanedSession] = []
    seen_qids = set()
    
    for subj, count in quota.items():
        candidates = sessions_by_subject.get(subj, [])
        added = 0
        for s in candidates:
            if s.question_id not in seen_qids:
                seen_qids.add(s.question_id)
                selected_sessions.append(s)
                added += 1
                if added >= count:
                    break
        # 若题目有重叠则放宽题目去重限制填满配额
        if added < count:
            for s in candidates:
                if s not in selected_sessions:
                    selected_sessions.append(s)
                    added += 1
                    if added >= count:
                        break
                        
    logger.info(f"✅ 成功分层抽样 {len(selected_sessions)} 场典型会话！")
    return selected_sessions


def run_expansion(use_llm: bool = True, limit: Optional[int] = None):
    """
    执行知识库批量扩容与双引擎持久化。

    Args:
        use_llm: 必须为 True；模板知识卡抽取已禁用
        limit: 限制处理的 session 数量 (用于测试，None = 全量)
    """
    logger.info("🚀 [Step6_Expansion] 开始执行知识库扩容流程...")
    if load_dotenv is not None:
        load_dotenv()

    # 依赖配置在读取数据前即严格校验，避免昂贵处理后才暴露不可用状态。
    if not use_llm:
        raise LLMUnavailableError("LLM 抽取不可禁用：模板知识卡路径已永久禁用")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise LLMUnavailableError("知识库扩容缺少 LLM_API_KEY，拒绝模板降级")
    if not model_name:
        raise LLMUnavailableError("知识库扩容缺少 LLM_MODEL，拒绝隐式模型默认值")
    try:
        import openai
    except ImportError as exc:
        raise LLMUnavailableError("知识库扩容缺少 openai 依赖") from exc
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    llm_client = openai.OpenAI(**client_kwargs)
    logger.info(
        "✅ LLM 严格抽取模式已启用 (model=%s, prompt_version=%s, evidence_audit=false)",
        model_name,
        EXTRACTION_PROMPT_VERSION,
    )

    # 2. 分层抽样
    selected_sessions = select_stratified_100_sessions()
    if limit:
        selected_sessions = selected_sessions[:limit]
        logger.info(f"🔧 测试模式: 仅处理前 {limit} 个 session")

    if not selected_sessions:
        raise ValueError("分层抽样未返回任何 session，拒绝写入空扩容结果")

    # 3. 在创建存储对象前完成全量抽取、严格校验与切块。
    chunker = SessionChunker(window_size=6, step=3, default_strategy="hybrid")
    all_extracted: List[ExtractedPIU] = []
    all_chunks: List[Chunk] = []
    count_success = 0
    count_failed = 0

    for idx, s in enumerate(selected_sessions, 1):
        piu = extract_piu_with_llm(s, llm_client, model_name)
        if piu.extraction_status != "success" or piu.misconception is None or piu.tutor_strategy is None:
            count_failed += 1
            logger.warning(f"  [{idx}/{len(selected_sessions)}] Session #{s.intervention_id} → LLM 抽取失败，跳过 ❌")
            continue
        count_success += 1
        all_extracted.append(piu)
        chunks = chunker.chunk_session(s, piu)
        all_chunks.extend(chunks)
        logger.info(
            "  [%s/%s] Session #%s → LLM 抽取通过 ✅",
            idx,
            len(selected_sessions),
            s.intervention_id,
        )

    if not all_extracted:
        raise ValueError("所有 session 抽取均失败，拒绝写入空库")

    # 真实性审计报告
    total = len(selected_sessions)
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 数据真实性审计报告:")
    logger.info(f"  总处理 session: {total}")
    logger.info(f"  LLM 抽取成功: {count_success} ({count_success/total*100:.1f}%)")
    logger.info(f"  LLM 抽取失败: {count_failed} ({count_failed/total*100:.1f}%)")
    logger.info(f"  模板降级: 0 (已禁用)")
    logger.info(f"{'='*60}")

    # 4. 持久化至双引擎数据库
    db_path = "data/db/tutoring_knowledge.duckdb"
    chroma_dir = "data/chroma"

    storage = DualEngineStorageManager(
        db_path=db_path,
        chroma_dir=chroma_dir,
        embedding_backend="deterministic",
    )

    try:
        storage.ingest_all(
            cleaned_sessions=selected_sessions,
            extracted_pius=all_extracted,
            chunks=all_chunks
        )
        audit = storage.get_storage_audit_snapshot()
    finally:
        storage.close()

    logger.info("🎉 [Step6_Expansion] 知识库扩容完成！")
    return audit


if __name__ == "__main__":
    run_expansion()
