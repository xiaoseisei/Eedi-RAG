"""
================================================================================
模块名称: src/chunker.py
业务定位: Step 2 - 生产级双基座 RAG 切块引擎 (Dual-Baseline Chunking Engine)
核心职责:
  1. 实现【主线基座】：知识卡片切块器 (KnowledgeDistilledChunker)，产出学生错因卡与导师策略卡。
     严格遵循反假可用原则：必须消费合法的 ExtractedPIU，若入参为空则立即抛出异常 (Fail-Fast)。
  2. 实现【兜底基座】：6 轮滑动窗口切块器 (SlidingWindowChunker, W6_S3)，纯规则算法冷备。
  3. 提供统一门面调度类 (SessionChunker)，支持 'primary' (主线卡片), 'fallback' (纯窗口), 'hybrid' (双模混合) 三种策略模式。
  4. 支持批量切块、Token 统计与 JSONL 序列化。
================================================================================
"""

import re
import sys
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    CleanedSession,
    Chunk,
    ExtractedPIU
)

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    轻量估算中英文混合文本的 Token 消耗量。
    
    规则:
      - 英文单词与数字按空白分词 (约 1 词 = 1.3 Tokens)
      - 中文字符按字符计数 (约 1 汉字 = 1 Token)
    """
    if not text:
        return 0
    words = re.findall(r'[A-Za-z0-9_]+', text)
    cjk_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return int(len(words) * 1.3) + len(cjk_chars)


class BaseChunker(ABC):
    """切块器统一抽象基类。"""
    
    @abstractmethod
    def chunk(self, session: CleanedSession, extracted: Optional[ExtractedPIU] = None) -> List[Chunk]:
        """对单个会话执行切块，返回 Chunk 列表。"""
        pass


class KnowledgeDistilledChunker(BaseChunker):
    """
    主线切块器：基于 LLM 真实结构化抽取的知识卡片切块器。
    
    产出两大物理隔离的高密度卡片:
      1. 学生认知误区卡 (Misconception Profile Card)
      2. 名师启发式策略卡 (Tutor Pedagogical Strategy Card)
      
    真实性契约:
      必须传入有效的 ExtractedPIU 对象。严禁在 extracted 为空时静默伪造假卡片。
    """
    
    def chunk(self, session: CleanedSession, extracted: Optional[ExtractedPIU] = None) -> List[Chunk]:
        if extracted is None:
            raise ValueError(
                f"KnowledgeDistilledChunker 需要有效的 ExtractedPIU 输入 (Session ID: {session.intervention_id})。"
                f"根据通用反假可用守则，严禁在无大模型真实抽取时静默生成假卡片。"
                f"请传入真实提取的 ExtractedPIU，或将切块策略切换为 'fallback' (纯规则滑动窗口)。"
            )
            
        chunks = []
        subject_path = session.subjects.paths[0] if session.subjects.paths else "Mathematics"
        
        # 1. 构建学生错因卡 Chunk
        if extracted.misconception:
            misc = extracted.misconception
            quotes_str = "\n".join(f'  - "{q}"' for q in misc.verbatim_student_quotes) if misc.verbatim_student_quotes else "  (无直接引用)"
            triggers_str = ", ".join(misc.confusion_triggers) if misc.confusion_triggers else "无"
            
            misc_content = f"""[知识卡片类型]: 学生认知误区卡 (Student Misconception Profile)
[学科考点]: {subject_path}
[考题原题]: {session.question.question_text}
[误选选项]: {misc.error_choice or '未确定'}
[错因标准命名]: {misc.misconception_name}
[深层认知机理]: {misc.deep_mechanism}
[困惑触发概念]: {triggers_str}
[学生原声引用]:
{quotes_str}"""
            
            chunks.append(Chunk(
                chunk_id=f"session_{session.intervention_id}_misconception",
                session_id=session.intervention_id,
                question_id=session.question_id,
                chunk_type="misconception",
                content=misc_content,
                metadata={
                    "intervention_id": session.intervention_id,
                    "question_id": session.question_id,
                    "subject_path": subject_path,
                    "misconception_name": misc.misconception_name,
                    "error_choice": misc.error_choice,
                    "confusion_triggers": misc.confusion_triggers,
                    "source_turn_ids": misc.source_turn_ids
                },
                token_count=estimate_tokens(misc_content),
                source_turn_ids=misc.source_turn_ids
            ))
            
        # 2. 构建导师启发式策略卡 Chunk
        if extracted.tutor_strategy:
            strat = extracted.tutor_strategy
            steps_str = "\n".join(f"  {step}" for step in strat.scaffolding_steps) if strat.scaffolding_steps else "  (分步拆解引导)"
            moves_str = ", ".join(strat.talk_moves) if strat.talk_moves else "无"
            
            strat_content = f"""[知识卡片类型]: 名师启发式策略卡 (Tutor Pedagogical Strategy)
[学科考点]: {subject_path}
[考题原题]: {session.question.question_text}
[教学引导目标]: {strat.pedagogical_goal}
[教学策略分类]: {strat.strategy_category}
[名师破局一问]: {strat.key_aha_question}
[脚手架步骤链]:
{steps_str}
[启发比喻案例]: {strat.analogy_or_metaphor or '无'}
[涉及教学动作]: {moves_str}
[最终辅导成效]: {strat.resolution_outcome}"""
            
            chunks.append(Chunk(
                chunk_id=f"session_{session.intervention_id}_tutor_strategy",
                session_id=session.intervention_id,
                question_id=session.question_id,
                chunk_type="tutor_strategy",
                content=strat_content,
                metadata={
                    "intervention_id": session.intervention_id,
                    "question_id": session.question_id,
                    "subject_path": subject_path,
                    "strategy_category": strat.strategy_category,
                    "key_aha_question": strat.key_aha_question,
                    "talk_moves": strat.talk_moves,
                    "source_turn_ids": strat.source_turn_ids
                },
                token_count=estimate_tokens(strat_content),
                source_turn_ids=strat.source_turn_ids
            ))
            
        return chunks


class SlidingWindowChunker(BaseChunker):
    """
    兜底切块器：6 轮滑动窗口、步长 3 (SlidingWindow_W6_S3)。
    
    业务背景:
      纯规则冷备方案。每个切块前置注入考纲路径与原题题干 Header，
      保证切块自包含性，覆盖多轮完整因果链条。
    """
    
    def __init__(self, window_size: int = 6, step: int = 3):
        self.window_size = window_size
        self.step = step
        
    def chunk(self, session: CleanedSession, extracted: Optional[ExtractedPIU] = None) -> List[Chunk]:
        chunks = []
        turns = session.turns
        total_turns = len(turns)
        
        if total_turns == 0:
            return chunks
            
        subject_path = session.subjects.paths[0] if session.subjects.paths else "Mathematics"
        header_prefix = f"[学科考纲]: {subject_path}\n[考题原题]: {session.question.question_text}\n"
        
        i = 0
        while i < total_turns:
            window_turns = turns[i : i + self.window_size]
            start_turn = window_turns[0].turn_id
            end_turn = window_turns[-1].turn_id
            
            turn_lines = []
            turn_ids = []
            for t in window_turns:
                turn_ids.append(t.turn_id)
                speaker_tag = "[Tutor]" if t.is_tutor else "[Student]"
                turn_lines.append(f"[Turn {t.turn_id}] {speaker_tag}: {t.text}")
                
            content = f"{header_prefix}[对话片段 (Turn {start_turn} ~ {end_turn})]:\n" + "\n".join(turn_lines)
            
            chunk = Chunk(
                chunk_id=f"session_{session.intervention_id}_win_{start_turn}_{end_turn}",
                session_id=session.intervention_id,
                question_id=session.question_id,
                chunk_type="sliding_window",
                content=content,
                metadata={
                    "intervention_id": session.intervention_id,
                    "question_id": session.question_id,
                    "subject_path": subject_path,
                    "window_start_turn": start_turn,
                    "window_end_turn": end_turn,
                    "window_size": self.window_size,
                    "step": self.step,
                    "source_turn_ids": turn_ids
                },
                token_count=estimate_tokens(content),
                source_turn_ids=turn_ids
            )
            chunks.append(chunk)
            
            if i + self.window_size >= total_turns:
                break
            i += self.step
            
        return chunks


class SessionChunker:
    """
    统一 RAG 切块门面类 (Unified Session Chunker Facade)。
    
    支持在主线知识卡片 ('primary')、纯滑动窗口 ('fallback') 与混合双模 ('hybrid') 之间灵活路由。
    
    降级原则:
      1. 若显式指定 strategy='primary' 但未传入 extracted，立即抛出异常 (Fail-Fast)。
      2. strategy='fallback' 始终基于纯规则执行，无需外部依赖。
      3. strategy='hybrid' 在有 extracted 时产出卡片+窗口；无 extracted 时透明降级为仅产出滑动窗口切块，
         并记录结构化 Warning 日志，绝不捏造假卡片。
    """
    
    def __init__(self, window_size: int = 6, step: int = 3, default_strategy: str = "fallback"):
        self.primary_chunker = KnowledgeDistilledChunker()
        self.fallback_chunker = SlidingWindowChunker(window_size=window_size, step=step)
        self.default_strategy = default_strategy
        
    def chunk_session(
        self,
        session: CleanedSession,
        extracted: Optional[ExtractedPIU] = None,
        strategy: Optional[str] = None
    ) -> List[Chunk]:
        """
        对单场会话执行切块。
        
        参数:
          session (CleanedSession): 输入会话
          extracted (Optional[ExtractedPIU]): 结构化抽取卡片
          strategy (Optional[str]): 'primary' (主线知识卡片), 'fallback' (6轮窗口), 'hybrid' (双模并存)
        """
        strat = strategy or self.default_strategy
        if strat == "primary":
            if extracted is None:
                raise ValueError(
                    f"会话 {session.intervention_id} 请求了 'primary' 切块策略，但未提供有效的 ExtractedPIU。"
                    f"根据工程真实性守则，拒绝伪造执行。请传入抽取资产或切换 strategy='fallback'。"
                )
            return self.primary_chunker.chunk(session, extracted)
        elif strat == "fallback":
            return self.fallback_chunker.chunk(session)
        elif strat == "hybrid":
            fallback_chunks = self.fallback_chunker.chunk(session)
            if extracted is not None:
                primary_chunks = self.primary_chunker.chunk(session, extracted)
                return primary_chunks + fallback_chunks
            else:
                logger.warning(
                    f"会话 {session.intervention_id} 未提供 ExtractedPIU，Hybrid 模式透明降级为纯滑动窗口切块。"
                )
                return fallback_chunks
        else:
            raise ValueError(f"未知的切分策略: {strat}，可选值为 'primary', 'fallback', 'hybrid'")
            
    def batch_chunk_sessions(
        self,
        sessions: List[CleanedSession],
        extracted_map: Optional[Dict[int, ExtractedPIU]] = None,
        strategy: Optional[str] = None
    ) -> List[Chunk]:
        """批量对多个会话执行切块。"""
        all_chunks = []
        for sess in sessions:
            ext = extracted_map.get(sess.intervention_id) if extracted_map else None
            chunks = self.chunk_session(sess, extracted=ext, strategy=strategy)
            all_chunks.extend(chunks)
        return all_chunks

    def save_chunks_jsonl(self, chunks: List[Chunk], output_file: Path) -> None:
        """将切块列表持久化保存为 JSONL 格式。"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
