# -*- coding: utf-8 -*-
"""
tests/chunk/chunkers/knowledge_distilled.py
------------------------------------------
策略 4: 结构化教学知识蒸馏切块器 (Knowledge-Distilled Pedagogical Card Chunker)。
针对长对话进行教学机理提纯，产出两个物理隔离的高密度知识卡片：
1. 学生认知误区卡片 (Misconception Card): 汇聚考点、错选、深层思维机理与学生原声。
2. 名师破局策略卡片 (Tutor Strategy Card): 汇聚核心破局提问、引导步骤链与教学动作。
"""

from typing import List, Dict, Any
from tests.chunk.chunkers.base import BaseChunker, Chunk


class KnowledgeDistilledChunker(BaseChunker):
    """
    教学知识结构化蒸馏切块器

    不做简单的文本切分，而是对长对话进行教学机理提纯，
    产出两个物理隔离的高密度知识卡片：

    1. 学生认知误区卡片 (Misconception Card):
       - 汇聚考点路径、题目选项、学生原声中的典型错解
       - 用途: 帮助 LLM 理解"学生在哪里卡住了、为什么卡住"

    2. 名师破局策略卡片 (Tutor Strategy Card):
       - 汇聚导师的启发式提问、分步脚手架引导、教学动作标签
       - 用途: 帮助 LLM 学习"好的教学引导长什么样"

    设计哲学: 一场对话 → 两张知识卡片，从"过程记录"蒸馏为"结构化教学知识"。
    """

    def __init__(self):
        # 固定策略名，因为该策略不依赖滑动窗口参数
        super().__init__(strategy_name="KnowledgeDistilled_Cards")

    def chunk_session(self, session: Dict[str, Any]) -> List[Chunk]:
        # 从 session 中提取核心字段
        intervention_id = session.get("intervention_id", 0)
        question_id = session.get("question_id", 0)
        turns = session.get("turns", [])
        subjects = session.get("subjects", {})
        question = session.get("question", {})

        if not turns:
            return []

        # 提取考纲路径和题目信息（两张卡片共享的元数据）
        subject_paths = subjects.get("paths", [])
        path_str = " | ".join(subject_paths) if subject_paths else "Unknown Subject"
        question_text = question.get("question_text", "")
        options = question.get("options", {})
        options_str = ", ".join([f"{k}: {v}" for k, v in sorted(options.items())])

        # 按角色分离轮次：过滤掉寒暄/噪声轮次
        # 学生轮次: is_tutor=False 且非噪声
        student_turns = [t for t in turns if not t.get("is_tutor", False) and not t.get("is_greeting_or_noise", False)]
        # 导师轮次: is_tutor=True 且非噪声
        tutor_turns = [t for t in turns if t.get("is_tutor", True) and not t.get("is_greeting_or_noise", False)]

        chunks: List[Chunk] = []

        # === 卡片 A: 学生认知误区卡 (Misconception Profile Chunk) ===
        # 仅取前 3 条学生原声作为典型错解样本（避免过长）
        student_quotes = [f"Turn {t.get('turn_id')}: \"{t.get('text')}\"" for t in student_turns[:3]]
        student_turn_ids = [t.get("turn_id", 0) for t in student_turns]

        # 组装误区卡文本：结构化呈现学生的困惑点
        misconception_text = (
            f"【学生认知错因卡】\n"
            f"[考点路径]: {path_str}\n"
            f"[原题干]: {question_text}\n"
            f"[备选选项]: {options_str}\n"
            f"[学生疑惑与典型错解原声]:\n" + "\n".join(student_quotes)
        )

        chunks.append(
            Chunk(
                chunk_id=f"{intervention_id}_distilled_misconception",
                intervention_id=intervention_id,
                question_id=question_id,
                strategy_name=self.strategy_name,
                text=misconception_text,
                turn_ids=student_turn_ids,
                metadata={
                    "card_type": "misconception",   # 卡片类型标记
                    "subject_path": path_str,
                    "target_role": "Student",       # 目标角色：学生视角
                },
                token_count_approx=self.estimate_token_count(misconception_text),
            )
        )

        # === 卡片 B: 名师破局引导策略卡 (Tutor Strategy Profile Chunk) ===
        # 过滤掉过短的导师发言（<15字符通常是确认/寒暄），保留有教学价值的内容
        # talk_moves 是导师的教学动作标签（如 "scaffolding", "questioning", "feedback"）
        tutor_prompts = [
            f"Turn {t.get('turn_id')} ({','.join(t.get('talk_moves', []))}): \"{t.get('text')}\""
            for t in tutor_turns
            if len(t.get("text", "")) > 15
        ]
        tutor_turn_ids = [t.get("turn_id", 0) for t in tutor_turns]

        # 组装策略卡文本：结构化呈现导师的教学引导过程
        strategy_text = (
            f"【名师破局引导策略卡】\n"
            f"[考点路径]: {path_str}\n"
            f"[原题干]: {question_text}\n"
            f"[核心启发式提问与分步脚手架实录]:\n" + "\n".join(tutor_prompts)
        )

        chunks.append(
            Chunk(
                chunk_id=f"{intervention_id}_distilled_strategy",
                intervention_id=intervention_id,
                question_id=question_id,
                strategy_name=self.strategy_name,
                text=strategy_text,
                turn_ids=tutor_turn_ids,
                metadata={
                    "card_type": "tutor_strategy",  # 卡片类型标记
                    "subject_path": path_str,
                    "target_role": "Tutor",         # 目标角色：导师视角
                },
                token_count_approx=self.estimate_token_count(strategy_text),
            )
        )

        return chunks
