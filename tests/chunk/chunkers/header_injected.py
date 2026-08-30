# -*- coding: utf-8 -*-
"""
tests/chunk/chunkers/header_injected.py
--------------------------------------
策略 2: 考纲与题干元数据 Header 注入切块器 (Header-Injected Turn Chunker)。
在每个精细的师生交互轮次对（如 2 轮）前列，强行注入学科考纲路径、题目题干和选项内容，
解决局部对话脱离题目背景后语义悬空的问题。
"""

from typing import List, Dict, Any
from tests.chunk.chunkers.base import BaseChunker, Chunk


class HeaderInjectedChunker(BaseChunker):
    """
    元数据 Header 注入切块器

    核心思路：在每个 Chunk 的对话文本前，强制注入考纲路径、题干和选项作为全局 Header，
    解决切块后局部对话脱离题目背景导致语义悬空的问题。

    效果对比:
        - 普通切块: "[Turn 3 - Student]: 这个怎么解？"  ← 缺少题目上下文
        - Header注入: "[考纲考点]: ... \n[题目题干]: ... \n[Turn 3 - Student]: 这个怎么解？"  ← 语义完整

    Args:
        pair_size: 每个 Chunk 包含的对话轮次数 (默认 2 轮，一组标准师生问答对)
        step_size: 滑动步长 (默认 2 轮，即无重叠)
    """

    def __init__(self, pair_size: int = 2, step_size: int = 2):
        # 策略名编码 pair 和 step 参数
        strategy_name = f"HeaderInjected_P{pair_size}_S{step_size}"
        super().__init__(strategy_name=strategy_name)
        self.pair_size = pair_size
        self.step_size = step_size

    def chunk_session(self, session: Dict[str, Any]) -> List[Chunk]:
        # 从 session 中提取核心字段
        intervention_id = session.get("intervention_id", 0)
        question_id = session.get("question_id", 0)
        turns = session.get("turns", [])
        subjects = session.get("subjects", {})
        question = session.get("question", {})

        if not turns:
            return []

        # === 第一步：构造全局 Header（所有 Chunk 共享同一份）===
        # 考纲路径，如 "函数 | 三角函数 | 三角恒等变换"
        subject_paths = subjects.get("paths", [])
        path_str = " | ".join(subject_paths) if subject_paths else "Unknown Subject"

        # 题干原文
        question_text = question.get("question_text", "")
        # 选项文本，格式化为 "A: xxx, B: yyy, ..."
        options = question.get("options", {})
        options_str = ", ".join([f"{k}: {v}" for k, v in sorted(options.items())]) if options else "No Options"

        # 组装标准 Header 模板
        header = (
            f"[考纲考点]: {path_str}\n"
            f"[题目题干]: {question_text}\n"
            f"[选项分布]: {options_str}\n"
            f"--- [师生互动实录] ---"
        )

        chunks: List[Chunk] = []
        n_turns = len(turns)

        # 以 step_size 为步长，每次取 pair_size 个轮次
        for i in range(0, n_turns, self.step_size):
            window_turns = turns[i : i + self.pair_size]
            if not window_turns:
                break

            turn_ids = [t.get("turn_id", i + idx + 1) for idx, t in enumerate(window_turns)]

            # 格式化对话内容
            lines = []
            for t in window_turns:
                speaker = t.get("speaker", "Unknown")
                text = t.get("text", "")
                turn_id = t.get("turn_id", 0)
                lines.append(f"[Turn {turn_id} - {speaker}]: {text}")

            dialogue_body = "\n".join(lines)
            # 将 Header 与对话体拼接：Header 在前，保证 LLM 优先读取上下文
            full_chunk_text = f"{header}\n{dialogue_body}"
            # chunk_id 命名规则: {会话ID}_hi_{首轮ID}_{末轮ID}
            chunk_id = f"{intervention_id}_hi_{turn_ids[0]}_{turn_ids[-1]}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    intervention_id=intervention_id,
                    question_id=question_id,
                    strategy_name=self.strategy_name,
                    text=full_chunk_text,
                    turn_ids=turn_ids,
                    metadata={
                        "subject_path": path_str,       # 考纲路径，方便按考点筛选
                        "question_text": question_text,  # 原始题干，便于回溯
                        "start_turn_id": turn_ids[0],
                        "end_turn_id": turn_ids[-1],
                        "has_header_injected": True,     # 标记该 Chunk 已注入 Header
                    },
                    token_count_approx=self.estimate_token_count(full_chunk_text),
                )
            )

            # 窗口已覆盖到对话末尾，终止
            if i + self.pair_size >= n_turns:
                break

        return chunks
