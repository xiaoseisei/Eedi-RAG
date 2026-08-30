# -*- coding: utf-8 -*-
"""
tests/chunk/chunkers/sliding_window.py
-------------------------------------
策略 1: 基于对话轮次的滑动窗口切块器 (Turn-Based Sliding Window Chunker)。
按照固定的轮次数 window_size 进行切分，滑动步长为 step_size，保留前后重叠以维持问答因果链。
"""

from typing import List, Dict, Any
from tests.chunk.chunkers.base import BaseChunker, Chunk


class TurnSlidingWindowChunker(BaseChunker):
    """
    基于对话轮次的滑动窗口切块器

    按固定轮次数 window_size 滑动切分对话，步长为 step_size。
    当 step_size < window_size 时，相邻 Chunk 之间有重叠轮次，
    保证对话的上下文连贯性不因切块而断裂。

    示例 (window_size=4, step_size=2):
        Chunk1: [Turn 1, 2, 3, 4]
        Chunk2: [Turn 3, 4, 5, 6]  ← 与 Chunk1 重叠 Turn 3,4
        Chunk3: [Turn 5, 6, 7, 8]

    Args:
        window_size: 每个 Chunk 包含的逻辑轮次数 (如 4 轮)
        step_size: 每次滑动的轮次步长 (如 2 轮，产生 2 轮重叠)
    """

    def __init__(self, window_size: int = 4, step_size: int = 2):
        # 策略名编码窗口参数，方便评测时按策略名分组对比
        strategy_name = f"SlidingWindow_W{window_size}_S{step_size}"
        super().__init__(strategy_name=strategy_name)
        self.window_size = window_size
        self.step_size = step_size

    def chunk_session(self, session: Dict[str, Any]) -> List[Chunk]:
        # 从 session 中提取核心字段
        intervention_id = session.get("intervention_id", 0)
        question_id = session.get("question_id", 0)
        turns = session.get("turns", [])

        if not turns:
            return []

        chunks: List[Chunk] = []
        n_turns = len(turns)

        # 以 step_size 为步长遍历所有轮次，每次取 window_size 个轮次
        for i in range(0, n_turns, self.step_size):
            # 截取当前窗口：若剩余轮次不足 window_size，取到末尾即可
            window_turns = turns[i : i + self.window_size]
            if not window_turns:
                break

            # 提取每个 turn 的 ID（若缺失则按位置补全）
            turn_ids = [t.get("turn_id", i + idx + 1) for idx, t in enumerate(window_turns)]

            # 格式化组装对话文本：[Turn ID - Speaker]: Text
            lines = []
            for t in window_turns:
                speaker = t.get("speaker", "Unknown")
                text = t.get("text", "")
                turn_id = t.get("turn_id", 0)
                lines.append(f"[Turn {turn_id} - {speaker}]: {text}")

            combined_text = "\n".join(lines)
            # chunk_id 命名规则: {会话ID}_sw_{首轮ID}_{末轮ID}
            chunk_id = f"{intervention_id}_sw_{turn_ids[0]}_{turn_ids[-1]}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    intervention_id=intervention_id,
                    question_id=question_id,
                    strategy_name=self.strategy_name,
                    text=combined_text,
                    turn_ids=turn_ids,
                    metadata={
                        "start_turn_id": turn_ids[0],   # 窗口起始轮次
                        "end_turn_id": turn_ids[-1],     # 窗口结束轮次
                        "num_turns": len(window_turns),  # 本窗口实际包含的轮次数
                    },
                    token_count_approx=self.estimate_token_count(combined_text),
                )
            )

            # 窗口已覆盖到对话末尾，无需继续滑动
            if i + self.window_size >= n_turns:
                break

        return chunks
