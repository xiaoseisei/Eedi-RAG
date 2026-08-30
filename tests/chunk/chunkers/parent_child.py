# -*- coding: utf-8 -*-
"""
tests/chunk/chunkers/parent_child.py
-----------------------------------
策略 3: 层次化父子索引切块器 (Hierarchical Parent-Child Chunker)。
核心思想：“小块（Child）用于精准向量检索，大块（Parent）用于大模型富上下文生成”。
每个 Child Chunk 仅包含 2 轮紧凑问答，但在 Metadata 中完整绑定全局 Parent 会话上下文。
"""

from typing import List, Dict, Any
from tests.chunk.chunkers.base import BaseChunker, Chunk


class ParentChildChunker(BaseChunker):
    """
    层次化父子索引切块器

    核心思想：检索时用小块（Child），生成时用大块（Parent）。
    - Child Chunk: 仅 2 轮紧凑问答，高语义密度，用于精准向量检索命中
    - Parent Context: 完整会话上下文，存储在 metadata 中，检索命中后供 LLM 富上下文生成

    检索流程:
        1. 用户问题 → 向量化 → 与 Child Chunk 的 text 字段匹配
        2. 命中 Child → 从 metadata 取出 parent_context
        3. 将 parent_context + 用户问题一起送入 LLM 生成回答

    Args:
        child_turn_size: Child Chunk 的轮次大小 (默认 2 轮，一组紧密师生问答)
        child_step_size: Child Chunk 滑动步长 (默认 2 轮)
    """

    def __init__(self, child_turn_size: int = 2, child_step_size: int = 2):
        strategy_name = f"ParentChild_C{child_turn_size}_S{child_step_size}"
        super().__init__(strategy_name=strategy_name)
        self.child_turn_size = child_turn_size
        self.child_step_size = child_step_size

    def chunk_session(self, session: Dict[str, Any]) -> List[Chunk]:
        # 从 session 中提取核心字段
        intervention_id = session.get("intervention_id", 0)
        question_id = session.get("question_id", 0)
        turns = session.get("turns", [])
        subjects = session.get("subjects", {})
        question = session.get("question", {})

        if not turns:
            return []

        # === 第一步：构造全局 Parent Context（完整会话大上下文）===
        # Parent Context 包含整个会话的全部对话，作为检索命中后的富上下文输入
        subject_paths = subjects.get("paths", [])
        path_str = " | ".join(subject_paths) if subject_paths else "Unknown Subject"
        question_text = question.get("question_text", "")
        options = question.get("options", {})
        options_str = ", ".join([f"{k}: {v}" for k, v in sorted(options.items())])

        # 将所有轮次格式化为完整对话实录
        all_dialogue_lines = [
            f"[Turn {t.get('turn_id', idx + 1)} - {t.get('speaker', 'Unknown')}]: {t.get('text', '')}"
            for idx, t in enumerate(turns)
        ]

        # Parent Context 模板：包含考点、题干、选项、完整对话
        parent_context = (
            f"=== PARENT SESSION ({intervention_id}) ===\n"
            f"[考点]: {path_str}\n"
            f"[题干]: {question_text}\n"
            f"[选项]: {options_str}\n"
            f"--- 完整辅导实录 ---\n" + "\n".join(all_dialogue_lines)
        )

        chunks: List[Chunk] = []
        n_turns = len(turns)

        # === 第二步：切分精细的 Child Chunks（用于向量检索）===
        for i in range(0, n_turns, self.child_step_size):
            child_turns = turns[i : i + self.child_turn_size]
            if not child_turns:
                break

            turn_ids = [t.get("turn_id", i + idx + 1) for idx, t in enumerate(child_turns)]

            # Child 检索文本：仅包含紧凑问答 + 简明考点前缀（高语义密度）
            # 去掉 turn_id 标注以节省 token，仅保留说话人和内容
            child_dialogue = "\n".join([
                f"[{t.get('speaker', 'Unknown')}]: {t.get('text', '')}"
                for t in child_turns
            ])
            # 拼接考点路径作为前缀，提升向量检索的语义匹配精度
            child_search_text = f"[{path_str}] {child_dialogue}"
            # chunk_id 命名规则: {会话ID}_child_{首轮ID}_{末轮ID}
            chunk_id = f"{intervention_id}_child_{turn_ids[0]}_{turn_ids[-1]}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    intervention_id=intervention_id,
                    question_id=question_id,
                    strategy_name=self.strategy_name,
                    text=child_search_text,        # 检索时用的高密度文本
                    turn_ids=turn_ids,
                    metadata={
                        "parent_id": f"parent_{intervention_id}",   # 父级标识，用于关联
                        "parent_context": parent_context,           # 完整会话上下文，供 LLM 生成
                        "start_turn_id": turn_ids[0],
                        "end_turn_id": turn_ids[-1],
                        "is_child_chunk": True,                     # 标记为 Child Chunk
                    },
                    token_count_approx=self.estimate_token_count(child_search_text),
                )
            )

            # 窗口已覆盖到对话末尾，终止
            if i + self.child_turn_size >= n_turns:
                break

        return chunks
