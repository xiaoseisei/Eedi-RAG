# -*- coding: utf-8 -*-
"""
tests/chunk/chunkers/base.py
---------------------------
长对话分块策略抽象基类与数据契约定义。
定义了所有切块器统一遵循的输入输出规范与 Chunk 数据模型。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    统一切块数据模型 (Chunk Data Model)

    所有切块策略的输出统一为该结构，便于下游向量化、检索与评测流程无差别对接。
    每个 Chunk 代表一段可独立检索的文本片段，附带完整的溯源信息（会话/题目/轮次）。

    Attributes:
        chunk_id: 切块全局唯一标识符，命名规则因策略而异 (如 "10_sw_1_4" 表示滑动窗口)
        intervention_id: 关联的教学会话 ID，用于回溯完整辅导对话
        question_id: 关联的考题 ID，用于按题目维度聚合分析
        strategy_name: 产生该 Chunk 的切块策略名称 (如 "SlidingWindow_W4_S2")
        text: 经过策略处理后、用于向量化与文本检索的核心文本内容
        turn_ids: 该 Chunk 包含或引用的对话轮次 ID 列表 (如 [1, 2, 3, 4])
        metadata: 附带的结构化元数据，字段因策略而异（如考点路径、父级 ID、卡片类型等）
        token_count_approx: 近似 Token / 单词数统计，用于预估存储与推理成本
    """
    chunk_id: str = Field(..., description="切块全局唯一 ID")
    intervention_id: int = Field(..., description="所属会话 ID")
    question_id: int = Field(..., description="所属题目 ID")
    strategy_name: str = Field(..., description="切块策略名称")
    text: str = Field(..., description="检索核心文本内容")
    turn_ids: List[int] = Field(default_factory=list, description="涉及的对话轮次序号列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="结构化元数据")
    token_count_approx: int = Field(0, description="近似 Token 数量")


class BaseChunker(ABC):
    """
    分块策略抽象基类 (Base Chunker Interface)

    所有具体切块策略（滑动窗口、Header 注入、父子索引、知识蒸馏）均需继承并实现该接口。
    提供统一的 strategy_name 属性与轻量级 Token 估算工具方法。
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def chunk_session(self, session: Dict[str, Any]) -> List[Chunk]:
        """
        对单场 CleanedSession 执行分块操作。

        Args:
            session: CleanedSession 字典对象，预期包含:
                - intervention_id (int): 会话唯一 ID
                - question_id (int): 题目 ID
                - subjects (dict): 学科信息，含 paths 列表
                - question (dict): 题目信息，含 question_text 和 options
                - turns (list[dict]): 清洗后的对话轮次列表

        Returns:
            List[Chunk]: 切分出的统一 Chunk 对象列表
        """
        pass

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        轻量级 Token / 词数估算（支持中英文混合）。

        估算逻辑:
            - 英文按空格分割，每个词视为一个 token
            - 中文每个字符视为一个 token（通过总字符数 / 3 来近似覆盖）
            - 取两者较大值作为最终估算
        注: 这是粗略估算，仅用于评测阶段的 Token 膨胀比预估，不用于精确计费。

        Args:
            text: 待估算的文本内容

        Returns:
            int: 近似 token 数量
        """
        if not text:
            return 0
        # 英文按空格分割得到词数
        words = text.split()
        # 取英文词数和中英混合估算（字符数/3）的较大值
        return max(len(words), int(len(text) / 3))
