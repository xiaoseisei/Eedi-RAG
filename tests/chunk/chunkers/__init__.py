# -*- coding: utf-8 -*-
"""
tests/chunk/chunkers/__init__.py
---------------------------------
分块策略模块入口，统一导出所有 Chunker 实现与基础数据模型。

使用方式:
    from tests.chunk.chunkers import TurnSlidingWindowChunker, Chunk

支持的切块策略:
    1. TurnSlidingWindowChunker  — 滑动窗口切块（按轮次固定窗口滑动）
    2. HeaderInjectedChunker     — Header 注入切块（在对话前注入考纲/题干元数据）
    3. ParentChildChunker        — 父子索引切块（小块检索 + 大块生成）
    4. KnowledgeDistilledChunker — 知识蒸馏切块（从对话中蒸馏出结构化教学卡片）
"""

# 基础类：数据模型与抽象接口
from tests.chunk.chunkers.base import BaseChunker, Chunk
# 策略 1: 滑动窗口
from tests.chunk.chunkers.sliding_window import TurnSlidingWindowChunker
# 策略 2: Header 注入
from tests.chunk.chunkers.header_injected import HeaderInjectedChunker
# 策略 3: 父子索引
from tests.chunk.chunkers.parent_child import ParentChildChunker
# 策略 4: 知识蒸馏
from tests.chunk.chunkers.knowledge_distilled import KnowledgeDistilledChunker

__all__ = [
    "BaseChunker",
    "Chunk",
    "TurnSlidingWindowChunker",
    "HeaderInjectedChunker",
    "ParentChildChunker",
    "KnowledgeDistilledChunker",
]
