"""
================================================================================
模块名称: src/models.py
业务定位: Eedi-RAG 数据清洗、知识蒸馏与切块阶段核心数据结构契约定义 (Data Contract)
核心职责:
  1. 定义强类型 Pydantic 数据模型，确保从非结构化 CSV 到清洗后会话、知识卡片与切块各环节具备严格模式约束。
  2. 规范化学科树层级 (SubjectHierarchy)、原题与选项 (Question)、合并对话轮次 (DialogueTurn) 与完整辅导会话 (CleanedSession)。
  3. 规范化学生认知误区卡片 (StudentMisconceptionProfile) 与导师启发式策略卡片 (TutorStrategyProfile)，构成标准结构化教学资产 (ExtractedPIU)。
  4. 定义统一切块数据结构 (Chunk)，支撑主线知识卡片与滑动窗口兜底的双基座 RAG 切分流水线。
================================================================================
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SubjectHierarchy(BaseModel):
    """
    学科与考点知识树层级对象。
    
    业务背景:
      Eedi 数据集将学科划分为三级结构:
        Level 1 (Subject): 一级学科 (例如: 'Number', 'Geometry and Measure')
        Level 2 (Topic): 二级主题/模块 (例如: 'Rounding and Estimating', 'Basic Arithmetic')
        Level 3 (Subtopic): 三级具体考点/微知识点 (例如: 'Rounding to Decimal Places')
    
    本模型将分散的节点关系拼装为直观的层级路径列表，便于后续元数据过滤与知识检索。
    """
    paths: List[str] = Field(
        default_factory=list,
        description="格式化后的全路径字符串列表，例如: ['Number > Rounding and Estimating > Rounding to Decimal Places']"
    )
    subjects: List[str] = Field(
        default_factory=list,
        description="一级学科名称集合 (Level 1 Subject)"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="二级主题名称集合 (Level 2 Topic)"
    )
    subtopics: List[str] = Field(
        default_factory=list,
        description="三级考点名称集合 (Level 3 Subtopic)"
    )


class Question(BaseModel):
    """
    题干与选项透视数据对象。
    
    业务背景:
      在原始数据 dq-question-metadata.csv 中，题目题干可能分多行(Sequence)存储，
      且选项 A/B/C/D 与图片是以行标签(Label)形式离散存在的。
      本模型将其透视为一个结构完整的题目实体，完整保留 LaTeX 公式与选项映射。
    """
    question_id: int = Field(
        description="题目全局唯一标识符 (QuestionId_DQ)"
    )
    question_text: str = Field(
        description="按 Sequence 时序拼接后的完整题干文本 (包含 LaTeX 数学公式，例如 \\( \\mathbf{5 . 4 5 9 8} \\))"
    )
    options: Dict[str, str] = Field(
        default_factory=dict,
        description="选项字典映射，键为选项标识 ('A', 'B', 'C', 'D')，值为选项文本"
    )
    images: Dict[str, str] = Field(
        default_factory=dict,
        description="题目或选项附带的图片 URL 或占位符 (例如: {'Question': 'url...', 'A': 'url...'})"
    )


class DialogueTurn(BaseModel):
    """
    合并与清洗后的单轮会话实体 (Dialogue Turn)。
    
    业务背景:
      学生或导师在真实在线打字聊天中，常常一句话分多次按回车发送 (如连续发送 3~4 条碎句)。
      通过状态机压缩 (Turn Compression) 后，同一角色连续发送的碎句被合并为一个逻辑 Turn。
      同时在此层完成 PII 姓名脱敏、保留 TalkMove 教学动作标签、标记开场/客套寒暄。
    """
    turn_id: int = Field(
        description="基于时间的 1-based 递增轮次序号"
    )
    speaker: str = Field(
        description="说话人角色标识: 'tutor' (导师) 或 'student' (学生)"
    )
    is_tutor: bool = Field(
        description="布尔类型角色标识，True 为导师发言，False 为学生发言"
    )
    raw_messages: List[str] = Field(
        description="状态机压缩前原始分散的消息列表 (保留 100% 原始字面以供溯源比对)"
    )
    text: str = Field(
        description="清洗、脱敏、合并后的最终呈现文本"
    )
    talk_moves: List[str] = Field(
        default_factory=list,
        description="本轮次中包含的官方教学动作预测标签 (例如: ['<Press for Accuracy>', '<Keep Together>'])"
    )
    is_greeting_or_noise: bool = Field(
        default=False,
        description="寒暄与噪音标记: 若为 True 表示仅包含问好/再见/纯语气词，非核心教学问答"
    )


class CleanedSession(BaseModel):
    """
    清洗融合后的完整教学交互单元 (PIU, Pedagogical Interaction Unit)。
    
    业务背景:
      将单次 1v1 辅导的所有上下文（题目事实 + 考纲树 + 时序对话流）打包为自包含的宽对象。
      这是整个 RAG 系统的核心输入基石，下游 LLM 抽取与数据库存储均以该对象为原子输入。
    """
    intervention_id: int = Field(
        description="辅导会话全局唯一业务 ID (InterventionId)"
    )
    question_id: int = Field(
        description="辅导所绑定的考题 ID (QuestionId_DQ)"
    )
    tutor_id: int = Field(
        description="授课导师 ID (TutorId)"
    )
    subjects: SubjectHierarchy = Field(
        description="该会话考题所属的学科考纲层级知识树"
    )
    question: Question = Field(
        description="该会话考题的完整题干与选项信息"
    )
    turns: List[DialogueTurn] = Field(
        description="按时序排列的完整清洗后对话轮次列表"
    )
    total_turns: int = Field(
        description="会话包含的有效对话轮次总数"
    )
    student_turn_count: int = Field(
        description="学生发言轮次数"
    )
    tutor_turn_count: int = Field(
        description="导师发言轮次数"
    )
    has_valid_tutoring: bool = Field(
        default=True,
        description="质量门禁标志: 是否构成实质性双向多轮辅导 (例如师生双方均至少发言 1 次且总轮次 >= 2)"
    )


# ================================================================================
# Step 2: 结构化教学知识卡片与切块模型契约 (Knowledge Distillation & Chunking Contracts)
# ================================================================================

class StudentMisconceptionProfile(BaseModel):
    """
    学生认知误区卡片 (Student Misconception Profile Card)。
    
    业务背景:
      从整场辅导对话中提纯学生暴露出的深层思维认知障碍、误选选项、困惑触发概念，
      并提取 100% 原始发言字面量子串作为绝对溯源证据 (Grounding Citation)。
    """
    session_id: int = Field(
        description="关联辅导会话唯一业务 ID (InterventionId)"
    )
    question_id: int = Field(
        description="关联考题 ID (QuestionId_DQ)"
    )
    subject_path: str = Field(
        default="",
        description="学科考纲完整路径"
    )
    misconception_name: str = Field(
        description="学术化标准错因命名 (例如: '四舍五入数位保留规则混淆' 或 '小数位数与数值大小误判')"
    )
    error_choice: Optional[str] = Field(
        default=None,
        description="学生初试或对话中误选的选项字母 (例如: 'B', 'D' 或 'Sophie')"
    )
    deep_mechanism: str = Field(
        description="深层思维认知障碍机理剖析 (解释学生为什么会产生这个误区，其思维模型错在哪里)"
    )
    confusion_triggers: List[str] = Field(
        default_factory=list,
        description="触发学生困惑的核心术语、概念词或关键数值 (例如: ['5.45', '1 decimal place'])"
    )
    verbatim_student_quotes: List[str] = Field(
        default_factory=list,
        description="学生暴露出该错误时的原始发言直接引用 (Exact Quotes，必须为真实发言子串)"
    )
    source_turn_ids: List[int] = Field(
        default_factory=list,
        description="对应的学生发言轮次序号列表 (1-based Turn IDs)"
    )


class TutorStrategyProfile(BaseModel):
    """
    名师启发式策略卡片 (Tutor Pedagogical Strategy Card)。
    
    业务背景:
      提炼授课导师在辅导过程中展现出的核心破局灵魂提问 (Aha Moment Prompt)、
      分步引导脚手架 (Scaffolding)、生活化比喻、教学动作与引导效果。
    """
    session_id: int = Field(
        description="关联辅导会话唯一业务 ID (InterventionId)"
    )
    question_id: int = Field(
        description="关联考题 ID (QuestionId_DQ)"
    )
    pedagogical_goal: str = Field(
        description="阶段性教学引导目标 (例如: '引导学生理解 1 位小数的精确定义并完成进位')"
    )
    strategy_category: str = Field(
        default="Scaffolding",
        description="教学法标准分类 (如: 'Socratic_Questioning', 'Scaffolding', 'Counter_Example', 'Analogy', 'Revoicing')"
    )
    key_aha_question: str = Field(
        description="名师破局核心提问 (Aha Moment Prompt / 灵魂一问，促成学生顿悟的关键句)"
    )
    scaffolding_steps: List[str] = Field(
        default_factory=list,
        description="分步搭建的脚手架引导步骤链 (例如: ['Step 1: 圈出目标小数位', 'Step 2: 观察后一位是舍还是入'])"
    )
    analogy_or_metaphor: Optional[str] = Field(
        default=None,
        description="导师采用的通俗比喻、现实情境或金钱案例 (若有)"
    )
    talk_moves: List[str] = Field(
        default_factory=list,
        description="涉及的官方教学动作预测标签 (例如: ['<Press for Accuracy>', '<Keep Together>'])"
    )
    resolution_outcome: str = Field(
        default="",
        description="最终辅导结果与转化状态 (例如: '学生自主推导出正确选项并理解四舍五入规则')"
    )
    source_turn_ids: List[int] = Field(
        default_factory=list,
        description="涉及的导师关键引导轮次序号列表 (1-based Turn IDs)"
    )


class ExtractedPIU(BaseModel):
    """
    单场辅导会话结构化蒸馏聚合对象 (Extracted Pedagogical Interaction Unit)。
    
    业务背景:
      将单场 CleanedSession 经 LLM 抽取后的两大物理隔离卡片与基础元数据统一聚合，
      作为 Step 2 的标准输出物与 Step 3 写入双引擎存储的直接输入源。
    """
    session_id: int = Field(
        description="会话全局唯一 ID (InterventionId)"
    )
    question_id: int = Field(
        description="考题 ID (QuestionId_DQ)"
    )
    misconception: Optional[StudentMisconceptionProfile] = Field(
        default=None,
        description="学生认知误区卡片"
    )
    tutor_strategy: Optional[TutorStrategyProfile] = Field(
        default=None,
        description="名师启发式策略卡片"
    )
    extraction_status: str = Field(
        default="success",
        description="抽取状态: 'success' (成功), 'partial' (部分成功), 'fallback' (规则降级抽取)"
    )


class Chunk(BaseModel):
    """
    统一 RAG 切块数据契约实体 (Unified Chunk Contract)。
    
    业务背景:
      无论是主线知识卡片 (Knowledge Distilled Cards) 还是兜底的滑动窗口切块 (Sliding Window Chunks)，
      均统一封装为此 Chunk 模型，携带标准化的向量索引文本、溯源元数据与 Token 统计。
    """
    chunk_id: str = Field(
        description="全局唯一切块 ID (例如: 'session_104614_misconception' 或 'session_104614_win_1_6')"
    )
    session_id: int = Field(
        description="所属辅导会话全局唯一 ID (InterventionId)"
    )
    question_id: int = Field(
        description="关联考题 ID (QuestionId_DQ)"
    )
    chunk_type: str = Field(
        description="切块类型标识: 'misconception' (错因卡), 'tutor_strategy' (策略卡), 'sliding_window' (滑动窗口切块)"
    )
    content: str = Field(
        description="供向量化 (Embedding) 与 LLM 检索召回消费的高密度文本内容"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="富元数据字典 (包含学科路径、题干、关键标签、策略类别、父会话外键等)"
    )
    token_count: int = Field(
        default=0,
        description="切块估计 Token 数 (按英文字词与中文字符综合估算)"
    )
    source_turn_ids: List[int] = Field(
        default_factory=list,
        description="该切块所覆盖的原始轮次序号列表 (用于 100% 精确行号高亮与溯源)"
    )


# ================================================================================
# Step 4: 在线多视角查询改写契约 (Multi-Perspective Query Rewriting Contracts)
# ================================================================================

class MultiPerspectiveQueries(BaseModel):
    """
    多视角改写查询集契约 (Multi-Perspective Rewritten Queries Contract)。
    
    业务背景:
      接收用户的口语短提问，通过注入领域专有名词与学科考纲，
      同时生成 3 个正交视角的专业检索表达，配合 RRF 融合彻底攻克词表鸿沟与语义弥散。
    """
    raw_query: str = Field(
        description="用户原始输入提问 (Raw User Query)"
    )
    misconception_query: str = Field(
        description="视角 1 (学情错因诊断): 聚焦学生深层认知障碍、易混淆规则与典型错选机理"
    )
    strategy_query: str = Field(
        description="视角 2 (名师教法启发): 聚焦名师破局一问、引导脚手架步骤链与苏格拉底反问"
    )
    curriculum_query: str = Field(
        description="视角 3 (学科考纲题型): 聚焦考纲完整层级路径、中英专有名词对照与核心数值约束"
    )
    extracted_keywords: List[str] = Field(
        default_factory=list,
        description="提取出的核心数值、考点术语、题号与中英对照词 (例如: ['LCM', '最小公倍数', '5.4598'])"
    )
    injected_domain_context: str = Field(
        default="",
        description="前置动态注入的学科知识树与中英术语映射上下文"
    )

