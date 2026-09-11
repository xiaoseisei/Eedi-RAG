"""
================================================================================
模块名称: src/business_models.py
业务定位: 业务感知层的 Eedi-RAG 查询边界严格契约 (Strict contracts)

本文件定义从"业务查询请求"到"业务响应信封"的整条链路契约:
  1. 请求入口契约     : BusinessQueryRequest          —— 上层业务方如何"提一个问题"
  2. 查询理解契约     : QueryUnderstanding            —— 意图/作用域/置信度如何被解读
  3. 能力探测契约     : DataCapabilitySnapshot / Registry —— 底层数据能否支撑该查询
  4. 执行计划契约     : QueryPlan                      —— 决定"怎么查"(聚合/检索/拒绝)
  5. 结果与证据契约   : BusinessResultStub / EvidencePointer —— 回答 + 可核验引用
  6. 响应信封契约     : BusinessResponseEnvelope       —— 对外统一返回结构

核心设计原则:
  - strict=True + extra="forbid": 契约两侧严禁多余字段, 防止前后端静默漂移;
  - str_strip_whitespace=True   : 入参自动去空白, 避免"看着相同但不同"的脏 token;
  - model_validator(mode="after"): 把跨字段的业务不变量在装配边界上直接咬死,
    非法组合一进 Pydantic 模型就抛错, 而不是拖到下游才暴露。
================================================================================
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic import BaseModel


class StrictBusinessModel(BaseModel):
    """业务契约共享的基类配置。

    - extra="forbid"             : 出现未声明字段直接报错（两端契约同时收紧）;
    - strict=True                : 不做隐式类型转换（如 str->bool、int->str）;
    - str_strip_whitespace=True  : 字符串字段自动去掉首尾空白。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# 枚举: 置于模块顶层, 既充当可复用的"话术", 也作为 plan / response 的状态机
# ---------------------------------------------------------------------------


class EvidencePerspective(str, Enum):
    """证据视角: 同一条真实对白可以从谁的角度来引证。"""

    STUDENT = "STUDENT"  # 学生视角证据（原话困惑、错因）
    TUTOR = "TUTOR"      # 导师视角证据（引导追问、策略原话）


class ComputationScope(str, Enum):
    """计算范围: 回答需要聚合统计、逐案例细查, 还是两者兼有。

    MACRO  -> 跨会话/学段的宏观聚合分析（需要 analytics 能力）;
    MICRO  -> 单案例(单会话/单题)的细粒度追溯（不需要聚合）;
    HYBRID -> 先聚合定位热点, 再下钻到具体会话求证;
    REJECT -> 当前数据范围内无法计算, 下游应直接拒绝。
    """

    MACRO = "MACRO"
    MICRO = "MICRO"
    HYBRID = "HYBRID"
    REJECT = "REJECT"


class QueryIntent(str, Enum):
    """业务查询意图分类: 决定后续走聚合报表、案例剧本还是单题辅导。"""

    STUDENT_FREQUENT_QUESTIONS = "STUDENT_FREQUENT_QUESTIONS"  # 学生高频问题清单
    DIFFICULT_CONCEPTS = "DIFFICULT_CONCEPTS"                  # 学生疑难概念
    RECURRING_MISCONCEPTIONS = "RECURRING_MISCONCEPTIONS"      # 反复出现的顽固误区
    EXAM_ERROR_PATTERNS = "EXAM_ERROR_PATTERNS"                # 考试错题模式分析
    STUDENT_COHORT_PATTERNS = "STUDENT_COHORT_PATTERNS"        # 学生群体/分层规律
    TUTOR_STRATEGY_PATTERNS = "TUTOR_STRATEGY_PATTERNS"        # 导师干预策略归纳
    CONTENT_IMPROVEMENT = "CONTENT_IMPROVEMENT"                # 内容/题目改进建议
    QUESTION_TUTORING = "QUESTION_TUTORING"                    # 针对某道题的一对一个性化辅导
    UNSUPPORTED = "UNSUPPORTED"                                # 超出系统能力范围
    UNRESOLVED = "UNRESOLVED"                                  # 待二层模型判定/候选冲突未决


class QueryScope(str, Enum):
    """查询作用域: 回答要覆盖多大的范围。"""

    CORPUS = "CORPUS"        # 整个知识库 / 全量语料
    STUDENT = "STUDENT"      # 学生提问/学习行为视角的业务子域
    SUBJECT = "SUBJECT"      # 某个学科 / 某条考纲路径之下
    CASE = "CASE"            # 某一次具体辅导案例（单会话）
    QUESTION = "QUESTION"    # 某一道具体题目


class RouteConfidence(str, Enum):
    """意图路由置信度: 区分"规则路确定"与"LLM 分类器猜测", 供下游保守降级。"""

    HIGH = "HIGH"        # 规则命中 / 高置信
    MEDIUM = "MEDIUM"    # 边界情况, 需谨慎
    LOW = "LOW"          # 归类存疑, 优先降级而非硬答


class QueryPlanType(str, Enum):
    """执行计划类型: 一个理解后的查询最终被编排成哪一类任务。"""

    ANALYTICS_REPORT = "ANALYTICS_REPORT"                  # 纯聚合报表（无需逐条证据）
    ANALYTICS_WITH_EVIDENCE = "ANALYTICS_WITH_EVIDENCE"    # 聚合分析 + 真实对白证据支撑
    CASE_PLAYBOOK = "CASE_PLAYBOOK"                        # 单案例可复用的辅导剧本
    CONTENT_BRIEF = "CONTENT_BRIEF"                        # 内容改进简报
    SOCRATIC_TUTORING = "SOCRATIC_TUTORING"                # 苏格拉底式单题辅导
    REJECT = "REJECT"                                      # 能力不足, 拒绝执行


class BusinessResponseStatus(str, Enum):
    """业务响应状态: 对成功/失败/降级做显式声明, 而非静默失败。"""

    SUCCESS = "SUCCESS"              # 正常成功
    NO_EVIDENCE = "NO_EVIDENCE"      # 有分析结论但缺真实证据支撑
    UNSUPPORTED = "UNSUPPORTED"      # 查询不被支持, 返回说明
    DEGRADED = "DEGRADED"            # 部分能力缺失, 已降级完成
    FAILED = "FAILED"                # 执行失败
    UNMEASURED = "UNMEASURED"        # 意图未测量/待二层消歧阻断


class DetailLevel(str, Enum):
    """回答详细程度: 由请求方决定, 用于规划生成回答的长度与结构。"""

    BRIEF = "BRIEF"          # 一句话结论
    STANDARD = "STANDARD"    # 标准教研答复
    FULL = "FULL"            # 完整诊断报告


# ---------------------------------------------------------------------------
# 入参契约: 上层业务方唯一需要关心的"问题该怎么提"
# ---------------------------------------------------------------------------


class BusinessQueryRequest(StrictBusinessModel):
    """业务层入参: 把业务问题转成受约束的结构化请求。"""

    query: str = Field(min_length=1, max_length=2000)   # 原始业务问题文本
    subject_filters: list[str] = Field(default_factory=list)  # 学科/考纲路径过滤, 空列表=不限
    top_k: int = Field(default=3, ge=1, le=10)          # 期望返回的证据/卡片条数
    detail_level: DetailLevel = DetailLevel.STANDARD    # 回答详细程度
    debug: bool = False                                 # 是否附带调试溯源信息


class QueryUnderstanding(StrictBusinessModel):
    """查询理解: 意图识别后的结构化解释, 是后续计划编排的唯一输入依据。

    raw_query 必须原样保留, 保证整条链路可以向上溯源到用户到底问的是什么。
    """

    raw_query: str = Field(min_length=1, max_length=2000)  # 原始问题（保留溯源）
    primary_intent: QueryIntent                            # 主意图（唯一）
    secondary_intents: list[QueryIntent] = Field(default_factory=list, max_length=1)  # 次意图, 至多 1 个
    scope: QueryScope                                      # 作用域（语料/学科/案例/题目）
    computation_scope: ComputationScope                    # 宏观聚合 / 微观细查 / 混合
    evidence_perspectives: list[EvidencePerspective] = Field(min_length=1, max_length=2)  # 所需证据视角, 1~2 个
    subject_filters: list[str] = Field(default_factory=list)  # 继承自请求的过滤条件
    protected_tokens: list[str] = Field(default_factory=list) # 查询改写时须保持原样的 token（如题号/会话号）
    requested_top_k: int = Field(default=3, ge=1, le=10)  # 期望返回条数
    requires_aggregation: bool                            # 是否需要跨会话聚合统计
    requires_case_retrieval: bool                         # 是否需要取回真实案例对白
    response_type: str = Field(min_length=1)              # 结果类型标签, 供 result 多态反序列化
    confidence: RouteConfidence                           # 本次理解的置信度
    route_source: Literal["deterministic", "llm_classifier"]  # 路由来源: 规则 或 LLM 分类器
    ambiguity_detected: bool = False                     # 是否存在多意图/非强指向歧义
    matched_intent_signals: list[str] = Field(default_factory=list)
    routing_status: Literal["RESOLVED", "NEED_LLM", "UNSUPPORTED", "UNMEASURED"] = "RESOLVED"
    candidate_intents: list[QueryIntent] = Field(default_factory=list)  # 冲突或备选意图列表
    # Orthogonal semantic hints. These never authorize SQL by themselves.
    ranking_signal: str | None = None
    artifact_target: str | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 能力契约: 查询是否可执行, 必须先探测底层数据能力
# ---------------------------------------------------------------------------


class DataCapabilitySnapshot(StrictBusinessModel):
    """单项数据能力的探测快照: 标记某能力是否可用, 缺什么就明说缺什么。"""

    capability: str = Field(min_length=1)           # 能力名（如 DIALOGUE_EVIDENCE）
    available: bool                                 # 该能力是否可用
    coverage: float = Field(ge=0.0, le=1.0)         # 数据覆盖率 0~1
    missing_requirements: list[str] = Field(default_factory=list)  # 缺失的依赖项清单
    source_artifact: str = Field(min_length=1)      # 能力来源（真实库信息 / synthetic 测试数据）

    @model_validator(mode="after")
    def availability_matches_requirements(self) -> "DataCapabilitySnapshot":
        # 不变量: 能力"可用"与"存在缺失项"不能同时成立, 防止自相矛盾的探测结果。
        # 注: 生产线的真实探测走 information_schema, 这里只把矛盾数据挡在入口之外。
        if self.available and self.missing_requirements:
            raise ValueError("available capability cannot have missing requirements")
        return self


class DataCapabilityRegistry:
    """测试辅助: 为四种核心能力构造合成快照; 生产线的 inspect() 读真实元数据。"""

    @staticmethod
    def synthetic_snapshot(
        *, dialogue: bool, analytics: bool, retake: bool, ability: bool
    ) -> dict[str, DataCapabilitySnapshot]:
        """生成仅用于测试的合成快照, 方便在无真实数据源时跑通计划编排逻辑。"""

        return {
            # 真实师生对白证据: 支撑 MICRO 细查与证据引用能力
            "DIALOGUE_EVIDENCE": DataCapabilitySnapshot(
                capability="DIALOGUE_EVIDENCE",
                available=dialogue,
                coverage=1.0 if dialogue else 0.0,
                missing_requirements=[] if dialogue else ["session_dialogue_turns"],
                source_artifact="synthetic",
            ),
            # 业务聚合分析: 支撑 MACRO 聚合与 analytics_operations 能力
            "BUSINESS_ANALYTICS": DataCapabilitySnapshot(
                capability="BUSINESS_ANALYTICS",
                available=analytics,
                coverage=1.0 if analytics else 0.0,
                missing_requirements=[] if analytics else ["business_analytics_manifest"],
                source_artifact="synthetic",
            ),
            # 复读/重做群体: 识别同一学生的再次作答, 用于分析进步与反复
            "RETAKE_COHORT": DataCapabilitySnapshot(
                capability="RETAKE_COHORT",
                available=retake,
                coverage=1.0 if retake else 0.0,
                missing_requirements=[] if retake else ["student_key", "attempt_number"],
                source_artifact="synthetic",
            ),
            # 能力分层群体: 以历史正确率为依据划分学生能力层, 支撑群体规律分析
            "ABILITY_COHORT": DataCapabilitySnapshot(
                capability="ABILITY_COHORT",
                available=ability,
                coverage=1.0 if ability else 0.0,
                missing_requirements=[] if ability else ["student_key", "historic_correctness"],
                source_artifact="synthetic",
            ),
        }


# ---------------------------------------------------------------------------
# 执行计划契约: "怎么查"
# ---------------------------------------------------------------------------


class QueryPlan(StrictBusinessModel):
    """执行计划: 把一个已理解的查询编排成具体的执行意图与所需能力集合。"""

    understanding: QueryUnderstanding                  # 基于哪份理解产生（冗余保存, 便于审计归来）
    plan_type: QueryPlanType                           # 计划类型（聚合/案例/辅导/拒绝…）
    analytics_operations: list[str] = Field(default_factory=list)  # 需执行的聚合运算清单
    retrieval_perspectives: list[EvidencePerspective] = Field(default_factory=list)  # 需检索的证据视角
    required_capabilities: list[str] = Field(default_factory=list)  # 执行本计划必须可用的能力清单
    response_type: str = Field(min_length=1)           # 结果类型标签, 用于 result 多态分发
    rejection_reason: str | None = None                # 仅 REJECT 计划携带的拒绝原因

    @model_validator(mode="after")
    def validate_plan_semantics(self) -> "QueryPlan":
        # 不变量一: 可执行的聚合查询必须给出具体聚合运算, 不允许"说要聚合却不列运算";
        # REJECT 是能力门禁结果, 它必须清空 execution operations, 因而不应被该规则误拦截。
        if (
            self.plan_type != QueryPlanType.REJECT
            and self.understanding.requires_aggregation
            and not self.analytics_operations
        ):
            raise ValueError("aggregation query requires analytics operations")
        # 不变量二: REJECT 计划必须先想清楚拒绝原因, 不允许空口拒绝;
        if self.plan_type == QueryPlanType.REJECT and not self.rejection_reason:
            raise ValueError("REJECT plan requires rejection_reason")
        # 不变量三: 可执行的计划严禁夹带拒绝原因, 避免跨状态污染。
        if self.plan_type != QueryPlanType.REJECT and self.rejection_reason is not None:
            raise ValueError("executable plan cannot carry rejection_reason")
        return self


# ---------------------------------------------------------------------------
# 数据范围与证据契约: 回答必须"可溯源、可核验"
# ---------------------------------------------------------------------------


class DataScope(StrictBusinessModel):
    """本次回答实际覆盖的数据范围: 声明我看到了什么, 才谈得上"没看到的"。

    这是"全透明"的一部分——消费方据此判断统计分析的口径与局限。
    """

    dataset_version: str = Field(min_length=1)      # 数据集版本号
    index_version: str = Field(min_length=1)        # 向量索引版本号
    total_sessions_considered: int = Field(ge=0)    # 全局纳入统计的会话总数
    matched_sessions: int = Field(ge=0)             # 命中过滤条件的会话数
    subject_filters: list[str] = Field(default_factory=list)   # 生效的学科过滤
    other_filters: dict[str, str] = Field(default_factory=dict)  # 其它过滤条件的键值
    unmeasured_metrics: list[str] = Field(default_factory=list)  # 本应计算但因能力缺失而未算的指标


class EvidencePointer(StrictBusinessModel):
    """证据指针: 指向一条可精确核验的真实对白引文, 是防伪审计的载体。"""

    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")  # 证据编号, 强制 E + 至少 3 位数字
    source_event_id: str = Field(min_length=1)        # 来源事件 ID（底层事件表的唯一标识）
    source_card_id: str | None = None                 # 来源知识卡片 ID（无卡片来源可为空）
    session_id: int = Field(ge=1)                     # 所属会话
    turn_id: int = Field(ge=1)                        # 会话内轮次
    speaker: Literal["student", "tutor"]              # 说话人角色
    exact_quote: str = Field(min_length=1)            # 逐字引用的原文（禁止改写）
    subject_path: str = Field(min_length=1)           # 学科考纲路径
    verified: bool                                    # 是否已通过逐字段核验


class BusinessResultStub(StrictBusinessModel):
    """结果外壳: 用统一的外层结构承载业务结果负载。

    payload 保持 dict 不透明——具体业务结果由 response_type 决定如何解释,
    避免一个类膨胀出所有业务的字段。
    """

    response_type: str = Field(min_length=1)              # 结果类型标签（多态判别字段）
    payload: dict[str, Any] = Field(default_factory=dict) # 业务结果的实际内容


# response_type 同时充当"判别字段", 下游可据此对多态结果做安全反序列化
ResponseResult = Annotated[BusinessResultStub, Field(discriminator="response_type")]


class BusinessResponseEnvelope(StrictBusinessModel):
    """对外统一响应信封: 自解释的契约, 消费方无需猜测字段含义。"""

    schema_version: Literal["business-response/v1"] = "business-response/v1"  # 响应契约版本
    query: str = Field(min_length=1)                  # 原始问题（回显溯源）
    understanding: QueryUnderstanding                 # 系统对该问题的理解
    status: BusinessResponseStatus                    # 本次响应状态
    summary: str = Field(min_length=1)                # 面向用户的摘要
    data_scope: DataScope                             # 本次回答覆盖的数据范围
    result: BusinessResultStub | None = None          # 业务结果（按 status 可能为空）
    evidence: list[EvidencePointer] = Field(default_factory=list)  # 可核验的真实证据列表
    limitations: list[str] = Field(default_factory=list)  # 能力/数据局限的显式声明
    audit_status: Literal[                            # 证据审计状态
        "PENDING",                # 未审计, 仅供内部过渡使用
        "AUDITED_100_VERIFIED",   # 100% 逐字段核验通过
        "GROUNDING_DEGRADED",     # 存在引用无法精确落地, 已降级
        "NOT_APPLICABLE",         # 无历史证据场景, 审计不适用
    ]
    trace_id: str = Field(min_length=1)               # 链路追踪 ID（日志/排障用）

    @model_validator(mode="after")
    def status_matches_payload(self) -> "BusinessResponseEnvelope":
        # 不变量一: SUCCESS 必须携带结果; 且一旦带历史证据, 必须先过 100% 审计——
        #     防止"声称成功却引用了无法核验的对白"这种最严重的信任破坏;
        if self.status == BusinessResponseStatus.SUCCESS:
            if self.result is None:
                raise ValueError("SUCCESS requires a result")
            if self.evidence and self.audit_status != "AUDITED_100_VERIFIED":
                raise ValueError("SUCCESS with historical evidence requires verified audit")
        # 不变量二: 非 SUCCESS 响应严禁夹带业务结果, 避免降级/失败时返回误导性数据;
        elif self.result is not None:
            raise ValueError("non-SUCCESS responses cannot carry a business result")
        # 不变量三: 明确不支持或未测量的查询必须用 NOT_APPLICABLE 审计态, 保持语义自洽。
        if self.status in {BusinessResponseStatus.UNSUPPORTED, BusinessResponseStatus.UNMEASURED} and self.audit_status != "NOT_APPLICABLE":
            raise ValueError(f"{self.status.value} must use NOT_APPLICABLE audit")
        return self


# 公共导出面: 业务层只与这些符号打交道, 其余视为内部实现
__all__ = [
    "BusinessQueryRequest",
    "BusinessResponseEnvelope",
    "BusinessResponseStatus",
    "BusinessResultStub",
    "ComputationScope",
    "DataCapabilityRegistry",
    "DataCapabilitySnapshot",
    "DataScope",
    "DetailLevel",
    "EvidencePerspective",
    "EvidencePointer",
    "QueryIntent",
    "QueryPlan",
    "QueryPlanType",
    "QueryScope",
    "QueryUnderstanding",
    "ResponseResult",
    "RouteConfidence",
    "StrictBusinessModel",
]
