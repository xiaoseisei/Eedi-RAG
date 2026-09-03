# Eedi-RAG Query 意图路由与业务响应可执行设计规范

> 文档编号：`EEDI-RAG-QUERY-RESPONSE-001`
> 版本：`v1.0-design`
> 日期：2026-09-02（Asia/Shanghai）
> 状态：已确认设计方向，待按工作包施工
> 适用范围：当前 100 Session 开发快照，以及后续约 1,000 场有效课堂目标基线

## 1. 文档目的

本规范把已经确认的产品方向转化为可编码、可测试、可评测的工程契约。

核心决定是：

> 系统不根据“用户是谁”决定答案，而是根据用户当前问题的业务意图、分析范围和期望动作，自动选择统计分析、案例检索、内容建议或单步辅导路径，并返回与任务相匹配的结构化结果。

本规范解决以下现有问题：

1. 当前所有 Query 共用一种检索链和一种通用回答，无法覆盖真实业务问题。
2. 把 System Prompt 压缩成“一句话、80 字以内”虽然降低了耗时，却同时破坏了统计报告、教学策略和内容建议所需的结构。
3. “学生最常问什么”“哪些概念最难”等跨会话问题不能由 Top-1/Top-3 向量检索直接回答。
4. 当前模型调用未显式关闭 Mimo Thinking、未在 API 层限制输出、未测量真实 TTFT 和可见文本吞吐。
5. 当前回答引用仍需落实完整四元组与候选授权门禁。

## 2. 产品定位与边界

### 2.1 产品定位

Eedi-RAG 是基于真实辅导记录的教学智能与知识行动系统。它的核心价值不是“与记录聊天”，而是把课堂数据转化为：

- 可验证的跨课堂教学洞察；
- 可直接使用的导师教学策略；
- 可进入内容 Backlog 的课程改进任务；
- 未来可用于学生端的单步苏格拉底式辅导。

RAG 是底层证据引擎，不是最终产品形态。

### 2.2 不做身份与权限路由

当前版本不实现：

- 登录用户角色识别；
- 教研主管、新导师、课程研发、学生的权限分级；
- 根据身份强制限制问题类型；
- 多租户、RBAC 或 ACL。

原因：同一个人可能提出不同类型的问题。系统应回答问题本身，而不是推断提问者身份。

用户画像只用于发现业务场景和设计回复，不进入当前运行时路由条件。

### 2.3 当前数据边界

- 当前数据是 Eedi 数学辅导公开代理数据，不是 NBCOT 客户私有数据。
- 当前本地 DuckDB/Chroma 生产快照为 100 场会话，不是目标约 1,000 场规模。
- 跨会话回答必须展示本次实际查询的数据范围和过滤条件。
- 不得把 Eedi 数学结论写成 NBCOT 业务结论。
- 不得用当前观察数据声称某种导师策略具有因果效果。

## 3. 核心术语

| 术语 | 含义 |
|---|---|
| Query Intent | 用户问题希望完成的业务任务 |
| Query Scope | 问题覆盖全库、某考点、某场景还是具体题目 |
| Query Plan | 为回答问题选择的实际计算路径 |
| Analytics | DuckDB 上的确定性聚合、过滤、计数和排序 |
| Case Retrieval | 从错因卡、策略卡或原文窗口检索代表案例 |
| Evidence | 可回查至 DuckDB 权威 Turn 的真实原文 |
| Response Contract | 某一业务意图对应的强类型响应 Schema |
| Authorized Evidence | 来自本次检索候选及其声明 `source_turn_ids` 的证据 |
| Socratic Turn | 学生端一次只推进一步的教学响应 |

## 4. 总体架构

```text
Raw Query
   │
   ▼
Query Understanding
   ├─ intent：问什么
   ├─ scope：范围多大
   ├─ filters：考点/题型/条件
   └─ requested action：统计/解释/策略/建议/辅导
   │
   ▼
Query Planner
   ├─ ANALYTICS_REPORT
   ├─ ANALYTICS_WITH_EVIDENCE
   ├─ CASE_PLAYBOOK
   ├─ CONTENT_BRIEF
   └─ SOCRATIC_TUTORING
   │
   ▼
Execution
   ├─ DuckDB SQL aggregation
   ├─ Dense/Sparse retrieval + RRF
   ├─ Reranker + Context Assembler
   └─ Stateful tutoring controller
   │
   ▼
Response Builder
   ├─ deterministic facts and citations
   ├─ optional concise LLM synthesis
   └─ intent-specific response schema
   │
   ▼
Grounding Audit + Renderer
```

## 5. Query 意图契约

### 5.1 一级意图枚举

```python
from enum import Enum


class QueryIntent(str, Enum):
    STUDENT_FREQUENT_QUESTIONS = "STUDENT_FREQUENT_QUESTIONS"
    DIFFICULT_CONCEPTS = "DIFFICULT_CONCEPTS"
    RECURRING_MISCONCEPTIONS = "RECURRING_MISCONCEPTIONS"
    EXAM_ERROR_PATTERNS = "EXAM_ERROR_PATTERNS"
    TUTOR_STRATEGY_PATTERNS = "TUTOR_STRATEGY_PATTERNS"
    CONTENT_IMPROVEMENT = "CONTENT_IMPROVEMENT"
    QUESTION_TUTORING = "QUESTION_TUTORING"
    UNSUPPORTED = "UNSUPPORTED"
```

意图定义：

| Intent | 用户要解决的问题 | 典型 Query |
|---|---|---|
| `STUDENT_FREQUENT_QUESTIONS` | 学生最常提出什么问题 | 学生最常问什么？ |
| `DIFFICULT_CONCEPTS` | 哪些考点、题型最难 | 哪些概念卡壳时间最长？ |
| `RECURRING_MISCONCEPTIONS` | 哪些深层误区反复出现 | 分数题有哪些共性认知错误？ |
| `EXAM_ERROR_PATTERNS` | 有哪些稳定的考试错误模式 | 学生做选择题时常犯什么错误？ |
| `TUTOR_STRATEGY_PATTERNS` | 导师反复采用哪些方法 | 金牌导师常传授哪些思考策略？ |
| `CONTENT_IMPROVEMENT` | 教材、微课、题库如何改进 | 根据辅导记录应该补什么微课？ |
| `QUESTION_TUTORING` | 当前题目如何理解和推进 | 为什么 5/7-1/4 不能直接减分母？ |
| `UNSUPPORTED` | 当前数据不能可靠回答 | 谁是最优秀的导师？ |

### 5.2 Query 范围枚举

```python
class QueryScope(str, Enum):
    CORPUS = "CORPUS"
    SUBJECT = "SUBJECT"
    CASE = "CASE"
    QUESTION = "QUESTION"
```

| Scope | 判定说明 | 示例 |
|---|---|---|
| `CORPUS` | 跨整个当前索引快照 | 导师最常采用哪些策略？ |
| `SUBJECT` | 限定学科、Topic 或 Subtopic | 分数运算中哪些误区最常见？ |
| `CASE` | 限定教学困境或错误表现 | 学生在两个选项中纠结时如何引导？ |
| `QUESTION` | 当前具体题目、公式、选项或学生尝试 | 3.153 取最近 0.02 怎么理解？ |

### 5.3 查询理解模型

目标 Schema：

```python
class RouteConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QueryUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    raw_query: str = Field(min_length=1, max_length=2000)
    primary_intent: QueryIntent
    secondary_intents: list[QueryIntent] = Field(default_factory=list, max_length=1)
    scope: QueryScope
    subject_filters: list[str] = Field(default_factory=list)
    protected_tokens: list[str] = Field(default_factory=list)
    requested_top_k: int = Field(default=3, ge=1, le=10)
    requires_aggregation: bool
    requires_case_retrieval: bool
    response_type: str = Field(min_length=1)
    confidence: RouteConfidence
    route_source: Literal["deterministic", "llm_classifier"]
```

约束：

- `secondary_intents` 最多一个，避免一个 Query 触发无限扩展。
- 不从 Query 推断用户身份。
- 数字、公式、选项、题目 ID 必须放入 `protected_tokens`。
- `confidence` 是离散状态，不制造没有校准依据的概率分数。
- `UNSUPPORTED` 必须给出明确原因，不允许自动路由到通用生成。

### 5.4 统一请求契约

当前不传用户角色。调用方只提供 Query、可选过滤条件、展示详细度和学生端可选会话状态。

目标模型文件必须启用 `from __future__ import annotations`，因为 `BusinessQueryRequest`、`BusinessResponseEnvelope`、`TutorSessionState` 和 `ResponseResult` 存在前向引用；全部类型定义完成后调用 Pydantic `model_rebuild()`。

```python
class DetailLevel(str, Enum):
    BRIEF = "BRIEF"
    STANDARD = "STANDARD"
    FULL = "FULL"


class BusinessQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=2000)
    subject_filters: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=10)
    detail_level: DetailLevel = DetailLevel.STANDARD
    tutoring_state: TutorSessionState | None = None
    debug: bool = False
```

目标 Python 接口：

```python
def ask_business(request: BusinessQueryRequest) -> BusinessResponseEnvelope:
    ...
```

如果后续增加 HTTP API，目标接口为 `POST /v1/query`，请求和响应必须复用同一 Pydantic 契约，不能在 CLI、Web 和 API 中维护三套不同 Schema。

## 6. 意图识别与路由算法

### 6.1 两级路由

```text
Level 1：本地确定性规则
  ├─ 高置信度 → 直接产生 QueryUnderstanding
  └─ 低置信度 → Level 2

Level 2：轻量 LLM 分类
  ├─ thinking disabled
  ├─ max_tokens <= 96
  ├─ 严格 JSON Schema
  └─ 仍低置信度/失败 → UNSUPPORTED 或安全默认路径
```

默认不为每个 Query 调用分类模型，以免额外增加 TTFT 和成本。

### 6.2 文本预处理

路由前执行：

1. Unicode NFKC 标准化；
2. 保留原始 Query；
3. 提取数字、小数、分数、LaTeX、括号表达式、选项字母；
4. 英文统一小写用于匹配，但不得覆盖原文；
5. 将匹配到的学科术语映射为标准 `subject_filters`。

### 6.3 初版确定性规则

以下是路由信号，不是业务结论：

| 意图 | 强信号 |
|---|---|
| 高频问题 | 最常问、高频问题、经常问、反复提问、FAQ |
| 困难概念 | 最难、卡壳、耗时最长、难理解、困难主题、最费时间 |
| 共性误区 | 误区、认知错误、错误理解、错误概念、为什么总是 |
| 考试错误 | 考试错误、答题错误、误选、审题、粗心、错误模式 |
| 导师策略 | 导师、老师、如何教、怎么引导、策略、话术、脚手架、反例 |
| 内容改进 | 教材、微课、题库、解析、内容、课程、改进、开发什么 |
| 当前题目 | 这题、怎么算、为什么不能、我选、答案选项、具体公式或题干 |

Scope 信号：

| Scope | 信号 |
|---|---|
| `CORPUS` | 最常、反复、总体、全部、Top、有哪些模式 |
| `SUBJECT` | 明确考点路径或学科术语，且要求跨案例归纳 |
| `CASE` | “当学生……时”“遇到……怎么办”等情境表达 |
| `QUESTION` | 具体公式、数值、题干、选项或学生当前尝试 |

### 6.4 规则评分

初版实现使用可解释分数：

- 完整强短语命中：`+3`；
- 单个领域关键词命中：`+1`；
- 与 Scope 一致的语法模式：`+2`；
- 与意图冲突的强信号：`-2`。

判定：

```text
最高分 >= 4 且领先第二名 >= 2 → HIGH，直接路由
最高分 >= 3 且领先第二名 >= 1 → MEDIUM，直接路由并记录 Trace
其他 → LOW，调用 LLM classifier
```

这些阈值是初始实现参数，必须通过人工标注 Router 数据集校准，不能长期硬编码后不评测。

### 6.5 复合意图

仅支持一个主意图和最多一个次意图。

示例：

```text
Query：分数运算中学生最常卡在哪里，导师通常怎么引导？
primary_intent：DIFFICULT_CONCEPTS
secondary_intents：[TUTOR_STRATEGY_PATTERNS]
scope：SUBJECT
response_type：DIFFICULTY_WITH_STRATEGY_REPORT
```

执行时先完成主意图事实计算，再补充次意图证据；禁止让两个独立长回答简单拼接。

## 7. Query Plan 契约

```python
class QueryPlanType(str, Enum):
    ANALYTICS_REPORT = "ANALYTICS_REPORT"
    ANALYTICS_WITH_EVIDENCE = "ANALYTICS_WITH_EVIDENCE"
    CASE_PLAYBOOK = "CASE_PLAYBOOK"
    CONTENT_BRIEF = "CONTENT_BRIEF"
    SOCRATIC_TUTORING = "SOCRATIC_TUTORING"
    REJECT = "REJECT"
```

路由矩阵：

| Intent | 默认 Plan | SQL | 向量检索 | LLM 综合 |
|---|---|---:|---:|---:|
| 高频问题 | `ANALYTICS_WITH_EVIDENCE` | 是 | 只取代表案例 | 可选短总结 |
| 困难概念 | `ANALYTICS_WITH_EVIDENCE` | 是 | 只取代表案例 | 是 |
| 共性误区 | `ANALYTICS_WITH_EVIDENCE` | 是 | 是 | 是 |
| 考试错误 | `ANALYTICS_WITH_EVIDENCE` | 是 | 是 | 是 |
| 导师策略 | `ANALYTICS_WITH_EVIDENCE` 或 `CASE_PLAYBOOK` | Corpus/Subject 时是 | 是 | 是 |
| 内容改进 | `CONTENT_BRIEF` | 是 | 是 | 是 |
| 当前题目 | `SOCRATIC_TUTORING` | 否 | 是 | 是，短输出 |
| 不支持 | `REJECT` | 否 | 否 | 否 |

关键原则：

- “多少、最常、Top-N、反复出现”必须使用 SQL 或预计算聚合。
- LLM 不得自行估算数量、比例、排名和数据范围。
- 向量检索负责找相关案例，不负责全库计数。
- 统计结果必须先生成，代表证据必须从统计结果对应的 Session 范围内抽取。

## 8. 共用响应外壳

```python
class BusinessResponseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_EVIDENCE = "NO_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class DataScope(BaseModel):
    dataset_version: str
    index_version: str
    total_sessions_considered: int = Field(ge=0)
    matched_sessions: int = Field(ge=0)
    subject_filters: list[str] = Field(default_factory=list)
    other_filters: dict[str, str] = Field(default_factory=dict)


class EvidencePointer(BaseModel):
    evidence_id: str = Field(min_length=1)
    source_card_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    turn_id: int = Field(ge=1)
    speaker: Literal["student", "tutor"]
    exact_quote: str = Field(min_length=1)
    subject_path: str = Field(min_length=1)
    verified: bool


class BusinessResponseEnvelope(BaseModel):
    schema_version: Literal["business-response/v1"] = "business-response/v1"
    query: str = Field(min_length=1)
    understanding: QueryUnderstanding
    status: BusinessResponseStatus
    summary: str = Field(min_length=1)
    data_scope: DataScope
    result: ResponseResult | None = None
    evidence: list[EvidencePointer]
    limitations: list[str] = Field(default_factory=list)
    audit_status: Literal[
        "PENDING",
        "AUDITED_100_VERIFIED",
        "GROUNDING_DEGRADED",
        "NOT_APPLICABLE",
    ]
    trace_id: str = Field(min_length=1)
```

约束：

- `status=SUCCESS` 且回答使用历史事实时，`audit_status` 必须为 `AUDITED_100_VERIFIED`。
- `status=SUCCESS` 时 `result` 不得为空；其他状态可为空，但必须在 `summary/limitations` 中说明原因。
- `NO_EVIDENCE` 不得携带编造的结果项。
- `UNSUPPORTED` 不调用生成模型补答案。
- 学生端响应不向用户展示内部历史原声，但内部仍可保留经过授权的 evidence Trace。
- 所有跨会话报告必须提供 `DataScope`。

## 9. 七种业务响应契约

### 9.1 Student FAQ Report

```python
class FrequentQuestionItem(BaseModel):
    rank: int = Field(ge=1)
    cluster_id: str
    canonical_question: str
    question_function: Literal[
        "REQUEST_EXPLANATION",
        "REQUEST_CONFIRMATION",
        "ASK_NEXT_STEP",
        "COMPARE_OPTIONS",
        "RECALL_RULE",
        "ASK_DEFINITION",
        "EXPRESS_UNCERTAINTY",
        "OTHER",
    ]
    event_count: int = Field(ge=1)
    distinct_session_count: int = Field(ge=1)
    session_share: float = Field(ge=0.0, le=1.0)
    related_subjects: list[str]
    representative_evidence_ids: list[str] = Field(min_length=1, max_length=2)


class StudentFAQReport(BaseModel):
    response_type: Literal["STUDENT_FAQ_REPORT"] = "STUDENT_FAQ_REPORT"
    items: list[FrequentQuestionItem] = Field(min_length=1, max_length=10)
    cross_pattern_summary: str
```

默认展示 Top-3。`event_count` 与 `distinct_session_count` 同时展示，避免同一场长对话重复发问放大频次。

### 9.2 Difficulty Report

```python
class DifficultySignals(BaseModel):
    distinct_session_count: int = Field(ge=1)
    average_tutoring_turns: float = Field(ge=0.0)
    repeated_incorrect_attempt_sessions: int = Field(ge=0)
    repeated_help_sessions: int = Field(ge=0)
    strategy_switch_sessions: int | None = Field(default=None, ge=0)
    unresolved_sessions: int | None = Field(default=None, ge=0)


class DifficultConceptItem(BaseModel):
    rank: int = Field(ge=1)
    subject_path: str
    signals: DifficultySignals
    observed_bottlenecks: list[str] = Field(min_length=1, max_length=3)
    representative_evidence_ids: list[str] = Field(min_length=1, max_length=3)


class DifficultyReport(BaseModel):
    response_type: Literal["DIFFICULTY_REPORT"] = "DIFFICULTY_REPORT"
    items: list[DifficultConceptItem] = Field(min_length=1, max_length=10)
    ranking_method: str
```

初版不强制合成单一“难度分”。如果业务需要排序，`ranking_method` 必须透明记录各信号和权重，并经过人工校准。

### 9.3 Misconception Report

```python
class MisconceptionItem(BaseModel):
    rank: int = Field(ge=1)
    cluster_id: str
    canonical_name: str
    error_category: str
    distinct_session_count: int = Field(ge=1)
    typical_manifestations: list[str] = Field(min_length=1, max_length=3)
    deep_mechanism_summary: str
    representative_evidence_ids: list[str] = Field(min_length=1, max_length=3)


class MisconceptionReport(BaseModel):
    response_type: Literal["MISCONCEPTION_REPORT"] = "MISCONCEPTION_REPORT"
    items: list[MisconceptionItem] = Field(min_length=1, max_length=10)
```

统计使用规范化 `cluster_id`，原始 LLM `misconception_name` 只作为来源字段，不直接 `GROUP BY` 后冒充稳定分类。

### 9.4 Exam Error Pattern Report

```python
class ExamErrorPatternItem(BaseModel):
    rank: int = Field(ge=1)
    pattern_id: str
    pattern_name: str
    category: Literal[
        "CONCEPTUAL",
        "PROCEDURAL",
        "REPRESENTATION",
        "LANGUAGE_INTERPRETATION",
        "CONDITION_OMISSION",
        "OVERGENERALIZATION",
        "OPTION_COMPARISON",
        "UNCLASSIFIED",
    ]
    distinct_session_count: int = Field(ge=1)
    typical_manifestations: list[str] = Field(min_length=1)
    corrective_principle: str
    representative_evidence_ids: list[str] = Field(min_length=1, max_length=3)


class ExamErrorPatternReport(BaseModel):
    response_type: Literal["EXAM_ERROR_PATTERN_REPORT"] = "EXAM_ERROR_PATTERN_REPORT"
    items: list[ExamErrorPatternItem] = Field(min_length=1, max_length=10)
```

禁止仅凭一次错误将学生标记为“粗心”。`CARELESS` 不作为初版受控枚举，除非有明确行为证据和人工标签。

### 9.5 Tutor Strategy Report / Playbook

```python
class TutorStrategyItem(BaseModel):
    rank: int = Field(ge=1)
    strategy_id: str
    strategy_name: str
    strategy_category: str
    distinct_session_count: int | None = Field(default=None, ge=1)
    suitable_situations: list[str] = Field(min_length=1, max_length=3)
    first_question: str = Field(min_length=1)
    scaffolding_steps: list[str] = Field(min_length=1, max_length=4)
    recommended_talk_moves: list[str]
    avoid_actions: list[str] = Field(default_factory=list, max_length=3)
    representative_evidence_ids: list[str] = Field(min_length=1, max_length=2)


class TutorStrategyReport(BaseModel):
    response_type: Literal["TUTOR_STRATEGY_REPORT"] = "TUTOR_STRATEGY_REPORT"
    mode: Literal["PATTERN_REPORT", "CASE_PLAYBOOK"]
    items: list[TutorStrategyItem] = Field(min_length=1, max_length=5)
```

`CORPUS/SUBJECT` 输出 `PATTERN_REPORT`；`CASE` 输出 `CASE_PLAYBOOK`。当前数据只能证明策略被观察到和反复使用，不能证明它是“最有效策略”。

### 9.6 Content Improvement Brief

```python
class ContentOpportunityItem(BaseModel):
    opportunity_id: str
    observed_gap: str
    evidence_session_count: int = Field(ge=1)
    target_misconceptions: list[str] = Field(min_length=1)
    recommended_asset_types: list[Literal[
        "MICRO_LESSON",
        "WORKED_EXAMPLE",
        "DIAGNOSTIC_QUIZ",
        "DISTRACTOR_EXPLANATION",
        "VISUAL_AID",
        "TUTOR_GUIDE",
    ]]
    learning_objective: str
    suggested_content_outline: list[str] = Field(min_length=1, max_length=6)
    validation_method: str
    priority_basis: str
    representative_evidence_ids: list[str] = Field(min_length=1, max_length=3)


class ContentImprovementBrief(BaseModel):
    response_type: Literal["CONTENT_IMPROVEMENT_BRIEF"] = "CONTENT_IMPROVEMENT_BRIEF"
    items: list[ContentOpportunityItem] = Field(min_length=1, max_length=5)
```

禁止输出没有实验依据的“预计提升百分比”。优先级必须基于明确规则，例如会话覆盖、重复错误和现有内容缺口。

### 9.7 Socratic Turn

```python
class TutoringStage(str, Enum):
    DIAGNOSE = "DIAGNOSE"
    ELICIT_REASONING = "ELICIT_REASONING"
    GIVE_HINT = "GIVE_HINT"
    CHECK_UNDERSTANDING = "CHECK_UNDERSTANDING"
    TRANSFER = "TRANSFER"
    COMPLETE = "COMPLETE"


class SocraticTurn(BaseModel):
    response_type: Literal["SOCRATIC_TURN"] = "SOCRATIC_TURN"
    stage: TutoringStage
    student_visible_message: str = Field(min_length=1, max_length=240)
    teaching_move: str
    asks_exactly_one_question: bool
    answer_revealed: bool
    next_stage: TutoringStage
```

学生端规则：

- 一次最多提出一个核心问题；
- 默认不直接给最终答案；
- 先诊断学生现有思路，再给提示；
- 不展示其他学生的原声、Session ID 或内部 Debug Trace；
- 完成当前题后使用同构题检查迁移；
- 缺少题干或学生尝试时，先请求必要信息，不猜题意。

学生端属于后续独立工作包，不阻塞前三类内部业务报告上线。

### 9.8 ResponseResult Union

实现时使用带唯一 `response_type` 的 Union。这里使用前向类型名称表达目标契约；代码中应按 Pydantic 版本选择标准 discriminated-union 写法。

```python
ResponseResult = Annotated[
    StudentFAQReport
    | DifficultyReport
    | MisconceptionReport
    | ExamErrorPatternReport
    | TutorStrategyReport
    | ContentImprovementBrief
    | SocraticTurn,
    Field(discriminator="response_type"),
]

BusinessQueryRequest.model_rebuild()
BusinessResponseEnvelope.model_rebuild()
```

禁止把所有可选字段重新合并为一个万能 Response；否则会再次出现不同业务场景互相妥协、缺字段却仍能通过的问题。

## 10. 新增分析数据模型

当前两张知识卡能支持错因和策略案例，但不能完整支持高频问题和难度分析。需要新增离线抽取/派生资产。

### 10.1 学生问题事件

建议 DuckDB 表：

```sql
CREATE TABLE student_question_events (
    event_id VARCHAR PRIMARY KEY,
    session_id BIGINT NOT NULL,
    turn_id INTEGER NOT NULL,
    subject_path VARCHAR NOT NULL,
    raw_question VARCHAR NOT NULL,
    normalized_question VARCHAR NOT NULL,
    question_cluster_id VARCHAR NOT NULL,
    question_function VARCHAR NOT NULL,
    extraction_method VARCHAR NOT NULL,
    evidence_verified BOOLEAN NOT NULL
);
```

抽取边界：

- 来源必须是学生 Turn；
- `raw_question` 必须是 Turn 原文子串或完整原文；
- “I don't know”“Is it B?” 等可按功能归类，但不得伪造为语法完整的问题；
- 聚类名和原声分开保存；
- 同一事件不可跨 split。

### 10.2 难度信号

建议以事件和可重算视图为主，不优先存储不透明总分：

```sql
CREATE TABLE concept_difficulty_events (
    event_id VARCHAR PRIMARY KEY,
    session_id BIGINT NOT NULL,
    subject_path VARCHAR NOT NULL,
    signal_type VARCHAR NOT NULL,
    signal_value DOUBLE NOT NULL,
    source_turn_ids INTEGER[] NOT NULL,
    derivation_version VARCHAR NOT NULL
);
```

初版 Signal：

- `REPEATED_INCORRECT_ATTEMPT`；
- `REPEATED_HELP_REQUEST`；
- `LONG_TUTORING_SEQUENCE`；
- `STRATEGY_SWITCH`；
- `UNRESOLVED_ENDING`。

若某 Signal 无可靠提取条件，则返回 `UNMEASURED`，不能填 0 冒充没有发生。

### 10.3 误区规范化

```sql
CREATE TABLE misconception_taxonomy (
    cluster_id VARCHAR PRIMARY KEY,
    canonical_name VARCHAR NOT NULL,
    error_category VARCHAR NOT NULL,
    taxonomy_version VARCHAR NOT NULL
);

CREATE TABLE misconception_cluster_assignments (
    chunk_id VARCHAR PRIMARY KEY,
    cluster_id VARCHAR NOT NULL,
    assignment_source VARCHAR NOT NULL,
    review_status VARCHAR NOT NULL
);
```

Agent 自动分配只能标记为 `DRAFT/AUTO_ASSIGNED`；人工审核后才能成为签署基线。

### 10.4 导师策略规范化

现有 `strategy_category` 只有五类，仍需区分“策略类型”和“可复用教学动作”。建议新增稳定 `strategy_id`，原始 `key_aha_question` 继续保留为证据。

### 10.5 内容机会

内容机会是派生建议，不是事实表。建议先在运行时生成并带完整来源；经课程研发人工采纳后再写入单独的 `content_backlog`，不得自动写成已批准任务。

## 11. 各 Query Plan 的执行细节

### 11.1 Analytics With Evidence

适用于高频问题、困难概念、共性误区、考试错误和 Corpus/Subject 级导师策略。

```text
1. 解析 subject 和其他 filter
2. 在 DuckDB 计算独立 Session 计数、事件数及排名
3. 为 Top-N 项目选择 1~3 个代表 Session
4. 仅在对应 Session/cluster 内检索相关卡片
5. 从 source_turn_ids 回查 DuckDB Turn
6. 生成 authorized evidence map
7. 可选调用 LLM 生成短 summary/机制解释
8. 后端填充数量、比例、排名和 citation
9. 四元组审计
10. 按响应类型渲染
```

代表案例选择应兼顾相关性和会话去重，不能让一个长会话占据全部证据位。

### 11.2 Case Playbook

适用于“当学生出现某种表现时，老师如何引导”。

```text
1. 识别 CASE scope、考点和学生表现
2. 错因库检索潜在 misconception
3. 策略库检索对应 intervention
4. 通过 subject/session consistency 重排
5. 选择最多两个策略案例
6. 后端从策略卡读取 first question、steps、talk moves
7. LLM 只负责压缩为适用场景和避免事项
8. 审计导师原声引用
```

### 11.3 Content Brief

```text
1. 调用困难概念/误区聚合
2. 读取现有高频问题和导师策略
3. 形成结构化 Evidence Bundle
4. LLM 生成 learning objective 和 content outline
5. 后端保留真实 evidence count 和 citations
6. 明确 validation method
7. 输出“建议”，不得自动标记为批准
```

### 11.4 Socratic Tutoring

```text
1. 解析当前题目、学生答案和理由
2. 检索相似错因与导师策略，仅作为内部参考
3. 更新 TutorSessionState
4. 选择一个教学动作
5. 生成一个短反馈和一个问题
6. 检查是否泄露答案
7. 返回 SocraticTurn
```

初版状态模型：

```python
class TutorSessionState(BaseModel):
    tutoring_session_id: str
    question_text: str
    options: dict[str, str]
    student_attempt: str | None
    student_reasoning: str | None
    stage: TutoringStage
    hint_count: int = Field(ge=0)
    misconception_hypotheses: list[str]
    answer_revealed: bool
```

## 12. LLM 使用规范与性能预算

### 12.1 Mimo 当前实测基线

2026-09-02 对当前 `mimo-v2.5` endpoint 的合成探针：

| 配置 | 结果 |
|---|---|
| 默认 Thinking，`max_tokens=256` | 总耗时约 14.63s，256 completion tokens 全部消耗于 reasoning，未产生可见正文 |
| `thinking.type=disabled`，短回答 | TTFT 约 2.86s，总耗时约 3.04s |
| `thinking.type=disabled`，231 可见 tokens | TTFT 约 1.85s，总耗时约 7.56s，首 token 后约 41.16 token/s |

这是单次诊断证据，不是长期 SLA；服务器负载、套餐和网络会导致波动。

### 12.2 默认生成策略

面向事实归纳和格式整理，默认：

```python
temperature = 0.1
stream = True
extra_body = {"thinking": {"type": "disabled"}}
```

Thinking 只能按意图显式开启，且必须通过同一评测集证明质量收益高于延迟代价。不得静默启用。

### 12.3 初始输出预算

| Intent | API 最大输出 tokens | 目标可见长度 |
|---|---:|---|
| 高频问题 | 160 | 1 句总结；列表由后端渲染 |
| 困难概念 | 240 | 总结主要难点和限制 |
| 共性误区 | 220 | 机制归纳 |
| 考试错误 | 220 | 模式解释和纠正原则 |
| 导师策略 | 300 | 适用场景、步骤和避免事项 |
| 内容改进 | 420 | 一份短 Content Brief |
| 当前题目 | 96 | 一个反馈 + 一个问题 |
| 路由分类 | 96 | 纯 JSON |

Pydantic `max_length` 只是返回后校验，不是 API 输出限制。调用层必须设置 `max_tokens` 或供应商实际支持的等价参数。

### 12.4 Client 生命周期

OpenAI-compatible Client 在 Pipeline 初始化时创建并复用，不得每次 Query 重新初始化。需要记录：

- 实际响应 model；
- TTFT；
- 总请求耗时；
- 可见文本流耗时；
- prompt tokens；
- cached tokens；
- reasoning tokens；
- visible output tokens；
- retry count；
- finish reason。

禁止用 `completion_tokens / total_request_seconds` 命名为“模型输出速度”。应分别报告：

```text
end_to_end_throughput
visible_decode_throughput
time_to_first_visible_token
```

### 12.5 CLI 流式策略

结构化 JSON 在完整解析前不能当作可信业务响应逐 token 展示。当前 CLI 采用：

1. 显示阶段状态：理解问题、统计、检索、生成、审计；
2. 内部流式收集模型 JSON并记录 TTFT；
3. 完整解析和审计后一次性渲染业务卡片；
4. Debug Trace 默认关闭，通过 `:debug on` 显式开启。

后续 Web UI 可以通过 SSE 推送阶段事件，但仍应在审计完成后展示最终可信结果。

## 13. Prompt 设计原则

### 13.1 禁止一个 Prompt 处理所有任务

每种响应只允许模型生成其真正需要推理的字段。数量、比例、排名、引用和原始字段由后端注入。

### 13.2 Prompt 输入结构

```text
System：当前意图的职责、禁止事项、输出 Schema
User：原始 Query
Facts：SQL 计算结果，只读
Evidence candidates：带 evidence_id 的授权候选
Task：只生成指定字段，并从 evidence_id 中选择
```

### 13.3 Evidence ID 约束

模型只能返回后端提供的 `evidence_id`，不能自由输出 Session ID、Turn ID 或改写引用。后端根据授权映射还原完整四元组。

若模型返回未知 ID：

- 第一次：带明确校验错误重试一次；
- 第二次：返回 `FAILED` 或显式 deterministic response；
- 不得选一个相似 ID 偷偷替换。

### 13.4 语言规则

- 回答语言跟随用户 Query；
- 原声引用保留原文；
- 可附翻译，但翻译必须标记为翻译，不能替代 exact quote；
- 数学公式和选项不可改写；
- 不把学生错误原话写成标准知识。

## 14. Evidence 与引用安全

完整引用授权条件：

```text
citation.session_id 来自本次候选
AND citation.turn_id 属于该卡 source_turn_ids
AND citation.speaker 与权威 DuckDB Turn 一致
AND citation.exact_quote 与权威 Turn 精确/允许的 canonical substring 匹配
```

禁止：

- 仅因为 Session 存在就把整场前 6 个 Turn 标记为可信引用；
- 将 `verifiable_in_duckdb=True` 作为跳过审计的通行证；
- 引用未被本次检索授权的数据库内容；
- 将学生错误言论用作正确结论依据；
- 将 Gold/qrels 输入生产 Retriever 或 Generator。

Grounding 指标直接复用 L1 V2：

- Citation Precision = 100%；
- Citation Recall = 100%；
- Role Accuracy = 100%；
- Quote Grounding Precision = 100%；
- Candidate Authorization Rate = 100%。

## 15. 无证据、歧义和不支持场景

### 15.1 无证据

```json
{
  "status": "NO_EVIDENCE",
  "summary": "当前索引中没有足够证据回答这个问题。",
  "limitations": [
    "当前查询范围内没有经过验证的相关课堂记录"
  ],
  "audit_status": "NOT_APPLICABLE"
}
```

不得用模型常识补成“基于数据”的回答。

### 15.2 问题歧义

- 可安全选择主意图时，执行主意图并明确说明解释方式；
- 两个意图均强命中时使用主 + 次复合报告；
- 无法判断且不同解释会明显改变结果时，返回一个简短澄清问题；
- 不应在后台同时运行全部七条路径。

### 15.3 不支持

以下例子初版应拒答或明确限制：

- “谁是最优秀的导师？”：缺少因果效果标签，且涉及人员评价。
- “这个策略能提升多少分？”：没有 A/B 或纵向结果。
- “重考者有哪些模式？”：当前 `CleanedSession` 未保留可验证的 retaker 标签。
- “NBCOT 最难的内容是什么？”：当前数据是 Eedi 数学代理数据。

## 16. Trace 与可观测性

每个 Query 生成一个 `trace_id`，至少记录：

```text
raw_query
primary/secondary intent
scope and filters
route source and confidence
query plan
SQL statement ID and row counts
retrieval lanes and candidate IDs
final context IDs
authorized evidence IDs
response schema
thinking mode
max output tokens
TTFT / generation / total latency
prompt / cached / reasoning / visible tokens
retry count
status / degradation / failure reason
dataset / qrels / index / model / config versions
```

日志禁止包含 API key、Cookie、学生未脱敏 PII 或未经授权的完整原始数据。

## 17. 性能目标

以下是 0→1 阶段的目标，不是当前已达成绩：

| Plan | P50 | P95 | 备注 |
|---|---:|---:|---|
| 纯 SQL Analytics | ≤0.5s | ≤1.5s | 不调用 LLM |
| Analytics + 短总结 | ≤4s | ≤7s | 默认 Thinking off |
| Case Playbook | ≤6s | ≤10s | 最多两个案例 |
| Content Brief | ≤9s | ≤15s | 输出预算更高 |
| Socratic Turn | ≤4s | ≤6s | 单步短输出 |
| 首个可见 token | ≤2.5s | ≤4s | 单独测量 |

性能优化优先级：

1. 关闭不必要的 Thinking；
2. 在 API 层限制输出；
3. 复用 Client 连接；
4. 后端确定性渲染事实字段；
5. 缩减最终上下文；
6. 缓存重复聚合；
7. 并行独立检索 lane；
8. 最后才考虑更换模型或服务等级。

## 18. 测试设计

### 18.1 Router 人工标注集

新增：

```text
evals/datasets/query-router-v1.jsonl
```

最小字段：

```json
{
  "case_id": "router-001",
  "query": "分数运算中学生最常卡在哪里？",
  "primary_intent": "DIFFICULT_CONCEPTS",
  "secondary_intents": [],
  "scope": "SUBJECT",
  "subject_filters": ["Fractions"],
  "label_source": "human",
  "split": "dev"
}
```

至少覆盖：

- 每个一级意图不少于 15 条；
- 中英文和中英混合 Query；
- 数字、公式和选项；
- 复合意图；
- 无证据和不支持问题；
- 同一主题在不同 Scope 下的表达。

### 18.2 Router 门禁

初始验收：

| 指标 | 门槛 |
|---|---:|
| Primary intent accuracy | ≥95% |
| Scope accuracy | ≥95% |
| Protected token preservation | 100% |
| Unsupported precision | 100% |
| Multi-intent recall | ≥90% |
| Router P95（确定性命中） | ≤10ms |

### 18.3 Response Contract 测试

每种响应必须测试：

- 缺字段拒绝；
- 额外字段拒绝；
- 空结果不能标 `SUCCESS`；
- 排名、计数和比例边界；
- Evidence ID 必须存在于授权映射；
- `SUCCESS` 与 audit status 一致；
- Student response 一次只有一个核心问题；
- Student response 不泄露最终答案和历史 Session 信息。

### 18.4 Planner 测试

```text
统计 Query 不允许进入纯向量问答
QUESTION scope 不运行全库聚合
CONTENT_IMPROVEMENT 必须包含 analytics 和 evidence
UNSUPPORTED 不调用 LLM
复合意图最多执行两条协调后的子计划
```

### 18.5 Analytics 测试

- 所有计数与固定 DuckDB fixture 的 SQL 真值一致；
- 使用 `COUNT(DISTINCT session_id)`；
- 同 Session 重复事件不会错误放大覆盖率；
- 空集合返回 `NO_EVIDENCE`，不返回 0% 的“结论”；
- 过滤前后 DataScope 正确；
- 数据版本变化导致缓存失效。

### 18.6 Grounding 测试

复用 L1 V2 红灯用例：

- 错误 Session；
- 错误 Turn；
- 错误 speaker；
- 改写 quote；
- 未召回但数据库存在的证据；
- 空引用；
- 模型返回未知 evidence ID。

### 18.7 性能测试

每类 Intent 至少 20 条 Query，分别记录：

- Router；
- SQL；
- Retrieval；
- Assembly；
- TTFT；
- Visible generation；
- Audit/render；
- Total。

模型速度报告必须区分 reasoning 和可见输出。

## 19. 推荐文件结构

目标结构：

```text
src/
├── query_intent.py          # 枚举、规则 Router、LLM fallback classifier
├── query_planner.py         # QueryUnderstanding → QueryPlan
├── response_models.py       # 七种响应和统一 Envelope
├── analytics.py             # 高频、难度、误区、错误、策略 SQL
├── evidence.py              # Authorized evidence map 和四元组审计
├── generation.py            # 持久 Client、Mimo 参数、流式探针
├── tutoring.py              # 后续 Socratic 状态机
├── rag_pipeline.py          # 统一编排，逐步替换当前通用链路
└── cli.py                   # 意图感知渲染、detail/debug 指令

tests/
├── test_query_intent.py
├── test_query_planner.py
├── test_response_models.py
├── test_analytics.py
├── test_evidence.py
├── test_generation_latency.py
└── test_tutoring.py

evals/datasets/
├── query-router-v1.jsonl
├── business-response-v1/
└── l1-v2/
```

文件名是目标接口；施工前不存在的文件不得在使用文档中表述为已经可运行。

## 20. 与现有代码的迁移方案

### 20.1 保留并复用

- `src/models.py` 的 Session、知识卡和 Citation 基础模型；
- `src/data_fusion.py` 的三表融合与 Turn 压缩；
- `src/extract_knowledge.py` 的严格抽取和 grounding；
- `src/storage_manager.py` 的 DuckDB/Chroma 与证据回查；
- `src/retriever.py` 的现有检索作为 baseline lane；
- `src/reranker.py` 的 Assembler 作为待 Trace 化基线；
- `evals/` 的严格状态、graded qrels、Grounding 和报告框架。

### 20.2 逐步替换

当前：

```text
ask(query)
  → 同时检索错因和策略
  → 固定选择一张错因和一张策略
  → 单一 LLMGuidancePayload.answer
  → 通用 Markdown
```

目标：

```text
ask(query)
  → QueryUnderstanding
  → QueryPlan
  → intent-specific execution
  → BusinessResponseEnvelope
  → intent-specific renderer
```

### 20.3 兼容策略

第一阶段新增 `ask_business()`，保留当前 `ask()` 供既有测试和 CLI 对照；新链路通过 Router/Contract 测试后，再让 `ask()` 调用新编排器。不得一次性删除旧链路后失去 paired baseline。

旧 `PedagogicalGuidanceResponse` 标记为兼容模型，不继续扩充为七种业务响应的万能对象。

## 21. 工作包与施工顺序

### WP0：冻结业务测试集

交付：

- `query-router-v1.jsonl`；
- 每类响应不少于 5 个 Gold 示例；
- 明确 dev/holdout；
- 人工确认意图、Scope、响应结构。

完成条件：测试集版本化，Gold 不进入 SUT。

### WP1：响应契约与 Router

交付：

- `response_models.py`；
- `query_intent.py`；
- 确定性规则；
- 低置信度 Mimo classifier；
- Router Trace。

完成条件：Router 和 Contract 门禁通过。

### WP2：Analytics 数据资产

交付：

- 学生问题事件；
- 难度信号；
- 误区和策略规范化映射；
- 版本化派生脚本与 manifest；
- 对应 SQL 单测。

完成条件：统计值与人工 SQL 真值 100% 一致。

### WP3：前三类业务报告

优先实现：

1. `StudentFAQReport`；
2. `DifficultyReport`；
3. `TutorStrategyReport`。

原因：分别覆盖学生问题发现、教研难点诊断和导师知识复用，是当前最能证明业务价值的三个闭环。

完成条件：每份报告可回溯证据，数字字段不由 LLM 生成。

### WP4：误区、考试错误和内容建议

交付：

- `MisconceptionReport`；
- `ExamErrorPatternReport`；
- `ContentImprovementBrief`。

完成条件：内容建议有证据和验证方法，不承诺未经验证的效果。

### WP5：生成性能改造

交付：

- 复用 API Client；
- Thinking 显式配置；
- 动态输出预算；
- 内部 streaming；
- TTFT/Reasoning/Visible TPS Trace；
- `:debug` 和 `:detail` CLI 指令。

完成条件：各 Plan 的 P95 达到第 17 节初始目标，失败不静默降级。

### WP6：学生端 Socratic 状态机

在内部业务报告稳定后施工。交付状态机、答案泄露检测、同构迁移题和独立评测集。

### WP7：统一业务评测

将 Router、Analytics、Retrieval、Assembler、Grounding、Generator Trace 接入现有 L1 V2 控制面，自动生成不可覆盖报告。

## 22. 首批 TDD 测试名称

```text
test_router_does_not_infer_user_role
test_router_maps_corpus_strategy_question_to_strategy_report
test_router_maps_frequent_student_questions_to_analytics
test_router_maps_difficult_topic_question_to_difficulty_report
test_router_maps_specific_math_question_to_socratic_tutoring
test_router_preserves_numbers_formulas_and_options
test_low_confidence_router_uses_bounded_non_thinking_classifier
test_statistics_plan_never_uses_vector_results_as_counts
test_response_success_requires_verified_evidence
test_model_can_only_select_authorized_evidence_ids
test_no_evidence_does_not_generate_business_claims
test_strategy_report_does_not_claim_causal_effectiveness
test_difficulty_report_exposes_signals_not_fake_score
test_student_turn_asks_one_question_and_hides_internal_citations
test_generation_records_ttft_reasoning_and_visible_tokens_separately
test_generation_enforces_intent_specific_output_budget
```

## 23. 0→1 验收标准

首版业务链路只有同时满足以下条件才算完成：

- [ ] 不需要用户身份或权限配置即可正确路由典型业务问题；
- [ ] 七个 Intent 和四个 Scope 均有人工标注测试；
- [ ] 高频、困难、误区、错误和策略的数量来自 SQL；
- [ ] 至少完成 FAQ、Difficulty、Strategy 三种正式报告；
- [ ] 每条历史事实引用通过完整四元组和候选授权审计；
- [ ] 无证据时返回 `NO_EVIDENCE`，不调用常识补写；
- [ ] 不支持的因果、人员排名和 NBCOT 问题被正确限制；
- [ ] Mimo Thinking 与输出预算均显式记录；
- [ ] 延迟按 Router/SQL/Retrieval/TTFT/Generation/Audit 分段；
- [ ] 报告展示真实 DataScope、dataset/index version；
- [ ] 所有测试通过，任何失败或未测状态如实报告；
- [ ] 优化前后使用同一 dev/holdout 做 paired comparison。

## 24. 当前不属于完成状态的事项

以下内容目前只是本规范的目标，不得在 README、演示或交付中表述为已经实现：

- 七类 Query Intent Router；
- 学生问题事件和规范化问题聚类；
- 可审计的困难概念信号；
- 七种强类型业务响应；
- SQL/RAG 自动 Query Planner；
- Mimo Thinking 关闭和动态输出预算的生产接入；
- 真实用户可见的流式体验；
- 学生端有状态 Socratic Tutor；
- 当前 100 Session 之外的约 1,000 场正式索引；
- NBCOT 领域效果。

## 25. 最终开发定调

后续所有功能讨论都应先回答四个问题：

1. 这条 Query 属于哪个业务 Intent？
2. 它要求全库统计、主题归纳、案例检索还是具体题目辅导？
3. 哪些字段必须由数据库确定性计算，哪些字段才需要 LLM？
4. 最终结论如何回到真实 Session/Turn 证据？

如果一个功能无法回答这四个问题，就不应直接通过增加 Prompt、Top-K 或新 Agent 进入生产链路。

本项目的目标输出不是“一段看起来聪明的回答”，而是：

> 与业务任务匹配、计算路径正确、长度受控、速度可测、证据可审计、失败状态真实的教学响应。
