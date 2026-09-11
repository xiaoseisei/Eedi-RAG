"""
================================================================================
模块名称: src/query_intent.py
业务定位: 业务查询的确定性一级路由 (Deterministic business-query routing)

本模块是业务感知层 Eedi-RAG 查询边界的"第一步解读器": 把一个结构化
BusinessQueryRequest 路由成 QueryUnderstanding(意图/作用域/计算范围/证据视角等)。

设计上刻意让两个决策互相独立, 不互相绑架:
  * 计算范围 (computation scope): 宏观聚合(analytics) / 微观案例检索(micro) / 两者混合;
  * 证据视角 (evidence perspective): 学生对白 / 导师对白 / 两者都要。

三条铁律:
  1. 不推断"提问者是谁"——是谁问的属于业务上下文, 不属于查询文本本身;
  2. 不用向量相似度替代分析决策——聚合与否是数学模型问题, 不是语义近似问题;
  3. 信号即规则, 显式可审计——本模块是规则引擎, 不是语言模型, 所有打分都可回查。

输出喂给下游 (如 business_models.QueryPlan 编排), 决定后续走哪条执行链路。
================================================================================
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from src.business_models import (
    BusinessQueryRequest,
    ComputationScope,
    EvidencePerspective,
    QueryIntent,
    QueryScope,
    QueryUnderstanding,
    RouteConfidence,
)
from src.intent_classifier import IntentClassifier


# =============================================================================
# 受保护 token 提取: 公式与标识符必须原样保留
# -----------------------------------------------------------------------------
# 匹配在归一化后的文本上进行, 但返回的 token 一律是"调用方原始查询里的切片",
# 绝不改写用户写的数学表达式(那是后续查询改写器的事, 且对数学必须放行)。
# =============================================================================

# 括号表达式(如 (-2)^2、(54+58)/2): 最长语义单元优先, 内部碎片不再单独拆分。
_PAREN_EXPRESSION_RE = re.compile(
    r"\([^\r\n()]{1,80}\)(?:\s*(?:\^|/)\s*[+-]?\d+(?:\.\d+)?)?",
    re.UNICODE,
)
# 钟表时刻(如 10:12): 必须先于普通数字识别, 否则会丢失冒号结构。
_CLOCK_TIME_RE = re.compile(r"\d{1,2}:\d{2}", re.UNICODE)
# 二元分数表达式(如 5/7-1/4): 先整体保留, 避免把减号误读成第二个分数的负号。
_FRACTION_BINARY_EXPRESSION_RE = re.compile(
    r"[+-]?\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?",
    re.UNICODE,
)
# 数字(含正负号、小数、可选的分数形式 ±a/b)。
_NUMBER_RE = re.compile(
    r"[+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?",
    re.UNICODE,
)
# 单个小写字母变量(可选幂次, 如 x、-q、x^2): 只保留常见独立代数变量,
# 避免把英语冠词 a、介词等自然语言碎片误判为数学 token。
_VARIABLE_EXPRESSION_RE = re.compile(
    r"(?<![A-Za-z])-?[pqnxyz](?:\s*\^\s*\d+)?(?![A-Za-z])",
    re.UNICODE,
)
# 选择题选项位 (选项A ~ 选项D / option A ~ option D)。
_OPTION_RE = re.compile(r"(?:选项|option)\s*[A-D]", re.IGNORECASE)


def _append_unique(tokens: list[str], value: str) -> None:
    """去空白 + 去重地追加一个 token 到结果列表。"""
    value = value.strip()
    if value and value not in tokens:
        tokens.append(value)


def extract_protected_tokens(text: str) -> list[str]:
    """按源文本出现顺序返回公式、数字与答案选项片段。

    刻意返回"原始字符串切片"(不经归一化、不改写), 供查询理解与后续检索使用;
    数学内容必须原样保留, 这也是本函数存在的根本理由。
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matches: list[tuple[int, int, str]] = []       # (start, end, value) 命中区间
    occupied: list[tuple[int, int]] = []           # 已被更长单元占用的区间, 用于去重叠

    def add_matches(pattern: re.Pattern[str], *, skip_occupied: bool = True) -> None:
        """收集某一条正则的命中; skip_occupied=True 时跳过与已占区间重叠的命中。"""
        for match in pattern.finditer(text):
            start, end = match.span()
            if skip_occupied and any(start < other_end and end > other_start for other_start, other_end in occupied):
                continue  # 与更长的语义单元重叠, 放弃(它只是那个单元的内部碎片)
            value = match.group(0).strip()
            if not value:
                continue
            matches.append((start, end, value))
            occupied.append((start, end))

    # 长语义单元优先于其内部散落的数字/变量, 例如 `(-q)^2` 绝不能拆成 `q` + `2`。
    add_matches(_PAREN_EXPRESSION_RE)
    add_matches(_OPTION_RE)
    add_matches(_CLOCK_TIME_RE)
    add_matches(_FRACTION_BINARY_EXPRESSION_RE)
    add_matches(_VARIABLE_EXPRESSION_RE)
    add_matches(_NUMBER_RE)
    matches.sort(key=lambda item: (item[0], item[1]))  # 按源位置恢复自然顺序

    tokens: list[str] = []
    for _, _, value in matches:
        _append_unique(tokens, value)
    return tokens


# =============================================================================
# 意图信号表 (INTENT_SIGNALS) 与打分规则
# -----------------------------------------------------------------------------
# 信号词刻意"显式、可审阅": 强短语计 +3 分, 短领域词计 +1 分。
# 本模块是规则引擎不是语言模型, 因此打分必须透明可回溯, 不做隐式语义。
# =============================================================================
INTENT_SIGNALS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.STUDENT_FREQUENT_QUESTIONS: (
        "最常问",
        "常问",
        "提问最多",
        "最常提出的问题",
        "学生常问",
        "提问密度",
        "most frequent question",
        "most asked",
        "frequently asked",
        "common student questions",
        "ask most often",
    ),
    QueryIntent.DIFFICULT_CONCEPTS: (
        "最难",
        "理解难",
        "困难概念",
        "最难理解",
        "卡壳",
        "困难主题",
        "核心难点",
        "最费时间",
        "most difficult",
        "hardest concept",
        "hardest for students",
        "difficult concepts",
        "struggle with",
        "最常卡住",
        "困难信号",
    ),
    QueryIntent.RECURRING_MISCONCEPTIONS: (
        "共性误区",
        "认知错误",
        "误解反复",
        "错误理解",
        "认知盲区",
        "容易混淆",
        "混淆",
        "recurring misconception",
        "common misconception",
        "common misunderstanding",
        "misconceptions recur",
        "最常见的误解",
        "常见误解",
        "错因",
        "错题讲解",
        "误区",
        "统计误区",
        "舍入误区",
        "图谱分析",
        "宏观误区",
        "macro graph",
        "执行该统计",
        "图谱",
    ),
    QueryIntent.EXAM_ERROR_PATTERNS: (
        "考试错误",
        "错误模式",
        "误选",
        "审题错误",
        "答题错误",
        "exam error",
        "test mistake",
        "wrong answer pattern",
        "考试中最常",
        "最常犯",
        "遗漏条件",
        "最容易遗漏",
        "漏掉",
    ),
    QueryIntent.STUDENT_COHORT_PATTERNS: (
        "多次重考",
        "重考者",
        "基础薄弱",
        "低基础",
        "学生群体",
        "重考学生",
        "基础薄弱学生",
        "students with a weak foundation",
        "retakers",
        "weak foundation",
        "low foundation",
        "重考",
    ),
    QueryIntent.TUTOR_STRATEGY_PATTERNS: (
        "导师反复",
        "老师反复",
        "导师如何",
        "导师怎么",
        "怎么引导",
        "如何引导",
        "教学策略",
        "引导策略",
        "导师策略",
        "脚手架策略",
        "脚手架",
        "tutor strategy",
        "tutor strategies",
        "how do tutors",
        "teaching strategy",
        "teaching strategies",
        "scaffolding",
        "提问方式",
        "启发式教学动作",
    ),
    QueryIntent.CONTENT_IMPROVEMENT: (
        "教材",
        "微课",
        "题库",
        "内容改进",
        "教育内容",
        "课程改进",
        "新增教学内容",
        "新增哪些教学内容",
        "micro lesson",
        "content improvement",
        "curriculum improvement",
        "curriculum improve",
        "内容缺口",
        "清晰的教学材料",
        "学生错误和导师策略",
        "干预",
        "intervention",
        "跨课堂",
        "跨会话",
        "不同课堂",
        "cross session",
    ),
    QueryIntent.QUESTION_TUTORING: (
        "这题",
        "怎么算",
        "为什么不能",
        "我选",
        "答案选项",
        "如何解",
        "this question",
        "how do i solve",
        "why can't",
        "i chose",
        "answer option",
        "推理过程",
        "学生推理",
        "reasoning process",
        "单题辅导",
        "单题解析",
        "student error",
        "misconception intervention",
        "class review",
    ),
}

# Strong signals are semantic phrases that identify one business operation,
# not generic words such as "student", "strategy", or "error".  Only a single
# unconflicted strong-intent hit qualifies for deterministic classification.
STRONG_INTENT_SIGNALS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.STUDENT_FREQUENT_QUESTIONS: (
        "最常提出的问题", "学生常问", "提问密度", "most common student questions", "最常问的问题",
    ),
    QueryIntent.DIFFICULT_CONCEPTS: (
        "最难理解", "核心难点", "困难主题", "hardest concept", "difficult concepts",
    ),
    QueryIntent.RECURRING_MISCONCEPTIONS: (
        "共性误区", "反复出现的误区", "常见误解", "recurring misconception", "macro graph", "执行该统计",
    ),
    QueryIntent.EXAM_ERROR_PATTERNS: (
        "考试错误模式", "审题错误", "错误选项模式", "wrong answer pattern",
    ),
    QueryIntent.STUDENT_COHORT_PATTERNS: (
        "多次重考", "重考学生", "基础薄弱学生", "retakers", "weak foundation", "重考",
    ),
    QueryIntent.TUTOR_STRATEGY_PATTERNS: (
        "教学策略", "引导策略", "提问方式", "tutor strategies", "teaching strategies",
    ),
    QueryIntent.CONTENT_IMPROVEMENT: (
        "改进教材", "优化教材", "教材应优先改", "新增教学内容", "内容缺口", "content improvement", "cross session", "题库内容改进", "内容改进",
    ),
    QueryIntent.QUESTION_TUTORING: (
        "这道题怎么解", "如何解这道题", "how do i solve", "answer option", "student error", "misconception intervention",
    ),
}

# 强信号集合: 计 +3 分的词。规则 = 长度 ≥3 的短语, 或明确的高价值短领域词。
_STRONG_SIGNALS: frozenset[str] = frozenset(
    signal
    for intent, signals in INTENT_SIGNALS.items()
    for signal in signals
    if len(signal) >= 3
    or signal in {"最难", "卡壳", "重考者", "导师", "老师", "策略", "引导", "教材", "微课", "题库"}
)

# 分组信号: 用于判断"计算范围"与"证据视角"(而不是意图打分本身)。
MACRO_SIGNALS = ("最常", "多少", "top", "反复", "总体", "全部", "共性", "模式", "高频", "most", "common", "recurring", "overall", "macro", "宏观")   # 群体级/宏观聚合标记
STUDENT_SIGNALS = ("学生", "困惑", "误区", "错误", "重考", "基础薄弱", "混淆", "student", "confusion", "misconception", "retaker")             # 学生视角标记
TUTOR_SIGNALS = ("导师", "老师", "教学", "引导", "策略", "话术", "脚手架", "tutor", "teacher", "teach", "guidance", "strategy", "scaffolding")                # 导师视角标记
CASE_SIGNALS = (
    "当学生", "遇到", "如何引导", "怎么办", "纠结", "两个选项", "when a student", "which option",
    "这节课", "这堂课", "回顾这节", "回顾课堂", "本节辅导", "单节辅导", "具体案例", "真实案例", "围绕真实题目", "case review",
)

# ---------------------------------------------------------------------------
# 显式不支持信号: 在通用意图打分之前短路, 防止相邻能力伪装成可回答问题
# ---------------------------------------------------------------------------
# 这些不是"低置信度"的普通问题, 而是当前数据/业务边界已经明确禁止的结论:
# 人员优劣需要因果效果标签; 分数提升需要实验或纵向结果; NBCOT 不属于 Eedi 数学语料。
UNSUPPORTED_SIGNALS = (
    "最优秀的导师",
    "最好的导师",
    "最佳导师",
    "最优秀的老师",
    "最好的老师",
    "谁是最优秀",
    "谁教得最好",
    "教得最好",
    "效果最好",
    "评选最佳",
    "能提升多少分",
    "提升多少分",
    "提高多少分",
    "因果效果",
    "nbcot",
)

# 意图平局时的优先级排序: 分数相同时, 靠前的意图胜出。
_INTENT_PRIORITY = (
    QueryIntent.STUDENT_COHORT_PATTERNS,
    QueryIntent.QUESTION_TUTORING,
    QueryIntent.STUDENT_FREQUENT_QUESTIONS,
    QueryIntent.DIFFICULT_CONCEPTS,
    QueryIntent.RECURRING_MISCONCEPTIONS,
    QueryIntent.EXAM_ERROR_PATTERNS,
    QueryIntent.CONTENT_IMPROVEMENT,
    QueryIntent.TUTOR_STRATEGY_PATTERNS,
)

# 意图 -> 结果类型标签, 与 business_models 的 response_type 判别字段对应。
_RESPONSE_TYPES: dict[QueryIntent, str] = {
    QueryIntent.STUDENT_FREQUENT_QUESTIONS: "STUDENT_FAQ_REPORT",
    QueryIntent.DIFFICULT_CONCEPTS: "DIFFICULTY_REPORT",
    QueryIntent.RECURRING_MISCONCEPTIONS: "MISCONCEPTION_REPORT",
    QueryIntent.EXAM_ERROR_PATTERNS: "EXAM_ERROR_PATTERN_REPORT",
    QueryIntent.STUDENT_COHORT_PATTERNS: "STUDENT_COHORT_PATTERN_REPORT",
    QueryIntent.TUTOR_STRATEGY_PATTERNS: "TUTOR_STRATEGY_REPORT",
    QueryIntent.CONTENT_IMPROVEMENT: "CONTENT_IMPROVEMENT_BRIEF",
    QueryIntent.QUESTION_TUTORING: "SOCRATIC_TURN",
    QueryIntent.UNSUPPORTED: "UNSUPPORTED",
    QueryIntent.UNRESOLVED: "UNRESOLVED_QUERY",
}


def _normalized(text: str) -> str:
    """统一归一化: NFKC(兼容性分解, 统一全半角/异体字符) + 小写。"""
    return unicodedata.normalize("NFKC", text).casefold()


def _contains_any(text: str, signals: Iterable[str]) -> bool:
    """信号词是否任一命中归一化文本(用于布尔判定, 不关心计分细节)。"""
    return any(_normalized(signal) in text for signal in signals)


def _matched_signals(text: str, signals: Iterable[str]) -> list[str]:
    """返回命中的信号词, 且长度长者优先去重。

    保证 `导师反复` 计作一个长信号, 而不是再叠加一个 `导师` 重复计分。
    """

    matched = sorted(
        (signal for signal in signals if _normalized(signal) in text),
        key=lambda value: (-len(value), value),  # 长信号排前, 同长按字典序
    )
    selected: list[str] = []
    for signal in matched:
        # 与已入选者互为包含关系则跳过(含它的长短语已覆盖它)。
        if any(signal in existing or existing in signal for existing in selected):
            continue
        selected.append(signal)
    return selected


def _score_intents(text: str) -> dict[QueryIntent, int]:
    """对归一化文本逐意图计分: 强信号 +3, 弱信号 +1, 求和为该意图得分。"""
    scores: dict[QueryIntent, int] = {}
    for intent, signals in INTENT_SIGNALS.items():
        matches = _matched_signals(text, signals)
        if not matches:
            continue
        scores[intent] = sum(3 if signal in _STRONG_SIGNALS else 1 for signal in matches)
    return scores


def _strong_intent_matches(text: str) -> dict[QueryIntent, list[str]]:
    """Return strong semantic matches, retaining only longest phrases per intent."""
    matches = {
        intent: _matched_signals(text, signals)
        for intent, signals in STRONG_INTENT_SIGNALS.items()
        if _matched_signals(text, signals)
    }
    # Structural strong signals are phrasing-independent: a concrete math
    # expression plus an explanatory request identifies single-question
    # tutoring even when no benchmark phrase is reused.
    has_question_content = (
        bool(extract_protected_tokens(text))
        or any(marker in text for marker in ("questionid", "题号", "题目id"))
    )
    if (
        has_question_content
        and not _contains_any(text, MACRO_SIGNALS)
        and _contains_any(text, ("为什么", "为何", "如何", "怎么", "why", "how", "是什么", "what", "解析", "分析", "回顾", "review"))
    ):
        matches.setdefault(QueryIntent.QUESTION_TUTORING, []).append(
            "具体数学内容+解释请求"
        )
    if (
        _contains_any(text, ("这道", "该题", "此题", "this question"))
        and _contains_any(text, ("长方体", "几何", "分数", "小数", "代数", "表面积", "体积", "cuboid", "surface area", "volume"))
    ):
        matches.setdefault(QueryIntent.QUESTION_TUTORING, []).append(
            "特定题目上下文"
        )
    if _contains_any(text, ("most common question", "ask most often", "最常问的问题", "最常提出的问题")):
        matches.setdefault(QueryIntent.STUDENT_FREQUENT_QUESTIONS, []).append(
            "频率型问题询问"
        )
    if _contains_any(text, ("most difficult concept", "hardest concept", "最费时间")):
        matches.setdefault(QueryIntent.DIFFICULT_CONCEPTS, []).append(
            "困难概念询问"
        )
    if _contains_any(text, ("新增哪些教学内容", "新增教学内容", "what should be added")):
        matches.setdefault(QueryIntent.CONTENT_IMPROVEMENT, []).append(
            "新增教学内容询问"
        )
    if (
        _contains_any(text, ("how should a tutor", "导师如何", "导师怎么"))
        and _contains_any(
            text,
            (
                "distinguish",
                "区分",
                "解释",
                "solve",
                "解",
                "volume",
                "surface area",
                "cuboid",
                "分数",
                "小数",
                "几何",
            ),
        )
        and not _contains_any(text, MACRO_SIGNALS)
    ):
        matches.setdefault(QueryIntent.QUESTION_TUTORING, []).append(
            "导师+具体题目辨析请求"
        )
    return matches


def _subject_filters(raw_query: str) -> list[str]:
    """把已知的 Eedi 数学名词(考纲术语)映射成稳定的考纲路径(subject_path)。

    从 query_rewriter 的 MATH_DOMAIN_GLOSSARY 查找, 规避循环依赖采用惰性导入;
    词表不可用则返回空列表(不阻断路由)。
    """

    try:
        from src.query_rewriter import MATH_DOMAIN_GLOSSARY
    except ImportError:
        return []
    query = _normalized(raw_query)
    # “耗时/最费时间”描述分析负担, 不等于钟表读法中的数学“时间”考点。
    # 若不排除, 字符串包含关系会把任意困难概念问题错误限制到 Clock subject。
    elapsed_time_phrases = ("最费时间", "耗时", "用时", "花费时间", "时间最长", "卡壳时间", "时间长")
    is_elapsed_time_context = any(phrase in query for phrase in elapsed_time_phrases)
    result: list[str] = []
    # Resolve high-specificity domain terms before the broad glossary scan.
    # This prevents a shared token such as “range” from stealing a median or
    # rounding query's subject path.
    explicit_domains = (
        (("舍入", "rounding", "rounded"), "Number > Rounding and Estimating > Rounding to Decimal Places"),
        (("median", "中位数"), "Data and Statistics > Data Processing > Averages (mean, median, mode) from a List of Data"),
    )
    for markers, subject_path in explicit_domains:
        if any(_normalized(marker) in query for marker in markers):
            result.append(subject_path)
    for entry in MATH_DOMAIN_GLOSSARY:
        keywords = entry.get("keywords", [])
        if any(
            _normalized(keyword) in query
            and not (is_elapsed_time_context and _normalized(keyword) == "时间")
            for keyword in keywords
        ):
            subject_path = str(entry.get("subject_path", "")).strip()
            if subject_path and subject_path not in result and not any(
                subject_path != selected and any(
                    _normalized(marker) in query
                    for marker in markers
                )
                for markers, selected in explicit_domains
            ):
                result.append(subject_path)
    return result


def _requested_top_k(request: BusinessQueryRequest) -> int:
    """解析用户文本里的显式数量请求(top N / 前 N / 列出 N), 封顶 10。

    查不到显式请求时回退到请求自带的 top_k。
    """

    query = _normalized(request.query)
    query_match = re.search(r"(?:top|前|列出)\s*([0-9]{1,3})", query, re.IGNORECASE)
    if query_match:
        requested = int(query_match.group(1))
        if requested >= 1:
            return min(requested, 10)
    return request.top_k


def _has_specific_math_token(protected_tokens: list[str]) -> bool:
    """判断受保护 token 是否真是题目数学内容, 而非仅表示 Top-N 的排名数字。"""

    return any(
        re.fullmatch(r"\d+", token.strip()) is None
        for token in protected_tokens
    )


def _semantic_hints(text: str) -> tuple[str | None, str | None, dict[str, str]]:
    """Extract non-authoritative metric/artifact hints without changing scope."""
    ranking = None
    if _contains_any(text, ("时间最长", "最费时间", "耗时最长", "duration", "longest")):
        ranking = "duration_desc"
    elif _contains_any(text, ("最常", "高频", "最多", "most common", "frequent")):
        ranking = "frequency_desc"
    elif _contains_any(text, ("最难", "困难", "卡壳", "hardest", "difficult")):
        ranking = "difficulty_desc"
    artifact = None
    if _contains_any(text, ("误区卡", "错因卡", "misconception card")):
        artifact = "misconception_card"
    elif _contains_any(text, ("策略卡", "教学策略", "strategy card")):
        artifact = "strategy_card"
    elif _contains_any(text, ("问题卡", "question card")):
        artifact = "question_card"
    sources: dict[str, str] = {}
    if ranking:
        sources["ranking_signal"] = "semantic_rule"
    if artifact:
        sources["artifact_target"] = "explicit_term"
    return ranking, artifact, sources


def _scope_for(
    text: str,
    intent: QueryIntent,
    protected_tokens: list[str],
    subject_filters: list[str],
) -> QueryScope:
    """推断查询作用域: 语料级 / 学科级 / 案例级 / 题目级。

    判定是自顶向下的通用业务规则链, 每条规则假设"更具体的情境优先"。
    """

    # 显式全库/跨课堂/跨会话标记优先归全库
    if _contains_any(
        text,
        (
            "课堂原声证据",
            "manifest",
            "版本不一致",
            "全库",
            "全量",
            "总体",
            "按学科统计",
            "跨课堂",
            "跨会话",
            "不同课堂",
            "cross session",
        ),
    ):
        return QueryScope.CORPUS
    # 群体/分层规律必然跨全库统计;
    if intent == QueryIntent.STUDENT_COHORT_PATTERNS:
        return QueryScope.CORPUS
    # 宏观图谱/统计在未限定学科过滤时属于全库
    if not subject_filters and _contains_any(text, ("macro graph", "图谱分析", "全图谱", "macro", "宏观")):
        return QueryScope.CORPUS
    # 单题辅导 + 有具体题目 token(题号/公式) -> 题目级;
    if intent in {QueryIntent.QUESTION_TUTORING, QueryIntent.TUTOR_STRATEGY_PATTERNS} and protected_tokens:
        return QueryScope.QUESTION
    # 案例化措辞优先归为 CASE: 包含情境问法与单节辅导回顾
    if _contains_any(text, CASE_SIGNALS):
        return QueryScope.CASE
    # 已被 Router 判定为单题辅导但没有显式公式时, 仍保持 QUESTION 粒度。
    # subject_filters 只是检索过滤条件, 不能反向把“怎么教这一题”扩大为主题报告。
    if intent == QueryIntent.QUESTION_TUTORING:
        return QueryScope.QUESTION
    # 用户明确点名考点时, 统计边界限定在该学科
    if subject_filters:
        return QueryScope.SUBJECT
    # 有具体 token 且带宏观聚合词 + 递归误区/错题两类 -> 收敛到学科级(避免全库空泛);
    if _has_specific_math_token(protected_tokens) and _contains_any(text, MACRO_SIGNALS) and intent in {
        QueryIntent.RECURRING_MISCONCEPTIONS,
        QueryIntent.EXAM_ERROR_PATTERNS,
    }:
        return QueryScope.SUBJECT
    # 明确指向某道题的话术;
    if _contains_any(text, ("这题", "这道题", "怎么算", "我选", "答案选项", "具体题", "题干", "this question")):
        return QueryScope.QUESTION
    # 点名的数学领域词 -> 学科级;
    if _contains_any(text, ("分数", "小数", "代数", "几何", "函数", "rounding", "fractions")):
        return QueryScope.SUBJECT
    # 宏观聚合词 -> 语料级;
    if _contains_any(text, MACRO_SIGNALS):
        return QueryScope.CORPUS
    # 兜底: 有具体 token 倾向是单题, 否则归全库。
    return QueryScope.QUESTION if protected_tokens else QueryScope.CORPUS


def _primary_intent(scores: dict[QueryIntent, int], text: str, protected_tokens: list[str]) -> tuple[QueryIntent, int, int]:
    # 关键捷径: 有具体公式/数字 + 直接解释型问句(为什么/如何/怎么) -> 判定为微观单题辅导,
    #   即使文本同时提到导师——先帮学生解题, 导师策略作为次意图。
    # 但出现种群级标记(Top-N、最常、反复…)时禁用该捷径, 保持为聚合分析。
    if (
        protected_tokens
        and not _contains_any(text, MACRO_SIGNALS)
        and _contains_any(text, ("为什么", "如何", "怎么", "what", "why", "how"))
    ):
        return QueryIntent.QUESTION_TUTORING, 4, 0
    # 无任何信号命中: 有具体 token 或解释问句则按单题处理, 否则判不支持。
    if not scores:
        if protected_tokens or _contains_any(text, ("为什么", "如何", "怎么", "what", "why", "how")):
            return QueryIntent.QUESTION_TUTORING, 1, 0
        return QueryIntent.UNSUPPORTED, 0, 0
    # 常规: 按(得分降序, 优先级升序)取主意图, 次高分作为 second_score 供置信度用。
    priority = {intent: index for index, intent in enumerate(_INTENT_PRIORITY)}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], priority.get(item[0], 999)))
    primary, primary_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    return primary, primary_score, second_score


def _need_llm_understanding(
    request: BusinessQueryRequest,
    protected_tokens: list[str],
    subject_filters: list[str],
    *,
    candidate_intents: list[QueryIntent] | None = None,
    matched_signals: list[str],
    routing_status: Literal["RESOLVED", "NEED_LLM", "UNSUPPORTED", "UNMEASURED"] = "NEED_LLM",
) -> QueryUnderstanding:
    """Return an explicit unresolved route when query requires second-layer LLM disambiguation."""
    ranking_signal, artifact_target, field_sources = _semantic_hints(request.query)
    field_sources["intent"] = "deterministic"
    field_sources["subject_filters"] = "explicit_term" if subject_filters else "none"
    return QueryUnderstanding(
        raw_query=request.query,
        primary_intent=QueryIntent.UNRESOLVED,
        secondary_intents=[],
        scope=QueryScope.CORPUS,
        computation_scope=ComputationScope.HYBRID,
        evidence_perspectives=[EvidencePerspective.STUDENT],
        subject_filters=subject_filters,
        protected_tokens=protected_tokens,
        requested_top_k=_requested_top_k(request),
        requires_aggregation=True,
        requires_case_retrieval=True,
        response_type=_RESPONSE_TYPES[QueryIntent.UNRESOLVED],
        confidence=RouteConfidence.LOW,
        route_source="deterministic",
        ambiguity_detected=True,
        matched_intent_signals=matched_signals,
        candidate_intents=candidate_intents or [],
        routing_status=routing_status,
        ranking_signal=ranking_signal,
        artifact_target=artifact_target,
        field_sources=field_sources,
    )


# 保留别名以兼容旧调用
_unmeasured_understanding = _need_llm_understanding


def _confidence(primary_score: int, second_score: int, intent: QueryIntent) -> RouteConfidence:
    """由"得分差"决定置信度: 主次意图差距越大越可信。"""

    if intent == QueryIntent.UNSUPPORTED:
        return RouteConfidence.LOW  # 不支持本身就是低置信, 供上游侧重判定
    lead = primary_score - second_score
    if primary_score >= 4 and lead >= 2:
        return RouteConfidence.HIGH
    if primary_score >= 3 and lead >= 1:
        return RouteConfidence.MEDIUM
    return RouteConfidence.LOW


def _perspectives(intent: QueryIntent, text: str) -> list[EvidencePerspective]:
    """决定需要哪些证据视角(学生/导师), 供检索阶段决定取回哪些角色对白。"""
    has_tutor = _contains_any(text, TUTOR_SIGNALS)
    has_student = _contains_any(text, STUDENT_SIGNALS)
    has_mixed = _contains_any(text, ("mixed", "双视角", "学生和导师", "学生与导师", "共现"))

    # 学生分层分析是纯学生群体视角
    if intent == QueryIntent.STUDENT_COHORT_PATTERNS:
        return [EvidencePerspective.STUDENT]

    # 策略类以导师视角为主; 若同时提到学生对白或标注混合则双视角
    if intent == QueryIntent.TUTOR_STRATEGY_PATTERNS:
        if has_student or has_mixed:
            return [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR]
        return [EvidencePerspective.TUTOR]

    # 内容改进: 结合学生错因与导师干预策略，默认双视角; 若纯学生误区统计则为学生视角
    if intent == QueryIntent.CONTENT_IMPROVEMENT:
        if has_tutor or has_mixed or _contains_any(text, ("干预", "intervention", "策略", "教学", "教材", "题库", "跨课堂", "跨会话")):
            return [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR]
        if has_student and not has_tutor:
            return [EvidencePerspective.STUDENT]
        return [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR]

    # 单题辅导: 纯导师动作则导师视角，纯学生推理/错因则学生视角，兼具或通用则双视角
    if intent == QueryIntent.QUESTION_TUTORING:
        # Explicit tutor-action questions remain tutor-scoped even when the
        # object is a student.  The requested evidence is how the tutor
        # guides, not the student's dialogue itself.
        if (has_tutor and _contains_any(text, ("导师如何", "导师怎么", "怎么引导", "如何引导", "导师如何处理"))
                and not _contains_any(text, ("学生为什么", "学生为何", "学生错因", "学生如何"))):
            return [EvidencePerspective.TUTOR]
        if has_mixed:
            return [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR]
        if has_tutor and not has_student and _contains_any(text, ("导师如何", "导师怎么", "怎么引导", "如何引导", "导师如何处理")):
            return [EvidencePerspective.TUTOR]
        if has_student and not has_tutor and _contains_any(text, ("student error", "学生为什么", "学生错因", "学生推理", "原话", "选项推理")):
            return [EvidencePerspective.STUDENT]
        return [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR]

    # 常见误区 / 困难概念 / 考试错误
    if intent in {QueryIntent.RECURRING_MISCONCEPTIONS, QueryIntent.EXAM_ERROR_PATTERNS, QueryIntent.DIFFICULT_CONCEPTS}:
        if has_mixed or (has_tutor and has_student):
            return [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR]
        if has_tutor and not has_student:
            return [EvidencePerspective.TUTOR]
        return [EvidencePerspective.STUDENT]

    # 通用后备: 包含双方则双视角，纯导师则导师，其余学生视角
    if has_mixed or (has_tutor and has_student):
        return [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR]
    if has_tutor and not has_student:
        return [EvidencePerspective.TUTOR]
    return [EvidencePerspective.STUDENT]


class BusinessQueryRouter:
    """业务查询的确定性一级路由: BusinessQueryRequest -> QueryUnderstanding。"""

    def __init__(self, llm_classifier: IntentClassifier | None = None) -> None:
        self.llm_classifier = llm_classifier

    def route(self, request: BusinessQueryRequest) -> QueryUnderstanding:
        """把结构化请求解读为查询理解; 纯规则、无网络调用、可单元测试。"""
        if not isinstance(request, BusinessQueryRequest):
            raise TypeError("request must be a BusinessQueryRequest")

        raw_query = request.query
        normalized_query = _normalized(raw_query)                      # 用于信号匹配
        protected_tokens = extract_protected_tokens(raw_query)        # 用于保留数学原文

        # 先处理业务边界已明确拒绝的请求。此处不调用评分、向量检索或 LLM,
        # 以免"相关"的导师策略/困难概念被误写成可证明的人员或因果结论。
        if _contains_any(normalized_query, UNSUPPORTED_SIGNALS):
            return QueryUnderstanding(
                raw_query=raw_query,
                primary_intent=QueryIntent.UNSUPPORTED,
                secondary_intents=[],
                scope=QueryScope.CORPUS,
                computation_scope=ComputationScope.REJECT,
                evidence_perspectives=[EvidencePerspective.STUDENT],
                subject_filters=_subject_filters(raw_query),
                protected_tokens=protected_tokens,
                requested_top_k=_requested_top_k(request),
                requires_aggregation=False,
                requires_case_retrieval=False,
                response_type=_RESPONSE_TYPES[QueryIntent.UNSUPPORTED],
                confidence=RouteConfidence.HIGH,
                route_source="deterministic",
                ambiguity_detected=False,
                matched_intent_signals=[],
                candidate_intents=[],
                routing_status="UNSUPPORTED",
            )

        scores = _score_intents(normalized_query)                     # 意图打分
        route_source = "deterministic"
        llm_secondary: QueryIntent | None = None
        strong_matches = _strong_intent_matches(normalized_query)
        ambiguity_detected = len(strong_matches) != 1
        routing_status: Literal["RESOLVED", "NEED_LLM", "UNSUPPORTED", "UNMEASURED"] = "RESOLVED"
        candidate_intents: list[QueryIntent] = []
        # Cross-session misconception/strategy co-occurrence is a content-
        # improvement question in the approved business matrix.
        cooccurrence_markers = (
            "共现", "共同出现", "伴随", "跨会话", "跨课堂", "不同课堂", "co-occurrence", "cooccur"
        )
        mixed_intervention = (
            _contains_any(normalized_query, STUDENT_SIGNALS)
            and _contains_any(normalized_query, TUTOR_SIGNALS)
            and _contains_any(normalized_query, ("干预", "intervention"))
        )

        competing_candidates: list[QueryIntent] = []
        if len(strong_matches) == 1:
            sole_strong = next(iter(strong_matches))
            for intent, score in scores.items():
                if intent == sole_strong or score < 2:
                    continue
                # When sole_strong is single-question tutoring, generic tutor strategy or intervention does not compete
                if sole_strong == QueryIntent.QUESTION_TUTORING and intent in {
                    QueryIntent.TUTOR_STRATEGY_PATTERNS,
                    QueryIntent.CONTENT_IMPROVEMENT,
                }:
                    continue
                competing_candidates.append(intent)

        if len(strong_matches) > 1 or competing_candidates:
            candidate_intents = (
                list(strong_matches.keys())
                if len(strong_matches) > 1
                else [next(iter(strong_matches))] + competing_candidates
            )
            if self.llm_classifier is None:
                return _need_llm_understanding(
                    request,
                    protected_tokens,
                    list(dict.fromkeys([*request.subject_filters, *_subject_filters(raw_query)])),
                    candidate_intents=candidate_intents,
                    matched_signals=[
                        signal for signals in strong_matches.values() for signal in signals
                    ] or [sig for intent in candidate_intents for sig in _matched_signals(normalized_query, INTENT_SIGNALS.get(intent, ()))],
                )
            else:
                # Candidate signals are diagnostic only; the second layer
                # must retain global visibility across all supported intents.
                candidates = tuple(
                    intent for intent in QueryIntent
                    if intent not in (QueryIntent.UNSUPPORTED, QueryIntent.UNRESOLVED)
                )
                try:
                    classification = self.llm_classifier.classify(raw_query, candidates)
                    primary = classification.primary_intent
                    primary_score = 4 if classification.confidence >= 0.8 else 3
                    second_score = 0
                    llm_secondary = classification.secondary_intent
                    route_source = "llm_classifier"
                    routing_status = "RESOLVED" if primary != QueryIntent.UNRESOLVED else "UNMEASURED"
                    candidate_intents = [primary]
                except Exception:
                    return _need_llm_understanding(
                        request,
                        protected_tokens,
                        list(dict.fromkeys([*request.subject_filters, *_subject_filters(raw_query)])),
                        candidate_intents=candidate_intents,
                        matched_signals=[
                            signal for signals in strong_matches.values() for signal in signals
                        ] or [sig for intent in candidate_intents for sig in _matched_signals(normalized_query, INTENT_SIGNALS.get(intent, ()))],
                        routing_status="UNMEASURED",
                    )
        elif mixed_intervention:
            if _contains_any(normalized_query, ("围绕真实题目", "真实题目", "这道题", "该题", "此题")) or protected_tokens:
                primary = QueryIntent.QUESTION_TUTORING
            else:
                primary = QueryIntent.CONTENT_IMPROVEMENT
            primary_score, second_score = 4, 0
            candidate_intents = [primary]
            routing_status = "RESOLVED"
        elif (_contains_any(normalized_query, ("反复出现", "反复发生", "recurring", "recur"))
              and _contains_any(normalized_query, ("误区", "误解", "misconception"))):
            primary = QueryIntent.RECURRING_MISCONCEPTIONS
            primary_score = max(scores.get(QueryIntent.RECURRING_MISCONCEPTIONS, 0), 4)
            second_score = 0
            candidate_intents = [primary]
            routing_status = "RESOLVED"
        elif any(marker in normalized_query for marker in cooccurrence_markers) and any(
            marker in normalized_query for marker in ("策略", "导师", "教学", "misconception", "strategy", "误区")
        ):
            primary = QueryIntent.CONTENT_IMPROVEMENT
            primary_score = max(scores.get(QueryIntent.CONTENT_IMPROVEMENT, 0), 4)
            second_score = 0
            candidate_intents = [primary]
            routing_status = "RESOLVED"
        elif len(strong_matches) == 1:
            primary = next(iter(strong_matches))
            primary_score = 4
            second_score = 0
            candidate_intents = [primary]
            routing_status = "RESOLVED"
        else:
            if not scores:
                # No deterministic intent signal: delegate to the configured
                # second-layer classifier instead of guessing QUESTION_TUTORING.
                # Explicit unsupported requests were rejected above.
                candidate_intents = []
                if self.llm_classifier is None:
                    return _need_llm_understanding(
                        request, protected_tokens,
                        list(dict.fromkeys([*request.subject_filters, *_subject_filters(raw_query)])),
                        candidate_intents=candidate_intents, matched_signals=[],
                    )
                else:
                    candidates = tuple(
                            intent for intent in QueryIntent if intent not in (QueryIntent.UNSUPPORTED, QueryIntent.UNRESOLVED)
                        )
                    try:
                            classification = self.llm_classifier.classify(raw_query, candidates)
                            primary = classification.primary_intent
                            primary_score = 4 if classification.confidence >= 0.8 else 3
                            second_score = 0
                            llm_secondary = classification.secondary_intent
                            route_source = "llm_classifier"
                            routing_status = "RESOLVED" if primary != QueryIntent.UNRESOLVED else "UNMEASURED"
                            candidate_intents = [primary]
                    except Exception:
                        return _need_llm_understanding(
                                request,
                                protected_tokens,
                                list(dict.fromkeys([*request.subject_filters, *_subject_filters(raw_query)])),
                                candidate_intents=list(candidates),
                                matched_signals=[],
                                routing_status="UNMEASURED",
                        )
            else:
                ordered = sorted(scores.items(), key=lambda item: (-item[1], _INTENT_PRIORITY.index(item[0]) if item[0] in _INTENT_PRIORITY else 999))
                top_score = ordered[0][1]
                if len(ordered) == 1 or (top_score - ordered[1][1] >= 2):
                    primary = ordered[0][0]
                    primary_score = top_score
                    second_score = ordered[1][1] if len(ordered) > 1 else 0
                    candidate_intents = [primary]
                    routing_status = "RESOLVED"
                else:
                    candidate_intents = [k for k, v in ordered if v >= top_score - 1]
                    if self.llm_classifier is None:
                        return _need_llm_understanding(
                            request,
                            protected_tokens,
                            list(dict.fromkeys([*request.subject_filters, *_subject_filters(raw_query)])),
                            candidate_intents=candidate_intents,
                            matched_signals=[],
                        )
                    else:
                        try:
                            all_supported = tuple(
                                intent for intent in QueryIntent
                                if intent not in (QueryIntent.UNSUPPORTED, QueryIntent.UNRESOLVED)
                            )
                            classification = self.llm_classifier.classify(raw_query, all_supported)
                            primary = classification.primary_intent
                            primary_score = 4 if classification.confidence >= 0.8 else 3
                            second_score = 0
                            llm_secondary = classification.secondary_intent
                            route_source = "llm_classifier"
                            routing_status = "RESOLVED" if primary != QueryIntent.UNRESOLVED else "UNMEASURED"
                            candidate_intents = [primary]
                        except Exception:
                            return _need_llm_understanding(
                                request,
                                protected_tokens,
                                list(dict.fromkeys([*request.subject_filters, *_subject_filters(raw_query)])),
                                candidate_intents=candidate_intents,
                                matched_signals=[],
                                routing_status="UNMEASURED",
                            )

        subject_filters = list(
            dict.fromkeys([*request.subject_filters, *_subject_filters(raw_query)])
        )
        scope = _scope_for(
            normalized_query,
            primary,
            protected_tokens,
            subject_filters,
        )

        # 计算范围/是否需要聚合: 与意图强绑定——
        #   单题辅导 = 微观细查, 不需要聚合, 但要取回案例对白;
        #   不支持  = 直接拒绝, 两者都不做;
        #   其余    = 宏观+微观混合: 既聚合定位, 又要真实对白支撑。
        if primary == QueryIntent.QUESTION_TUTORING or (
            scope in (QueryScope.CASE, QueryScope.QUESTION)
            and _contains_any(normalized_query, ("micro", "单案例", "单节", "单会话"))
        ):
            computation_scope = ComputationScope.MICRO
            requires_aggregation = False
            requires_case_retrieval = True
        elif primary == QueryIntent.UNSUPPORTED or (
            primary == QueryIntent.STUDENT_COHORT_PATTERNS
            and any(marker in normalized_query for marker in ("分数是否提高", "成绩是否提高", "提分", "提高多少分"))
        ):
            computation_scope = ComputationScope.REJECT
            requires_aggregation = False
            requires_case_retrieval = False
        else:
            computation_scope = ComputationScope.HYBRID
            requires_aggregation = True
            requires_case_retrieval = True

        # 次意图: 除"不支持/导师策略本身"外, 若文本带导师标记就补一个导师策略次意图
        # (内容改进类的双意图已由 _perspectives 覆盖, 不再重复追加)。
        secondary: list[QueryIntent] = []
        if llm_secondary is not None and llm_secondary != primary:
            secondary.append(llm_secondary)
        if primary not in {QueryIntent.UNSUPPORTED, QueryIntent.TUTOR_STRATEGY_PATTERNS}:
            if _contains_any(normalized_query, TUTOR_SIGNALS) and primary != QueryIntent.CONTENT_IMPROVEMENT:
                if QueryIntent.TUTOR_STRATEGY_PATTERNS not in secondary:
                    secondary.append(QueryIntent.TUTOR_STRATEGY_PATTERNS)

        confidence = _confidence(primary_score, second_score, primary)
        response_type = _RESPONSE_TYPES[primary]
        ranking_signal, artifact_target, field_sources = _semantic_hints(normalized_query)
        field_sources["intent"] = route_source
        field_sources["subject_filters"] = "explicit_term" if subject_filters else "none"
        return QueryUnderstanding(
            raw_query=raw_query,                                     # 原样保留, 供全链路溯源
            primary_intent=primary,
            secondary_intents=secondary[:1],                         # 契约约束: 至多 1 个次意图
            scope=scope,
            computation_scope=computation_scope,
            evidence_perspectives=_perspectives(primary, normalized_query),
            # 请求自带过滤 + 词表映射过滤, 合并去重。
            subject_filters=subject_filters,
            protected_tokens=protected_tokens,
            requested_top_k=_requested_top_k(request),
            requires_aggregation=requires_aggregation,
            requires_case_retrieval=requires_case_retrieval,
            response_type=response_type,
            confidence=confidence,
            route_source=route_source,                              # 规则或二层分类器
            ambiguity_detected=ambiguity_detected,
            matched_intent_signals=[
                signal for signals in strong_matches.values() for signal in signals
            ],
            candidate_intents=candidate_intents,
            routing_status=routing_status,
            ranking_signal=ranking_signal,
            artifact_target=artifact_target,
            field_sources=field_sources,
        )


# 公共导出面: 外部只与路由器打交道, 其余辅助函数视为内部实现。
__all__ = [
    "BusinessQueryRouter",
    "INTENT_SIGNALS",
    "STRONG_INTENT_SIGNALS",
    "extract_protected_tokens",
]
