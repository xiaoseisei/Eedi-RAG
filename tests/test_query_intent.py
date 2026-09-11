"""Part 1 Task 2: two-axis deterministic business query routing tests."""

import pytest

from src.business_models import (
    BusinessQueryRequest,
    ComputationScope,
    EvidencePerspective,
    QueryIntent,
    QueryScope,
)
from src.query_intent import BusinessQueryRouter
from src.query_intent import extract_protected_tokens


@pytest.mark.parametrize(
    ("query", "intent", "scope", "computation", "perspectives"),
    [
        (
            "学生在辅导中最常提出的问题是什么？列出10条",
            QueryIntent.STUDENT_FREQUENT_QUESTIONS,
            QueryScope.CORPUS,
            ComputationScope.HYBRID,
            [EvidencePerspective.STUDENT],
        ),
        (
            "导师反复使用哪些引导策略？",
            QueryIntent.TUTOR_STRATEGY_PATTERNS,
            QueryScope.CORPUS,
            ComputationScope.HYBRID,
            [EvidencePerspective.TUTOR],
        ),
        (
            "学生为什么把 5.4598 四舍五入成 5.45，导师如何引导？",
            QueryIntent.QUESTION_TUTORING,
            QueryScope.QUESTION,
            ComputationScope.MICRO,
            [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ),
        (
            "那些多次重考或基础薄弱的学生有什么共性模式？",
            QueryIntent.STUDENT_COHORT_PATTERNS,
            QueryScope.CORPUS,
            ComputationScope.HYBRID,
            [EvidencePerspective.STUDENT],
        ),
    ],
)
def test_router_models_two_independent_axes(
    query, intent, scope, computation, perspectives
) -> None:
    result = BusinessQueryRouter().route(BusinessQueryRequest(query=query, top_k=10))
    assert result.primary_intent == intent
    assert result.scope == scope
    assert result.computation_scope == computation
    assert result.evidence_perspectives == perspectives


def test_router_preserves_formula_and_requested_top_k() -> None:
    result = BusinessQueryRouter().route(
        BusinessQueryRequest(
            query="Top 5：为什么 (-q)^2 和 -q^2 容易混淆？",
            top_k=5,
        )
    )
    assert result.requested_top_k == 5
    assert "5" in result.protected_tokens
    assert "(-q)^2" in result.protected_tokens
    assert "-q^2" in result.protected_tokens


def test_router_uses_subject_scope_when_macro_question_names_a_topic() -> None:
    """明确考点的宏观问题应限定在该学科范围, 不能错误扩展到全库。"""

    result = BusinessQueryRouter().route(
        BusinessQueryRequest(query="分数加减中学生最常问什么？")
    )

    assert result.primary_intent == QueryIntent.STUDENT_FREQUENT_QUESTIONS
    assert result.scope == QueryScope.SUBJECT
    assert result.subject_filters == [
        "Number > Fractions and Decimals > Adding and Subtracting Fractions"
    ]


@pytest.mark.parametrize(
    ("query", "intent", "scope", "perspectives"),
    [
        (
            "What do students ask most often?",
            QueryIntent.STUDENT_FREQUENT_QUESTIONS,
            QueryScope.CORPUS,
            [EvidencePerspective.STUDENT],
        ),
        (
            "What tutor strategies recur most often?",
            QueryIntent.TUTOR_STRATEGY_PATTERNS,
            QueryScope.CORPUS,
            [EvidencePerspective.TUTOR],
        ),
        (
            "What is the most difficult concept in fractions?",
            QueryIntent.DIFFICULT_CONCEPTS,
            QueryScope.SUBJECT,
            [EvidencePerspective.STUDENT],
        ),
    ],
)
def test_router_supports_english_business_queries(
    query: str,
    intent: QueryIntent,
    scope: QueryScope,
    perspectives: list[EvidencePerspective],
) -> None:
    """英文业务问题与中文问题遵循同一意图、范围和证据视角边界。"""

    result = BusinessQueryRouter().route(BusinessQueryRequest(query=query))

    assert result.primary_intent == intent
    assert result.scope == scope
    assert result.evidence_perspectives == perspectives


def test_router_does_not_infer_user_role() -> None:
    result = BusinessQueryRouter().route(
        BusinessQueryRequest(query="我是主管，学生最常问什么？")
    )
    assert result.primary_intent == QueryIntent.STUDENT_FREQUENT_QUESTIONS
    assert not hasattr(result, "user_role")


@pytest.mark.parametrize(
    "query",
    [
        "谁是最优秀的导师？",
        "这个策略能提升多少分？",
        "NBCOT 最难的内容是什么？",
    ],
)
def test_router_marks_explicitly_unsupported_business_claims(query: str) -> None:
    """人员排序、因果分数与跨领域结论不能伪装成可执行的 RAG 问题。"""

    result = BusinessQueryRouter().route(BusinessQueryRequest(query=query))

    assert result.primary_intent == QueryIntent.UNSUPPORTED
    assert result.computation_scope == ComputationScope.REJECT
    assert result.evidence_perspectives == [EvidencePerspective.STUDENT]


@pytest.mark.parametrize(
    ("query", "intent", "scope", "perspectives"),
    [
        (
            "What are the most common student questions?",
            QueryIntent.STUDENT_FREQUENT_QUESTIONS,
            QueryScope.CORPUS,
            [EvidencePerspective.STUDENT],
        ),
        (
            "Top 5 difficult concepts",
            QueryIntent.DIFFICULT_CONCEPTS,
            QueryScope.CORPUS,
            [EvidencePerspective.STUDENT],
        ),
        (
            "重考学生有什么共性表现？",
            QueryIntent.STUDENT_COHORT_PATTERNS,
            QueryScope.CORPUS,
            [EvidencePerspective.STUDENT],
        ),
        (
            "What teaching strategies do tutors use most often?",
            QueryIntent.TUTOR_STRATEGY_PATTERNS,
            QueryScope.CORPUS,
            [EvidencePerspective.TUTOR],
        ),
        (
            "基于数据应该新增哪些教学内容？",
            QueryIntent.CONTENT_IMPROVEMENT,
            QueryScope.CORPUS,
            [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ),
        (
            "How should a tutor distinguish volume from surface area for a cuboid?",
            QueryIntent.QUESTION_TUTORING,
            QueryScope.QUESTION,
            [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ),
    ],
)
def test_router_handles_business_word_forms_without_falling_to_question_mode(
    query: str,
    intent: QueryIntent,
    scope: QueryScope,
    perspectives: list[EvidencePerspective],
) -> None:
    """英文复数、群体称谓与内容改进表达必须映射到对应业务路径。"""

    result = BusinessQueryRouter().route(BusinessQueryRequest(query=query))

    assert result.primary_intent == intent
    assert result.scope == scope
    assert result.evidence_perspectives == perspectives
    assert result.routing_status == "RESOLVED"


def test_conflicting_query_requires_llm_and_resolves_when_provided() -> None:
    """冲突查询必须标记为 NEED_LLM，配置 LLM 后成功消歧为 RESOLVED。"""
    query = "重考学生最常问什么？"
    rule_res = BusinessQueryRouter().route(BusinessQueryRequest(query=query))
    assert rule_res.routing_status == "NEED_LLM"
    assert rule_res.primary_intent == QueryIntent.UNRESOLVED
    assert QueryIntent.STUDENT_COHORT_PATTERNS in rule_res.candidate_intents
    assert QueryIntent.STUDENT_FREQUENT_QUESTIONS in rule_res.candidate_intents

    class MockClassifier:
        def classify(self, q, candidates):
            from src.intent_classifier import IntentClassification
            return IntentClassification(
                primary_intent=QueryIntent.STUDENT_COHORT_PATTERNS,
                confidence=0.9,
                rationale="用户关注的是重考学生群体的特殊提问，归属于学生群体模式",
            )

    llm_res = BusinessQueryRouter(llm_classifier=MockClassifier()).route(
        BusinessQueryRequest(query=query)
    )
    assert llm_res.routing_status == "RESOLVED"
    assert llm_res.primary_intent == QueryIntent.STUDENT_COHORT_PATTERNS


def test_router_does_not_treat_elapsed_time_as_clock_subject_filter() -> None:
    """“最费时间”是难度信号, 不是钟表/时间读法考点。"""

    result = BusinessQueryRouter().route(
        BusinessQueryRequest(query="哪些概念让学生最费时间？")
    )

    assert result.primary_intent == QueryIntent.DIFFICULT_CONCEPTS
    assert result.scope == QueryScope.CORPUS
    assert result.subject_filters == []


def test_protected_tokens_keep_compound_math_and_clock_expressions() -> None:
    """保护 token 应保留完整数学/钟表表达, 不把英语 I 识别成数学变量。"""

    assert extract_protected_tokens("为什么 (54+58)/2 不能先算 58/2？") == [
        "(54+58)/2",
        "58/2",
    ]
    assert extract_protected_tokens("学生把 10 to 12 读成 10:12") == [
        "10",
        "12",
        "10:12",
    ]
    assert extract_protected_tokens("How do I compare option A and option B for 5/7-1/4?") == [
        "option A",
        "option B",
        "5/7-1/4",
    ]
