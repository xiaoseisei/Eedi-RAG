"""
================================================================================
测试模块: tests/test_query_planner.py
业务定位: 验证业务查询计划是否正确区分宏观聚合、微观 RAG 与数据能力拒绝

本测试关注 Query Planner 的三条边界:
  1. 高频问题必须先走 SQL 聚合, 再用学生视角证据支撑;
  2. 具体公式问题只走微观双视角 RAG, 不要求 analytics sidecar;
  3. 当前缺少学生纵向字段时, 重考者问题必须显式 REJECT;
     严禁静默退化成普通 misconception 检索。
================================================================================
"""

from src.business_models import (
    BusinessQueryRequest,
    QueryPlanType,
)
from src.data_capabilities import DataCapabilityRegistry
from src.query_intent import BusinessQueryRouter
from src.query_planner import BusinessQueryPlanner
from src.business_models import QueryUnderstanding


def _caps(dialogue=True, analytics=True, retake=False, ability=False):
    """构造仅供 Planner 单测使用的能力快照, 不读取生产资产。"""

    return DataCapabilityRegistry.synthetic_snapshot(
        dialogue=dialogue,
        analytics=analytics,
        retake=retake,
        ability=ability,
    )


def test_frequent_question_plan_uses_sql_then_student_evidence() -> None:
    """高频问题必须是宏观聚合 + 学生原声证据的混合计划。"""

    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="学生最常提出的问题是什么？", top_k=10)
    )
    plan = BusinessQueryPlanner().plan(understanding, _caps())

    assert plan.plan_type == QueryPlanType.ANALYTICS_WITH_EVIDENCE
    assert plan.analytics_operations == ["student_frequent_questions"]
    assert [item.value for item in plan.retrieval_perspectives] == ["STUDENT"]


def test_current_eedi_retaker_question_is_rejected_before_retrieval_or_llm() -> None:
    """没有 attempt_number 时, 重考者群体问题必须在执行前拒绝。"""

    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="重考者最常见的思维卡点是什么？")
    )
    plan = BusinessQueryPlanner().plan(understanding, _caps(retake=False))

    assert plan.plan_type == QueryPlanType.REJECT
    assert plan.analytics_operations == []
    assert plan.retrieval_perspectives == []
    assert "attempt_number" in plan.rejection_reason


def test_specific_question_uses_existing_micro_rag_without_analytics() -> None:
    """具体数学问题只需要对白证据能力, 不应强制 analytics sidecar。"""

    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="为什么 (-q)^2 和 -q^2 不同？导师怎么引导？")
    )
    plan = BusinessQueryPlanner().plan(understanding, _caps(analytics=False))

    assert plan.plan_type == QueryPlanType.SOCRATIC_TUTORING
    assert plan.analytics_operations == []


def test_unmeasurable_duration_ranking_is_rejected_explicitly() -> None:
    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="那个卡的时间最长")
    )
    plan = BusinessQueryPlanner().plan(understanding, _caps())
    assert plan.plan_type == QueryPlanType.REJECT
    assert plan.response_type == "UNMEASURED"
    assert "duration_desc" in plan.rejection_reason


def test_unknown_artifact_target_is_not_defaulted_to_a_card_table() -> None:
    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="学生最常问的问题是什么？")
    ).model_copy(update={"artifact_target": "unknown_card"})
    plan = BusinessQueryPlanner().plan(understanding, _caps())
    assert plan.plan_type == QueryPlanType.REJECT
    assert plan.response_type == "UNMEASURED"
