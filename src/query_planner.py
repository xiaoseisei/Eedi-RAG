"""
================================================================================
模块名称: src/query_planner.py
业务定位: 将查询理解结果编排为可执行的业务 Query Plan (Query Planner)

本模块位于 Query Router 与具体执行器之间, 不负责执行 SQL、向量检索或 LLM:
  1. 读取 Router 产出的 intent / scope / computation_scope / evidence_perspectives;
  2. 根据显式计划矩阵选择宏观分析、分析+证据、微观辅导或内容简报路径;
  3. 先检查 DataCapabilitySnapshot, 缺少必要字段时返回 REJECT;
  4. 保留可审计的 analytics_operations 与 retrieval_perspectives, 供下游执行。

核心设计原则:
  - 统计问题不能降级为 vector-only: 所有 requires_aggregation 计划必须列出 SQL 操作;
  - 具体题目不能强依赖 analytics sidecar: QUESTION_TUTORING 只要求对白证据能力;
  - 群体问题不能使用相邻数据冒充: retaker / ability 能力缺失时明确拒绝执行;
  - 计划只是"怎么查"的声明, 不在这里读取 Gold/qrels 或调用外部模型。
================================================================================
"""

from __future__ import annotations

from typing import Mapping

from src.business_models import (
    DataCapabilitySnapshot,
    EvidencePerspective,
    QueryIntent,
    QueryPlan,
    QueryPlanType,
    QueryUnderstanding,
)


# ---------------------------------------------------------------------------
# 显式计划矩阵: 每种业务意图对应固定的分析操作、证据视角和响应类型
# ---------------------------------------------------------------------------


PLAN_MATRIX: dict[
    QueryIntent,
    tuple[
        QueryPlanType,
        list[str],
        list[EvidencePerspective],
        list[str],
        str,
    ],
] = {
    # 学生视角宏观频率: SQL 先统计, RAG 再提供代表性学生原声。
    QueryIntent.STUDENT_FREQUENT_QUESTIONS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["student_frequent_questions"],
        [EvidencePerspective.STUDENT],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "STUDENT_FAQ_REPORT",
    ),
    # 难点需要学生困惑与导师处理两种证据, 但排名仍由分析层决定。
    QueryIntent.DIFFICULT_CONCEPTS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["difficult_concepts"],
        [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "DIFFICULTY_REPORT",
    ),
    # 共性误区与考试错误均从学生视角聚合。
    QueryIntent.RECURRING_MISCONCEPTIONS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["recurring_misconceptions"],
        [EvidencePerspective.STUDENT],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "MISCONCEPTION_REPORT",
    ),
    QueryIntent.EXAM_ERROR_PATTERNS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["exam_error_patterns"],
        [EvidencePerspective.STUDENT],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "EXAM_ERROR_PATTERN_REPORT",
    ),
    # 导师策略宏观报告只执行 Tutor lane, 避免无关学生卡片污染统计。
    QueryIntent.TUTOR_STRATEGY_PATTERNS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["tutor_strategy_patterns"],
        [EvidencePerspective.TUTOR],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "TUTOR_STRATEGY_REPORT",
    ),
    # 内容建议需要学生需求与导师策略两侧的分析/证据共同支撑。
    QueryIntent.CONTENT_IMPROVEMENT: (
        QueryPlanType.CONTENT_BRIEF,
        ["content_opportunities"],
        [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "CONTENT_IMPROVEMENT_BRIEF",
    ),
    # 具体问题直接进入既有微观 RAG/辅导执行器, 不强制 analytics sidecar。
    QueryIntent.QUESTION_TUTORING: (
        QueryPlanType.SOCRATIC_TUTORING,
        [],
        [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ["DIALOGUE_EVIDENCE"],
        "SOCRATIC_TURN",
    ),
}


class BusinessQueryPlanner:
    """依据固定矩阵和能力快照生成可审计的 QueryPlan。"""

    @staticmethod
    def _capability(
        capabilities: Mapping[str, DataCapabilitySnapshot],
        name: str,
    ) -> DataCapabilitySnapshot:
        """读取能力快照; 缺少注册项本身视为能力缺失, 不使用宽松默认值。"""

        snapshot = capabilities.get(name)
        if snapshot is None:
            raise ValueError(f"capability snapshot missing required entry: {name}")
        return snapshot

    @classmethod
    def _missing_requirements(
        cls,
        capabilities: Mapping[str, DataCapabilitySnapshot],
        names: list[str],
    ) -> list[str]:
        """收集所有不可用能力的去重缺失项, 保持输出顺序稳定。"""

        missing: list[str] = []
        for name in names:
            snapshot = cls._capability(capabilities, name)
            if snapshot.available:
                continue
            for requirement in snapshot.missing_requirements or [name]:
                if requirement not in missing:
                    missing.append(requirement)
        return missing

    @staticmethod
    def _cohort_capabilities(understanding: QueryUnderstanding) -> list[str]:
        """根据原始问题中的明确群体条件选择所需纵向数据能力。"""

        query = understanding.raw_query.casefold()
        # Any cohort claim requires a stable longitudinal cohort capability;
        # without it, even wording such as "long-term gain" must be rejected
        # explicitly instead of falling through to an unimplemented aggregate.
        required: list[str] = ["RETAKE_COHORT"]
        if any(token in query for token in ("重考", "retake", "再考")):
            required.append("RETAKE_COHORT")
        if any(token in query for token in ("基础薄弱", "低基础", "weak foundation", "low ability")):
            required.append("ABILITY_COHORT")
        return required

    def plan(
        self,
        understanding: QueryUnderstanding,
        capabilities: Mapping[str, DataCapabilitySnapshot],
    ) -> QueryPlan:
        """生成单一明确计划; 能力不足时只返回带原因的 REJECT。"""

        if not isinstance(understanding, QueryUnderstanding):
            raise TypeError("understanding must be a QueryUnderstanding")
        if not isinstance(capabilities, Mapping):
            raise TypeError("capabilities must be a mapping of snapshots")

        # Ranking and artifact hints are orthogonal to intent.  Never compile
        # an unsupported metric into an arbitrary SQL operator.
        if understanding.ranking_signal == "duration_desc":
            return QueryPlan(
                understanding=understanding,
                plan_type=QueryPlanType.REJECT,
                analytics_operations=[],
                retrieval_perspectives=[],
                required_capabilities=[],
                response_type="UNMEASURED",
                rejection_reason=(
                    "排序指标 duration_desc 当前无可审计的源字段/统计算子，"
                    "Planner 拒绝猜测或改用其他指标。"
                ),
            )
        if understanding.artifact_target not in {None, "misconception_card", "strategy_card", "question_card"}:
            return QueryPlan(
                understanding=understanding,
                plan_type=QueryPlanType.REJECT,
                analytics_operations=[],
                retrieval_perspectives=[],
                required_capabilities=[],
                response_type="UNMEASURED",
                rejection_reason="artifact_target 未被支持的执行器注册，无法安全选择卡片来源。",
            )

        intent = understanding.primary_intent
        if intent == QueryIntent.STUDENT_COHORT_PATTERNS:
            # Cohort intent 需要先根据原始条件决定 retake/ability/both 门禁。
            required_capabilities = ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"]
            required_capabilities.extend(self._cohort_capabilities(understanding))
            missing = self._missing_requirements(capabilities, required_capabilities)
            if missing:
                reason = (
                    "当前数据能力不足以执行学生群体分析; "
                    f"缺失字段/资产: {', '.join(missing)}"
                )
                return QueryPlan(
                    understanding=understanding,
                    plan_type=QueryPlanType.REJECT,
                    analytics_operations=[],
                    retrieval_perspectives=[],
                    required_capabilities=required_capabilities,
                    response_type="UNSUPPORTED",
                    rejection_reason=reason,
                )
            return QueryPlan(
                understanding=understanding,
                plan_type=QueryPlanType.ANALYTICS_WITH_EVIDENCE,
                analytics_operations=["student_cohort_patterns"],
                retrieval_perspectives=[EvidencePerspective.STUDENT],
                required_capabilities=required_capabilities,
                response_type="STUDENT_COHORT_PATTERN_REPORT",
            )

        matrix_entry = PLAN_MATRIX.get(intent)
        if matrix_entry is None:
            # Router 已明确标记 UNSUPPORTED 或出现未来未登记意图时, 保持 fail-fast。
            reason = f"query intent {intent.value} has no executable plan"
            return QueryPlan(
                understanding=understanding,
                plan_type=QueryPlanType.REJECT,
                analytics_operations=[],
                retrieval_perspectives=[],
                required_capabilities=[],
                response_type="UNSUPPORTED",
                rejection_reason=reason,
            )

        plan_type, operations, perspectives, required_capabilities, response_type = matrix_entry
        missing = self._missing_requirements(capabilities, required_capabilities)
        if missing:
            reason = (
                f"执行 {intent.value} 所需数据能力不可用; "
                f"缺失字段/资产: {', '.join(missing)}"
            )
            return QueryPlan(
                understanding=understanding,
                plan_type=QueryPlanType.REJECT,
                analytics_operations=[],
                retrieval_perspectives=[],
                required_capabilities=required_capabilities,
                response_type="UNSUPPORTED",
                rejection_reason=reason,
            )

        return QueryPlan(
            understanding=understanding,
            plan_type=plan_type,
            analytics_operations=list(operations),
            retrieval_perspectives=list(perspectives),
            required_capabilities=list(required_capabilities),
            response_type=response_type,
        )


__all__ = ["BusinessQueryPlanner", "PLAN_MATRIX"]
