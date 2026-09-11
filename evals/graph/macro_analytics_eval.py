"""
================================================================================
模块名称: evals/graph/macro_analytics_eval.py
业务定位: Eedi-RAG 知识图谱宏观统计与拓扑查询准确度评测套件 (Macro Analytics Evaluator)
核心职责:
  1. 对 src/graph_analytics.py 中的 6 大宏观聚合操作进行确定性数学核验。
  2. 严守 AGENTS.md 真实性第一铁律与评测防假阳性七大铁律，建立双轨独立原生 SQL 金标引擎，
     杜绝“自验自证”、“宽容放行”与“学科越界泄露”。

五大核心指标 (The 5 Core Macro Metrics):
  1. 聚合结果与真值一致率 (aggregation_parity):
     - 评测独立 SQL 引擎算出的 Top-K 结果与业务代码是否 100% 一致 (Hard Gate == 1.0)。
  2. 学科过滤纯洁度与零泄露率 (subject_filter_accuracy):
     - 校验学科切片下匹配 session 规模精准性，且代表性证据零学科泄露 (Hard Gate == 1.0)。
  3. 共现拓扑会话同源精度 (cooccurrence_topology_precision):
     - 校验错因-策略对是否 100% 诞生于同一 session，杜绝跨会话拓扑幻觉 (Hard Gate == 1.0)。
  4. 代表性证据引用闭环率 (representative_evidence_integrity):
     - 校验代表性证据在 graph_nodes、session_id、turn_ids 及说话人角色上的闭环有效性 (Hard Gate == 1.0)。
  5. 未分类数据透明度 (unclassified_visibility_rate):
     - 校验 UNCLASSIFIED 数据与不可度量指标是否如实透明暴露 (Hard Gate == 1.0)。
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.graph_analytics import GraphAnalytics, GraphRepository
from src.graph_store import GraphStore


# =============================================================================
# 一、 基础数据结构与评估契约
# =============================================================================

class StrictEvalModel(BaseModel):
    """严格契约基类: 禁止未知多余字段与弱类型放行。"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetricStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ERROR = "ERROR"


class ComparisonOperator(str, Enum):
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"

    def passes(self, value: float, threshold: float) -> bool:
        if self is ComparisonOperator.GTE:
            return value >= threshold or math.isclose(value, threshold, rel_tol=1e-6, abs_tol=1e-6)
        if self is ComparisonOperator.LTE:
            return value <= threshold or math.isclose(value, threshold, rel_tol=1e-6, abs_tol=1e-6)
        return math.isclose(value, threshold, rel_tol=1e-6, abs_tol=1e-6)


class MacroAnalyticsMetricResult(StrictEvalModel):
    """单个宏观分析评测指标结果。"""
    metric_name: str = Field(description="指标唯一标识名")
    display_name: str = Field(description="中文人类可读名称")
    value: float = Field(description="实测数值")
    threshold: float = Field(description="门禁阈值")
    operator: ComparisonOperator = Field(description="比较运算符")
    status: MetricStatus = Field(description="判定状态 (SUCCESS/FAILED)")
    hard_gate: bool = Field(description="是否为硬门禁")
    passed_cases: int = Field(description="通过样本数")
    total_cases: int = Field(description="总样本数")
    details: Dict[str, Any] = Field(default_factory=dict, description="结构化诊断细节")
    error_message: Optional[str] = Field(default=None, description="失败原因摘要")

    @model_validator(mode="after")
    def validate_gate(self) -> "MacroAnalyticsMetricResult":
        passed = self.operator.passes(self.value, self.threshold)
        expected_status = MetricStatus.SUCCESS if passed else MetricStatus.FAILED
        if self.status != expected_status:
            raise ValueError(
                f"状态矛盾: 指标 {self.metric_name} 实测值 {self.value} 对比阈值 {self.threshold} "
                f"应为 {expected_status}, 却标为 {self.status}"
            )
        return self


class MacroAnalyticsReport(StrictEvalModel):
    """宏观统计与拓扑查询准确度全量评测报告。"""
    run_id: str = Field(description="本次评测唯一 Run ID")
    evaluated_at_utc: str = Field(description="评测执行 UTC 时间戳")
    git_commit_sha: str = Field(description="当前代码 Git Commit SHA")
    git_tree_dirty: bool = Field(description="当前工作区是否存在未提交修改")
    graph_version: str = Field(description="图谱版本标识")
    graph_db_path: str = Field(description="图谱 DuckDB 文件路径")
    source_db_path: str = Field(description="数据源 DuckDB 文件路径")
    release_status: Literal["PASSED", "FAILED"] = Field(description="总体发布状态")
    hard_gates_passed: bool = Field(description="是否全部硬门禁通过")
    total_metrics_evaluated: int = Field(description="评估指标总数")
    passed_metrics_count: int = Field(description="通过的指标总数")
    failed_metrics_count: int = Field(description="失败的指标总数")
    metrics: Dict[str, MacroAnalyticsMetricResult] = Field(description="核心指标结果字典")
    execution_duration_ms: float = Field(description="评测总耗时 (毫秒)")


# 显式构建 Pydantic 模型
MacroAnalyticsMetricResult.model_rebuild()
MacroAnalyticsReport.model_rebuild()


# =============================================================================
# 二、 核心评测执行器 (MacroAnalyticsEvaluator)
# =============================================================================

class MacroAnalyticsEvaluator:
    """
    负责连接只读图谱 DuckDB 与源数据库 DuckDB，
    运行双轨独立金标 SQL，对 GraphAnalytics 执行无假阳性的数学核验。
    """

    def __init__(self, graph_db_path: str | Path, source_db_path: str | Path) -> None:
        self.graph_db_path = Path(graph_db_path).resolve()
        self.source_db_path = Path(source_db_path).resolve()

        if not self.graph_db_path.exists():
            raise FileNotFoundError(f"图谱数据库文件不存在: {self.graph_db_path}")
        if not self.source_db_path.exists():
            raise FileNotFoundError(f"数据源数据库文件不存在: {self.source_db_path}")

    @staticmethod
    def _get_git_info() -> tuple[str, bool]:
        """获取当前工作区 Git 提交与干净状态，实现审计可追溯。"""
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
            ).decode().strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
            ).decode().strip()
            return sha, len(status) > 0
        except Exception:
            return "UNKNOWN", True

    def _verify_manifest_defensive(self, graph_conn: duckdb.DuckDBPyConnection) -> str:
        """
        防假阳性铁律 6: 防御式元数据单行断言。
        强制校验 graph_manifest 必须恰好只有 1 条记录，且状态为 READY。
        """
        rows = graph_conn.execute(
            "SELECT graph_version, source_artifact_hash, build_status FROM graph_manifest"
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(f"Manifest 契约违规: 必须且仅能有 1 行记录，实测存在 {len(rows)} 行")
        version, src_hash, status = rows[0]
        if status != "READY":
            raise ValueError(f"图谱构建未就绪: build_status 为 {status}")
        return str(src_hash)

    # -------------------------------------------------------------------------
    # 指标 1: 聚合结果与真值一致率 (aggregation_parity)
    # -------------------------------------------------------------------------
    def _eval_aggregation_parity(
        self, analytics: GraphAnalytics, graph_conn: duckdb.DuckDBPyConnection, source_conn: duckdb.DuckDBPyConnection
    ) -> MacroAnalyticsMetricResult:
        """
        防假阳性铁律 3: 独立金标双算铁律。
        评测器直接在底层只读库上执行独立的纯原生 SQL，与业务 API 的 Top-K 逐字段比对。
        """
        operations = [
            ("frequent_student_questions", "QUESTION_FUNCTION", analytics.frequent_student_questions),
            ("difficult_concepts", "DIFFICULTY_SIGNAL", analytics.difficult_concepts),
            ("recurring_misconceptions", "MISCONCEPTION", analytics.recurring_misconceptions),
            ("exam_error_patterns", "EXAM_ERROR", analytics.exam_error_patterns),
            ("tutor_strategy_patterns", "TUTOR_STRATEGY", analytics.tutor_strategy_patterns),
        ]

        total_checks = 0
        passed_checks = 0
        mismatches: List[str] = []

        total_sessions = source_conn.execute("SELECT COUNT(DISTINCT intervention_id) FROM tutoring_sessions").fetchone()[0]

        for op_name, label_type, func in operations:
            # 1. 业务函数调用
            biz_res = func(top_k=5)

            # 2. 独立金标 SQL 计算 (完全独立重写，不调用业务私有逻辑)
            gold_sql = """
                SELECT a.canonical_label AS canonical_name,
                       COUNT(*) AS event_count,
                       COUNT(DISTINCT a.source_session_id) AS distinct_session_count
                FROM graph_label_assignments a
                WHERE a.label_type = ?
                GROUP BY a.canonical_label
                ORDER BY distinct_session_count DESC, event_count DESC, canonical_name ASC
                LIMIT 5
            """
            gold_rows = graph_conn.execute(gold_sql, [label_type]).fetchall()

            # 3. 逐行比对 (rank, canonical_name, event_count, distinct_session_count, session_share)
            for i, gold_r in enumerate(gold_rows, 1):
                total_checks += 1
                if i > len(biz_res.rows):
                    mismatches.append(f"[{op_name}] 业务结果缺失 Rank {i} 行")
                    continue

                biz_r = biz_res.rows[i - 1]
                gold_name, gold_event_cnt, gold_session_cnt = gold_r
                gold_share = gold_session_cnt / total_sessions if total_sessions > 0 else 0.0

                match = (
                    biz_r["rank"] == i
                    and biz_r["canonical_name"] == gold_name
                    and biz_r["event_count"] == gold_event_cnt
                    and biz_r["distinct_session_count"] == gold_session_cnt
                    and math.isclose(biz_r["session_share"], gold_share, rel_tol=1e-5, abs_tol=1e-5)
                )

                if match:
                    passed_checks += 1
                else:
                    mismatches.append(
                        f"[{op_name}] Rank {i} 不吻合: 期望(name={gold_name}, events={gold_event_cnt}, sessions={gold_session_cnt}), "
                        f"实际(name={biz_r['canonical_name']}, events={biz_r['event_count']}, sessions={biz_r['distinct_session_count']})"
                    )

            # 双向核对：业务侧额外生成的行同样是契约违规，不能只验证
            # 独立 SQL 返回的行而放过多余结果。
            if len(biz_res.rows) > len(gold_rows):
                for i in range(len(gold_rows) + 1, len(biz_res.rows) + 1):
                    total_checks += 1
                    extra = biz_res.rows[i - 1]
                    mismatches.append(
                        f"[{op_name}] 业务结果多出 Rank {i} 行: "
                        f"实际(name={extra.get('canonical_name')}, events={extra.get('event_count')}, "
                        f"sessions={extra.get('distinct_session_count')})"
                    )

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return MacroAnalyticsMetricResult(
            metric_name="aggregation_parity",
            display_name="聚合结果与真值一致率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "operations_tested": [op[0] for op in operations],
                "total_rows_evaluated": total_checks,
                "passed_rows": passed_checks,
                "mismatches": mismatches[:5],
            },
            error_message="; ".join(mismatches[:2]) if mismatches else None,
        )

    # -------------------------------------------------------------------------
    # 指标 2: 学科过滤纯洁度与零泄露率 (subject_filter_accuracy)
    # -------------------------------------------------------------------------
    def _eval_subject_filter_accuracy(
        self, analytics: GraphAnalytics, source_conn: duckdb.DuckDBPyConnection
    ) -> MacroAnalyticsMetricResult:
        """
        校验在学科切片过滤下，匹配的 session 规模必须精准，且抽取的代表性证据必须 100% 属于该学科。
        """
        test_filters = [
            ["Number"],
            ["Algebra"],
            ["NonExistentSubjectForTesting"],  # 边界对抗：空匹配
        ]

        total_checks = 0
        passed_checks = 0
        violations: List[str] = []

        for s_filter in test_filters:
            # 1. 源数据库独立计算匹配数
            cond = " AND " + " AND ".join("s.subject_path LIKE ?" for _ in s_filter)
            params = [f"%{v}%" for v in s_filter]
            gold_matched = source_conn.execute(
                f"SELECT COUNT(DISTINCT s.intervention_id) FROM tutoring_sessions s WHERE 1=1 {cond}", params
            ).fetchone()[0]

            # 2. 业务调用
            res = analytics.recurring_misconceptions(top_k=3, subject_filters=s_filter)

            # 核验匹配规模一致性
            total_checks += 1
            if res.data_scope.matched_sessions == gold_matched:
                passed_checks += 1
            else:
                violations.append(f"学科过滤 {s_filter} matched_sessions 不一致: 期望 {gold_matched}, 实际 {res.data_scope.matched_sessions}")

            # 核验代表性证据的学科归属零泄露
            for ref in res.representative_refs:
                total_checks += 1
                # 查询源表真实 subject_path
                real_subject = source_conn.execute(
                    "SELECT subject_path FROM tutoring_sessions WHERE intervention_id = ?", [ref.session_id]
                ).fetchone()

                if not real_subject or not all(f in real_subject[0] for f in s_filter):
                    violations.append(f"跨学科泄露! session {ref.session_id} 真实学科 {real_subject} 不满足过滤器 {s_filter}")
                else:
                    passed_checks += 1

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return MacroAnalyticsMetricResult(
            metric_name="subject_filter_accuracy",
            display_name="学科过滤纯洁度与零泄露率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "filters_tested": test_filters,
                "violations_count": len(violations),
                "violation_samples": violations[:5],
            },
            error_message="; ".join(violations[:2]) if violations else None,
        )

    # -------------------------------------------------------------------------
    # 指标 3: 共现拓扑会话同源精度 (cooccurrence_topology_precision)
    # -------------------------------------------------------------------------
    def _eval_cooccurrence_topology(
        self, analytics: GraphAnalytics, graph_conn: duckdb.DuckDBPyConnection
    ) -> MacroAnalyticsMetricResult:
        """
        核验 misconception_strategy_pairs：
        返回的每一对 (misconception, tutor_strategy) 必须真实产生于同一个 session，
        杜绝跨 session 幻觉配对。
        """
        res = analytics.misconception_strategy_pairs(top_k=5)

        # 独立 SQL 计算共现对
        gold_sql = """
            SELECT m.canonical_label AS misc,
                   t.canonical_label AS strat,
                   COUNT(*) AS event_count,
                   COUNT(DISTINCT m.source_session_id) AS distinct_session_count
            FROM graph_label_assignments m
            JOIN graph_label_assignments t ON m.source_session_id = t.source_session_id
            WHERE m.label_type = 'MISCONCEPTION' AND t.label_type = 'TUTOR_STRATEGY'
            GROUP BY m.canonical_label, t.canonical_label
            ORDER BY distinct_session_count DESC, event_count DESC, misc ASC, strat ASC
            LIMIT 5
        """
        gold_pairs = graph_conn.execute(gold_sql).fetchall()

        total_checks = 0
        passed_checks = 0
        violations: List[str] = []

        for i, gold_p in enumerate(gold_pairs, 1):
            total_checks += 1
            if i > len(res.rows):
                violations.append(f"共现结果缺失 Rank {i}")
                continue

            biz_p = res.rows[i - 1]
            gold_misc, gold_strat, gold_ev_cnt, gold_sess_cnt = gold_p

            match = (
                biz_p["rank"] == i
                and biz_p["misconception"] == gold_misc
                and biz_p["strategy"] == gold_strat
                and biz_p["event_count"] == gold_ev_cnt
                and biz_p["distinct_session_count"] == gold_sess_cnt
            )

            if match:
                passed_checks += 1
            else:
                violations.append(
                    f"共现对 Rank {i} 不一致: 期望({gold_misc}, {gold_strat}, ev={gold_ev_cnt}, sess={gold_sess_cnt}), "
                    f"实际({biz_p['misconception']}, {biz_p['strategy']}, ev={biz_p['event_count']}, sess={biz_p['distinct_session_count']})"
                )

        # 进一步核验代表性证据是否确实存在于同一 session
        for ref in res.representative_refs:
            total_checks += 1
            # 查该 session 是否同时有错因与策略标签
            has_both = graph_conn.execute("""
                SELECT COUNT(DISTINCT label_type)
                FROM graph_label_assignments
                WHERE source_session_id = ? AND label_type IN ('MISCONCEPTION', 'TUTOR_STRATEGY')
            """, [ref.session_id]).fetchone()[0]

            if has_both == 2:
                passed_checks += 1
            else:
                violations.append(f"代表性证据 session {ref.session_id} 并非同时包含错因与策略！")

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return MacroAnalyticsMetricResult(
            metric_name="cooccurrence_topology_precision",
            display_name="共现拓扑会话同源精度",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "pairs_evaluated": len(gold_pairs),
                "refs_evaluated": len(res.representative_refs),
                "violations": violations[:5],
            },
            error_message="; ".join(violations[:2]) if violations else None,
        )

    # -------------------------------------------------------------------------
    # 指标 4: 代表性证据引用闭环率 (representative_evidence_integrity)
    # -------------------------------------------------------------------------
    def _eval_representative_evidence_integrity(
        self, analytics: GraphAnalytics, graph_conn: duckdb.DuckDBPyConnection, source_conn: duckdb.DuckDBPyConnection
    ) -> MacroAnalyticsMetricResult:
        """
        防假阳性铁律 1: 三方闭环内连接原则。
        全量核验所有操作抽取的 RepresentativeEvidenceRef：
        1. source_event_id 在 graph_nodes 中真实存在；
        2. session_id 与事件严格一致；
        3. turn_ids 在事件声明范围且在源对话表中真实存在；
        4. 说话人角色与 perspective (STUDENT/TUTOR) 100% 吻合。
        """
        all_results = [
            analytics.frequent_student_questions(top_k=3),
            analytics.difficult_concepts(top_k=3),
            analytics.recurring_misconceptions(top_k=3),
            analytics.tutor_strategy_patterns(top_k=3),
            analytics.misconception_strategy_pairs(top_k=3),
        ]

        total_checks = 0
        passed_checks = 0
        violations: List[str] = []

        for res in all_results:
            for ref in res.representative_refs:
                total_checks += 1
                ref_valid = True

                # 1. 查 graph_nodes 节点存在性与 session 一致性
                node_row = graph_conn.execute(
                    "SELECT source_session_id, source_turn_ids, node_type FROM graph_nodes WHERE node_id = ?",
                    [ref.source_event_id],
                ).fetchone()

                if not node_row:
                    violations.append(f"虚构节点! source_event_id {ref.source_event_id} 不在 graph_nodes 中")
                    ref_valid = False
                    continue

                node_sid, node_turns, node_type = node_row
                if int(node_sid) != int(ref.session_id):
                    violations.append(f"会话错位! ref session={ref.session_id}, 节点真实 session={node_sid}")
                    ref_valid = False
                    continue

                if not set(ref.turn_ids).issubset(set(node_turns)):
                    violations.append(f"轮次越界! ref turns={ref.turn_ids}, 节点声明 turns={node_turns}")
                    ref_valid = False
                    continue

                # 2. 批量查 source_conn 该 session 的对白真实存在性与角色 (避免逐轮循环单行查询)
                session_turns = {
                    r[0]: (r[1], r[2], r[3])
                    for r in source_conn.execute(
                        "SELECT turn_id, is_tutor, text, is_greeting_or_noise FROM session_dialogue_turns WHERE intervention_id = ?",
                        [ref.session_id],
                    ).fetchall()
                }

                for tid in ref.turn_ids:
                    if tid not in session_turns:
                        violations.append(f"源表无此轮次: session {ref.session_id}, turn {tid}")
                        ref_valid = False
                        break

                    is_tutor, text, is_noise = session_turns[tid]
                    # 强角色校验：错因事件必须为学生，策略事件必须为导师，单个轮次按 perspective 校验
                    if node_type == "MISCONCEPTION_EVENT" and bool(is_tutor) is True:
                        violations.append(f"角色冲突! 错因卡包含导师发言: session {ref.session_id}, turn {tid}")
                        ref_valid = False
                        break
                    elif node_type == "TUTOR_STRATEGY_EVENT" and bool(is_tutor) is False:
                        violations.append(f"角色冲突! 策略卡包含学生发言: session {ref.session_id}, turn {tid}")
                        ref_valid = False
                        break
                    elif node_type == "TURN":
                        expected_tutor = (ref.perspective == "TUTOR")
                        if bool(is_tutor) != expected_tutor:
                            violations.append(f"角色冲突! ref视角={ref.perspective}, 但源表 is_tutor={is_tutor}")
                            ref_valid = False
                            break

                    # 实质性教学对白文本严禁为空 (源数据中已标为 noise 的除外)
                    if (not text or not text.strip()) and not is_noise:
                        violations.append(f"空文本实质性证据! session {ref.session_id}, turn {tid}")
                        ref_valid = False
                        break

                if ref_valid:
                    passed_checks += 1

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return MacroAnalyticsMetricResult(
            metric_name="representative_evidence_integrity",
            display_name="代表性证据引用闭环率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "total_refs_checked": total_checks,
                "passed_refs": passed_checks,
                "violations_count": len(violations),
                "violation_samples": violations[:5],
            },
            error_message="; ".join(violations[:2]) if violations else None,
        )

    # -------------------------------------------------------------------------
    # 指标 5: 未分类数据透明度 (unclassified_visibility_rate)
    # -------------------------------------------------------------------------
    def _eval_unclassified_visibility(self, analytics: GraphAnalytics) -> MacroAnalyticsMetricResult:
        """
        防假阳性铁律 2: 闭合失败与透明度铁律。
        校验当底层存在 UNCLASSIFIED 时，宏观分析是否如实保留，且不可度量指标如实声明。
        """
        res = analytics.recurring_misconceptions(top_k=10)

        total_checks = 2
        passed_checks = 0
        details: Dict[str, Any] = {}

        # 1. 检查 unmeasured_metrics 是否如实公开 UNRESOLVED_ENDING
        if "UNRESOLVED_ENDING" in res.data_scope.unmeasured_metrics:
            passed_checks += 1
            details["unmeasured_metrics_declared"] = True
        else:
            details["unmeasured_metrics_declared"] = False

        # 2. 检查 UNCLASSIFIED 是否未被静默吞掉
        has_unclassified = any(r["canonical_name"] == "UNCLASSIFIED" for r in res.rows)
        # 在本数据集中已知存在 UNCLASSIFIED
        if has_unclassified:
            passed_checks += 1
            details["unclassified_visible"] = True
        else:
            # 未分类节点未出现在结果中时，缺失证据必须失败，不能静默放行。
            details["unclassified_visible"] = False

        rate = passed_checks / total_checks
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return MacroAnalyticsMetricResult(
            metric_name="unclassified_visibility_rate",
            display_name="未分类数据透明度",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details=details,
            error_message=None if status == MetricStatus.SUCCESS else "未分类或不可度量维度被隐瞒",
        )

    # -------------------------------------------------------------------------
    # 全量评测执行
    # -------------------------------------------------------------------------
    def evaluate(self) -> MacroAnalyticsReport:
        start_time = time.perf_counter()
        run_id = f"macro-eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        git_sha, git_dirty = self._get_git_info()

        # 1. 打开只读连接并执行防御性校验
        graph_conn = duckdb.connect(str(self.graph_db_path), read_only=True)
        source_conn = duckdb.connect(str(self.source_db_path), read_only=True)

        try:
            src_hash = self._verify_manifest_defensive(graph_conn)
            graph_manifest = graph_conn.execute("SELECT graph_version FROM graph_manifest").fetchone()
            graph_version = str(graph_manifest[0])

            # 2. 组装只读被测对象
            graph_store = GraphStore.open_ready(self.graph_db_path, src_hash)
            repo = GraphRepository(graph_store=graph_store, source_connection=source_conn)
            analytics = GraphAnalytics(repo)

            # 3. 执行五大核心评测
            m1 = self._eval_aggregation_parity(analytics, graph_conn, source_conn)
            m2 = self._eval_subject_filter_accuracy(analytics, source_conn)
            m3 = self._eval_cooccurrence_topology(analytics, graph_conn)
            m4 = self._eval_representative_evidence_integrity(analytics, graph_conn, source_conn)
            m5 = self._eval_unclassified_visibility(analytics)

            metrics = {
                m1.metric_name: m1,
                m2.metric_name: m2,
                m3.metric_name: m3,
                m4.metric_name: m4,
                m5.metric_name: m5,
            }

            # 4. 发布裁决
            hard_gates = [m for m in metrics.values() if m.hard_gate]
            all_passed = all(m.status == MetricStatus.SUCCESS for m in hard_gates)
            release_status: Literal["PASSED", "FAILED"] = "PASSED" if all_passed else "FAILED"

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return MacroAnalyticsReport(
                run_id=run_id,
                evaluated_at_utc=datetime.now(UTC).isoformat(),
                git_commit_sha=git_sha,
                git_tree_dirty=git_dirty,
                graph_version=graph_version,
                graph_db_path=str(self.graph_db_path),
                source_db_path=str(self.source_db_path),
                release_status=release_status,
                hard_gates_passed=all_passed,
                total_metrics_evaluated=len(metrics),
                passed_metrics_count=sum(1 for m in metrics.values() if m.status == MetricStatus.SUCCESS),
                failed_metrics_count=sum(1 for m in metrics.values() if m.status == MetricStatus.FAILED),
                metrics=metrics,
                execution_duration_ms=round(duration_ms, 2),
            )
        finally:
            graph_conn.close()
            source_conn.close()


# =============================================================================
# 三、 格式化呈现与报告输出函数
# =============================================================================

def format_macro_analytics_markdown(report: MacroAnalyticsReport) -> str:
    """生成对齐 AGENTS.md 规范的 Markdown 审计报告。"""
    lines: List[str] = [
        "# 知识图谱宏观统计与拓扑查询准确度评测报告",
        "",
        f"> **评测编号 (Run ID)**: `{report.run_id}`  ",
        f"> **执行时间 (UTC)**: `{report.evaluated_at_utc}`  ",
        f"> **Git 审计锚点**: `{report.git_commit_sha}` ({'工作区未提交Dirty' if report.git_tree_dirty else 'Clean'})  ",
        f"> **图谱版本**: `{report.graph_version}`  ",
        f"> **发布裁决 (Release Status)**: **{report.release_status}** ({'全部硬门禁通过' if report.hard_gates_passed else '存在硬门禁未达标'})  ",
        f"> **评测耗时**: `{report.execution_duration_ms} ms`  ",
        "",
        "---",
        "",
        "## 宏观分析层核心指标审计表 (Macro Analytics Health Metrics)",
        "",
        "| 指标标识 | 中文指标名称 | 实测值 | 门禁阈值 | 门禁性质 | 判定结果 | 样本与核验说明 |",
        "|---|---|:---:|:---:|:---:|:---:|---|",
    ]

    for m in report.metrics.values():
        val_str = f"{m.value:.4%}"
        thresh_str = f"{m.threshold:.4%}"
        gate_str = "**Hard Gate**" if m.hard_gate else "Soft Gate"
        res_str = "**SUCCESS**" if m.status == MetricStatus.SUCCESS else "**FAILED**"
        extra = f"核验通过: {m.passed_cases:,} / {m.total_cases:,}"
        if m.error_message:
            extra = f"**错误**: {m.error_message}"

        lines.append(f"| `{m.metric_name}` | **{m.display_name}** | **{val_str}** | {thresh_str} | {gate_str} | {res_str} | {extra} |")

    lines.extend([
        "",
        "---",
        "> **工程审计结论**: 严格遵循防假阳性七大铁律，采用双轨独立 SQL 金标比对，严禁自验自证与单侧宽松放行。",
    ])

    return "\n".join(lines)


# =============================================================================
# 四、 命令行入口与默认执行
# =============================================================================

def find_latest_graph_db(staging_dir: Path) -> Path:
    candidates = glob.glob(str(staging_dir / "graph*.duckdb"))
    if not candidates:
        raise FileNotFoundError(f"在 {staging_dir} 中未找到任何 graph*.duckdb 文件")
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行 Eedi-RAG 知识图谱宏观统计与拓扑查询准确度评测")
    parser.add_argument(
        "--graph-db",
        type=Path,
        default=None,
        help="待评估图谱 DuckDB 路径 (默认自动寻找 reports/staging 最新构建)",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb",
        help="权威数据源 DuckDB 路径 (默认 data/db/tutoring_knowledge.duckdb)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选输出完整 JSON 评估报告路径",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="可选输出 Markdown 格式评估报告路径",
    )
    args = parser.parse_args(argv)

    graph_db_path = args.graph_db
    if graph_db_path is None:
        staging_dir = PROJECT_ROOT / "reports" / "staging" / "session-card-graph-v1"
        graph_db_path = find_latest_graph_db(staging_dir)

    print(f"\n🚀 开始执行宏观统计与拓扑查询准确度评测...")
    print(f"📁 图谱路径: {graph_db_path}")
    print(f"📁 数据源路径: {args.source_db}")

    evaluator = MacroAnalyticsEvaluator(
        graph_db_path=graph_db_path,
        source_db_path=args.source_db,
    )

    report = evaluator.evaluate()
    md_content = format_macro_analytics_markdown(report)
    print("\n" + md_content + "\n")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"✅ JSON 报告已保存至: {args.output_json}")

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_content, encoding="utf-8")
        print(f"✅ Markdown 报告已保存至: {args.output_md}")

    return 0 if report.release_status == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
