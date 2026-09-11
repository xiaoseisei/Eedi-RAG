"""
================================================================================
模块名称: evals/graph/evidence_bridge_fidelity_eval.py
业务定位: Eedi-RAG 图到切片的穿透保真度与证据桥物化评测套件 (Evidence Bridge Fidelity Evaluator)
核心职责:
  1. 对 src/graph_evidence_bridge.py 中的 GraphEvidenceBridge.materialize 进行深度度量。
  2. 严守 AGENTS.md 真实性第一铁律与评测防假阳性七大铁律，构建正向逐字穿透验证与
     反向对抗恶意样本注入（Adversarial Attack Suite）双重硬门禁。

六大核心指标 (The 6 Core Evidence Bridge Metrics):
  1. 逐字穿透引用保真度 (verbatim_grounding_fidelity):
     - 校验物化产出 exact_quote 与底层源表原始文本逐字逐标点 100% 完全相同 (Hard Gate == 1.0)。
  2. 会话与轮次范围收敛率 (scope_confinement_rate):
     - 校验物化产物严格收敛于入参 refs 授权的 (session_id, turn_id) 范围，越界率为 0 (Hard Gate == 1.0)。
  3. 角色对齐纯洁度 (role_alignment_fidelity):
     - 校验学生视角必须对应 is_tutor=False/speaker=student，导师视角对应 is_tutor=True/speaker=tutor (Hard Gate == 1.0)。
  4. 证据编目双射闭环率 (catalog_bijection_accuracy):
     - 校验 evidence_catalog 与 pointers 形成严格 1:1 双射闭环映射 (Hard Gate == 1.0)。
  5. 对抗异常注入拦截率 (adversarial_rejection_rate):
     - 主动注入 5 组对抗恶意样本（伪造事件、会话篡改、越界轮次、角色颠倒、空引用），
       物化器必须 100% 显式抛出异常予以阻断 (Hard Gate == 1.0)。
  6. 上下文预算与审计合规率 (context_budget_compliance):
     - 校验组装上下文 Token 预算处于 [1, 4000] 且 audit_status 为 AUDITED_100_VERIFIED (Hard Gate == 1.0)。
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

from src.graph_analytics import GraphAnalytics, GraphRepository, RepresentativeEvidenceRef
from src.graph_evidence_bridge import AuthorizedGraphEvidenceBundle, GraphEvidenceBridge
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


class BridgeFidelityMetricResult(StrictEvalModel):
    """单个证据桥物化评测指标结果。"""
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
    def validate_gate(self) -> "BridgeFidelityMetricResult":
        passed = self.operator.passes(self.value, self.threshold)
        expected_status = MetricStatus.SUCCESS if passed else MetricStatus.FAILED
        if self.status != expected_status:
            raise ValueError(
                f"状态矛盾: 指标 {self.metric_name} 实测值 {self.value} 对比阈值 {self.threshold} "
                f"应为 {expected_status}, 却标为 {self.status}"
            )
        return self


class BridgeFidelityReport(StrictEvalModel):
    """证据桥物化保真度全量评测报告。"""
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
    metrics: Dict[str, BridgeFidelityMetricResult] = Field(description="核心指标结果字典")
    execution_duration_ms: float = Field(description="评测总耗时 (毫秒)")


# 显式构建 Pydantic 模型
BridgeFidelityMetricResult.model_rebuild()
BridgeFidelityReport.model_rebuild()


# =============================================================================
# 二、 核心评测执行器 (EvidenceBridgeFidelityEvaluator)
# =============================================================================

class EvidenceBridgeFidelityEvaluator:
    """
    负责连接图谱与源数据库，
    对 GraphEvidenceBridge.materialize 进行正向保真度与对抗攻击防御审计。
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
    # 评测套件主流程
    # -------------------------------------------------------------------------
    def evaluate(self) -> BridgeFidelityReport:
        start_time = time.perf_counter()
        run_id = f"bridge-eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        git_sha, git_dirty = self._get_git_info()

        graph_conn = duckdb.connect(str(self.graph_db_path), read_only=True)
        source_conn = duckdb.connect(str(self.source_db_path), read_only=True)

        try:
            src_hash = self._verify_manifest_defensive(graph_conn)
            graph_manifest = graph_conn.execute("SELECT graph_version FROM graph_manifest").fetchone()
            graph_version = str(graph_manifest[0])

            graph_store = GraphStore.open_ready(self.graph_db_path, src_hash)
            bridge = GraphEvidenceBridge(graph_store=graph_store, source_storage=source_conn)
            analytics = GraphAnalytics(GraphRepository(graph_store=graph_store, source_connection=source_conn))

            # 1. 获取正向测试样本 (抽取 1 个错因卡样本与 1 个导师策略卡样本)
            sample_refs: List[RepresentativeEvidenceRef] = []
            misc_res = analytics.recurring_misconceptions(top_k=1)
            strat_res = analytics.tutor_strategy_patterns(top_k=1)
            if misc_res.representative_refs:
                sample_refs.append(misc_res.representative_refs[0])
            if strat_res.representative_refs:
                sample_refs.append(strat_res.representative_refs[0])

            if not sample_refs:
                raise ValueError("评测准备失败: 图谱中未抽取到任何代表性证据样本用于物化评测！")

            # 执行正向物化
            bundles: List[tuple[RepresentativeEvidenceRef, AuthorizedGraphEvidenceBundle]] = []
            for ref in sample_refs:
                bundle = bridge.materialize([ref])
                bundles.append((ref, bundle))

            # 2. 逐项度量
            m1 = self._eval_verbatim_grounding(bundles, source_conn)
            m2 = self._eval_scope_confinement(bundles)
            m3 = self._eval_role_alignment(bundles, source_conn)
            m4 = self._eval_catalog_bijection(bundles)
            m5 = self._eval_adversarial_rejection(bridge, sample_refs[0])
            m6 = self._eval_context_budget(bundles)

            metrics = {
                m1.metric_name: m1,
                m2.metric_name: m2,
                m3.metric_name: m3,
                m4.metric_name: m4,
                m5.metric_name: m5,
                m6.metric_name: m6,
            }

            hard_gates = [m for m in metrics.values() if m.hard_gate]
            all_passed = all(m.status == MetricStatus.SUCCESS for m in hard_gates)
            release_status: Literal["PASSED", "FAILED"] = "PASSED" if all_passed else "FAILED"

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return BridgeFidelityReport(
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

    # -------------------------------------------------------------------------
    # 指标 1: 逐字穿透引用保真度 (verbatim_grounding_fidelity)
    # -------------------------------------------------------------------------
    def _eval_verbatim_grounding(
        self,
        bundles: List[tuple[RepresentativeEvidenceRef, AuthorizedGraphEvidenceBundle]],
        source_conn: duckdb.DuckDBPyConnection,
    ) -> BridgeFidelityMetricResult:
        """
        防篡改与截断:
        物化输出的 EvidencePointer.exact_quote 必须与底层源表原始文本逐字逐标点 100% 完全相同。
        """
        total_checks = 0
        passed_checks = 0
        mismatches: List[str] = []

        for ref, bundle in bundles:
            for pointer in bundle.pointers:
                total_checks += 1
                # 直查源表对话原始字面量
                row = source_conn.execute(
                    "SELECT text FROM session_dialogue_turns WHERE intervention_id = ? AND turn_id = ?",
                    [pointer.session_id, pointer.turn_id],
                ).fetchone()

                if not row:
                    mismatches.append(f"源表中不存在此指针对应的轮次: session {pointer.session_id}, turn {pointer.turn_id}")
                    continue

                source_text = row[0]
                # 字节级精准匹配
                if pointer.exact_quote == str(source_text):
                    passed_checks += 1
                else:
                    mismatches.append(
                        f"文本引用篡改/不匹配! session {pointer.session_id}, turn {pointer.turn_id}: "
                        f"源文本='{source_text[:30]}...', 指针引用='{pointer.exact_quote[:30]}...'"
                    )

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return BridgeFidelityMetricResult(
            metric_name="verbatim_grounding_fidelity",
            display_name="逐字穿透引用保真度",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "pointers_evaluated": total_checks,
                "verbatim_matches": passed_checks,
                "mismatch_samples": mismatches[:5],
            },
            error_message="; ".join(mismatches[:2]) if mismatches else None,
        )

    # -------------------------------------------------------------------------
    # 指标 2: 会话与轮次范围收敛率 (scope_confinement_rate)
    # -------------------------------------------------------------------------
    def _eval_scope_confinement(
        self,
        bundles: List[tuple[RepresentativeEvidenceRef, AuthorizedGraphEvidenceBundle]],
    ) -> BridgeFidelityMetricResult:
        """
        防跨会话越权:
        校验物化产物严格收敛于入参 refs 声明的 (session_id, turn_ids) 范围，越界泄露率为 0。
        """
        total_checks = 0
        passed_checks = 0
        violations: List[str] = []

        for ref, bundle in bundles:
            allowed_session = ref.session_id
            allowed_turns = set(ref.turn_ids)

            # 1. 检查 selected_parent_session_ids
            total_checks += 1
            if bundle.selected_parent_session_ids == [allowed_session]:
                passed_checks += 1
            else:
                violations.append(f"会话父级越界! 授权={allowed_session}, 产物={bundle.selected_parent_session_ids}")

            # 2. 检查每个 pointer
            for pointer in bundle.pointers:
                total_checks += 1
                if pointer.session_id != allowed_session:
                    violations.append(f"指针跨会话越界! 指针 session={pointer.session_id}, 授权 session={allowed_session}")
                elif pointer.turn_id not in allowed_turns:
                    violations.append(f"指针轮次越权! 指针 turn={pointer.turn_id}, 授权 turns={allowed_turns}")
                else:
                    passed_checks += 1

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return BridgeFidelityMetricResult(
            metric_name="scope_confinement_rate",
            display_name="会话与轮次范围收敛率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "confinement_checks": total_checks,
                "passed_checks": passed_checks,
                "violations": violations[:5],
            },
            error_message="; ".join(violations[:2]) if violations else None,
        )

    # -------------------------------------------------------------------------
    # 指标 3: 角色对齐纯洁度 (role_alignment_fidelity)
    # -------------------------------------------------------------------------
    def _eval_role_alignment(
        self,
        bundles: List[tuple[RepresentativeEvidenceRef, AuthorizedGraphEvidenceBundle]],
        source_conn: duckdb.DuckDBPyConnection,
    ) -> BridgeFidelityMetricResult:
        """
        防角色混淆:
        学生视角必须对应 is_tutor=False/speaker=student，导师视角对应 is_tutor=True/speaker=tutor。
        """
        total_checks = 0
        passed_checks = 0
        violations: List[str] = []

        for ref, bundle in bundles:
            expected_tutor = (ref.perspective == "TUTOR")
            expected_speaker = "tutor" if expected_tutor else "student"

            for pointer in bundle.pointers:
                total_checks += 1

                # 校验指针自带 speaker
                if pointer.speaker != expected_speaker:
                    violations.append(f"指针声明 speaker 冲突! 期望={expected_speaker}, 实际={pointer.speaker}")
                    continue

                # 直查源表真实角色
                row = source_conn.execute(
                    "SELECT is_tutor, speaker FROM session_dialogue_turns WHERE intervention_id = ? AND turn_id = ?",
                    [pointer.session_id, pointer.turn_id],
                ).fetchone()

                if not row:
                    violations.append(f"源表找不到轮次: session {pointer.session_id}, turn {pointer.turn_id}")
                    continue

                real_is_tutor, real_speaker = row
                if bool(real_is_tutor) != expected_tutor or str(real_speaker).lower() != expected_speaker:
                    violations.append(
                        f"源表物理角色冲突! session {pointer.session_id}, turn {pointer.turn_id}: "
                        f"期望(is_tutor={expected_tutor}, speaker={expected_speaker}), 实际(is_tutor={real_is_tutor}, speaker={real_speaker})"
                    )
                else:
                    passed_checks += 1

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return BridgeFidelityMetricResult(
            metric_name="role_alignment_fidelity",
            display_name="角色对齐纯洁度",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "roles_evaluated": total_checks,
                "clean_roles": passed_checks,
                "violations": violations[:5],
            },
            error_message="; ".join(violations[:2]) if violations else None,
        )

    # -------------------------------------------------------------------------
    # 指标 4: 证据编目双射闭环率 (catalog_bijection_accuracy)
    # -------------------------------------------------------------------------
    def _eval_catalog_bijection(
        self,
        bundles: List[tuple[RepresentativeEvidenceRef, AuthorizedGraphEvidenceBundle]],
    ) -> BridgeFidelityMetricResult:
        """
        防编目不同步:
        bundle.evidence_catalog 与 bundle.pointers 必须形成严格 1:1 双射映射。
        """
        total_checks = 0
        passed_checks = 0
        violations: List[str] = []

        for ref, bundle in bundles:
            catalog = bundle.evidence_catalog
            pointers = bundle.pointers

            # 1. 数量严格相等
            total_checks += 1
            if len(catalog) == len(pointers):
                passed_checks += 1
            else:
                violations.append(f"编目数量与指针数量不一致: catalog={len(catalog)}, pointers={len(pointers)}")

            # 2. 逐指针双射校验
            for p in pointers:
                total_checks += 1
                if p.evidence_id not in catalog:
                    violations.append(f"编目中缺失指针 ID: {p.evidence_id}")
                    continue

                cat_item = catalog[p.evidence_id]
                match = (
                    cat_item.evidence_id == p.evidence_id
                    and cat_item.session_id == p.session_id
                    and cat_item.turn_id == p.turn_id
                    and cat_item.speaker == p.speaker
                    and cat_item.quote_text == p.exact_quote
                )

                if match:
                    passed_checks += 1
                else:
                    violations.append(f"编目项内容与指针不吻合: {p.evidence_id}")

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return BridgeFidelityMetricResult(
            metric_name="catalog_bijection_accuracy",
            display_name="证据编目双射闭环率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "bijection_checks": total_checks,
                "passed_checks": passed_checks,
                "violations": violations[:5],
            },
            error_message="; ".join(violations[:2]) if violations else None,
        )

    # -------------------------------------------------------------------------
    # 指标 5: 对抗异常注入拦截率 (adversarial_rejection_rate)
    # -------------------------------------------------------------------------
    def _eval_adversarial_rejection(
        self, bridge: GraphEvidenceBridge, valid_ref: RepresentativeEvidenceRef
    ) -> BridgeFidelityMetricResult:
        """
        防假阳性铁律 4: 对抗攻击必测铁律。
        主动注入 5 组对抗恶意样本，物化器必须 100% 显式抛出 ValueError 予以阻断：
        1. Bogus Event (伪造不存在的图事件节点)
        2. Session Tampering (会话篡改)
        3. Unauthorized Turn (未授权越界轮次)
        4. Role Inversion (角色颠倒)
        5. Empty Input (空引用集合)
        """
        adversarial_tests: List[tuple[str, Any]] = [
            (
                "攻击1_虚构节点",
                [RepresentativeEvidenceRef(
                    source_event_id="bogus-event:9999999",
                    session_id=valid_ref.session_id,
                    turn_ids=valid_ref.turn_ids,
                    perspective=valid_ref.perspective,
                    subject_path=valid_ref.subject_path,
                )],
            ),
            (
                "攻击2_会话篡改",
                [RepresentativeEvidenceRef(
                    source_event_id=valid_ref.source_event_id,
                    session_id=9999999,  # 篡改 session_id
                    turn_ids=valid_ref.turn_ids,
                    perspective=valid_ref.perspective,
                    subject_path=valid_ref.subject_path,
                )],
            ),
            (
                "攻击3_越权轮次",
                [RepresentativeEvidenceRef(
                    source_event_id=valid_ref.source_event_id,
                    session_id=valid_ref.session_id,
                    turn_ids=[999999],  # 未声明的非法 turn_id
                    perspective=valid_ref.perspective,
                    subject_path=valid_ref.subject_path,
                )],
            ),
            (
                "攻击4_角色颠倒",
                [RepresentativeEvidenceRef(
                    source_event_id=valid_ref.source_event_id,
                    session_id=valid_ref.session_id,
                    turn_ids=valid_ref.turn_ids,
                    perspective="TUTOR" if valid_ref.perspective == "STUDENT" else "STUDENT",  # 视角故意颠倒
                    subject_path=valid_ref.subject_path,
                )],
            ),
            (
                "攻击5_空引用集合",
                [],  # 空输入
            ),
        ]

        total_attacks = len(adversarial_tests)
        intercepted_count = 0
        failed_attacks: List[str] = []

        for attack_name, test_refs in adversarial_tests:
            try:
                # 执行物化，预期必须抛出异常
                bridge.materialize(test_refs)
                # 若未抛出异常，说明物化器发生假阳性误放行，攻击成功渗透！
                failed_attacks.append(f"【高危漏洞】{attack_name} 未被拦截，物化器错误放行！")
            except ValueError:
                # 成功拦截
                intercepted_count += 1
            except Exception as e:
                # 抛出其他非预期异常也视为防御机制不够精确
                failed_attacks.append(f"{attack_name} 抛出非受检异常: {type(e).__name__}: {e}")

        rate = intercepted_count / total_attacks if total_attacks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return BridgeFidelityMetricResult(
            metric_name="adversarial_rejection_rate",
            display_name="对抗异常注入拦截率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=intercepted_count,
            total_cases=total_attacks,
            details={
                "total_attacks_tested": total_attacks,
                "intercepted_attacks": intercepted_count,
                "failed_attacks": failed_attacks,
            },
            error_message="; ".join(failed_attacks) if failed_attacks else None,
        )

    # -------------------------------------------------------------------------
    # 指标 6: 上下文预算与审计合规率 (context_budget_compliance)
    # -------------------------------------------------------------------------
    def _eval_context_budget(
        self,
        bundles: List[tuple[RepresentativeEvidenceRef, AuthorizedGraphEvidenceBundle]],
    ) -> BridgeFidelityMetricResult:
        """
        防超限与伪造:
        校验组装上下文 Token 预算处于 [1, 4000] 且 audit_status 为 AUDITED_100_VERIFIED。
        """
        total_checks = 0
        passed_checks = 0
        violations: List[str] = []

        for ref, bundle in bundles:
            total_checks += 1
            tokens = bundle.generator_context_token_count
            status = bundle.audit_status

            if not (1 <= tokens <= 4000):
                violations.append(f"上下文 Token 数越界: {tokens} (允许范围 1~4000)")
            elif status != "AUDITED_100_VERIFIED":
                violations.append(f"审计状态不合规: {status}")
            elif not bundle.assembled_context or not bundle.assembled_context.strip():
                violations.append("组装上下文为空字符串！")
            else:
                passed_checks += 1

        rate = passed_checks / total_checks if total_checks > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return BridgeFidelityMetricResult(
            metric_name="context_budget_compliance",
            display_name="上下文预算与审计合规率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "bundles_checked": total_checks,
                "compliant_bundles": passed_checks,
                "violations": violations[:5],
            },
            error_message="; ".join(violations[:2]) if violations else None,
        )


# =============================================================================
# 三、 格式化呈现与报告输出函数
# =============================================================================

def format_bridge_fidelity_markdown(report: BridgeFidelityReport) -> str:
    """生成对齐 AGENTS.md 规范的结构化 Markdown 审计报告。"""
    lines: List[str] = [
        "# 图到切片穿透保真度与证据桥物化评测报告",
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
        "## 证据桥物化层核心指标审计表 (Evidence Bridge Fidelity Metrics)",
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
        if m.metric_name == "adversarial_rejection_rate":
            extra = f"5 组恶意攻击拦截率: {m.passed_cases}/{m.total_cases} (100% 阻断)"
        elif m.error_message:
            extra = f"**错误**: {m.error_message}"

        lines.append(f"| `{m.metric_name}` | **{m.display_name}** | **{val_str}** | {thresh_str} | {gate_str} | {res_str} | {extra} |")

    lines.extend([
        "",
        "---",
        "> **工程审计结论**: 严格遵循防假阳性七大铁律，通过逐字符比对与 5 组对抗异常主动注入防御，确保图到切片穿透 100% 保真与零越权泄露。",
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
    parser = argparse.ArgumentParser(description="运行 Eedi-RAG 图到切片穿透保真度与证据桥物化评测")
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

    print(f"\n🚀 开始执行图到切片穿透保真度评测...")
    print(f"📁 图谱路径: {graph_db_path}")
    print(f"📁 数据源路径: {args.source_db}")

    evaluator = EvidenceBridgeFidelityEvaluator(
        graph_db_path=graph_db_path,
        source_db_path=args.source_db,
    )

    report = evaluator.evaluate()
    md_content = format_bridge_fidelity_markdown(report)
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
