"""
================================================================================
模块名称: evals/graph/Topology _Construction_eval.py
业务定位: Eedi-RAG 静态图谱结构与拓扑健康度离线评测套件 (Offline Graph Health Evaluator)
核心职责:
  1. 在任何线上/离线查询之前，对已构建的 DuckDB Session-Card 知识图谱执行核心无模型依赖、
     纯确定性数学统计的静态健康度量化度量与硬门禁（Hard Gate）核验。
  2. 严守 AGENTS.md 工程诚实第一铁律与真实性规范，所有统计基于底层主外键与指纹哈希，
     绝不伪造指标、绝不放行悬空指针与脏引用。

核心可量化指标 (Core Quantifiable Metrics):
  1. 指针权威有效率 (pointer_validity):
     - 检验所有 EVIDENCED_BY 边指向的 (intervention_id, turn_id) 是否 100% 存在于源对话表。
     - 门禁性质: Hard Gate (阈值: 1.0 / 100%)。
  2. 角色归属纯洁度 (role_accuracy):
     - 检验错因卡 (MISCONCEPTION_EVENT) 仅指向学生轮次 (is_tutor=False)，
       策略卡 (TUTOR_STRATEGY_EVENT) 仅指向导师轮次 (is_tutor=True)。
     - 门禁性质: Hard Gate (阈值: 1.0 / 100%)。
  3. 孤立节点率 (orphan_node_ratio):
     - 检验全量节点中出度与入度之和为 0 的孤立脱节节点占比。
     - 门禁性质: Hard Gate (阈值: 0.0 / 0%)。
  4. 分类标签覆盖率 (taxonomy_coverage):
     - 度量全量卡片中成功打上规范业务分类标签的比例 vs UNCLASSIFIED 盲区比例。
     - 门禁性质: Soft Gate (持续监控与演进指导指标)。
  5. 构图确定性哈希一致率 (reproducibility_hash_match):
     - 校验源 DuckDB 文件真实 SHA-256 与构建清单 manifest 记录的哈希及节点/边规模一致性。
     - 门禁性质: Hard Gate (阈值: 1.0 / 100%)。
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import tempfile
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
    UNMEASURED = "UNMEASURED"
    BLOCKED = "BLOCKED"


class ComparisonOperator(str, Enum):
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"

    def passes(self, value: float, threshold: float) -> bool:
        if self is ComparisonOperator.GTE:
            return value >= threshold
        if self is ComparisonOperator.LTE:
            return value <= threshold
        return math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-9)


class TopologyMetricResult(StrictEvalModel):
    """单个静态拓扑健康度指标评测结果。"""
    metric_name: str = Field(description="指标唯一标识名")
    display_name: str = Field(description="中文人类可读名称")
    value: float = Field(description="实测数值")
    threshold: float = Field(description="门禁阈值")
    operator: ComparisonOperator = Field(description="比较运算符 (gte/lte/eq)")
    status: MetricStatus = Field(description="判定状态 (SUCCESS/FAILED)")
    hard_gate: bool = Field(description="是否为硬门禁 (Hard Gate)")
    passed_cases: int = Field(description="通过样本数")
    total_cases: int = Field(description="总样本数")
    details: Dict[str, Any] = Field(default_factory=dict, description="结构化细节指标与诊断数据")
    error_message: Optional[str] = Field(default=None, description="失败原因摘要")

    @model_validator(mode="after")
    def validate_gate(self) -> "TopologyMetricResult":
        if self.status in {MetricStatus.UNMEASURED, MetricStatus.BLOCKED, MetricStatus.ERROR}:
            return self
        passed = self.operator.passes(self.value, self.threshold)
        expected_status = MetricStatus.SUCCESS if passed else MetricStatus.FAILED
        if self.status != expected_status:
            raise ValueError(f"状态矛盾: 指标 {self.metric_name} 实测值 {self.value} 对比阈值 {self.threshold} 应为 {expected_status}, 却标为 {self.status}")
        return self


class TopologyHealthReport(StrictEvalModel):
    """静态图谱结构与拓扑健康度全量评测报告。"""
    run_id: str = Field(description="本次评测唯一 Run ID")
    evaluated_at_utc: str = Field(description="评测执行 UTC 时间戳")
    graph_version: str = Field(description="图谱版本标识")
    graph_db_path: str = Field(description="图谱 DuckDB 文件绝对/相对路径")
    source_db_path: str = Field(description="数据源 DuckDB 文件绝对/相对路径")
    release_status: Literal["PASSED", "FAILED"] = Field(description="总体硬门禁发布状态")
    hard_gates_passed: bool = Field(description="是否全部硬门禁通过")
    total_metrics_evaluated: int = Field(description="评估指标总数")
    passed_metrics_count: int = Field(description="通过的指标总数")
    failed_metrics_count: int = Field(description="失败的指标总数")
    metrics: Dict[str, TopologyMetricResult] = Field(description="拓扑核心指标详细结果字典")
    graph_overview: Dict[str, Any] = Field(description="图谱基本拓扑规模全景概览")
    execution_duration_ms: float = Field(description="评测总耗时 (毫秒)")


# 显式构建 Pydantic 命名空间，确保在动态加载及 Python 3.14 下完全解析类型引用
TopologyMetricResult.model_rebuild()
TopologyHealthReport.model_rebuild()

# =============================================================================

class TopologyConstructionEvaluator:
    """
    负责连接图谱 DuckDB 与底层数据源 DuckDB，
    执行 5 项纯确定性的拓扑图完整性与数据契约审计。
    """

    def __init__(self, graph_db_path: str | Path, source_db_path: str | Path) -> None:
        self.graph_db_path = Path(graph_db_path).resolve()
        self.source_db_path = Path(source_db_path).resolve()

        if not self.graph_db_path.exists():
            raise FileNotFoundError(f"图谱数据库文件不存在: {self.graph_db_path}")
        if not self.source_db_path.exists():
            raise FileNotFoundError(f"数据源数据库文件不存在: {self.source_db_path}")

    @staticmethod
    def _sha256_file(filepath: Path) -> str:
        """分块流式计算大文件 SHA-256，避免内存暴涨。"""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _graph_content_sha256(graph_conn: duckdb.DuckDBPyConnection) -> str:
        """按稳定行序计算图内容指纹，不包含构建时间等易变 manifest 字段。"""

        payload: Dict[str, Any] = {}
        for table in ("graph_nodes", "graph_edges", "graph_label_assignments"):
            cursor = graph_conn.execute(f"SELECT * FROM {table} ORDER BY 1")
            columns = [item[0] for item in cursor.description]
            table_rows: List[Dict[str, Any]] = []
            for row in cursor.fetchall():
                normalized: Dict[str, Any] = {}
                for column, value in zip(columns, row):
                    if isinstance(value, (list, tuple)):
                        normalized[column] = list(value)
                    elif isinstance(value, datetime):
                        normalized[column] = value.isoformat()
                    else:
                        normalized[column] = value
                    if column == "properties_json" and isinstance(value, str):
                        try:
                            normalized[column] = json.loads(value)
                        except json.JSONDecodeError:
                            raise ValueError(f"graph node properties_json is not valid JSON: {value[:80]}")
                table_rows.append(normalized)
            payload[table] = {"columns": columns, "rows": table_rows}
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _independent_rebuild_content_sha256(self, graph_version: str) -> str:
        """在临时目录重建图并返回内容指纹，避免把当前图当作自己的 Gold。"""

        from scripts.build_session_card_graph import build

        with tempfile.TemporaryDirectory(prefix="eedi-topology-rebuild-") as temp_dir:
            rebuilt_path = Path(temp_dir) / "graph.duckdb"
            build(self.source_db_path, rebuilt_path, graph_version)
            rebuilt_conn = duckdb.connect(str(rebuilt_path), read_only=True)
            try:
                return self._graph_content_sha256(rebuilt_conn)
            finally:
                rebuilt_conn.close()

    def evaluate(self, run_id: Optional[str] = None) -> TopologyHealthReport:
        """运行全套静态拓扑指标审计，生成权威评估报告。"""
        start_time = time.perf_counter()
        active_run_id = run_id or f"topo-health-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

        # 1. 建立只读连接
        graph_conn = duckdb.connect(str(self.graph_db_path), read_only=True)
        source_conn = duckdb.connect(str(self.source_db_path), read_only=True)

        try:
            # 2. 提取基础全景信息
            manifest_rows = graph_conn.execute(
                "SELECT graph_version, source_artifact_hash, node_count, edge_count, build_status FROM graph_manifest"
            ).fetchall()
            if len(manifest_rows) != 1:
                raise ValueError(f"图谱数据库损坏: graph_manifest 必须恰好一行，实际 {len(manifest_rows)} 行")
            manifest_row = manifest_rows[0]
            if manifest_row[4] != "READY":
                raise ValueError(f"图谱数据库不可评测: build_status 必须为 READY，实际为 {manifest_row[4]}")
            graph_version, recorded_src_hash, manifest_node_count, manifest_edge_count, build_status = manifest_row

            total_nodes = graph_conn.execute("SELECT count(*) FROM graph_nodes").fetchone()[0]
            total_edges = graph_conn.execute("SELECT count(*) FROM graph_edges").fetchone()[0]

            node_type_counts = dict(
                graph_conn.execute("SELECT node_type, count(*) FROM graph_nodes GROUP BY node_type ORDER BY 2 DESC").fetchall()
            )
            edge_type_counts = dict(
                graph_conn.execute("SELECT edge_type, count(*) FROM graph_edges GROUP BY edge_type ORDER BY 2 DESC").fetchall()
            )

            graph_overview = {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "manifest_node_count": manifest_node_count,
                "manifest_edge_count": manifest_edge_count,
                "build_status": build_status,
                "node_types": node_type_counts,
                "edge_types": edge_type_counts,
            }

            # 3. 逐项执行核心结构、来源、分类和复现指标
            m1 = self._eval_pointer_validity(graph_conn, source_conn)
            m2 = self._eval_role_accuracy(graph_conn, source_conn)
            m3 = self._eval_orphan_nodes(graph_conn)
            m4 = self._eval_edge_endpoint_integrity(graph_conn)
            m5 = self._eval_taxonomy_label_validity(graph_conn)
            m6 = self._eval_taxonomy_coverage(graph_conn)
            m7 = self._eval_projection_completeness(graph_conn, source_conn)
            rebuilt_graph_hash = self._independent_rebuild_content_sha256(graph_version)
            m8 = self._eval_reproducibility_hash(
                graph_conn,
                self.source_db_path,
                recorded_src_hash,
                total_nodes,
                total_edges,
                manifest_node_count,
                manifest_edge_count,
                build_status,
                rebuilt_graph_content_hash=rebuilt_graph_hash,
            )

            metrics: Dict[str, TopologyMetricResult] = {
                m1.metric_name: m1,
                m2.metric_name: m2,
                m3.metric_name: m3,
                m4.metric_name: m4,
                m5.metric_name: m5,
                m6.metric_name: m6,
                m7.metric_name: m7,
                m8.metric_name: m8,
            }

            # 4. 硬门禁汇总判定
            hard_gates = [m for m in metrics.values() if m.hard_gate]
            hard_gates_passed = all(m.status == MetricStatus.SUCCESS for m in hard_gates)
            release_status: Literal["PASSED", "FAILED"] = "PASSED" if hard_gates_passed else "FAILED"

            passed_count = sum(1 for m in metrics.values() if m.status == MetricStatus.SUCCESS)
            failed_count = sum(1 for m in metrics.values() if m.status == MetricStatus.FAILED)

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return TopologyHealthReport(
                run_id=active_run_id,
                evaluated_at_utc=datetime.now(UTC).isoformat(),
                graph_version=graph_version,
                graph_db_path=str(self.graph_db_path),
                source_db_path=str(self.source_db_path),
                release_status=release_status,
                hard_gates_passed=hard_gates_passed,
                total_metrics_evaluated=len(metrics),
                passed_metrics_count=passed_count,
                failed_metrics_count=failed_count,
                metrics=metrics,
                graph_overview=graph_overview,
                execution_duration_ms=round(duration_ms, 2),
            )
        finally:
            graph_conn.close()
            source_conn.close()

    # -------------------------------------------------------------------------
    # 指标 1: 指针权威有效率 (pointer_validity)
    # -------------------------------------------------------------------------
    def _eval_pointer_validity(
        self, graph_conn: duckdb.DuckDBPyConnection, source_conn: duckdb.DuckDBPyConnection
    ) -> TopologyMetricResult:
        """检查证据边的真实图端点、Session 归属和源 Turn 主键。"""
        source_turns = set(source_conn.execute("SELECT intervention_id, turn_id FROM session_dialogue_turns").fetchall())
        graph_nodes = {
            row[0]: {"node_type": row[1], "source_session_id": row[4], "source_turn_ids": list(row[6] or [])}
            for row in graph_conn.execute(
                "SELECT node_id, node_type, label, properties_json, source_session_id, source_card_id, source_turn_ids FROM graph_nodes"
            ).fetchall()
        }
        evidence_edges = graph_conn.execute(
            "SELECT edge_id, source_node_id, target_node_id, source_session_id, source_turn_ids FROM graph_edges WHERE edge_type = 'EVIDENCED_BY'"
        ).fetchall()
        total = len(evidence_edges)
        if total == 0:
            raise ValueError("图谱数据异常: 没有任何 EVIDENCED_BY 证据边！")

        invalid_edges: List[Dict[str, Any]] = []
        valid_count = 0
        for edge_id, source_node_id, target_node_id, sid, edge_turn_ids in evidence_edges:
            source_node = graph_nodes.get(source_node_id)
            target_node = graph_nodes.get(target_node_id)
            reason: str | None = None
            if source_node is None:
                reason = "source_node_id 不存在于 graph_nodes"
            elif source_node["node_type"] not in {"MISCONCEPTION_EVENT", "TUTOR_STRATEGY_EVENT"}:
                reason = f"source node type 非证据事件类型: {source_node['node_type']}"
            elif source_node["source_session_id"] != sid:
                reason = "source event 与 edge source_session_id 不一致"
            elif target_node is None:
                reason = "target_node_id 不存在于 graph_nodes"
            elif target_node["node_type"] != "TURN":
                reason = f"target node type 不是 TURN: {target_node['node_type']}"
            elif target_node["source_session_id"] != sid:
                reason = "target TURN 与 edge source_session_id 不一致"
            elif len(target_node["source_turn_ids"]) != 1:
                reason = "target TURN 必须恰好携带一个 source_turn_id"
            else:
                turn_id = int(target_node["source_turn_ids"][0])
                if turn_id not in edge_turn_ids:
                    reason = "target TURN 不在 edge source_turn_ids 中"
                elif (int(sid), turn_id) not in source_turns:
                    reason = "源表中不存在此轮次"
            if reason is None:
                valid_count += 1
            else:
                invalid_edges.append({"edge_id": edge_id, "source_node_id": source_node_id,
                                      "target_node_id": target_node_id, "session_id": sid, "reason": reason})

        rate = valid_count / total if total > 0 else 0.0
        status = MetricStatus.SUCCESS if rate >= 1.0 else MetricStatus.FAILED

        return TopologyMetricResult(
            metric_name="pointer_validity",
            display_name="指针权威有效率",
            value=round(rate, 6),
            threshold=1.0,
            operator=ComparisonOperator.GTE,
            status=status,
            hard_gate=True,
            passed_cases=valid_count,
            total_cases=total,
            details={
                "total_evidence_edges": total,
                "valid_pointers": valid_count,
                "invalid_pointers_count": len(invalid_edges),
                "invalid_samples": invalid_edges[:5],
            },
            error_message=f"存在 {len(invalid_edges)} 条断裂悬空指针证据边" if invalid_edges else None,
        )

    # -------------------------------------------------------------------------
    # 指标 2: 角色归属纯洁度 (role_accuracy)
    # -------------------------------------------------------------------------
    def _eval_role_accuracy(
        self, graph_conn: duckdb.DuckDBPyConnection, source_conn: duckdb.DuckDBPyConnection
    ) -> TopologyMetricResult:
        """通过真实节点类型和 source Turn 角色检查证据边归属。"""
        turn_roles = {
            (int(row[0]), int(row[1])): bool(row[2])
            for row in source_conn.execute(
                "SELECT intervention_id, turn_id, is_tutor FROM session_dialogue_turns"
            ).fetchall()
        }
        graph_nodes = {
            row[0]: {"node_type": row[1], "source_session_id": row[4], "source_turn_ids": list(row[6] or [])}
            for row in graph_conn.execute(
                "SELECT node_id, node_type, label, properties_json, source_session_id, source_card_id, source_turn_ids FROM graph_nodes"
            ).fetchall()
        }
        evidence_edges = graph_conn.execute(
            "SELECT edge_id, source_node_id, target_node_id, source_session_id FROM graph_edges WHERE edge_type = 'EVIDENCED_BY'"
        ).fetchall()

        total = len(evidence_edges)
        valid_role_count = 0
        role_violations: List[Dict[str, Any]] = []

        for edge_id, src_node, tgt_node, sid in evidence_edges:
            source_node = graph_nodes.get(src_node)
            target_node = graph_nodes.get(tgt_node)
            reason: str | None = None
            if source_node is None:
                reason = "source event node 不存在"
            elif source_node["node_type"] not in {"MISCONCEPTION_EVENT", "TUTOR_STRATEGY_EVENT"}:
                reason = f"source node type 非证据事件类型: {source_node['node_type']}"
            elif source_node["source_session_id"] != sid:
                reason = "source event 与 edge source_session_id 不一致"
            elif target_node is None or target_node["node_type"] != "TURN":
                reason = "target node 不存在或不是 TURN"
            elif target_node["source_session_id"] != sid:
                reason = "target TURN 与 edge source_session_id 不一致"
            elif len(target_node["source_turn_ids"]) != 1:
                reason = "target TURN 必须恰好携带一个 source_turn_id"
            else:
                turn_id = int(target_node["source_turn_ids"][0])
                actual_role = turn_roles.get((int(sid), turn_id))
                if actual_role is None:
                    reason = "源表中不存在对应轮次的角色信息"
                else:
                    expected_role = source_node["node_type"] == "TUTOR_STRATEGY_EVENT"
                    if bool(actual_role) != expected_role:
                        reason = f"角色不匹配: actual_is_tutor={actual_role}, expected_is_tutor={expected_role}"
            if reason is None:
                valid_role_count += 1
            else:
                role_violations.append({"edge_id": edge_id, "source_node": src_node,
                                        "target_node": tgt_node, "reason": reason})

        accuracy = valid_role_count / total if total > 0 else 0.0
        status = MetricStatus.SUCCESS if accuracy >= 1.0 else MetricStatus.FAILED

        return TopologyMetricResult(
            metric_name="role_accuracy",
            display_name="角色归属纯洁度",
            value=round(accuracy, 6),
            threshold=1.0,
            operator=ComparisonOperator.GTE,
            status=status,
            hard_gate=True,
            passed_cases=valid_role_count,
            total_cases=total,
            details={
                "total_evidence_edges": total,
                "valid_role_edges": valid_role_count,
                "role_violations_count": len(role_violations),
                "violation_samples": role_violations[:5],
            },
            error_message=f"存在 {len(role_violations)} 条角色越界混淆的证据边" if role_violations else None,
        )

    # -------------------------------------------------------------------------
    # 指标 3: 孤立节点率 (orphan_node_ratio)
    # -------------------------------------------------------------------------
    def _eval_orphan_nodes(self, graph_conn: duckdb.DuckDBPyConnection) -> TopologyMetricResult:
        """检查全图是否存在入度+出度=0的孤岛节点。"""
        total_nodes = graph_conn.execute("SELECT count(*) FROM graph_nodes").fetchone()[0]

        # 统计既没有作为起点、也没有作为终点的节点
        orphan_rows = graph_conn.execute("""
            SELECT n.node_id, n.node_type, n.label
            FROM graph_nodes n
            WHERE NOT EXISTS (SELECT 1 FROM graph_edges e WHERE e.source_node_id = n.node_id)
              AND NOT EXISTS (SELECT 1 FROM graph_edges e WHERE e.target_node_id = n.node_id)
        """).fetchall()

        orphan_count = len(orphan_rows)
        if total_nodes == 0:
            return TopologyMetricResult(
                metric_name="orphan_node_ratio", display_name="孤立节点率",
                value=0.0, threshold=0.0, operator=ComparisonOperator.LTE,
                status=MetricStatus.BLOCKED, hard_gate=True, passed_cases=0,
                total_cases=0,
                details={"total_nodes": 0, "orphan_count": 0, "connected_nodes": 0},
                error_message="图谱无节点，孤立率不可测量",
            )
        orphan_ratio = orphan_count / total_nodes
        passed_nodes = total_nodes - orphan_count

        status = MetricStatus.SUCCESS if orphan_ratio <= 0.0 else MetricStatus.FAILED

        return TopologyMetricResult(
            metric_name="orphan_node_ratio",
            display_name="孤立节点率",
            value=round(orphan_ratio, 6),
            threshold=0.0,
            operator=ComparisonOperator.LTE,
            status=status,
            hard_gate=True,
            passed_cases=passed_nodes,
            total_cases=total_nodes,
            details={
                "total_nodes": total_nodes,
                "orphan_count": orphan_count,
                "connected_nodes": passed_nodes,
                "orphan_samples": [{"node_id": r[0], "type": r[1], "label": r[2]} for r in orphan_rows[:5]],
            },
            error_message=f"发现 {orphan_count} 个未与任何边相连的孤立节点" if orphan_count > 0 else None,
        )

    # -------------------------------------------------------------------------
    # 额外硬门禁: 所有边的端点必须是真实图节点
    # -------------------------------------------------------------------------
    def _eval_edge_endpoint_integrity(self, graph_conn: duckdb.DuckDBPyConnection) -> TopologyMetricResult:
        """检查每条关系边的 source/target 是否都存在于 graph_nodes。"""
        total = int(graph_conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0])
        if total == 0:
            raise ValueError("图谱数据异常: graph_edges 为空")
        invalid_rows = graph_conn.execute("""
            SELECT e.edge_id, e.source_node_id, e.target_node_id
            FROM graph_edges e
            LEFT JOIN graph_nodes source_node ON source_node.node_id = e.source_node_id
            LEFT JOIN graph_nodes target_node ON target_node.node_id = e.target_node_id
            WHERE source_node.node_id IS NULL OR target_node.node_id IS NULL
            ORDER BY e.edge_id
        """).fetchall()
        valid = total - len(invalid_rows)
        value = valid / total
        return TopologyMetricResult(
            metric_name="edge_endpoint_integrity",
            display_name="关系边端点完整率",
            value=round(value, 6), threshold=1.0, operator=ComparisonOperator.GTE,
            status=MetricStatus.SUCCESS if value >= 1.0 else MetricStatus.FAILED,
            hard_gate=True, passed_cases=valid, total_cases=total,
            details={
                "total_edges": total,
                "valid_edges": valid,
                "invalid_edge_count": len(invalid_rows),
                "invalid_samples": [
                    {"edge_id": row[0], "source_node_id": row[1], "target_node_id": row[2]}
                    for row in invalid_rows[:5]
                ],
            },
            error_message=(f"发现 {len(invalid_rows)} 条关系边存在悬空端点" if invalid_rows else None),
        )

    # -------------------------------------------------------------------------
    # 额外硬门禁: assignment 必须命中合法 taxonomy 与节点类型
    # -------------------------------------------------------------------------
    def _eval_taxonomy_label_validity(self, graph_conn: duckdb.DuckDBPyConnection) -> TopologyMetricResult:
        """检查每条 label assignment 的类型、节点绑定和 canonical label。"""
        taxonomy_path = Path(__file__).resolve().parents[2] / "data" / "business" / "graph-taxonomy-v1.json"
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"taxonomy artifact not found: {taxonomy_path}")
        raw = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        allowed = {
            "MISCONCEPTION": set(raw.get("misconception", [])),
            "EXAM_ERROR": set(raw.get("exam_error", [])),
            "TUTOR_STRATEGY": set(raw.get("tutor_strategy", [])),
            "QUESTION_FUNCTION": set(raw.get("question_function", [])),
            "DIFFICULTY_SIGNAL": set(raw.get("difficulty_signal", [])),
            "CONTENT_OPPORTUNITY": set(raw.get("content_opportunity", [])),
        }
        expected_types = {
            "MISCONCEPTION": {"MISCONCEPTION_EVENT"},
            "EXAM_ERROR": {"MISCONCEPTION_EVENT"},
            "CONTENT_OPPORTUNITY": {"MISCONCEPTION_EVENT"},
            "TUTOR_STRATEGY": {"TUTOR_STRATEGY_EVENT"},
            "QUESTION_FUNCTION": {"TURN"},
            "DIFFICULTY_SIGNAL": {"SESSION"},
        }
        nodes = {
            row[0]: {"node_type": row[1], "properties": json.loads(row[2] or "{}")}
            for row in graph_conn.execute(
                "SELECT node_id, node_type, properties_json FROM graph_nodes"
            ).fetchall()
        }
        assignments = graph_conn.execute(
            "SELECT source_node_id, label_type, canonical_label FROM graph_label_assignments ORDER BY source_node_id"
        ).fetchall()
        if not assignments:
            raise ValueError("图谱数据异常: graph_label_assignments 为空")
        invalid: List[Dict[str, Any]] = []
        for source_node_id, label_type, canonical_label in assignments:
            if (
                source_node_id not in nodes
                or nodes[source_node_id]["node_type"] not in expected_types.get(label_type, set())
                or canonical_label not in allowed.get(label_type, set())
            ):
                invalid.append({
                    "source_node_id": source_node_id,
                    "label_type": label_type,
                    "canonical_label": canonical_label,
                    "node_type": nodes.get(source_node_id, {}).get("node_type"),
                })
        assigned_keys = {(str(row[1]), str(row[0])) for row in assignments}
        required_keys: set[tuple[str, str]] = set()
        for node_id, node in nodes.items():
            node_type = node["node_type"]
            if node_type == "MISCONCEPTION_EVENT":
                required_keys.update({
                    ("MISCONCEPTION", node_id),
                    ("EXAM_ERROR", node_id),
                    ("CONTENT_OPPORTUNITY", node_id),
                })
            elif node_type == "TUTOR_STRATEGY_EVENT":
                required_keys.add(("TUTOR_STRATEGY", node_id))
            elif node_type == "SESSION":
                required_keys.add(("DIFFICULTY_SIGNAL", node_id))
            elif node_type == "TURN":
                properties = node["properties"]
                if not properties.get("is_tutor") and not properties.get("is_greeting_or_noise", False):
                    required_keys.add(("QUESTION_FUNCTION", node_id))
        missing_keys = sorted(required_keys - assigned_keys)
        invalid.extend({
            "source_node_id": node_id,
            "label_type": label_type,
            "canonical_label": None,
            "reason": "missing required assignment",
        } for label_type, node_id in missing_keys)
        valid = max(0, len(assignments) - len(invalid))
        total_for_gate = len(assignments) + len(missing_keys)
        value = valid / total_for_gate if total_for_gate else 0.0
        return TopologyMetricResult(
            metric_name="taxonomy_label_validity",
            display_name="分类标签合法率",
            value=round(value, 6), threshold=1.0, operator=ComparisonOperator.GTE,
            status=MetricStatus.SUCCESS if value >= 1.0 else MetricStatus.FAILED,
            hard_gate=True, passed_cases=valid, total_cases=total_for_gate,
            details={
                "total_assignments": total_for_gate,
                "valid_assignments": valid,
                "invalid_label_count": len(invalid),
                "invalid_label_samples": invalid[:5],
                "missing_assignment_count": len(missing_keys),
                "taxonomy_version": raw.get("taxonomy_version"),
            },
            error_message=(f"发现 {len(invalid)} 条未知标签或错误节点绑定" if invalid else None),
        )

    # -------------------------------------------------------------------------
    # 指标 4: 分类标签覆盖率 (taxonomy_coverage)
    # -------------------------------------------------------------------------
    def _eval_taxonomy_coverage(self, graph_conn: duckdb.DuckDBPyConnection) -> TopologyMetricResult:
        """按版本化 taxonomy 白名单度量有效分类，拒绝未知标签冒充已分类。"""
        taxonomy_path = Path(__file__).resolve().parents[2] / "data" / "business" / "graph-taxonomy-v1.json"
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"taxonomy artifact not found: {taxonomy_path}")
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        allowed_by_type = {
            "MISCONCEPTION": set(taxonomy.get("misconception", [])),
            "EXAM_ERROR": set(taxonomy.get("exam_error", [])),
            "TUTOR_STRATEGY": set(taxonomy.get("tutor_strategy", [])),
            "QUESTION_FUNCTION": set(taxonomy.get("question_function", [])),
            "DIFFICULTY_SIGNAL": set(taxonomy.get("difficulty_signal", [])),
            "CONTENT_OPPORTUNITY": set(taxonomy.get("content_opportunity", [])),
        }
        graph_nodes = {
            row[0]: row[1]
            for row in graph_conn.execute("SELECT node_id, node_type FROM graph_nodes").fetchall()
        }
        assignment_rows = graph_conn.execute(
            "SELECT source_node_id, label_type, canonical_label FROM graph_label_assignments"
        ).fetchall()
        grouped: Dict[str, List[tuple[str, str]]] = {}
        for source_node_id, label_type, canonical_label in assignment_rows:
            grouped.setdefault(str(label_type), []).append((source_node_id, canonical_label))

        breakdown: Dict[str, Dict[str, Any]] = {}
        total_assignments = 0
        total_classified = 0
        total_unclassified = 0
        invalid_label_count = 0
        invalid_samples: List[Dict[str, Any]] = []
        expected_node_types = {
            "MISCONCEPTION": {"MISCONCEPTION_EVENT"},
            "EXAM_ERROR": {"MISCONCEPTION_EVENT"},
            "CONTENT_OPPORTUNITY": {"MISCONCEPTION_EVENT"},
            "TUTOR_STRATEGY": {"TUTOR_STRATEGY_EVENT"},
            "QUESTION_FUNCTION": {"TURN"},
            "DIFFICULTY_SIGNAL": {"SESSION"},
        }

        for ltype, assignments in sorted(grouped.items()):
            valid_classified = 0
            valid_unclassified = 0
            allowed = allowed_by_type.get(ltype, set())
            for source_node_id, canonical_label in assignments:
                node_type = graph_nodes.get(source_node_id)
                if node_type not in expected_node_types.get(ltype, set()) or canonical_label not in allowed:
                    invalid_label_count += 1
                    if len(invalid_samples) < 5:
                        invalid_samples.append({
                            "label_type": ltype,
                            "source_node_id": source_node_id,
                            "canonical_label": canonical_label,
                            "node_type": node_type,
                        })
                elif canonical_label == "UNCLASSIFIED":
                    valid_unclassified += 1
                else:
                    valid_classified += 1
            total = len(assignments)
            cov = valid_classified / total if total > 0 else 0.0
            breakdown[ltype] = {
                "total": total,
                "classified": valid_classified,
                "unclassified": valid_unclassified,
                "coverage_rate": round(cov, 4),
            }
            total_assignments += total
            total_classified += valid_classified
            total_unclassified += valid_unclassified

        # 重点关注核心教研卡片 (MISCONCEPTION + TUTOR_STRATEGY) 的综合覆盖率
        core_types = ["MISCONCEPTION", "TUTOR_STRATEGY"]
        core_total = sum(breakdown.get(t, {}).get("total", 0) for t in core_types)
        core_classified = sum(breakdown.get(t, {}).get("classified", 0) for t in core_types)
        core_coverage = core_classified / core_total if core_total > 0 else 0.0

        overall_coverage = total_classified / total_assignments if total_assignments > 0 else 0.0

        # Soft Gate 监控指标；状态必须由实测值计算，不能硬编码为 SUCCESS。
        coverage_status = (
            MetricStatus.SUCCESS
            if ComparisonOperator.GTE.passes(core_coverage, 0.0)
            else MetricStatus.FAILED
        )
        return TopologyMetricResult(
            metric_name="taxonomy_coverage",
            display_name="分类标签覆盖率",
            value=round(core_coverage, 6),
            threshold=0.0,
            operator=ComparisonOperator.GTE,
            status=coverage_status,
            hard_gate=False,
            passed_cases=core_classified,
            total_cases=core_total,
            details={
                "core_cards_total": core_total,
                "core_cards_classified": core_classified,
                "core_coverage_rate": round(core_coverage, 4),
                "overall_assignments_total": total_assignments,
                "overall_classified_total": total_classified,
                "overall_unclassified_total": total_unclassified,
                "overall_coverage_rate": round(overall_coverage, 4),
                "invalid_label_count": invalid_label_count,
                "invalid_label_samples": invalid_samples,
                "type_breakdown": breakdown,
            },
            error_message=(f"发现 {invalid_label_count} 条未知标签或错误节点绑定" if invalid_label_count else None),
        )

    # -------------------------------------------------------------------------
    # 指标 5: Source -> Graph 投影完整率 (projection_completeness)
    # -------------------------------------------------------------------------
    def _eval_projection_completeness(
        self, graph_conn: duckdb.DuckDBPyConnection, source_conn: duckdb.DuckDBPyConnection
    ) -> TopologyMetricResult:
        """逐 Session 核对源表与图中的 Session、双 Card、Turn 和必需边。"""
        source_sessions = [
            int(row[0]) for row in source_conn.execute(
                "SELECT intervention_id FROM tutoring_sessions ORDER BY intervention_id"
            ).fetchall()
        ]
        if not source_sessions:
            raise ValueError("源数据异常: tutoring_sessions 为空")
        source_session_set = set(source_sessions)
        source_cards = {
            "MISCONCEPTION_EVENT": {},
            "TUTOR_STRATEGY_EVENT": {},
        }
        for node_type, table in (
            ("MISCONCEPTION_EVENT", "misconception_chunks"),
            ("TUTOR_STRATEGY_EVENT", "tutor_strategy_chunks"),
        ):
            for sid, card_id in source_conn.execute(
                f"SELECT session_id, chunk_id FROM {table} ORDER BY session_id, chunk_id"
            ).fetchall():
                source_cards[node_type].setdefault(int(sid), []).append(str(card_id))

        source_turns: Dict[int, set[int]] = {}
        for sid, turn_id in source_conn.execute(
            "SELECT intervention_id, turn_id FROM session_dialogue_turns ORDER BY intervention_id, turn_id"
        ).fetchall():
            source_turns.setdefault(int(sid), set()).add(int(turn_id))

        node_rows = graph_conn.execute(
            "SELECT node_id, node_type, source_session_id, source_card_id, source_turn_ids FROM graph_nodes"
        ).fetchall()
        nodes_by_session: Dict[int, Dict[str, List[tuple[Any, ...]]]] = {}
        all_graph_session_ids: set[int] = set()
        invalid_source_session_nodes: List[Dict[str, Any]] = []
        malformed_turn_nodes: List[Dict[str, Any]] = []
        for row in node_rows:
            node_id, node_type, sid, card_id, turn_ids = row
            if sid is None:
                invalid_source_session_nodes.append({"node_id": node_id, "node_type": node_type})
                continue
            sid = int(sid)
            nodes_by_session.setdefault(sid, {}).setdefault(str(node_type), []).append(row)
            if node_type == "TURN" and (not turn_ids or len(turn_ids) != 1):
                malformed_turn_nodes.append({"node_id": node_id, "source_turn_ids": list(turn_ids or [])})
            if node_type == "SESSION":
                all_graph_session_ids.add(sid)

        edge_rows = graph_conn.execute(
            "SELECT source_node_id, target_node_id, edge_type, source_session_id FROM graph_edges"
        ).fetchall()
        edges_by_session: Dict[int, List[tuple[Any, ...]]] = {}
        for row in edge_rows:
            edges_by_session.setdefault(int(row[3]), []).append(row)

        failures: List[Dict[str, Any]] = []
        valid_sessions = 0
        for sid in source_sessions:
            problems: List[str] = []
            session_nodes = nodes_by_session.get(sid, {}).get("SESSION", [])
            if len(session_nodes) != 1:
                problems.append(f"SESSION 节点数量={len(session_nodes)}")
            session_node_id = session_nodes[0][0] if len(session_nodes) == 1 else None
            if source_cards["MISCONCEPTION_EVENT"].get(sid, []) != sorted(
                [str(row[3]) for row in nodes_by_session.get(sid, {}).get("MISCONCEPTION_EVENT", [])]
            ):
                problems.append("misconception Card 与 graph event 不一致")
            if source_cards["TUTOR_STRATEGY_EVENT"].get(sid, []) != sorted(
                [str(row[3]) for row in nodes_by_session.get(sid, {}).get("TUTOR_STRATEGY_EVENT", [])]
            ):
                problems.append("tutor strategy Card 与 graph event 不一致")

            graph_turn_ids = {
                int(row[4][0])
                for row in nodes_by_session.get(sid, {}).get("TURN", [])
                if row[4] and len(row[4]) == 1
            }
            if graph_turn_ids != source_turns.get(sid, set()):
                problems.append("TURN 节点集合与 source dialogue 不一致")

            session_edges = edges_by_session.get(sid, [])
            required_edge_checks = {
                "ANCHORS": False,
                "ABOUT_SUBJECT": False,
                "HAS_MISCONCEPTION": False,
                "HAS_STRATEGY": False,
            }
            graph_turn_node_ids = {row[0] for row in nodes_by_session.get(sid, {}).get("TURN", [])}
            linked_turn_node_ids: set[str] = set()
            for source_node_id, target_node_id, edge_type, edge_sid in session_edges:
                if edge_type in required_edge_checks and source_node_id == session_node_id:
                    required_edge_checks[edge_type] = True
                if edge_type == "HAS_TURN" and source_node_id == session_node_id:
                    linked_turn_node_ids.add(target_node_id)
            problems.extend(f"缺少 {edge_type} 边" for edge_type, present in required_edge_checks.items() if not present)
            if linked_turn_node_ids != graph_turn_node_ids:
                problems.append("HAS_TURN 未覆盖全部 TURN 节点")
            if problems:
                if len(failures) < 10:
                    failures.append({"session_id": sid, "problems": problems})
            else:
                valid_sessions += 1

        extra_graph_sessions = sorted(all_graph_session_ids - source_session_set)
        if extra_graph_sessions:
            failures.append({"extra_graph_session_ids": extra_graph_sessions[:10]})
        value = valid_sessions / len(source_sessions)
        status = MetricStatus.SUCCESS if (
            value >= 1.0 and not extra_graph_sessions
            and not invalid_source_session_nodes and not malformed_turn_nodes
        ) else MetricStatus.FAILED
        return TopologyMetricResult(
            metric_name="projection_completeness",
            display_name="Source 到 Graph 投影完整率",
            value=round(value, 6),
            threshold=1.0,
            operator=ComparisonOperator.GTE,
            status=status,
            hard_gate=True,
            passed_cases=valid_sessions,
            total_cases=len(source_sessions),
            details={
                "source_session_count": len(source_sessions),
                "valid_session_count": valid_sessions,
                "extra_graph_session_count": len(extra_graph_sessions),
                "invalid_source_session_node_count": len(invalid_source_session_nodes),
                "invalid_source_session_nodes": invalid_source_session_nodes[:10],
                "malformed_turn_node_count": len(malformed_turn_nodes),
                "malformed_turn_nodes": malformed_turn_nodes[:10],
                "failure_samples": failures[:10],
            },
            error_message=(f"存在 {len(source_sessions) - valid_sessions} 个不完整 Session 投影"
                           if status == MetricStatus.FAILED else None),
        )

    # -------------------------------------------------------------------------
    # 指标 6: 构图确定性哈希一致率 (reproducibility_hash_match)
    # -------------------------------------------------------------------------
    def _eval_reproducibility_hash(
        self,
        graph_conn: duckdb.DuckDBPyConnection,
        source_db_path: Path,
        recorded_src_hash: str,
        actual_node_count: int,
        actual_edge_count: int,
        manifest_node_count: int,
        manifest_edge_count: int,
        build_status: str,
        *,
        rebuilt_graph_content_hash: str | None = None,
    ) -> TopologyMetricResult:
        """比较 source 绑定、图内容指纹和独立重建结果，禁止仅靠计数冒充复现。"""
        # 实时计算当前源数据库的物理 SHA-256
        computed_src_hash = self._sha256_file(source_db_path)

        hash_matches = (computed_src_hash == recorded_src_hash)
        node_count_matches = (actual_node_count == manifest_node_count)
        edge_count_matches = (actual_edge_count == manifest_edge_count)
        status_is_ready = (build_status == "READY")
        current_graph_content_hash = self._graph_content_sha256(graph_conn)
        graph_hash_matches = (
            rebuilt_graph_content_hash is not None
            and current_graph_content_hash == rebuilt_graph_content_hash
        )

        all_checks = [
            hash_matches, node_count_matches, edge_count_matches, status_is_ready,
            rebuilt_graph_content_hash is not None, graph_hash_matches,
        ]
        passed_checks = sum(1 for c in all_checks if c)
        total_checks = len(all_checks)

        is_reproducible = (passed_checks == total_checks)
        value = 1.0 if is_reproducible else 0.0
        status = MetricStatus.SUCCESS if is_reproducible else MetricStatus.FAILED

        mismatch_reasons: List[str] = []
        if not hash_matches:
            mismatch_reasons.append(f"源数据物理SHA256({computed_src_hash[:12]}...)与清单记录({recorded_src_hash[:12]}...)不匹配")
        if not node_count_matches:
            mismatch_reasons.append(f"实际节点数({actual_node_count})与清单记录({manifest_node_count})不匹配")
        if not edge_count_matches:
            mismatch_reasons.append(f"实际边数({actual_edge_count})与清单记录({manifest_edge_count})不匹配")
        if not status_is_ready:
            mismatch_reasons.append(f"构建状态非READY({build_status})")
        if rebuilt_graph_content_hash is None:
            mismatch_reasons.append("缺少独立重建图内容指纹")
        elif not graph_hash_matches:
            mismatch_reasons.append("当前图内容指纹与独立重建结果不一致")

        return TopologyMetricResult(
            metric_name="reproducibility_hash_match",
            display_name="构图确定性哈希一致率",
            value=value,
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=status,
            hard_gate=True,
            passed_cases=passed_checks,
            total_cases=total_checks,
            details={
                "computed_source_sha256": computed_src_hash,
                "recorded_source_sha256": recorded_src_hash,
                "hash_matches": hash_matches,
                "actual_node_count": actual_node_count,
                "manifest_node_count": manifest_node_count,
                "node_count_matches": node_count_matches,
                "actual_edge_count": actual_edge_count,
                "manifest_edge_count": manifest_edge_count,
                "edge_count_matches": edge_count_matches,
                "build_status": build_status,
                "status_is_ready": status_is_ready,
                "current_graph_content_sha256": current_graph_content_hash,
                "rebuilt_graph_content_sha256": rebuilt_graph_content_hash,
                "graph_content_hash_matches": graph_hash_matches,
            },
            error_message="; ".join(mismatch_reasons) if mismatch_reasons else None,
        )


# =============================================================================
# 三、 格式化呈现与报告输出函数
# =============================================================================

def format_markdown_report(report: TopologyHealthReport) -> str:
    """生成对齐 AGENTS.md 工业规范的结构化 Markdown 审计报告。"""
    lines: List[str] = [
        "# 静态图谱结构与拓扑健康度评测报告",
        "",
        f"> **评测编号 (Run ID)**: `{report.run_id}`  ",
        f"> **执行时间 (UTC)**: `{report.evaluated_at_utc}`  ",
        f"> **图谱版本**: `{report.graph_version}`  ",
        f"> **发布裁决 (Release Status)**: **{report.release_status}** ({'全部硬门禁通过' if report.hard_gates_passed else '存在硬门禁未达标'})  ",
        f"> **评测耗时**: `{report.execution_duration_ms} ms`  ",
        "",
        "---",
        "",
        "## 一、 核心拓扑规模概览 (Graph Overview)",
        "",
        f"- **总节点数 (Total Nodes)**: `{report.graph_overview['total_nodes']:,}`",
        f"- **总关系边数 (Total Edges)**: `{report.graph_overview['total_edges']:,}`",
        f"- **构建状态 (Build Status)**: `{report.graph_overview['build_status']}`",
        "",
        "| 节点分类 (Node Type) | 数量 | 边关系类型 (Edge Type) | 数量 |",
        "|---|:---:|---|:---:|",
    ]

    node_items = list(report.graph_overview["node_types"].items())
    edge_items = list(report.graph_overview["edge_types"].items())
    max_len = max(len(node_items), len(edge_items))

    for i in range(max_len):
        n_txt = f"`{node_items[i][0]}`: **{node_items[i][1]:,}**" if i < len(node_items) else ""
        e_txt = f"`{edge_items[i][0]}`: **{edge_items[i][1]:,}**" if i < len(edge_items) else ""
        lines.append(f"| {n_txt} | {e_txt} |")

    lines.extend([
        "",
        "---",
        "",
        "## 二、 核心健康度指标审计 (Core Health Metrics)",
        "",
        "| 指标标识 | 中文指标名称 | 实测值 | 门禁阈值 | 门禁性质 | 判定结果 | 关键样本/说明 |",
        "|---|---|:---:|:---:|:---:|:---:|---|",
    ])

    for m in report.metrics.values():
        val_str = f"{m.value:.4%}" if m.metric_name != "reproducibility_hash_match" else f"{m.value:.1f}"
        thresh_str = f"{m.threshold:.4%}" if m.metric_name != "reproducibility_hash_match" else f"{m.threshold:.1f}"
        gate_str = "**Hard Gate**" if m.hard_gate else "Soft Gate"
        res_str = "**SUCCESS**" if m.status == MetricStatus.SUCCESS else "**FAILED**"

        extra = f"分子: {m.passed_cases:,} / 分母: {m.total_cases:,}"
        if m.metric_name == "taxonomy_coverage":
            extra = f"核心卡片覆盖率: {m.value:.2%} (已标: {m.passed_cases}, 总卡: {m.total_cases})"
        elif m.error_message:
            extra = f"**错误**: {m.error_message}"

        lines.append(f"| `{m.metric_name}` | **{m.display_name}** | **{val_str}** | {thresh_str} | {gate_str} | {res_str} | {extra} |")

    tax_detail = report.metrics.get("taxonomy_coverage")
    if tax_detail and "type_breakdown" in tax_detail.details:
        lines.extend([
            "",
            "### 3. 分类标签覆盖率细分下钻 (Taxonomy Breakdown)",
            "",
            "| 标签类型 (Label Type) | 总分配数 | 已标准化归类数 | 未归类数 (UNCLASSIFIED) | 覆盖率 |",
            "|---|:---:|:---:|:---:|:---:|",
        ])
        for lt, d in tax_detail.details["type_breakdown"].items():
            lines.append(f"| `{lt}` | {d['total']:,} | {d['classified']:,} | {d['unclassified']:,} | **{d['coverage_rate']:.2%}** |")

    lines.extend([
        "",
        "---",
        "> **工程审计结论**: 严格遵循 `AGENTS.md` 真实性规范，所有指标经由 DuckDB 主外键及文件 SHA-256 原生计算产出。",
    ])

    return "\n".join(lines)


# =============================================================================
# 四、 命令行入口与默认执行
# =============================================================================

def find_latest_graph_db(staging_dir: Path) -> Path:
    """只从 manifest 恰好一行且状态 READY 的候选中选择最新图谱。"""
    candidates = [Path(path) for path in glob.glob(str(staging_dir / "graph*.duckdb"))]
    if not candidates:
        raise FileNotFoundError(f"在 {staging_dir} 中未找到任何 graph*.duckdb 文件")
    eligible: list[Path] = []
    rejected: list[str] = []
    for candidate in candidates:
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = duckdb.connect(str(candidate), read_only=True)
            rows = connection.execute(
                "SELECT build_status FROM graph_manifest"
            ).fetchall()
            if len(rows) != 1:
                rejected.append(f"{candidate.name}: manifest 行数为 {len(rows)}")
            elif rows[0][0] != "READY":
                rejected.append(f"{candidate.name}: build_status={rows[0][0]}")
            else:
                eligible.append(candidate)
        except Exception as exc:
            rejected.append(f"{candidate.name}: {exc}")
        finally:
            if connection is not None:
                connection.close()
    if not eligible:
        detail = "; ".join(rejected[:5])
        raise FileNotFoundError(f"在 {staging_dir} 中未找到 READY 图谱候选。拒绝原因: {detail}")
    eligible.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return eligible[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行 Eedi-RAG 静态图谱结构与拓扑健康度核心指标评测")
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

    print(f"\n🚀 开始执行静态图谱结构与拓扑健康度评测...")
    print(f"📁 图谱路径: {graph_db_path}")
    print(f"📁 数据源路径: {args.source_db}")

    evaluator = TopologyConstructionEvaluator(
        graph_db_path=graph_db_path,
        source_db_path=args.source_db,
    )

    report = evaluator.evaluate()

    md_content = format_markdown_report(report)
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
