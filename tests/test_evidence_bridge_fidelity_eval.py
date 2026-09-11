"""
单元测试: tests/test_evidence_bridge_fidelity_eval.py
业务定位: 测试图到切片的穿透保真度与证据桥物化评测套件 (EvidenceBridgeFidelityEvaluator) 的
逐字穿透校验、会话范围收敛、角色对齐、编目双射及对抗异常注入拦截能力。
"""

from __future__ import annotations

from pathlib import Path
import pytest

from evals.graph.evidence_bridge_fidelity_eval import (
    BridgeFidelityMetricResult,
    BridgeFidelityReport,
    ComparisonOperator,
    EvidenceBridgeFidelityEvaluator,
    MetricStatus,
    find_latest_graph_db,
    format_bridge_fidelity_markdown,
)


def test_bridge_metric_contract_validation():
    res = BridgeFidelityMetricResult(
        metric_name="test_bridge",
        display_name="测试证据桥指标",
        value=1.0,
        threshold=1.0,
        operator=ComparisonOperator.EQ,
        status=MetricStatus.SUCCESS,
        hard_gate=True,
        passed_cases=10,
        total_cases=10,
    )
    assert res.status == MetricStatus.SUCCESS

    with pytest.raises(ValueError, match="状态矛盾"):
        BridgeFidelityMetricResult(
            metric_name="test_bridge",
            display_name="测试证据桥指标",
            value=0.9,
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=MetricStatus.SUCCESS,
            hard_gate=True,
            passed_cases=9,
            total_cases=10,
        )


def test_real_evidence_bridge_fidelity_evaluation():
    """在真实全量知识图谱资产上运行证据桥物化保真度评测，断言 6 大指标（包含 5 组对抗攻击拦截）全过。"""
    project_root = Path(__file__).resolve().parents[1]
    staging_dir = project_root / "reports" / "staging" / "session-card-graph-v1"
    latest_graph = find_latest_graph_db(staging_dir)
    source_db = project_root / "data" / "db" / "tutoring_knowledge.duckdb"

    evaluator = EvidenceBridgeFidelityEvaluator(
        graph_db_path=latest_graph,
        source_db_path=source_db,
    )

    report = evaluator.evaluate()

    assert isinstance(report, BridgeFidelityReport)
    assert report.release_status == "PASSED"
    assert report.hard_gates_passed is True
    assert set(report.metrics.keys()) == {
        "verbatim_grounding_fidelity",
        "scope_confinement_rate",
        "role_alignment_fidelity",
        "catalog_bijection_accuracy",
        "adversarial_rejection_rate",
        "context_budget_compliance",
    }

    # 6 大核心指标全过
    assert report.metrics["verbatim_grounding_fidelity"].value == 1.0
    assert report.metrics["scope_confinement_rate"].value == 1.0
    assert report.metrics["role_alignment_fidelity"].value == 1.0
    assert report.metrics["catalog_bijection_accuracy"].value == 1.0
    assert report.metrics["adversarial_rejection_rate"].value == 1.0
    assert report.metrics["context_budget_compliance"].value == 1.0

    # 验证 Markdown 格式输出
    md = format_bridge_fidelity_markdown(report)
    assert "# 图到切片穿透保真度与证据桥物化评测报告" in md
    assert "PASSED" in md
