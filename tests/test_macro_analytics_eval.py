"""
单元测试: tests/test_macro_analytics_eval.py
业务定位: 测试宏观统计与拓扑查询准确度评测套件 (MacroAnalyticsEvaluator) 的度量契约、
独立金标 SQL 双算逻辑、学科过滤零泄露核验及未分类数据透明度。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest
import duckdb

from evals.graph.macro_analytics_eval import (
    ComparisonOperator,
    MacroAnalyticsEvaluator,
    MacroAnalyticsMetricResult,
    MacroAnalyticsReport,
    MetricStatus,
    find_latest_graph_db,
    format_macro_analytics_markdown,
)


def test_macro_metric_contract_validation():
    # 正常构造
    res = MacroAnalyticsMetricResult(
        metric_name="test_macro",
        display_name="测试宏观指标",
        value=1.0,
        threshold=1.0,
        operator=ComparisonOperator.EQ,
        status=MetricStatus.SUCCESS,
        hard_gate=True,
        passed_cases=5,
        total_cases=5,
    )
    assert res.status == MetricStatus.SUCCESS

    # 状态矛盾抛出异常
    with pytest.raises(ValueError, match="状态矛盾"):
        MacroAnalyticsMetricResult(
            metric_name="test_macro",
            display_name="测试宏观指标",
            value=0.5,
            threshold=1.0,
            operator=ComparisonOperator.EQ,
            status=MetricStatus.SUCCESS,
            hard_gate=True,
            passed_cases=2,
            total_cases=5,
        )


def test_unclassified_visibility_fails_when_unclassified_is_not_visible() -> None:
    evaluator = MacroAnalyticsEvaluator.__new__(MacroAnalyticsEvaluator)
    analytics = SimpleNamespace(
        recurring_misconceptions=lambda top_k: SimpleNamespace(
            data_scope=SimpleNamespace(unmeasured_metrics={"UNRESOLVED_ENDING"}),
            rows=[{"canonical_name": "KNOWN"}],
        )
    )

    result = evaluator._eval_unclassified_visibility(analytics)

    assert result.status == MetricStatus.FAILED
    assert result.value == 0.5
    assert result.details["unclassified_visible"] is False


def test_aggregation_parity_rejects_extra_business_rows():
    evaluator = MacroAnalyticsEvaluator.__new__(MacroAnalyticsEvaluator)
    graph = duckdb.connect(":memory:")
    graph.execute("""CREATE TABLE graph_label_assignments (
        canonical_label VARCHAR, label_type VARCHAR, source_session_id BIGINT
    )""")
    for label_type in ("QUESTION_FUNCTION", "DIFFICULTY_SIGNAL", "MISCONCEPTION", "EXAM_ERROR", "TUTOR_STRATEGY"):
        graph.execute("INSERT INTO graph_label_assignments VALUES (?, ?, 1)", ["KNOWN", label_type])
    source = duckdb.connect(":memory:")
    source.execute("CREATE TABLE tutoring_sessions (intervention_id BIGINT)")
    source.execute("INSERT INTO tutoring_sessions VALUES (1)")
    extra_rows = [
        {"rank": 1, "canonical_name": "KNOWN", "event_count": 1, "distinct_session_count": 1, "session_share": 1.0},
        {"rank": 2, "canonical_name": "EXTRA", "event_count": 1, "distinct_session_count": 1, "session_share": 1.0},
    ]
    analytics = SimpleNamespace(**{
        name: (lambda top_k, rows=extra_rows: SimpleNamespace(rows=rows))
        for name in (
            "frequent_student_questions", "difficult_concepts", "recurring_misconceptions",
            "exam_error_patterns", "tutor_strategy_patterns",
        )
    })

    result = evaluator._eval_aggregation_parity(analytics, graph, source)

    assert result.status == MetricStatus.FAILED
    assert any("多出 Rank 2" in item for item in result.details["mismatches"])
    graph.close()
    source.close()


def test_real_macro_analytics_evaluation():
    """在真实全量知识图谱资产上运行宏观分析评测，断言 5 大核心指标全过。"""
    project_root = Path(__file__).resolve().parents[1]
    staging_dir = project_root / "reports" / "staging" / "session-card-graph-v1"
    latest_graph = find_latest_graph_db(staging_dir)
    source_db = project_root / "data" / "db" / "tutoring_knowledge.duckdb"

    evaluator = MacroAnalyticsEvaluator(
        graph_db_path=latest_graph,
        source_db_path=source_db,
    )

    report = evaluator.evaluate()

    assert isinstance(report, MacroAnalyticsReport)
    assert report.release_status == "PASSED"
    assert report.hard_gates_passed is True
    assert set(report.metrics.keys()) == {
        "aggregation_parity",
        "subject_filter_accuracy",
        "cooccurrence_topology_precision",
        "representative_evidence_integrity",
        "unclassified_visibility_rate",
    }

    # 校验具体数值
    assert report.metrics["aggregation_parity"].value == 1.0
    assert report.metrics["subject_filter_accuracy"].value == 1.0
    assert report.metrics["cooccurrence_topology_precision"].value == 1.0
    assert report.metrics["representative_evidence_integrity"].value == 1.0
    assert report.metrics["unclassified_visibility_rate"].value == 1.0

    # 校验 Markdown 输出
    md = format_macro_analytics_markdown(report)
    assert "# 知识图谱宏观统计与拓扑查询准确度评测报告" in md
    assert "PASSED" in md
