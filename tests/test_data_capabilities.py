"""
================================================================================
测试模块: tests/test_data_capabilities.py
业务定位: 验证业务查询所依赖的数据能力探测与缺失能力门禁

本测试只使用 DuckDB 内存连接, 重点确认:
  1. schema 探测是只读的, 不会为了"让能力可用"自动创建表;
  2. 当前仅有会话/对白表时, 只能声明真实对白证据能力;
  3. 重考者/基础薄弱群体分析缺少纵向字段时必须显式失败;
  4. 缺失能力异常携带稳定的 capability 与 missing_requirements 字段。
================================================================================
"""

import duckdb
import pytest

from src.data_capabilities import DataCapabilityRegistry, MissingDataCapabilityError


def test_current_eedi_schema_exposes_dialogue_but_not_cohort_capabilities() -> None:
    """只有真实会话与对白表时, 不得虚构学生群体能力。"""

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE tutoring_sessions(intervention_id BIGINT)")
    conn.execute(
        "CREATE TABLE session_dialogue_turns("
        "intervention_id BIGINT, turn_id INTEGER, speaker VARCHAR, text VARCHAR)"
    )
    conn.execute("INSERT INTO session_dialogue_turns VALUES (1, 1, 'student', 'x')")

    registry = DataCapabilityRegistry(source_conn=conn, analytics_conn=None)
    snapshot = registry.inspect()

    assert snapshot["DIALOGUE_EVIDENCE"].available is True
    assert snapshot["RETAKE_COHORT"].available is False
    assert "attempt_number" in snapshot["RETAKE_COHORT"].missing_requirements
    assert snapshot["ABILITY_COHORT"].available is False


def test_empty_required_table_is_not_reported_as_full_coverage() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE session_dialogue_turns("
        "intervention_id BIGINT, turn_id INTEGER, speaker VARCHAR, text VARCHAR)"
    )

    snapshot = DataCapabilityRegistry(source_conn=conn).inspect()["DIALOGUE_EVIDENCE"]

    assert snapshot.available is False
    assert snapshot.coverage == 0.0
    assert "session_dialogue_turns:non_empty" in snapshot.missing_requirements


def test_require_raises_with_machine_readable_missing_fields() -> None:
    """调用方强制要求缺失能力时, 返回结构化异常而不是默认放行。"""

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE tutoring_sessions(intervention_id BIGINT)")
    registry = DataCapabilityRegistry(source_conn=conn, analytics_conn=None)

    with pytest.raises(MissingDataCapabilityError) as captured:
        registry.require(["RETAKE_COHORT"])

    assert captured.value.capability == "RETAKE_COHORT"
    assert "student_key" in captured.value.missing_requirements
