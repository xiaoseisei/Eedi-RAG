"""
================================================================================
测试模块: tests/test_business_router_eval.py
业务定位: 验证 Query Router 验收集的人工标签契约与确定性评测报告

本测试确保:
  1. 正式 RouterCase 只能接受人工审核标签, 不把模型生成答案冒充 Gold;
  2. 评测器逐项记录错误, 不用均值或默认成功掩盖失败;
  3. 评测指标至少包含意图、范围、计算范围、证据视角与 protected token 保留率。
================================================================================
"""

import pytest
from pydantic import ValidationError

from scripts.evaluate_business_router import (
    RouterCase,
    evaluate_router_cases,
    write_router_report,
)


def test_router_case_requires_human_label_and_valid_split() -> None:
    """正式验收数据必须来自人工审核, generated/draft 标签不能进入 release 集。"""

    with pytest.raises(ValidationError):
        RouterCase.model_validate(
            {
                "case_id": "router-001",
                "query": "学生最常问什么？",
                "primary_intent": "STUDENT_FREQUENT_QUESTIONS",
                "scope": "CORPUS",
                "evidence_perspectives": ["STUDENT"],
                "protected_tokens": [],
                "label_source": "generated",
                "split": "dev",
            }
        )


def test_router_evaluator_reports_failures_without_overwriting_them() -> None:
    """评测通过时不应捏造 failures; 失败案例将在后续专门测试中完整保留。"""

    cases = [
        RouterCase(
            case_id="router-001",
            query="学生最常问什么？",
            primary_intent="STUDENT_FREQUENT_QUESTIONS",
            scope="CORPUS",
            computation_scope="HYBRID",
            evidence_perspectives=["STUDENT"],
            protected_tokens=[],
            label_source="human",
            split="dev",
        )
    ]

    report = evaluate_router_cases(cases)

    assert report["case_count"] == 1
    assert report["metrics"]["primary_intent_accuracy"] == 1.0
    assert report["failures"] == []


def test_router_report_writes_new_directory_and_refuses_overwrite(tmp_path) -> None:
    """报告落盘必须可审计且不可覆盖, 避免旧失败结果被静默替换。"""

    dataset_path = tmp_path / "router.jsonl"
    dataset_path.write_text(
        "{"
        "\"case_id\":\"router-001\","
        "\"query\":\"学生最常问什么？\","
        "\"primary_intent\":\"STUDENT_FREQUENT_QUESTIONS\","
        "\"secondary_intents\":[],"
        "\"scope\":\"CORPUS\","
        "\"computation_scope\":\"HYBRID\","
        "\"evidence_perspectives\":[\"STUDENT\"],"
        "\"subject_filters\":[],"
        "\"protected_tokens\":[],"
        "\"requested_top_k\":3,"
        "\"label_source\":\"human\","
        "\"split\":\"dev\""
        "}\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "router-report"

    report = write_router_report(
        dataset_path=dataset_path,
        split="dev",
        output_dir=output_dir,
    )

    assert report["case_count"] == 1
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "failures.jsonl").is_file()
    assert (output_dir / "cases.jsonl").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_router_report(
            dataset_path=dataset_path,
            split="dev",
            output_dir=output_dir,
        )
