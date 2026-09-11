"""
================================================================================
测试模块: tests/test_router_review_promotion.py
业务定位: 验证 Router 草案验收集提升为正式人工 Gold 的受控流程

提升边界:
  1. 只接受完整 135 条、全部 label_source=draft 的 v2 队列;
  2. 只改变 label_source, 不改变 query、期望标签或 case_id;
  3. 源草案保持不变, 正式文件已存在时拒绝覆盖。
================================================================================
"""

import json

import pytest

from scripts.generate_router_review_queue import build_review_cases
from scripts.promote_router_review_queue import promote_review_queue


def test_promotion_changes_only_label_source_and_preserves_draft(tmp_path) -> None:
    """人工确认后提升 135 条队列, 源文件内容必须保持完全不变。"""

    rows = build_review_cases()
    source = tmp_path / "review-v2.jsonl"
    source_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    source.write_text(source_text, encoding="utf-8")
    target = tmp_path / "query-router-v1.jsonl"

    result = promote_review_queue(source, target)

    assert result["case_count"] == 135
    assert result["label_source"] == "human"
    assert source.read_text(encoding="utf-8") == source_text
    promoted = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(promoted) == 135
    assert all(row["label_source"] == "human" for row in promoted)
    assert [row["case_id"] for row in promoted] == [row["case_id"] for row in rows]


def test_promotion_refuses_non_draft_input(tmp_path) -> None:
    """已经标记 human 的输入不能再次被提升, 防止隐藏二次修改。"""

    rows = build_review_cases()
    rows[0]["label_source"] = "human"
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="draft"):
        promote_review_queue(source, tmp_path / "target.jsonl")


def test_promotion_refuses_existing_target(tmp_path) -> None:
    """正式 Gold 文件存在时必须拒绝覆盖。"""

    rows = build_review_cases()
    source = tmp_path / "review.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    target = tmp_path / "target.jsonl"
    target.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        promote_review_queue(source, target)
