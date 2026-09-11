"""
================================================================================
测试模块: tests/test_router_review_queue.py
业务定位: 验证 Router 人工审核前的 draft 审阅队列覆盖面与真实性标记

这不是 Router 发布评测。它只验证 AI 起草的候选队列:
  - 覆盖全部 9 个业务意图;
  - 样本数量与语言/范围/公式分布达到人工审核的最低准备条件;
  - 每条都明确标记 draft, 不会被误认成已审核的人类 Gold。
================================================================================
"""

from collections import Counter

from scripts.generate_router_review_queue import build_review_cases


def test_review_queue_has_required_coverage_and_stays_draft() -> None:
    """审阅队列必须覆盖 Part 1 验收面, 但所有标签仍是待人审状态。"""

    cases = build_review_cases()
    intent_counts = Counter(case["primary_intent"] for case in cases)

    assert len(cases) >= 135
    assert len(intent_counts) == 9
    assert all(count >= 15 for count in intent_counts.values())
    assert all(case["label_source"] == "draft" for case in cases)
    assert sum(case["scope"] == "SUBJECT" for case in cases) >= 15
    assert sum(case["scope"] == "QUESTION" for case in cases) >= 15
    assert sum(bool(case["protected_tokens"]) for case in cases) >= 20
    assert sum(
        any(ord(character) < 128 and character.isalpha() for character in case["query"])
        for case in cases
    ) >= 27
