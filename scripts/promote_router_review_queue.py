"""
================================================================================
脚本名称: scripts/promote_router_review_queue.py
业务定位: 将已获人工确认的 Router draft 队列提升为正式 human Gold

本脚本是一个受控的"标签状态转换器", 不是标签生成器:
  1. 读取指定 review-v2 JSONL 并要求每行 label_source="draft";
  2. 通过正式 RouterCase schema 验证所有业务字段与 135 条覆盖要求;
  3. 只在内存中把 label_source 改为 "human", 不修改 query/期望标签/case_id;
  4. 以独占创建方式写入 query-router-v1.jsonl, 源草案保持不变;
  5. 输出可追溯的源文件与目标文件路径、数量和状态。

工程真实性约束:
  - 只有调用方明确确认人工审核完成后才允许运行本脚本;
  - 不接受已混入 human/generated 的输入, 防止重复提升或伪造来源;
  - 不覆盖任何既有目标文件, 不生成半成品或默认标签。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_business_router import RouterCase


EXPECTED_CASE_COUNT = 135
EXPECTED_INTENT_COUNT = 15


def _load_draft_cases(source: Path) -> list[dict[str, Any]]:
    """读取并验证 draft 输入; 任一行不满足人工待审状态即立即失败。"""

    resolved = source.resolve(strict=True)
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            raw_case = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {resolved}:{line_number}: {exc}") from exc
        if not isinstance(raw_case, dict):
            raise ValueError(f"router case must be a JSON object at {resolved}:{line_number}")
        if raw_case.get("label_source") != "draft":
            raise ValueError(
                f"promotion input must contain only draft labels at {resolved}:{line_number}"
            )
        promoted_candidate = {**raw_case, "label_source": "human"}
        try:
            validated = RouterCase.model_validate(promoted_candidate)
        except ValidationError as exc:
            raise ValueError(f"invalid RouterCase at {resolved}:{line_number}: {exc}") from exc
        if validated.case_id in seen_ids:
            raise ValueError(f"duplicate router case_id: {validated.case_id}")
        seen_ids.add(validated.case_id)
        # model_dump() provides normalized enum/tuple values while preserving all
        # human-approved expectation fields; no Router call is made here.
        cases.append(validated.model_dump(mode="json"))

    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"promotion requires exactly {EXPECTED_CASE_COUNT} draft cases, got {len(cases)}"
        )
    counts = Counter(case["primary_intent"] for case in cases)
    if set(counts) != set(
        {
            "STUDENT_FREQUENT_QUESTIONS",
            "DIFFICULT_CONCEPTS",
            "RECURRING_MISCONCEPTIONS",
            "EXAM_ERROR_PATTERNS",
            "STUDENT_COHORT_PATTERNS",
            "TUTOR_STRATEGY_PATTERNS",
            "CONTENT_IMPROVEMENT",
            "QUESTION_TUTORING",
            "UNSUPPORTED",
        }
    ) or any(count != EXPECTED_INTENT_COUNT for count in counts.values()):
        raise ValueError(
            "promotion requires all 9 registered intents to contain exactly 15 cases: "
            f"{dict(sorted(counts.items()))}"
        )
    return cases


def promote_review_queue(source: Path, target: Path) -> dict[str, Any]:
    """将完整 draft 队列提升为正式 human 文件, 不覆盖源或目标。"""

    resolved_source = source.resolve(strict=True)
    resolved_target = target.resolve()
    if resolved_source == resolved_target:
        raise ValueError("source and target must be different files")
    if resolved_target.exists():
        raise FileExistsError(f"refusing to overwrite promoted router dataset: {resolved_target}")

    cases = _load_draft_cases(resolved_source)
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    with resolved_target.open("x", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return {
        "source": str(resolved_source),
        "target": str(resolved_target),
        "case_count": len(cases),
        "label_source": "human",
        "release_ready": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口: 所有验证成功后返回 0, 否则以参数错误退出且不创建目标。"""

    parser = argparse.ArgumentParser(
        description="Promote a human-reviewed draft router queue to the formal Gold dataset"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = promote_review_queue(args.source, args.target)
    except (FileExistsError, OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
