"""
================================================================================
脚本名称: scripts/evaluate_business_router.py
业务定位: 业务 Query Router 的人工标签验收集评测与不可覆盖报告生成器

本脚本验证 Router 是否正确把用户问题映射为:
  1. 业务意图 (primary_intent);
  2. 查询范围 (scope);
  3. 计算范围 (macro / micro / hybrid);
  4. 学生/导师证据视角;
  5. 数字、公式、选项等 protected tokens。

真实性边界:
  - 正式 RouterCase 只接受 label_source="human" 的人工审核样本;
  - 脚本绝不调用被测 Router 生成期望标签;
  - 一个失败样本就是失败样本, 写入 failures.jsonl, 不换成默认分数;
  - 输出目录必须是新目录, 防止覆盖先前报告并制造"回归通过"假象。
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# 允许从项目根目录通过 ``python scripts/evaluate_business_router.py`` 运行。
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business_models import BusinessQueryRequest, QueryIntent, QueryScope
from src.query_intent import BusinessQueryRouter


# ---------------------------------------------------------------------------
# 正式验收数据契约与发布门禁
# ---------------------------------------------------------------------------


class RouterCase(BaseModel):
    """
    一条经过人工审核的 Router 验收样本。

    字段保留为字符串而不是复用业务 Enum, 因为 JSONL 是跨语言评测资产;
    field validator 仍严格限制为当前业务契约中的可用枚举值。
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=2000)
    primary_intent: str = Field(min_length=1)
    secondary_intents: list[str] = Field(default_factory=list, max_length=1)
    scope: str = Field(min_length=1)
    computation_scope: Literal["MACRO", "MICRO", "HYBRID", "REJECT"]
    evidence_perspectives: list[Literal["STUDENT", "TUTOR"]] = Field(
        min_length=1,
        max_length=2,
    )
    subject_filters: list[str] = Field(default_factory=list)
    protected_tokens: list[str] = Field(default_factory=list)
    requested_top_k: int = Field(default=3, ge=1, le=10)
    label_source: Literal["human"]
    split: Literal["dev", "holdout"]

    @field_validator("primary_intent")
    @classmethod
    def primary_intent_is_registered(cls, value: str) -> str:
        """禁止手写不存在的意图名称进入正式评测集。"""

        if value not in {intent.value for intent in QueryIntent}:
            raise ValueError(f"unknown primary_intent: {value}")
        return value

    @field_validator("secondary_intents")
    @classmethod
    def secondary_intents_are_registered(cls, values: list[str]) -> list[str]:
        """次意图最多一个, 且也必须来自业务契约枚举。"""

        registered = {intent.value for intent in QueryIntent}
        unknown = [value for value in values if value not in registered]
        if unknown:
            raise ValueError(f"unknown secondary_intents: {unknown}")
        return values

    @field_validator("scope")
    @classmethod
    def scope_is_registered(cls, value: str) -> str:
        """禁止把自由文本范围标签当成可测期望。"""

        if value not in {scope.value for scope in QueryScope}:
            raise ValueError(f"unknown scope: {value}")
        return value

    @field_validator("evidence_perspectives")
    @classmethod
    def perspectives_are_unique_and_stable(
        cls,
        values: list[Literal["STUDENT", "TUTOR"]],
    ) -> list[Literal["STUDENT", "TUTOR"]]:
        """双视角固定 student -> tutor, 保证下游 evidence ID 顺序稳定。"""

        if len(set(values)) != len(values):
            raise ValueError("evidence_perspectives must be unique")
        if values == ["TUTOR", "STUDENT"]:
            raise ValueError("dual perspectives must be ordered STUDENT then TUTOR")
        return values


# 所有 gate 都基于正式 holdout 样本的真实计算结果; 不可测时不得伪造通过。
GATES: dict[str, float] = {
    "primary_intent_accuracy": 0.95,
    "scope_accuracy": 0.95,
    "computation_scope_accuracy": 0.95,
    "perspective_exact_match": 0.95,
    "protected_token_preservation": 1.0,
    "unsupported_precision": 1.0,
    "deterministic_router_p95_ms": 10.0,
}


# ---------------------------------------------------------------------------
# JSONL 读取、单案例比较与指标汇总
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    """按块计算数据集指纹, 报告可追溯至精确输入文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_router_cases(path: Path) -> list[RouterCase]:
    """严格读取正式人工标签 JSONL; 空行跳过, 其他格式问题立即失败。"""

    resolved = path.resolve(strict=True)
    cases: list[RouterCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            raw_case = json.loads(raw_line)
            case = RouterCase.model_validate(raw_case)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid router case at {resolved}:{line_number}: {exc}") from exc
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate router case_id: {case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"router dataset contains no cases: {resolved}")
    return cases


def _p95(values: Sequence[float]) -> float | None:
    """返回 nearest-rank P95; 空数组显式为 None, 不伪造 0ms。"""

    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return round(ordered[index], 6)


def _ratio(numerator: int, denominator: int) -> float | None:
    """安全计算比例; 无分母时返回 None 交由调用方定义业务语义。"""

    if denominator == 0:
        return None
    return numerator / denominator


def _case_result(case: RouterCase, router: BusinessQueryRouter) -> dict[str, Any]:
    """运行单条样本并完整保存 expected / actual / 时延 / 字段级错误。"""

    started = time.perf_counter()
    actual = router.route(
        BusinessQueryRequest(
            query=case.query,
            subject_filters=case.subject_filters,
            top_k=case.requested_top_k,
        )
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    actual_perspectives = [perspective.value for perspective in actual.evidence_perspectives]
    expected_protected = list(case.protected_tokens)
    missing_protected = [
        token for token in expected_protected if token not in actual.protected_tokens
    ]

    checks = {
        "primary_intent": actual.primary_intent.value == case.primary_intent,
        "scope": actual.scope.value == case.scope,
        "computation_scope": actual.computation_scope.value == case.computation_scope,
        "evidence_perspectives": actual_perspectives == case.evidence_perspectives,
        "protected_tokens": not missing_protected,
    }
    mismatches = {
        field: {
            "expected": expected,
            "actual": observed,
        }
        for field, expected, observed, passed in (
            ("primary_intent", case.primary_intent, actual.primary_intent.value, checks["primary_intent"]),
            ("scope", case.scope, actual.scope.value, checks["scope"]),
            (
                "computation_scope",
                case.computation_scope,
                actual.computation_scope.value,
                checks["computation_scope"],
            ),
            (
                "evidence_perspectives",
                case.evidence_perspectives,
                actual_perspectives,
                checks["evidence_perspectives"],
            ),
            ("protected_tokens", expected_protected, actual.protected_tokens, checks["protected_tokens"]),
        )
        if not passed
    }
    return {
        "case_id": case.case_id,
        "split": case.split,
        "query": case.query,
        "expected": {
            "primary_intent": case.primary_intent,
            "scope": case.scope,
            "computation_scope": case.computation_scope,
            "evidence_perspectives": case.evidence_perspectives,
            "protected_tokens": expected_protected,
            "requested_top_k": case.requested_top_k,
        },
        "actual": {
            "primary_intent": actual.primary_intent.value,
            "scope": actual.scope.value,
            "computation_scope": actual.computation_scope.value,
            "evidence_perspectives": actual_perspectives,
            "protected_tokens": actual.protected_tokens,
            "requested_top_k": actual.requested_top_k,
            "confidence": actual.confidence.value,
            "route_source": actual.route_source,
        },
        "checks": checks,
        "missing_protected_tokens": missing_protected,
        "elapsed_ms": round(elapsed_ms, 6),
        "mismatches": mismatches,
    }


def evaluate_router_cases(
    cases: Sequence[RouterCase],
    router: BusinessQueryRouter | None = None,
) -> dict[str, Any]:
    """评测正式人工标签案例, 失败明细与指标同时返回给 CLI/单测。"""

    if not cases:
        raise ValueError("at least one RouterCase is required")
    effective_router = router or BusinessQueryRouter()
    results = [_case_result(case, effective_router) for case in cases]

    count = len(results)
    correct_intent = sum(item["checks"]["primary_intent"] for item in results)
    correct_scope = sum(item["checks"]["scope"] for item in results)
    correct_computation = sum(item["checks"]["computation_scope"] for item in results)
    correct_perspective = sum(item["checks"]["evidence_perspectives"] for item in results)
    expected_protected_count = sum(
        len(item["expected"]["protected_tokens"]) for item in results
    )
    preserved_protected_count = expected_protected_count - sum(
        len(item["missing_protected_tokens"]) for item in results
    )

    predicted_unsupported = [
        item
        for item in results
        if item["actual"]["primary_intent"] == QueryIntent.UNSUPPORTED.value
    ]
    unsupported_true_positive = sum(
        item["expected"]["primary_intent"] == QueryIntent.UNSUPPORTED.value
        for item in predicted_unsupported
    )
    expected_unsupported_count = sum(
        item["expected"]["primary_intent"] == QueryIntent.UNSUPPORTED.value
        for item in results
    )
    unsupported_precision = _ratio(unsupported_true_positive, len(predicted_unsupported))
    # 没有预测任何 UNSUPPORTED 时: 若 Gold 也没有则精度语义上为 1; 否则为 0。
    if unsupported_precision is None:
        unsupported_precision = 1.0 if expected_unsupported_count == 0 else 0.0

    protected_preservation = _ratio(
        preserved_protected_count,
        expected_protected_count,
    )
    if protected_preservation is None:
        protected_preservation = 1.0

    failures = [item for item in results if item["mismatches"]]
    metrics = {
        "primary_intent_accuracy": correct_intent / count,
        "scope_accuracy": correct_scope / count,
        "computation_scope_accuracy": correct_computation / count,
        "perspective_exact_match": correct_perspective / count,
        "protected_token_preservation": protected_preservation,
        "unsupported_precision": unsupported_precision,
        "unsupported_recall": _ratio(unsupported_true_positive, expected_unsupported_count),
        "deterministic_router_p95_ms": _p95([item["elapsed_ms"] for item in results]),
    }
    return {
        "case_count": count,
        "metrics": metrics,
        "failures": failures,
        "cases": results,
    }


# ---------------------------------------------------------------------------
# 运行报告: 不覆盖既有目录, 对未满足门禁如实以非零状态退出
# ---------------------------------------------------------------------------


def _gate_result(metrics: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    """将指标与门禁逐一比对; 时延类指标是小于等于, 其余是大于等于。"""

    result: dict[str, dict[str, Any]] = {}
    for name, threshold in GATES.items():
        value = metrics.get(name)
        if value is None:
            passed = False
        elif name == "deterministic_router_p95_ms":
            passed = value <= threshold
        else:
            passed = value >= threshold
        result[name] = {
            "value": value,
            "threshold": threshold,
            "operator": "<=" if name == "deterministic_router_p95_ms" else ">=",
            "passed": passed,
        }
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    """渲染简明、可人工审核的 Markdown 报告; 完整样本细节仍保留在 JSONL。"""

    lines = [
        "# Business Router Evaluation",
        "",
        f"- Dataset: `{report['dataset']['path']}`",
        f"- Dataset SHA-256: `{report['dataset']['sha256']}`",
        f"- Split: `{report['split']}`",
        f"- Cases: `{report['case_count']}`",
        f"- Release decision: `{report['release_decision']}`",
        "",
        "| Metric | Value | Gate | Result |",
        "| --- | ---: | --- | --- |",
    ]
    for name, gate in report["gates"].items():
        value = gate["value"]
        rendered = "UNMEASURED" if value is None else f"{value:.6f}"
        symbol = "PASS" if gate["passed"] else "FAIL"
        lines.append(
            f"| {name} | {rendered} | {gate['operator']} {gate['threshold']} | {symbol} |"
        )
    lines.extend(
        [
            "",
            f"- Failure cases: `{report['failure_count']}`",
            "- Every failure is preserved in `failures.jsonl`; no metric is substituted for a failed case.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_router_report(
    *,
    dataset_path: Path,
    split: Literal["dev", "holdout"],
    output_dir: Path,
    router: BusinessQueryRouter | None = None,
) -> dict[str, Any]:
    """运行指定 split 并将不可覆盖的 report/failures/cases 写入新目录。"""

    resolved_dataset = dataset_path.resolve(strict=True)
    resolved_output = output_dir.resolve()
    if resolved_output.exists():
        raise FileExistsError(f"refusing to overwrite router evaluation output: {resolved_output}")

    all_cases = load_router_cases(resolved_dataset)
    selected_cases = [case for case in all_cases if case.split == split]
    if not selected_cases:
        raise ValueError(f"no {split} cases in router dataset: {resolved_dataset}")

    measured = evaluate_router_cases(selected_cases, router=router)
    gates = _gate_result(measured["metrics"])
    release_passed = all(item["passed"] for item in gates.values())
    report = {
        "schema_version": "business-router-eval/v1",
        "dataset": {
            "path": str(resolved_dataset),
            "sha256": _file_sha256(resolved_dataset),
            "total_case_count": len(all_cases),
        },
        "split": split,
        "case_count": measured["case_count"],
        "metrics": measured["metrics"],
        "gates": gates,
        "failure_count": len(measured["failures"]),
        "release_decision": "PASS" if release_passed else "BLOCKED",
    }

    resolved_output.mkdir(parents=True, exist_ok=False)
    (resolved_output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (resolved_output / "report.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    with (resolved_output / "failures.jsonl").open("x", encoding="utf-8") as handle:
        for failure in measured["failures"]:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    with (resolved_output / "cases.jsonl").open("x", encoding="utf-8") as handle:
        for case in measured["cases"]:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口: 成功返回 0, 未通过发布门禁返回 1, 输入错误返回 2。"""

    parser = argparse.ArgumentParser(
        description="Evaluate the human-labelled business query router dataset"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        report = write_router_report(
            dataset_path=args.dataset,
            split=args.split,
            output_dir=args.output,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValidationError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "release_decision": report["release_decision"],
                "case_count": report["case_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["release_decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
