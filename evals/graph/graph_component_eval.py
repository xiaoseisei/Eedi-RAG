"""
================================================================================
模块名称: evals/graph/graph_component_eval.py
业务定位: L1 组件层评测 —— Graph 编排链条件验证 (Conditional Component Evaluation)

背景与目的
--------------------------------------------------------------------------------
此前评测把 Router/Planner/Graph/Card/Window/Prompt/Citation 混在一个 83 条
总通过率里统计，结果是 55/83 条失败断在 ROUTER 阶段，Graph 及之后的所有
组件**根本没有被执行到**，因此指标无法回答"Graph 是否正常"。

本套件按 L1 组件层规范，把路由器固定为已知正确值（从 Gold 的 expected_router
注入 QueryUnderstanding），从而让所有下游指标变成**"路由正确前提下"的条件指标**。
这样得到的是纯粹的组件能力，而不是"路由会不会认词"的混合结果。

固定前提
--------------------------------------------------------------------------------
  router_exact_match 恒为平凡通过（因为注入的就是期望值），不参与判断。
  本套件衡量的是: 在正确意图/作用域/视角下，
    Planner -> GraphAnalytics -> GraphEvidenceBridge -> Card -> Window -> Prompt
  这条链路是否产出与 Gold 一致的证据编排。

不做的三件事
--------------------------------------------------------------------------------
  1. 不修改 Gold、不修改被测代码；
  2. 不因失败放宽阈值；
  3. 不把 BLOCKED(未执行) 折算为 FAIL 或 PASS。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business_graph_metrics import evaluate_case
from src.business_graph_pipeline import BusinessGraphPipeline, BusinessGraphRuntimeConfig
from src.business_models import (
    BusinessQueryRequest,
    ComputationScope,
    EvidencePerspective,
    QueryIntent,
    QueryScope,
    QueryUnderstanding,
    RouteConfidence,
)
from src.query_intent import _RESPONSE_TYPES

# 下游阶段（Router 之前的阶段不参与本套件判断）
DOWNSTREAM_STAGES = ("PLANNER", "ANALYTICS", "GRAPH", "CARD", "WINDOW", "ASSEMBLY", "CITATION", "STATUS")

# computation_scope -> (requires_aggregation, requires_case_retrieval)
_COMPUTATION_FLAGS = {
    ComputationScope.MACRO: (True, False),
    ComputationScope.MICRO: (False, True),
    ComputationScope.HYBRID: (True, True),
    ComputationScope.REJECT: (False, False),
}


def build_understanding(case: dict[str, Any]) -> QueryUnderstanding:
    """从 Gold 的 expected_router 构造 QueryUnderstanding，用于绕过路由器。"""

    exp = case.get("expected_router") or {}
    if not exp:
        raise ValueError(f"case {case.get('case_id')} 缺少 expected_router，无法固定路由")
    computation = ComputationScope(exp["computation_scope"])
    requires_aggregation, requires_case_retrieval = _COMPUTATION_FLAGS[computation]
    intent = QueryIntent(exp["intent"])
    return QueryUnderstanding(
        raw_query=str(case["query"]),
        primary_intent=intent,
        secondary_intents=[],
        scope=QueryScope(exp["scope"]),
        computation_scope=computation,
        evidence_perspectives=[EvidencePerspective(p) for p in exp["perspectives"]],
        subject_filters=list(case.get("subject_filters") or []),
        protected_tokens=[],
        requested_top_k=int(case.get("top_k") or 3),
        requires_aggregation=requires_aggregation,
        requires_case_retrieval=requires_case_retrieval,
        response_type=_RESPONSE_TYPES[intent],
        confidence=RouteConfidence.HIGH,
        route_source="deterministic",
    )


@dataclass
class StageOutcome:
    """单个下游阶段的条件通过统计。"""

    stage: str
    passed: int = 0
    total: int = 0
    blocked: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        return self.passed / self.total if self.total else None


def _load_cases(dataset_dir: Path) -> list[dict[str, Any]]:
    """载入 split 数据集（queries + 三个 evaluator-only 视图）。"""

    views: dict[str, list[dict[str, Any]]] = {}
    for name in ("queries", "expected_facts", "expected_graph_paths", "expected_evidence"):
        rows: list[dict[str, Any]] = []
        for line in (dataset_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"{name}.jsonl is empty")
        views[name] = rows
    by_name = {k: {str(r["case_id"]): r for r in v} for k, v in views.items()}
    ordered = [str(r["case_id"]) for r in views["queries"]]
    cases: list[dict[str, Any]] = []
    for cid in ordered:
        merged: dict[str, Any] = {"case_id": cid}
        for name in ("queries", "expected_facts", "expected_graph_paths", "expected_evidence"):
            merged.update({k: v for k, v in by_name[name][cid].items() if k != "case_id"})
        cases.append(merged)
    return cases


def evaluate_component(
    cases: list[dict[str, Any]],
    runtime: BusinessGraphPipeline,
) -> dict[str, Any]:
    """在固定路由前提下，评估下游组件链路。"""

    stages: dict[str, StageOutcome] = {s: StageOutcome(stage=s) for s in DOWNSTREAM_STAGES}
    per_case: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case in cases:
        understanding = build_understanding(case)
        trace = runtime.prepare_case(
            str(case["query"]),
            request=BusinessQueryRequest(query=str(case["query"])),
            understanding=understanding,
        )
        evaluation = evaluate_case(case, trace)

        # 注入的是期望路由，因此 router 检查必然通过；此处断言它，防止误用
        if not evaluation.checks.get("router_exact_match"):
            raise AssertionError(
                f"路由注入失效: {case['case_id']} 的 router 检查未通过，"
                "说明 prepare_case 未按注入的 understanding 执行"
            )

        first = evaluation.first_failure_stage
        # 逐阶段累计：某阶段之后的阶段在该 case 上视为未执行
        passed_all_stages = True
        for stage in DOWNSTREAM_STAGES:
            if not passed_all_stages:
                break
            outcome = stages[stage]
            stage_passed = _stage_passed(stage, evaluation)
            if stage_passed is None:
                # 该阶段在 Gold 中无期望，不计入分母
                continue
            outcome.total += 1
            if stage_passed:
                outcome.passed += 1
            else:
                passed_all_stages = False
                if len(outcome.failures) < 12:
                    outcome.failures.append(
                        {
                            "case_id": case["case_id"],
                            "reason_codes": list(evaluation.reason_codes),
                            "first_failure_stage": first,
                        }
                    )

        per_case.append(
            {
                "case_id": case["case_id"],
                "category": case.get("category"),
                "expected_intent": understanding.primary_intent.value,
                "expected_scope": understanding.scope.value,
                "first_failure_stage": first,
                "reason_codes": list(evaluation.reason_codes),
                "checks": dict(evaluation.checks),
            }
        )

    return {
        "run_id": f"graph-component-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(cases),
        "routing_policy": "FIXED_FROM_GOLD_EXPECTED_ROUTER (router bypassed)",
        "duration_seconds": round(time.perf_counter() - started, 3),
        "stages": {
            name: {
                "passed": o.passed,
                "total": o.total,
                "rate": (round(o.rate, 6) if o.rate is not None else None),
                "failures": o.failures,
            }
            for name, o in stages.items()
        },
        "cases": per_case,
    }


def _stage_passed(stage: str, evaluation: Any) -> bool | None:
    """把 CaseEvaluation 的 checks 映射到阶段；无期望时返回 None（不计分母）。"""

    c = evaluation.checks
    mapping = {
        "PLANNER": lambda: c.get("planner_exact_match"),
        "ANALYTICS": lambda: c.get("sql_graph_scope_confinement"),
        "GRAPH": lambda: c.get("sql_graph_scope_confinement"),
        "CARD": lambda: c.get("card_evidence_validity"),
        "WINDOW": lambda: c.get("window_coverage"),
        "ASSEMBLY": lambda: c.get("prompt_window_coverage"),
        "CITATION": lambda: all(
            c.get(k) for k in ("citation_precision", "citation_recall", "role_accuracy", "exact_quote_match")
        ),
        "STATUS": lambda: c.get("status_accuracy"),
    }
    fn = mapping.get(stage)
    return None if fn is None else bool(fn())


def format_report(result: dict[str, Any]) -> str:
    lines = [
        "# L1 组件层报告 —— Graph 编排链条件验证",
        "",
        f"> **Run ID**: `{result['run_id']}`  ",
        f"> **用例数**: `{result['case_count']}`  ",
        f"> **路由策略**: `{result['routing_policy']}`  ",
        f"> **耗时**: `{result['duration_seconds']}s`  ",
        "",
        "本报告固定路由为 Gold 期望值（绕过路由器），因此衡量的是",
        "**在正确意图前提下，下游组件链路是否产出与 Gold 一致的证据编排**。",
        "",
        "---",
        "",
        "## 阶段通过率（条件指标）",
        "",
        "| 阶段 | 通过 | 分母 | 通过率 |",
        "|---|---:|---:|---:|",
    ]
    for name, s in result["stages"].items():
        rate = "—" if s["rate"] is None else f"{s['rate']:.4%}"
        lines.append(f"| `{name}` | {s['passed']} | {s['total']} | {rate} |")

    lines.extend(["", "---", "", "## 各阶段失败样例", ""])
    for name, s in result["stages"].items():
        if not s["failures"]:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for f in s["failures"][:12]:
            lines.append(f"- `{f['case_id']}` — {', '.join(f['reason_codes'][:4])}")
        lines.append("")

    first = Counter(c["first_failure_stage"] for c in result["cases"] if c["first_failure_stage"])
    lines.extend(["---", "", "## 首个失败阶段分布", "", "```json", json.dumps(dict(first), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "evals/datasets/business-graph-v1")
    parser.add_argument("--source-db", type=Path, default=PROJECT_ROOT / "data/db/tutoring_knowledge.duckdb")
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--chroma-dir", type=Path, default=PROJECT_ROOT / "data/chroma")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = _load_cases(args.dataset_dir.resolve(strict=True))
    source_db = args.source_db.resolve(strict=True)
    graph_db = args.graph_db.resolve(strict=True)

    import hashlib

    source_sha = hashlib.sha256(source_db.read_bytes()).hexdigest()
    runtime = BusinessGraphPipeline(
        source_db,
        graph_db,
        args.chroma_dir.resolve(strict=True),
        BusinessGraphRuntimeConfig(source_sha256=source_sha),
    )
    try:
        result = evaluate_component(cases, runtime)
    finally:
        runtime.close()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(format_report(result), encoding="utf-8")
    print(json.dumps(
        {
            "output": str(output),
            "case_count": result["case_count"],
            "stages": {k: v["rate"] for k, v in result["stages"].items()},
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
