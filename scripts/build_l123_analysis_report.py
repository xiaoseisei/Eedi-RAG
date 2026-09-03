from __future__ import annotations

"""Build the evidence-linked L1/L2/L3 delivery analysis report."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _raw_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def build_report(
    *,
    output_dir: Path,
    l1_closeout_path: Path,
    l1_manifest_path: Path,
    l2_report_path: Path,
    l2_dataset_manifest_path: Path,
    l3_report_path: Path,
    l3_dataset_manifest_path: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite L1/L2/L3 report directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    l1 = _load(l1_closeout_path)
    l1_manifest = _load(l1_manifest_path)
    l2 = _load(l2_report_path)
    l2_dataset = _load(l2_dataset_manifest_path)
    l3 = _load(l3_report_path)
    l3_dataset = _load(l3_dataset_manifest_path)
    hashes = {
        "l1_qwen_combined": l1_manifest["artifact_hashes"]["qwen_combined_sha256"],
        "l2_artifact": l2["artifact_hash_before"],
        "l3_artifact": l3["artifact_hash_before"],
    }
    hash_consistent = len(set(hashes.values())) == 1
    raw_files = {
        "l1_closeout": _raw_file(l1_closeout_path),
        "l1_manifest": _raw_file(l1_manifest_path),
        "l1_observations": _raw_file(l1_closeout_path.parent / "observations.jsonl"),
        "l2_report": _raw_file(l2_report_path),
        "l2_traces": _raw_file(l2_report_path.parent / "traces.jsonl"),
        "l2_metrics": _raw_file(l2_report_path.parent / "metrics.jsonl"),
        "l2_dataset_manifest": _raw_file(l2_dataset_manifest_path),
        "l3_report": _raw_file(l3_report_path),
        "l3_raw_cases": _raw_file(l3_report_path.parent / "raw_cases.jsonl"),
        "l3_dataset_manifest": _raw_file(l3_dataset_manifest_path),
    }
    l1_aggregate = [item for item in l1.get("aggregate_metrics", [])]
    l2_aggregate = l2.get("metric_aggregates", [])
    l3_metrics = l3.get("metrics", {})

    def l2_mean(track: str, metric: str) -> Any:
        row = next(
            (item for item in l2_aggregate if item.get("track") == track and item.get("metric") == metric),
            None,
        )
        return None if row is None else row.get("mean")

    l2_metric_count = sum(int(item.get("measured", 0) or 0) for item in l2_aggregate)
    l2_gold_geval = l2_mean("gold", "pedagogical_geval")
    l2_real_faithfulness = l2_mean("real", "faithfulness")
    component_conclusions = [
        {
            "layer": "L1",
            "component": "Storage / Chunk Structure",
            "evidence": "ID parity, metadata, orphan, pointer, verbatim and boundary gates passed in the latest closeout.",
            "conclusion": "Keep current artifact contract and immutable read path; no structural rollback needed.",
        },
        {
            "layer": "L1",
            "component": "Retriever / Fusion",
            "evidence": "Qwen single-slot Recall@5/MRR/nDCG=1.0; selected raw_first fusion lift=0.0; retrieval P95 exceeded 300ms.",
            "conclusion": "Preserve raw_first as non-degrading diagnostic policy; optimize API/Chroma latency and replace single-positive qrels before claiming quality.",
        },
        {
            "layer": "L1",
            "component": "Fallback Chunk",
            "evidence": "Turn Recall@3=0.7944 and MRR=0.8461; dev lower than holdout.",
            "conclusion": "Improve dialogue-window query/representation and validate on larger independent graded qrels.",
        },
        {
            "layer": "L1",
            "component": "Grounding",
            "evidence": "Citation precision/role/quote≈0.9889, authorization=1.0, citation recall=0.5889.",
            "conclusion": "Keep quadruple/candidate authorization hard gates; improve generator citation coverage without copying gold evidence.",
        },
        {
            "layer": "L1",
            "component": "Assembler",
            "evidence": "Candidate retention/evidence recall=1.0, budget/evidence-empty=0, latency P95≈3.6ms after slot/qrels correction.",
            "conclusion": "Independent slots and raw rank boost are effective; global two-slot precision/noise remains diagnostic under one-slot qrels.",
        },
        {
            "layer": "L2",
            "component": "Gold/Real Generator + DeepEval",
            "evidence": f"{l2.get('tracks', {}).get('gold_case_count', 0)} Gold + {l2.get('tracks', {}).get('real_case_count', 0)} Real traces, {l2_metric_count} DeepEval metric observations, Judge readiness {l2.get('judge_readiness', {}).get('status')}; Gold pedagogical GEval={l2_gold_geval}, Real faithfulness={l2_real_faithfulness}.",
            "conclusion": "The semantic loop is runnable, but expand calibration/holdout sample and resolve L1 precondition before release.",
        },
        {
            "layer": "L3",
            "component": "Business proxy / feedback",
            "evidence": "30 proxy cases; retrieval/document recall and actionability=1.0, final evidence recall=0.6111; adoption/task/learning/online feedback UNMEASURED.",
            "conclusion": "Use proxy only for technical prioritization; collect real researcher/user events and pre/post learning outcomes before business claims.",
        },
    ]
    report = {
        "schema_version": "l123-analysis-report/v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "delivery_decision": "L1_BLOCKED_L2_VALIDATED_L3_BUSINESS_UNMEASURED",
        "l1": {
            "decision": l1.get("decision"),
            "status_counts": l1.get("status_counts"),
            "hard_blocker_count": l1.get("hard_blocker_count"),
            "aggregate_metrics": l1_aggregate,
            "closeout_path": str(l1_closeout_path.resolve()),
        },
        "l2": {
            "release_decision": l2.get("release_decision"),
            "judge_readiness": l2.get("judge_readiness"),
            "tracks": l2.get("tracks"),
            "metric_aggregates": l2_aggregate,
            "trace_status_counts": l2.get("trace_status_counts"),
            "report_path": str(l2_report_path.resolve()),
        },
        "l3": {
            "release_decision": l3.get("release_decision"),
            "case_count": l3.get("case_count"),
            "metrics": l3_metrics,
            "report_path": str(l3_report_path.resolve()),
        },
        "artifact_hashes": hashes,
        "artifact_hash_consistent": hash_consistent,
        "raw_data_files": raw_files,
        "dataset_manifests": {
            "l1": l1_manifest,
            "l2": l2_dataset,
            "l3": l3_dataset,
        },
        "component_optimization_conclusions": component_conclusions,
        "limitations": [
            "Eedi is a technical proxy, not NBCOT customer data.",
            "L1/L3 qrels are derived from human quote evidence and are not independent multi-document graded judgments.",
            "L2 smoke sample has 2 cases per track; calibration and full holdout semantic evaluation remain required.",
            "Real business adoption, task completion, learning gain, and online feedback are unmeasured.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Eedi-RAG L1/L2/L3 多维分析报告",
        "",
        f"最终交付状态：**{report['delivery_decision']}**",
        "",
        "## 1. 结论摘要",
        "",
        "L1 已完成可审计 closeout，但 Grounding citation recall、fallback Chunk Turn Recall 和 Retrieval latency 未过门禁；因此 L1 不能发布。L2 Gold/Real + DeepEval 闭环已在真实 API/Judge 上跑通，但只完成 2-case smoke，且受 L1 前置阻塞。L3 已完成 30-case Eedi 技术 proxy 召回测评，真实业务效果尚无数据。",
        "",
        "## 2. 原始数据与 provenance",
        "",
        f"- L1 dataset: `{l1_manifest['dataset_version']}`；L2 dataset: `{l2_dataset['dataset_version']}`；L3 dataset: `{l3_dataset['dataset_version']}`",
        f"- Artifact hash 一致：`{hash_consistent}`（{hashes['l1_qwen_combined']}）",
        f"- L1 raw observations: `{raw_files['l1_observations']['path']}`",
        f"- L2 traces/metrics: `{raw_files['l2_traces']['path']}` / `{raw_files['l2_metrics']['path']}`",
        f"- L3 per-case raw data: `{raw_files['l3_raw_cases']['path']}`",
        "",
        "## 3. L1 指标",
        "",
        "| Stage | Metric | Status | Value | Threshold | Hard gate |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in l1_aggregate:
        lines.append(f"| {item['stage']} | {item['metric_name']} | {item['status']} | {item['value']} | {item['threshold']} | {item['hard_gate']} |")
    lines.extend(["", "## 4. L2 Gold/Real DeepEval 指标", "", "| Track | Metric | Measured | Mean | Threshold |", "| --- | --- | ---: | ---: | ---: |"])
    for item in l2_aggregate:
        lines.append(f"| {item['track']} | {item['metric']} | {item['measured']} | {item['mean']} | {item['threshold']} |")
    lines.extend(["", "## 5. L3 业务 proxy 指标", "", "| Metric | Mean | Dev | Holdout | Status |", "| --- | ---: | ---: | ---: | --- |"])
    for name, value in l3_metrics.items():
        if isinstance(value, dict) and "mean" in value:
            lines.append(f"| {name} | {value['mean']:.4f} | {value['dev_mean']:.4f} | {value['holdout_mean']:.4f} | {value['status']} |")
        else:
            lines.append(f"| {name} | — | — | — | {value.get('status', 'UNMEASURED')} |")
    lines.extend(["", "## 6. 组件优化结论", "", "| Layer | Component | Evidence | Conclusion |", "| --- | --- | --- | --- |"])
    for item in component_conclusions:
        lines.append(f"| {item['layer']} | {item['component']} | {item['evidence']} | {item['conclusion']} |")
    lines.extend(["", "## 7. 限制与下一步", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l1-closeout", type=Path, required=True)
    parser.add_argument("--l1-manifest", type=Path, required=True)
    parser.add_argument("--l2-report", type=Path, required=True)
    parser.add_argument("--l2-dataset-manifest", type=Path, required=True)
    parser.add_argument("--l3-report", type=Path, required=True)
    parser.add_argument("--l3-dataset-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        output_dir=args.output_dir,
        l1_closeout_path=args.l1_closeout,
        l1_manifest_path=args.l1_manifest,
        l2_report_path=args.l2_report,
        l2_dataset_manifest_path=args.l2_dataset_manifest,
        l3_report_path=args.l3_report,
        l3_dataset_manifest_path=args.l3_dataset_manifest,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "delivery_decision": report["delivery_decision"], "artifact_hash_consistent": report["artifact_hash_consistent"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
