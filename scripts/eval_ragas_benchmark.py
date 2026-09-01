"""
================================================================================
脚本名称: scripts/eval_ragas_benchmark.py
业务定位: Step 6 - 30 题黄金基准集 RAGAS 量化评测与白盒诊断套件 (RAGAS Benchmark Evaluation Suite)
核心功能:
  1. 载入用户提供的 30 题 Golden Benchmark 黄金数据集 (data/golden_test_set.json)。
  2. 驱动端到端教研 RAG 管道 (EndToEndPedagogicalRAGPipeline) 逐题执行推理与知识检索。
  3. 零侵入提取四元组: (Question, Contexts, Answer, Ground Truth, Verbatim Quotes)。
  4. 计算 RAGAS 核心四大指标 (Faithfulness, Answer Relevance, Context Precision, Context Recall)
     以及教研垂直事实引用率 (Citation Hit) 与 意图对齐率 (Intent Accuracy)。
  5. 依据故障定位矩阵 (Root-Cause Matrix) 输出 30 题逐案诊断明细与全盘均值雷达报表。
  6. 导出 data/ragas_evaluation_report.json 与 data/ragas_evaluation_report.md。
================================================================================
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from src.reranker import PedagogicalGoldAssembler
from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
from src.evaluation import RagasEvaluatorEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s"
)
logger = logging.getLogger(__name__)


def _extract_predicted_intent(response: Any) -> Optional[str]:
    """Read a real pipeline prediction; never substitute a golden label."""
    value = getattr(response, "predicted_intent", None)
    return value if isinstance(value, str) and value.strip() else None


def _mean_measured(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    """Average only measured numeric values; all-unmeasured stays None."""
    values = [row.get(key) for row in rows]
    measured = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(measured) / len(measured), 4) if measured else None


def _format_metric(value: Optional[float], digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


def run_ragas_benchmark(
    golden_test_set_path: str = "data/golden_test_set.json",
    db_path: str = "data/db/tutoring_knowledge.duckdb",
    chroma_dir: str = "data/chroma",
    out_report_json: str = "data/ragas_evaluation_report.json",
    out_report_md: str = "data/ragas_evaluation_report.md",
    allow_heuristic_fallback: bool = False,
    generation_mode: Literal["llm", "deterministic"] = "llm",
):
    """
    运行 30 题黄金基准集全量评测并输出量化报表。
    """
    logger.info("🚀 [Step6_Eval] 启动 RAGAS 黄金基准集自动化量化评测...")
    if generation_mode not in {"llm", "deterministic"}:
        raise ValueError("generation_mode 必须是 'llm' 或 'deterministic'")
    
    # 1. 载入 30 题黄金基准集
    with open(golden_test_set_path, "r", encoding="utf-8") as f:
        cases: List[Dict[str, Any]] = json.load(f)
        
    logger.info(f"📋 成功载入 {len(cases)} 条 Golden Benchmark 测试用例！")
    
    # 2. 初始化双引擎与端到端管道
    storage = DualEngineStorageManager(
        db_path=db_path,
        chroma_dir=chroma_dir,
        embedding_backend="deterministic",
    )
    retriever = DualMetricRetriever(
        storage_manager=storage,
        query_rewrite_mode="deterministic",
    )
    assembler = PedagogicalGoldAssembler(lambda_diversity=0.7)
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=retriever,
        assembler=assembler
    )
    evaluator = RagasEvaluatorEngine(
        allow_heuristic_fallback=allow_heuristic_fallback,
    )
    
    case_results = []
    start_time = time.time()
    
    print("\n" + "=" * 105)
    print(f"🎯 {'Eedi-RAG 全链路 RAGAS 评测流水线正在执行 (30 题基准)':^95}")
    print("=" * 105)
    print(f"{'#':<3} | {'考点模块':<22} | {'意图':<15} | {'Recall':<7} | {'Prec':<7} | {'Faith':<7} | {'Relev':<7} | {'Citation':<8} | {'RAGAS总分':<8}")
    print("-" * 105)

    for idx, case in enumerate(cases, 1):
        q = case["question"]
        gt = case["ground_truth"]
        exp_cat = case.get("category")
        verbatim_quotes = case.get("verbatim_grounding_quotes", [])
        subj_leaf = case.get("subject_path", "通用数学").split(" > ")[-1][:18]
        
        # 1. 端到端推理
        resp = pipeline.ask(query=q, mode=generation_mode)
        
        # 2. 提取 Contexts 与 Answer
        contexts = [
            src.get("document_text", "")
            for src in resp.retrieved_sources_debug
            if src.get("document_text")
        ]
        ans = resp.answer_content or resp.misconception_diagnosis
        actual_intent = _extract_predicted_intent(resp)
        
        # 3. 评测
        eval_metrics = evaluator.evaluate_case(
            question=q,
            answer=ans,
            contexts=contexts,
            ground_truth=gt,
            verbatim_quotes=verbatim_quotes,
            actual_citations=resp.dialogue_citations,
            expected_category=exp_cat,
            actual_intent=actual_intent,
        )
        
        row_result = {
            "case_id": idx,
            "question": q,
            "subject_path": case.get("subject_path"),
            "category": exp_cat,
            "actual_intent": actual_intent,
            "ground_truth": gt,
            "generated_answer_snippet": ans[:120] + "...",
            "retrieved_cards_count": len(resp.retrieved_sources_debug),
            **eval_metrics
        }
        case_results.append(row_result)
        
        # 控制台实时打点输出
        cat_disp = f"[{exp_cat[:13]}]" if exp_cat else "[N/A]"
        print(
            f"{idx:<3} | {subj_leaf:<22} | {cat_disp:<15} | "
            f"{eval_metrics['context_recall']:<7.3f} | "
            f"{eval_metrics['context_precision']:<7.3f} | "
            f"{eval_metrics['faithfulness']:<7.3f} | "
            f"{eval_metrics['answer_relevance']:<7.3f} | "
            f"{eval_metrics['citation_hit']:<8.3f} | "
            f"{eval_metrics['ragas_score']:<8.3f}"
        )
        
    total_duration = time.time() - start_time
    print("-" * 105)
    
    # 4. 计算全局均值雷达
    avg_recall = _mean_measured(case_results, "context_recall")
    avg_prec = _mean_measured(case_results, "context_precision")
    avg_faith = _mean_measured(case_results, "faithfulness")
    avg_relev = _mean_measured(case_results, "answer_relevance")
    avg_citation = _mean_measured(case_results, "citation_hit")
    avg_intent = _mean_measured(case_results, "intent_match")
    avg_ragas = _mean_measured(case_results, "ragas_score")

    statuses = {row["eval_status"] for row in case_results}
    modes = {row["eval_mode"] for row in case_results}
    overall_status = (
        next(iter(statuses)) if len(statuses) == 1 else "PARTIAL"
    ) if statuses else "PARTIAL"
    overall_mode = next(iter(modes)) if len(modes) == 1 else "MIXED"
    generated_at = datetime.now(timezone.utc).isoformat()
    
    summary = {
        "total_test_cases": len(case_results),
        "total_duration_seconds": round(total_duration, 2),
        "generated_at": generated_at,
        "eval_mode": overall_mode,
        "eval_status": overall_status,
        "evaluation_config": evaluator.configuration(),
        "generation_config": {
            "mode": generation_mode,
            "embedding_backend": "deterministic",
            "query_rewrite_mode": "deterministic",
        },
        "average_metrics": {
            "context_recall": avg_recall,
            "context_precision": avg_prec,
            "faithfulness": avg_faith,
            "answer_relevance": avg_relev,
            "citation_accuracy": avg_citation,
            "intent_accuracy": avg_intent,
            "global_ragas_score": avg_ragas,
        },
        "case_details": case_results
    }
    
    # 5. 输出控制台全局雷达
    print(f"\n{'📊 【Eedi-RAG 30 题黄金基准集 RAGAS 评测总成绩单】':^80}")
    print("=" * 80)
    print(f"  * 评测用例总数:          {len(case_results)} 题")
    print(f"  * 全流程评测耗时:        {total_duration:.2f} 秒 (平均 {total_duration/len(case_results):.2f}s/题)")
    print(f"  * 评测模式/状态:         {overall_mode} / {overall_status}")
    print(f"  * 🌟 全局综合 RAGAS 总分:  {_format_metric(avg_ragas, 4)} / 1.0000")
    print("--------------------------------------------------------------------------------")
    print(f"  [检索侧] Context Recall (上下文召回率):      {_format_metric(None if avg_recall is None else avg_recall*100, 1)}%")
    print(f"  [检索侧] Context Precision (上下文精准度):   {_format_metric(None if avg_prec is None else avg_prec*100, 1)}%")
    print(f"  [生成侧] Faithfulness (事实忠实度/无幻觉率): {_format_metric(None if avg_faith is None else avg_faith*100, 1)}%")
    print(f"  [生成侧] Answer Relevance (回答相关性):      {_format_metric(None if avg_relev is None else avg_relev*100, 1)}%")
    print(f"  [溯源侧] Citation Accuracy (原声引用命中率):  {_format_metric(None if avg_citation is None else avg_citation*100, 1)}%")
    print(f"  [意图侧] Intent Accuracy (意图分类准确率):    {_format_metric(None if avg_intent is None else avg_intent*100, 1)}%")
    print("=" * 80)
    
    # 6. 保存 JSON 报告
    with open(out_report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    # 7. 保存 Markdown 深度报告
    md_lines = [
        "# 🎓 Eedi-RAG 30 题黄金基准集 RAGAS 全链路量化评测报告\n",
        f"> **生成时间**: {generated_at} | **生成模式**: {generation_mode} | **评测模式/状态**: {overall_mode}/{overall_status} | **测试集规模**: {len(case_results)}\n",
        "## 📊 一、 全局核心量化指标雷达 (Global Scorecard)\n",
        "| 指标分类 | 评测指标 (Metric) | 测量得分 | 工业级达标线 (Target) | 状态 |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **检索层** | **Context Recall (召回率)** | **{_format_metric(None if avg_recall is None else avg_recall*100, 1)}%** | ≥ 75.0% | {'未测' if avg_recall is None else ('✅ 达标' if avg_recall>=0.75 else '⚠️ 待调优')} |",
        f"| **检索层** | **Context Precision (精准度)** | **{_format_metric(None if avg_prec is None else avg_prec*100, 1)}%** | ≥ 70.0% | {'未测' if avg_prec is None else ('✅ 达标' if avg_prec>=0.70 else '⚠️ 待调优')} |",
        f"| **生成层** | **Faithfulness (事实忠实度)** | **{_format_metric(None if avg_faith is None else avg_faith*100, 1)}%** | ≥ 80.0% | {'未测' if avg_faith is None else ('✅ 达标' if avg_faith>=0.80 else '⚠️ 待调优')} |",
        f"| **生成层** | **Answer Relevance (回答相关度)** | **{_format_metric(None if avg_relev is None else avg_relev*100, 1)}%** | ≥ 75.0% | {'未测' if avg_relev is None else ('✅ 达标' if avg_relev>=0.75 else '⚠️ 待调优')} |",
        f"| **溯源层** | **Citation Accuracy (原声引用率)** | **{_format_metric(None if avg_citation is None else avg_citation*100, 1)}%** | ≥ 70.0% | {'未测' if avg_citation is None else ('✅ 达标' if avg_citation>=0.70 else '⚠️ 待调优')} |",
        f"| **意图层** | **Intent Accuracy (意图分类准确率)** | **{_format_metric(None if avg_intent is None else avg_intent*100, 1)}%** | ≥ 85.0% | {'未测' if avg_intent is None else ('✅ 达标' if avg_intent>=0.85 else '⚠️ 待调优')} |",
        f"| **综合** | **Global RAGAS Score** | **{_format_metric(avg_ragas, 4)}** | ≥ 0.7500 | {'未测' if avg_ragas is None else ('✅ 优秀' if avg_ragas>=0.75 else '⚠️ 待调优')} |",
        "\n---\n",
        "## 🔍 二、 30 题逐案诊断与根因分析明细表 (Case Breakdown)\n",
        "| # | 提问意图 | 考点模块 | Recall | Precision | Faithfulness | Relevance | 根因诊断 (Root-Cause Diagnosis) |",
        "| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :--- |"
    ]
    
    for r in case_results:
        md_lines.append(
            f"| {r['case_id']} | `{r['question'][:28]}...` | {r['subject_path'].split(' > ')[-1] if r['subject_path'] else '数学'} | "
            f"{r['context_recall']:.2f} | {r['context_precision']:.2f} | {r['faithfulness']:.2f} | {r['answer_relevance']:.2f} | "
            f"{r['root_cause_diagnosis']} |"
        )
        
    md_lines.append("\n---\n")
    md_lines.append("## 💡 三、 调优建议与下阶段行动项\n")
    md_lines.append("1. **学情原声引用进一步增强**：针对部分长尾考点，继续扩充 DuckDB 学生困惑检索模式。")
    md_lines.append(r"2. **MMR 多样性参数校准**：当前 $\lambda=0.7$ 在保持相关性与多样性之间表现稳健，可作为 Baseline 固化。")
    
    with open(out_report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    logger.info(f"✅ 评测报告已完整导出至: {out_report_json} 与 {out_report_md}！")
    storage.close()
    return summary


if __name__ == "__main__":
    run_ragas_benchmark()
