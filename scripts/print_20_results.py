"""
================================================================================
脚本名称: scripts/print_20_results.py
业务定位: 格式化输出 20 条测试问题的检索输出、文本片段与召回统计
================================================================================
"""

import os
import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from scripts.eval_20_queries import BENCHMARK_20_QUERIES

def main():
    storage = DualEngineStorageManager(
        db_path="data/db/tutoring_knowledge.duckdb",
        chroma_dir="data/chroma"
    )
    retriever = DualMetricRetriever(storage_manager=storage, alpha=0.5)

    print("================================================================================")
    print("📋 20 条真实教研提问端到端检索输出详单 (Dual-Metric: Cosine + Euclidean L2)")
    print("================================================================================\n")

    for item in BENCHMARK_20_QUERIES:
        q_id = item["id"]
        q = item["query"]
        target = item["target_session_id"]
        coll = item["target_collection"]
        topic = item["topic"]

        if coll == "student_misconceptions":
            hits = retriever.retrieve_misconceptions(q, top_k=3, fetch_evidence=True)
        else:
            hits = retriever.retrieve_strategies(q, top_k=3, fetch_evidence=True)

        hit_rank = None
        for r_idx, h in enumerate(hits, start=1):
            if h["metadata"].get("session_id") == target:
                hit_rank = r_idx
                break

        top = hits[0]
        top_sess = top["metadata"].get("session_id")
        top_title = top["metadata"].get("misconception_name") or top["metadata"].get("key_aha_question") or ""
        score = top["hybrid_score"]
        cos_score = top["cosine_score"]
        l2_score = top["euclidean_score"]
        turns = top.get("evidence_turns", [])

        status_icon = "✅ [首位命中 Hit@1]" if hit_rank == 1 else ("⚠️ [前三命中 Hit@3]" if hit_rank else "❌ [未命中 Miss]")

        print(f"【问题 #{q_id:02d}】 {status_icon} (目标会话: #{target} | 召回排名: {'#' + str(hit_rank) if hit_rank else 'Top-3外'})")
        print(f"  • 用户输入提问: \"{q}\"")
        print(f"  • 检索目标方向: {coll} ({topic})")
        print(f"  • 召回卡片标题: 【Session #{top_sess}】 {top_title}")
        print(f"  • 相似度评分:   综合={score:.4f} (余弦={cos_score:.4f}, 欧氏={l2_score:.4f})")
        print(f"  • 召回卡片摘要:\n    {top['document'][:180]}...")
        if turns:
            print(f"  • DuckDB 原始实录证据 ({len(turns)} 轮):")
            for t in turns[:2]:
                print(f"    - [Turn {t['turn_id']}] [{t['speaker']}]: {t['text']}")
        print("-" * 80 + "\n")

    storage.close()

if __name__ == "__main__":
    main()
