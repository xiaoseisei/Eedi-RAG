# -*- coding: utf-8 -*-
"""
scripts/eval_top5.py
--------------------
评估 20 条教研基准提问在 Top-1, Top-3, Top-5 下的召回率与排名明细。
支持 Raw Baseline 与 Multi-Perspective RRF 双模型横向对比。
"""

import time
import json
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from scripts.eval_20_queries import BENCHMARK_20_QUERIES

def evaluate_top_k(retriever: DualMetricRetriever, top_k: int = 5, use_rrf: bool = False):
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    ranks = []
    latencies = []
    details = []

    for item in BENCHMARK_20_QUERIES:
        q_id = item["id"]
        query = item["query"]
        target_sess = item["target_session_id"]
        target_coll = item["target_collection"]
        topic = item["topic"]
        
        t0 = time.perf_counter()
        if use_rrf:
            res_all = retriever.retrieve_multi_perspective_rrf(query, top_k_each=top_k, fetch_evidence=True)
            cands = res_all["misconceptions"] if target_coll == "student_misconceptions" else res_all["strategies"]
        else:
            if target_coll == "student_misconceptions":
                cands = retriever.retrieve_misconceptions(query, top_k=top_k, fetch_evidence=True)
            else:
                cands = retriever.retrieve_strategies(query, top_k=top_k, fetch_evidence=True)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        
        hit_rank = None
        for r_idx, c in enumerate(cands, start=1):
            if c["metadata"].get("session_id") == target_sess:
                hit_rank = r_idx
                break
                
        if hit_rank == 1:
            hit_1 += 1
        if hit_rank and hit_rank <= 3:
            hit_3 += 1
        if hit_rank and hit_rank <= 5:
            hit_5 += 1
            
        rr = (1.0 / hit_rank) if hit_rank else 0.0
        ranks.append(rr)
        
        top_cand = cands[0] if cands else None
        top_title = ""
        score = 0.0
        if top_cand:
            top_title = top_cand["metadata"].get("misconception_name") or top_cand["metadata"].get("key_aha_question") or ""
            score = top_cand.get("rrf_score") or top_cand.get("hybrid_score", 0.0)
            
        hit_card_title = ""
        hit_card_score = 0.0
        if hit_rank and hit_rank <= len(cands):
            hit_c = cands[hit_rank - 1]
            hit_card_title = hit_c["metadata"].get("misconception_name") or hit_c["metadata"].get("key_aha_question") or ""
            hit_card_score = hit_c.get("rrf_score") or hit_c.get("hybrid_score", 0.0)

        details.append({
            "id": q_id,
            "query": query,
            "target_session_id": target_sess,
            "target_collection": target_coll,
            "topic": topic,
            "hit_rank": hit_rank,
            "latency": lat,
            "top_title": top_title,
            "top_score": score,
            "hit_card_title": hit_card_title,
            "hit_card_score": hit_card_score,
            "all_candidates": [
                {
                    "rank": idx + 1,
                    "session_id": c["metadata"].get("session_id"),
                    "title": c["metadata"].get("misconception_name") or c["metadata"].get("key_aha_question") or "",
                    "score": c.get("rrf_score") or c.get("hybrid_score", 0.0)
                }
                for idx, c in enumerate(cands)
            ]
        })
        
    total = len(BENCHMARK_20_QUERIES)
    return {
        "total": total,
        "hit_1": hit_1, "hit_1_rate": hit_1 / total * 100,
        "hit_3": hit_3, "hit_3_rate": hit_3 / total * 100,
        "hit_5": hit_5, "hit_5_rate": hit_5 / total * 100,
        "mrr": sum(ranks) / total,
        "avg_lat": sum(latencies) / total,
        "details": details
    }

def print_report(raw_res, rrf_res):
    print("=" * 95)
    print("📊 【20 条真实教研提问端到端召回率 Top-1 / Top-3 / Top-5 对比总榜】")
    print("=" * 95)
    print(f"{'评测维度 / 指标':<30} | {'原始基线 (Raw Query)':<24} | {'多视角 RRF 升级版':<24} | {'提升幅度'}")
    print("-" * 95)
    
    h1_raw, h1_up = raw_res['hit_1'], rrf_res['hit_1']
    h3_raw, h3_up = raw_res['hit_3'], rrf_res['hit_3']
    h5_raw, h5_up = raw_res['hit_5'], rrf_res['hit_5']
    mrr_raw, mrr_up = raw_res['mrr'], rrf_res['mrr']
    lat_raw, lat_up = raw_res['avg_lat'], rrf_res['avg_lat']
    
    print(f"{'Top-1 首位命中率 (Hit@1)':<26} | {h1_raw}/20 ({raw_res['hit_1_rate']:.1f}%)"
          f"                | {h1_up}/20 ({rrf_res['hit_1_rate']:.1f}%)"
          f"                | +{rrf_res['hit_1_rate'] - raw_res['hit_1_rate']:.1f}%")
    print(f"{'Top-3 前三命中率 (Hit@3)':<26} | {h3_raw}/20 ({raw_res['hit_3_rate']:.1f}%)"
          f"                | {h3_up}/20 ({rrf_res['hit_3_rate']:.1f}%)"
          f"                | +{rrf_res['hit_3_rate'] - raw_res['hit_3_rate']:.1f}%")
    print(f"{'Top-5 前五命中率 (Hit@5)':<26} | {h5_raw}/20 ({raw_res['hit_5_rate']:.1f}%)"
          f"                | {h5_up}/20 ({rrf_res['hit_5_rate']:.1f}%)"
          f"                | +{rrf_res['hit_5_rate'] - raw_res['hit_5_rate']:.1f}%")
    print(f"{'平均倒数排名 (MRR)':<28} | {mrr_raw:.4f}"
          f"                      | {mrr_up:.4f}"
          f"                      | +{mrr_up - mrr_raw:.4f}")
    print(f"{'平均单次检索延迟 (Latency)':<24} | {lat_raw:.2f} ms"
          f"                    | {lat_up:.2f} ms"
          f"                    | 高吞吐极速")
    print("=" * 95)
    
    print("\n" + "=" * 95)
    print("📝 【20 条提问端到端检索明细详表 (Top-5 命中追踪与横向对照)】")
    print("=" * 95)
    print(f"{'ID':<4} | {'基线状态':<10} | {'基线排名':<8} | {'升级状态':<10} | {'升级排名':<8} | {'目标会话':<12} | {'用户输入提问'}")
    print("-" * 95)
    
    for r, u in zip(raw_res['details'], rrf_res['details']):
        r_rank = r['hit_rank']
        u_rank = u['hit_rank']
        
        r_status = f"✅ HIT@{r_rank}" if (r_rank and r_rank == 1) else (f"⚠️ HIT@{r_rank}" if r_rank else "❌ MISS")
        u_status = f"✅ HIT@{u_rank}" if (u_rank and u_rank == 1) else (f"⚠️ HIT@{u_rank}" if u_rank else "❌ MISS")
        
        r_rk_str = f"#{r_rank}" if r_rank else "N/A"
        u_rk_str = f"#{u_rank}" if u_rank else "N/A"
        
        print(f"#{r['id']:02d}  | {r_status:<10} | {r_rk_str:<8} | {u_status:<10} | {u_rk_str:<8} | Session #{r['target_session_id']:<4} | {r['query']}")

    print("=" * 95)


def main():
    storage = DualEngineStorageManager(db_path="data/db/tutoring_knowledge.duckdb", chroma_dir="data/chroma", embedding_backend="deterministic")
    retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic", alpha=0.5)

    raw_res = evaluate_top_k(retriever, top_k=5, use_rrf=False)
    rrf_res = evaluate_top_k(retriever, top_k=5, use_rrf=True)

    print_report(raw_res, rrf_res)

    storage.close()


if __name__ == "__main__":
    main()
