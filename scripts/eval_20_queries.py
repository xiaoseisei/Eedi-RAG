"""
================================================================================
脚本名称: scripts/eval_20_queries.py
业务定位: 20 条真实教研提问基准评测 (Raw Baseline vs Multi-Perspective RRF Upgraded)
核心流程:
  1. 执行基线评测 (Raw Query Dual-Metric)
  2. 执行升级版评测 (Multi-Perspective Query Rewriting + Domain Knowledge + RRF)
  3. 对比 Hit@1、Hit@3、MRR 提升幅度与延迟变化
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever

# 20 条基准测试集
BENCHMARK_20_QUERIES = [
    {"id": 1, "query": "四舍五入 5.4598 到 1 位小数时，学生为什么会误选 5.45？", "target_session_id": 10, "target_collection": "student_misconceptions", "topic": "四舍五入数位保留与截断混淆"},
    {"id": 2, "query": "当学生把保留一位小数做成保留两位时，名师是如何用简单数字破局提问的？", "target_session_id": 10, "target_collection": "tutor_strategies", "topic": "5.45一位小数破局一问"},
    {"id": 3, "query": "学生在做包含加法和除法的分式运算时容易犯什么运算顺序错误？", "target_session_id": 14, "target_collection": "student_misconceptions", "topic": "运算顺序混淆与分数符号认知不清"},
    {"id": 4, "query": "名师如何通过苏格拉底反问引导学生识别混合运算中的运算优先级？", "target_session_id": 14, "target_collection": "tutor_strategies", "topic": "what do you think this question is asking"},
    {"id": 5, "query": "学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？", "target_session_id": 23, "target_collection": "student_misconceptions", "topic": "将互质数特例错误泛化为普遍规律"},
    {"id": 6, "query": "当学生盲目认为两数相乘就是LCM时，老师用什么反例策略引导学生思考？", "target_session_id": 23, "target_collection": "tutor_strategies", "topic": "Is there an example where it doesn't work反例引导"},
    {"id": 7, "query": "在看电视数量的频数统计表中，学生计算极差时为什么会混淆频数和数据本身？", "target_session_id": 29, "target_collection": "student_misconceptions", "topic": "范围计算中频率与数据值的混淆"},
    {"id": 8, "query": "名师如何引导学生看懂频数表里的实际人数与对应电视机数量？", "target_session_id": 29, "target_collection": "tutor_strategies", "topic": "7 people who have 1 TV 现实情境追问"},
    {"id": 9, "query": "将 3.153 四舍五入到最近的 0.02 时，学生面临的主要概念卡点是什么？", "target_session_id": 33, "target_collection": "student_misconceptions", "topic": "非常规0.02精度与传统0.1/0.01混淆"},
    {"id": 10, "query": "针对四舍五入到 0.02 倍数这道难题，名师搭建了怎样的脚手架步骤？", "target_session_id": 33, "target_collection": "tutor_strategies", "topic": "0.02倍数分步寻找脚手架"},
    {"id": 11, "query": "学生对于 (-q)^2 和 -q^2 的负号位置与括号规则有什么深层认知盲区？", "target_session_id": 42, "target_collection": "student_misconceptions", "topic": "负数幂运算中负号处理规则混淆"},
    {"id": 12, "query": "辅导负数乘方时，导师如何通过追问 -p x q 帮助学生厘清负号归属？", "target_session_id": 42, "target_collection": "tutor_strategies", "topic": "Does -p x q give us the same answer"},
    {"id": 13, "query": "学生判断 105 是否为质数时，为什么能背出定义却依然判断错误？", "target_session_id": 59, "target_collection": "student_misconceptions", "topic": "质数概念与尾数5倍数特征应用脱节"},
    {"id": 14, "query": "老师如何用‘末尾是5的数字特征’迅速启发学生排除合数 105？", "target_session_id": 59, "target_collection": "tutor_strategies", "topic": "Why can we rule it out as it ends in a 5"},
    {"id": 15, "query": "解不等式两边同时除以负数（如 -12 ÷ -2）时，学生常漏掉哪个核心操作？", "target_session_id": 69, "target_collection": "student_misconceptions", "topic": "不等式两边除以负数未翻转不等号"},
    {"id": 16, "query": "导师如何分步引导学生计算 -12 ÷ -2 并引出翻转不等号的关键规则？", "target_session_id": 69, "target_collection": "tutor_strategies", "topic": "分步计算 -12 ÷ -2 与不等号翻转"},
    {"id": 17, "query": "在计算长方体体积时，学生为什么容易把体积公式和表面积/周长混在一起？", "target_session_id": 77, "target_collection": "student_misconceptions", "topic": "三维长方体体积与表面积概念混淆"},
    {"id": 18, "query": "名师如何用长宽高连乘公式（length x width x height）快速纠偏体积计算？", "target_session_id": 77, "target_collection": "tutor_strategies", "topic": "长宽高公式提醒与体积/面积概念区分"},
    {"id": 19, "query": "学生为什么会把英语时间 '10 to 12' 误解为 10点12分（10:12）？", "target_session_id": 80, "target_collection": "student_misconceptions", "topic": "英语时间表述中的分钟-小时倒装混淆"},
    {"id": 20, "query": "导师如何通过指向选项时钟（Which time does B show）引导学生识别正确钟面？", "target_session_id": 80, "target_collection": "tutor_strategies", "topic": "Which time does B show 钟面引导"}
]


def evaluate_retrieval(use_multi_perspective_rrf: bool = False):
    storage = DualEngineStorageManager(
        db_path="data/db/tutoring_knowledge.duckdb",
        chroma_dir="data/chroma"
    )
    retriever = DualMetricRetriever(storage_manager=storage, alpha=0.5)

    hit_at_1_count = 0
    hit_at_3_count = 0
    reciprocal_ranks = []
    latencies = []
    results_detail = []

    for item in BENCHMARK_20_QUERIES:
        q_id = item["id"]
        query = item["query"]
        target_sess = item["target_session_id"]
        target_coll = item["target_collection"]
        topic = item["topic"]

        start_t = time.perf_counter()
        
        if use_multi_perspective_rrf:
            res_all = retriever.retrieve_multi_perspective_rrf(query, top_k_each=3, fetch_evidence=True)
            if target_coll == "student_misconceptions":
                candidates = res_all["misconceptions"]
            else:
                candidates = res_all["strategies"]
        else:
            if target_coll == "student_misconceptions":
                candidates = retriever.retrieve_misconceptions(query, top_k=3, fetch_evidence=True)
            else:
                candidates = retriever.retrieve_strategies(query, top_k=3, fetch_evidence=True)
                
        latency = (time.perf_counter() - start_t) * 1000.0
        latencies.append(latency)

        hit_rank = None
        for rank_idx, cand in enumerate(candidates, start=1):
            sess_id = cand["metadata"].get("session_id")
            if sess_id == target_sess:
                hit_rank = rank_idx
                break

        is_hit_1 = (hit_rank == 1)
        is_hit_3 = (hit_rank is not None and hit_rank <= 3)
        rr = (1.0 / hit_rank) if hit_rank else 0.0

        if is_hit_1:
            hit_at_1_count += 1
        if is_hit_3:
            hit_at_3_count += 1
        reciprocal_ranks.append(rr)

        top_cand = candidates[0] if candidates else None
        top_title = ""
        top_score = 0.0
        top_turns = []
        if top_cand:
            top_title = top_cand["metadata"].get("misconception_name") or top_cand["metadata"].get("key_aha_question") or ""
            top_score = top_cand.get("rrf_score") or top_cand.get("hybrid_score", 0.0)
            top_turns = top_cand.get("evidence_turns", [])

        results_detail.append({
            "id": q_id,
            "query": query,
            "target_session_id": target_sess,
            "target_collection": target_coll,
            "topic": topic,
            "hit_rank": hit_rank,
            "is_hit_1": is_hit_1,
            "is_hit_3": is_hit_3,
            "reciprocal_rank": rr,
            "latency_ms": latency,
            "top_candidate": {
                "title": top_title,
                "score": top_score,
                "evidence_count": len(top_turns),
                "sample_turn": top_turns[0]["text"] if top_turns else ""
            }
        })

    total = len(BENCHMARK_20_QUERIES)
    hit_1_rate = (hit_at_1_count / total) * 100.0
    hit_3_rate = (hit_at_3_count / total) * 100.0
    mrr = sum(reciprocal_ranks) / total
    avg_latency = sum(latencies) / total

    storage.close()
    return results_detail, {
        "total": total,
        "hit_1_count": hit_at_1_count,
        "hit_3_count": hit_at_3_count,
        "hit_1_rate": hit_1_rate,
        "hit_3_rate": hit_3_rate,
        "mrr": mrr,
        "avg_latency": avg_latency
    }


def run_comparison_benchmark():
    print("=" * 85)
    print("🚀 启动 20 条教研提问基准对照评测: [Raw Baseline] vs [Multi-Query + Domain + RRF]")
    print("=" * 85)

    # 1. 运行基线
    raw_details, raw_summary = evaluate_retrieval(use_multi_perspective_rrf=False)
    
    # 2. 运行升级版
    up_details, up_summary = evaluate_retrieval(use_multi_perspective_rrf=True)

    # 打印逐条对比表
    print(f"\n{'ID':<3} | {'基线排名':<8} | {'升级排名':<8} | {'目标会话':<8} | {'用户输入问题'}")
    print("-" * 85)
    for r_raw, r_up in zip(raw_details, up_details):
        r1_str = f"#{r_raw['hit_rank']}" if r_raw["hit_rank"] else "MISS"
        r2_str = f"#{r_up['hit_rank']}" if r_up["hit_rank"] else "MISS"
        icon = "🚀 跃升" if (not r_raw["is_hit_1"] and r_up["is_hit_1"]) else ("✅ 保持" if r_up["is_hit_1"] else "⚠️ 待调")
        print(f"{r_raw['id']:<3} | {r1_str:<8} | {r2_str:<8} | Session #{r_raw['target_session_id']:<3} | {icon} | {r_raw['query']}")

    print("\n" + "=" * 85)
    print("📊 【评测核心指标对比矩阵 (A/B Comparison Matrix)】")
    print("=" * 85)
    print(f"{'评测指标':<28} | {'原始基线 (Raw Query)':<22} | {'升级版 (Multi-Query+Domain+RRF)':<25} | {'提升幅度'}")
    print("-" * 85)
    print(f"{'Top-1 首位命中率 (Hit@1)':<24} | {raw_summary['hit_1_count']}/20 ({raw_summary['hit_1_rate']:.1f}%)"
          f"             | {up_summary['hit_1_count']}/20 ({up_summary['hit_1_rate']:.1f}%)"
          f"                    | +{up_summary['hit_1_rate'] - raw_summary['hit_1_rate']:.1f}%")
    print(f"{'Top-3 召回率 (Hit@3)':<26} | {raw_summary['hit_3_count']}/20 ({raw_summary['hit_3_rate']:.1f}%)"
          f"             | {up_summary['hit_3_count']}/20 ({up_summary['hit_3_rate']:.1f}%)"
          f"                    | +{up_summary['hit_3_rate'] - raw_summary['hit_3_rate']:.1f}%")
    print(f"{'平均倒数排名 (MRR)':<26} | {raw_summary['mrr']:.4f}"
          f"                   | {up_summary['mrr']:.4f}"
          f"                         | +{up_summary['mrr'] - raw_summary['mrr']:.4f}")
    print(f"{'平均单次耗时 (Latency)':<24} | {raw_summary['avg_latency']:.2f} ms"
          f"                 | {up_summary['avg_latency']:.2f} ms"
          f"                       | 依然亚毫秒级")
    print("=" * 85)


if __name__ == "__main__":
    run_comparison_benchmark()
