# -*- coding: utf-8 -*-
"""
scripts/eval_50_queries.py
--------------------------
评估 50 条真实教研基准提问在 Top-1, Top-3, Top-4, Top-5 下的召回率与排名明细。
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

# 50 条全量教研基准测试集 (25 条错因诊断 + 25 条名师策略，覆盖 10 大核心教学场景)
BENCHMARK_50_QUERIES = [
    # --- Session 10: 四舍五入 5.4598 到 1 位小数 ---
    {"id": 1, "query": "四舍五入 5.4598 到 1 位小数时，学生为什么会误选 5.45？", "target_session_id": 10, "target_collection": "student_misconceptions", "topic": "四舍五入数位保留与截断混淆", "type": "misconception"},
    {"id": 2, "query": "当学生把保留一位小数做成保留两位时，名师是如何用简单数字破局提问的？", "target_session_id": 10, "target_collection": "tutor_strategies", "topic": "5.45一位小数破局一问", "type": "strategy"},
    {"id": 3, "query": "在将 5.4598 近似为一位小数时，学生容易混淆十分位与百分位的原因是什么？", "target_session_id": 10, "target_collection": "student_misconceptions", "topic": "十分位与百分位概念混淆", "type": "misconception"},
    {"id": 4, "query": "针对小数四舍五入时的数位截断误区，导师通过怎样的反问引导学生关注关键判定数字？", "target_session_id": 10, "target_collection": "tutor_strategies", "topic": "判定数字反问脚手架", "type": "strategy"},
    {"id": 5, "query": "面对四舍五入题目，学生为什么容易直接截断后面多余数字而不进行进位？", "target_session_id": 10, "target_collection": "student_misconceptions", "topic": "纯截断不进位认知缺陷", "type": "misconception"},

    # --- Session 14: 分式混合运算顺序 ---
    {"id": 6, "query": "学生在做包含加法和除法的分式运算时容易犯什么运算顺序错误？", "target_session_id": 14, "target_collection": "student_misconceptions", "topic": "运算顺序混淆与分数符号认知不清", "type": "misconception"},
    {"id": 7, "query": "名师如何通过苏格拉底反问引导学生识别混合运算中的运算优先级？", "target_session_id": 14, "target_collection": "tutor_strategies", "topic": "what do you think this question is asking", "type": "strategy"},
    {"id": 8, "query": "在计算 (54+58)/2 这类算式时，学生为什么会先算除法或者遗漏括号内的求和？", "target_session_id": 14, "target_collection": "student_misconceptions", "topic": "分式分子加法与除法优先级混淆", "type": "misconception"},
    {"id": 9, "query": "当学生对多步混合运算感到困惑时，导师采用什么脚手架步骤拆解计算过程？", "target_session_id": 14, "target_collection": "tutor_strategies", "topic": "混合运算分步拆解脚手架", "type": "strategy"},
    {"id": 10, "query": "导师如何通过提示题目真实要求来帮学生理清复合算式的计算次序？", "target_session_id": 14, "target_collection": "tutor_strategies", "topic": "审题意图澄清策略", "type": "strategy"},

    # --- Session 23: 最小公倍数 (LCM) 与两数乘积 ---
    {"id": 11, "query": "学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？", "target_session_id": 23, "target_collection": "student_misconceptions", "topic": "将互质数特例错误泛化为普遍规律", "type": "misconception"},
    {"id": 12, "query": "当学生盲目认为两数相乘就是LCM时，老师用什么反例策略引导学生思考？", "target_session_id": 23, "target_collection": "tutor_strategies", "topic": "Is there an example where it doesn't work反例引导", "type": "strategy"},
    {"id": 13, "query": "学生将 3 和 5 等互质数特例推广到所有数求最小公倍数时的思维定势是什么？", "target_session_id": 23, "target_collection": "student_misconceptions", "topic": "互质数规律过度泛化", "type": "misconception"},
    {"id": 14, "query": "名师如何通过提问“是否存在不适用的例子”促使学生自主反思最小公倍数计算方法？", "target_session_id": 23, "target_collection": "tutor_strategies", "topic": "反例反问促自主反思", "type": "strategy"},
    {"id": 15, "query": "求非互质数（如 4 和 6）的公倍数时，学生漏掉公因数导致结果偏大的根本原因？", "target_session_id": 23, "target_collection": "student_misconceptions", "topic": "忽略公因数导致LCM偏大", "type": "misconception"},

    # --- Session 29: 频数统计表中求极差 ---
    {"id": 16, "query": "在看电视数量的频数统计表中，学生计算极差时为什么会混淆频数和数据本身？", "target_session_id": 29, "target_collection": "student_misconceptions", "topic": "范围计算中频率与数据值的混淆", "type": "misconception"},
    {"id": 17, "query": "名师如何引导学生看懂频数表里的实际人数与对应电视机数量？", "target_session_id": 29, "target_collection": "tutor_strategies", "topic": "7 people who have 1 TV 现实情境追问", "type": "strategy"},
    {"id": 18, "query": "学生误将频数列表中的最大数值或零频数项作为极差极值计算的深层机理？", "target_session_id": 29, "target_collection": "student_misconceptions", "topic": "频数行最大值误用为数据最大值", "type": "misconception"},
    {"id": 19, "query": "导师如何结合现实生活情境帮助学生区分样本频数与变量测量值？", "target_session_id": 29, "target_collection": "tutor_strategies", "topic": "生活情境具象化拆解", "type": "strategy"},
    {"id": 20, "query": "在统计图表教学中，名师如何一步步引导学生定位数据集中的最大值与最小值？", "target_session_id": 29, "target_collection": "tutor_strategies", "topic": "定位最大最小值脚手架", "type": "strategy"},

    # --- Session 33: 非常规精度四舍五入到 0.02 倍数 ---
    {"id": 21, "query": "将 3.153 四舍五入到最近的 0.02 时，学生面临的主要概念卡点是什么？", "target_session_id": 33, "target_collection": "student_misconceptions", "topic": "非常规0.02精度与传统0.1/0.01混淆", "type": "misconception"},
    {"id": 22, "query": "针对四舍五入到 0.02 倍数这道难题，名师搭建了怎样的脚手架步骤？", "target_session_id": 33, "target_collection": "tutor_strategies", "topic": "0.02倍数分步寻找脚手架", "type": "strategy"},
    {"id": 23, "query": "学生把“四舍五入到 0.02”误当作常规保留一位或两位小数时的认知障碍？", "target_session_id": 33, "target_collection": "student_misconceptions", "topic": "非十进制倍数舍入认知盲区", "type": "misconception"},
    {"id": 24, "query": "导师如何通过列举 3.14 与 3.16 的区间端点启发学生判断距离 3.153 最近的数值？", "target_session_id": 33, "target_collection": "tutor_strategies", "topic": "区间端点夹逼启发法", "type": "strategy"},
    {"id": 25, "query": "面对非标准步长的舍入问题，学生缺乏数轴区间概念导致盲目进位的心理机制？", "target_session_id": 33, "target_collection": "student_misconceptions", "topic": "数轴区间感知缺失", "type": "misconception"},

    # --- Session 42: 负数乘方中负号位置与括号规则 ---
    {"id": 26, "query": "学生对于 (-q)^2 和 -q^2 的负号位置与括号规则有什么深层认知盲区？", "target_session_id": 42, "target_collection": "student_misconceptions", "topic": "负数幂运算中负号处理规则混淆", "type": "misconception"},
    {"id": 27, "query": "辅导负数乘方时，导师如何通过追问 -p x q 帮助学生厘清负号归属？", "target_session_id": 42, "target_collection": "tutor_strategies", "topic": "Does -p x q give us the same answer", "type": "strategy"},
    {"id": 28, "query": "学生认为负数参与任何乘方运算结果都必定带负号的错误直觉是什么？", "target_session_id": 42, "target_collection": "student_misconceptions", "topic": "偶次幂结果符号判断直觉错误", "type": "misconception"},
    {"id": 29, "query": "名师如何通过对比 (-p)*q 与 -(p*q) 的等价形式帮助学生建立代数式符号感知？", "target_session_id": 42, "target_collection": "tutor_strategies", "topic": "代数等价式对比教学法", "type": "strategy"},
    {"id": 30, "query": "当学生混淆乘方优先级与相反数符号时，导师如何用具体代数展开式进行验证？", "target_session_id": 42, "target_collection": "tutor_strategies", "topic": "代数展开式直观求证", "type": "strategy"},

    # --- Session 59: 质数与尾数特征判断 ---
    {"id": 31, "query": "学生判断 105 是否为质数时，为什么能背出定义却依然判断错误？", "target_session_id": 59, "target_collection": "student_misconceptions", "topic": "质数概念与尾数5倍数特征应用脱节", "type": "misconception"},
    {"id": 32, "query": "老师如何用‘末尾是5的数字特征’迅速启发学生排除合数 105？", "target_session_id": 59, "target_collection": "tutor_strategies", "topic": "Why can we rule it out as it ends in a 5", "type": "strategy"},
    {"id": 33, "query": "学生在判断较大奇数是否为质数时，忽视基本整除性规则的思维盲区是什么？", "target_session_id": 59, "target_collection": "student_misconceptions", "topic": "大奇数整除性验证盲区", "type": "misconception"},
    {"id": 34, "query": "导师如何引导学生观察 105 的个位数字并联想到 5 的倍数特征？", "target_session_id": 59, "target_collection": "tutor_strategies", "topic": "个位数特征观察引导", "type": "strategy"},
    {"id": 35, "query": "为什么学生容易把非偶数的奇数（如 105）直觉性地误判为质数？", "target_session_id": 59, "target_collection": "student_misconceptions", "topic": "奇数与质数直觉等同混淆", "type": "misconception"},

    # --- Session 69: 一元一次不等式两边除以负数 ---
    {"id": 36, "query": "解不等式两边同时除以负数（如 -12 ÷ -2）时，学生常漏掉哪个核心操作？", "target_session_id": 69, "target_collection": "student_misconceptions", "topic": "不等式两边除以负数未翻转不等号", "type": "misconception"},
    {"id": 37, "query": "导师如何分步引导学生计算 -12 ÷ -2 并引出翻转不等号的关键规则？", "target_session_id": 69, "target_collection": "tutor_strategies", "topic": "分步计算 -12 ÷ -2 与不等号翻转", "type": "strategy"},
    {"id": 38, "query": "学生在求解 -2m > 12 时，将等式移项法则生搬硬套到不等式的认知误区？", "target_session_id": 69, "target_collection": "student_misconceptions", "topic": "等式移项规则在不等式中的机械套用", "type": "misconception"},
    {"id": 39, "query": "导师如何通过带正负号的数值除法逐步引出“除以负数不等号方向改变”的核心定理？", "target_session_id": 69, "target_collection": "tutor_strategies", "topic": "负数除法定理渐进式推导", "type": "strategy"},
    {"id": 40, "query": "名师如何用肯定鼓励加渐进提问推动学生推进解题？", "target_session_id": 69, "target_collection": "tutor_strategies", "topic": "情感肯定与渐进启发协同", "type": "strategy"},

    # --- Session 77: 长方体体积公式 vs 表面积/周长 ---
    {"id": 41, "query": "在计算长方体体积时，学生为什么容易把体积公式和表面积/周长混在一起？", "target_session_id": 77, "target_collection": "student_misconceptions", "topic": "三维长方体体积与表面积概念混淆", "type": "misconception"},
    {"id": 42, "query": "名师如何用长宽高连乘公式（length x width x height）快速纠偏体积计算？", "target_session_id": 77, "target_collection": "tutor_strategies", "topic": "长宽高公式提醒与体积/面积概念区分", "type": "strategy"},
    {"id": 43, "query": "面对长方体三维尺寸（如 10cm, 4cm, 5cm），学生将加法与乘法混淆导致结果错误的原因？", "target_session_id": 77, "target_collection": "student_misconceptions", "topic": "棱长连加与连乘算子混淆", "type": "misconception"},
    {"id": 44, "query": "导师如何通过强调“这道题求的是体积不是面积”来明确几何计算目标？", "target_session_id": 77, "target_collection": "tutor_strategies", "topic": "空间量度目标再确认", "type": "strategy"},
    {"id": 45, "query": "学生对于“空间容积”与“外表平面积”物理意义区分不清的深层机理？", "target_session_id": 77, "target_collection": "student_misconceptions", "topic": "三维空间感与二维平面感脱节", "type": "misconception"},

    # --- Session 80: 英语时间表达与表盘钟面识别 ---
    {"id": 46, "query": "学生为什么会把英语时间 '10 to 12' 误解为 10点12分（10:12）？", "target_session_id": 80, "target_collection": "student_misconceptions", "topic": "英语时间表述中的分钟-小时倒装混淆", "type": "misconception"},
    {"id": 47, "query": "导师如何通过指向选项时钟（Which time does B show）引导学生识别正确钟面？", "target_session_id": 80, "target_collection": "tutor_strategies", "topic": "Which time does B show 钟面引导", "type": "strategy"},
    {"id": 48, "query": "学生在阅读英式时间表达 'X to Y' 时，因直译词序导致的时分倒装错误？", "target_session_id": 80, "target_collection": "student_misconceptions", "topic": "英语时间倒装句型直译偏差", "type": "misconception"},
    {"id": 49, "query": "导师如何通过逐个排除错误钟面选项来降低学生的认知负荷？", "target_session_id": 80, "target_collection": "tutor_strategies", "topic": "选项钟面排除启发法", "type": "strategy"},
    {"id": 50, "query": "名师如何通过提问具体的表盘时间（如 11:50）引导学生关联 '差10分到12点' 的概念？", "target_session_id": 80, "target_collection": "tutor_strategies", "topic": "钟面数字与差量时间关联引导", "type": "strategy"},
]


def evaluate_top_k_50(retriever: DualMetricRetriever, max_k: int = 5, use_rrf: bool = False):
    hit_1 = 0
    hit_3 = 0
    hit_4 = 0
    hit_5 = 0
    ranks = []
    latencies = []
    details = []

    for item in BENCHMARK_50_QUERIES:
        q_id = item["id"]
        query = item["query"]
        target_sess = item["target_session_id"]
        target_coll = item["target_collection"]
        topic = item["topic"]
        q_type = item["type"]
        
        t0 = time.perf_counter()
        if use_rrf:
            res_all = retriever.retrieve_multi_perspective_rrf(query, top_k_each=max_k, fetch_evidence=True)
            cands = res_all["misconceptions"] if target_coll == "student_misconceptions" else res_all["strategies"]
        else:
            if target_coll == "student_misconceptions":
                cands = retriever.retrieve_misconceptions(query, top_k=max_k, fetch_evidence=True)
            else:
                cands = retriever.retrieve_strategies(query, top_k=max_k, fetch_evidence=True)
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
        if hit_rank and hit_rank <= 4:
            hit_4 += 1
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

        details.append({
            "id": q_id,
            "query": query,
            "target_session_id": target_sess,
            "target_collection": target_coll,
            "topic": topic,
            "type": q_type,
            "hit_rank": hit_rank,
            "latency": lat,
            "top_title": top_title,
            "top_score": score
        })
        
    total = len(BENCHMARK_50_QUERIES)
    return {
        "total": total,
        "hit_1": hit_1, "hit_1_rate": hit_1 / total * 100,
        "hit_3": hit_3, "hit_3_rate": hit_3 / total * 100,
        "hit_4": hit_4, "hit_4_rate": hit_4 / total * 100,
        "hit_5": hit_5, "hit_5_rate": hit_5 / total * 100,
        "mrr": sum(ranks) / total,
        "avg_lat": sum(latencies) / total,
        "details": details
    }


def print_50_report(raw_res, rrf_res):
    print("=" * 100)
    print("📊 【50 条真实教研提问端到端召回率评测总榜 (Top-1 / Top-3 / Top-4 / Top-5)】")
    print("=" * 100)
    print(f"{'评测维度 / 指标':<30} | {'原始基线 (Raw Query)':<25} | {'多视角 RRF 升级版':<25} | {'提升幅度'}")
    print("-" * 100)
    
    h1_raw, h1_up = raw_res['hit_1'], rrf_res['hit_1']
    h3_raw, h3_up = raw_res['hit_3'], rrf_res['hit_3']
    h4_raw, h4_up = raw_res['hit_4'], rrf_res['hit_4']
    h5_raw, h5_up = raw_res['hit_5'], rrf_res['hit_5']
    mrr_raw, mrr_up = raw_res['mrr'], rrf_res['mrr']
    lat_raw, lat_up = raw_res['avg_lat'], rrf_res['avg_lat']
    
    print(f"{'Top-1 首位命中率 (Hit@1)':<26} | {h1_raw}/50 ({raw_res['hit_1_rate']:.1f}%)"
          f"                | {h1_up}/50 ({rrf_res['hit_1_rate']:.1f}%)"
          f"                | +{rrf_res['hit_1_rate'] - raw_res['hit_1_rate']:.1f}%")
    print(f"{'Top-3 前三命中率 (Hit@3)':<26} | {h3_raw}/50 ({raw_res['hit_3_rate']:.1f}%)"
          f"                | {h3_up}/50 ({rrf_res['hit_3_rate']:.1f}%)"
          f"                | +{rrf_res['hit_3_rate'] - raw_res['hit_3_rate']:.1f}%")
    print(f"{'Top-4 前四命中率 (Hit@4)':<26} | {h4_raw}/50 ({raw_res['hit_4_rate']:.1f}%)"
          f"                | {h4_up}/50 ({rrf_res['hit_4_rate']:.1f}%)"
          f"                | +{rrf_res['hit_4_rate'] - raw_res['hit_4_rate']:.1f}%")
    print(f"{'Top-5 前五命中率 (Hit@5)':<26} | {h5_raw}/50 ({raw_res['hit_5_rate']:.1f}%)"
          f"                | {h5_up}/50 ({rrf_res['hit_5_rate']:.1f}%)"
          f"                | +{rrf_res['hit_5_rate'] - raw_res['hit_5_rate']:.1f}%")
    print(f"{'平均倒数排名 (MRR)':<28} | {mrr_raw:.4f}"
          f"                      | {mrr_up:.4f}"
          f"                      | +{mrr_up - mrr_raw:.4f}")
    print(f"{'平均单次检索延迟 (Latency)':<24} | {lat_raw:.2f} ms"
          f"                    | {lat_up:.2f} ms"
          f"                    | 纯本地毫秒级")
    print("=" * 100)
    
    print("\n" + "=" * 100)
    print("📝 【50 条提问端到端检索明细详表 (Top-1 / Top-3 / Top-4 / Top-5 逐条横向对照)】")
    print("=" * 100)
    print(f"{'ID':<4} | {'基线状态':<10} | {'基线排名':<8} | {'升级状态':<10} | {'升级排名':<8} | {'目标会话':<12} | {'类别':<8} | {'用户输入提问'}")
    print("-" * 100)
    
    for r, u in zip(raw_res['details'], rrf_res['details']):
        r_rank = r['hit_rank']
        u_rank = u['hit_rank']
        
        r_status = f"✅ HIT@{r_rank}" if (r_rank and r_rank == 1) else (f"⚠️ HIT@{r_rank}" if r_rank else "❌ MISS")
        u_status = f"✅ HIT@{u_rank}" if (u_rank and u_rank == 1) else (f"⚠️ HIT@{u_rank}" if u_rank else "❌ MISS")
        
        r_rk_str = f"#{r_rank}" if r_rank else "N/A"
        u_rk_str = f"#{u_rank}" if u_rank else "N/A"
        
        q_type_str = "错因诊断" if r['type'] == "misconception" else "名师策略"
        print(f"#{r['id']:02d}  | {r_status:<10} | {r_rk_str:<8} | {u_status:<10} | {u_rk_str:<8} | Session #{r['target_session_id']:<4} | {q_type_str:<8} | {r['query']}")

    print("=" * 100)


def main():
    storage = DualEngineStorageManager(db_path="data/db/tutoring_knowledge.duckdb", chroma_dir="data/chroma", embedding_backend="deterministic")
    retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic", alpha=0.5)

    raw_res = evaluate_top_k_50(retriever, max_k=5, use_rrf=False)
    rrf_res = evaluate_top_k_50(retriever, max_k=5, use_rrf=True)

    print_50_report(raw_res, rrf_res)

    storage.close()


if __name__ == "__main__":
    main()
