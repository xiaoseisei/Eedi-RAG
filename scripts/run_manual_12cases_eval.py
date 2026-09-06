import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from src.rag_pipeline import EndToEndPedagogicalRAGPipeline

TEST_CASES = [
    {
        "id": 1,
        "query": "四舍五入 5.4598 到 1 位小数时，学生为什么会选 5.45？",
        "expected": "学生混淆 1dp 和 2dp，把四舍五入误当成截断；引用 Session 10 的学生原话",
    },
    {
        "id": 2,
        "query": "学生为什么会认为任意两个数的 LCM 就是乘积？导师如何用反例引导？",
        "expected": "说明互质数时乘积成立，再用 4 和 6 反例说明不总成立；引用导师的 “Can you prove it?”",
    },
    {
        "id": 3,
        "query": "“四舍五入到最近 0.02”和保留两位小数有什么区别？",
        "expected": "使用 0.02 乘法表找 3.14、3.16，再判断 3.153 更接近 3.16",
    },
    {
        "id": 4,
        "query": "学生为什么先计算 58÷2，而不是先计算 (54+58)？",
        "expected": "说明学生没有理解分数线的分组作用，混用了 BIDMAS；不能把学生错误说成正确规则",
    },
    {
        "id": 5,
        "query": "展开 -3(1-2p) 时学生最容易错在哪里？",
        "expected": "指出负数乘负数和乘法/加法混淆，并说明导师如何拆成两个乘法步骤",
    },
    {
        "id": 6,
        "query": "为什么 -2m > 12 会得到 m > -6？",
        "expected": "指出除以负数后不等号必须翻转，正确结果是 m < -6",
    },
    {
        "id": 7,
        "query": "导师如何帮助学生理解“水深—时间”图像对应的容器形状？",
        "expected": "把斜率变化和容器底部宽窄联系起来，不能只复述图像结论",
    },
    {
        "id": 8,
        "query": "已知平均数为 4，求缺失数时学生卡在哪里？",
        "expected": "说明平均数 × 数据个数 = 总和，并引用导师用旧题类比的脚手架",
    },
    {
        "id": 9,
        "query": "混合小数和分数求众数时，学生为什么找不到重复值？",
        "expected": "识别 0.4 = 2/5，说明导师让学生统一表示法",
    },
    {
        "id": 10,
        "query": "学生说“体积就是把所有面的面积加起来”说明什么？",
        "expected": "区分体积和表面积，并给出导师用长×宽×高的点拨",
    },
    {
        "id": 11,
        "query": "“ten to twelve” 为什么不是 10:12？",
        "expected": "说明 to 表示距离十二点还差十分钟，结果是 11:50 或 23:50",
    },
    {
        "id": 12,
        "query": "密度题中为什么 kg/m² 不是密度单位？",
        "expected": "密度是质量 ÷ 体积，体积单位应为立方单位，不能把面积单位当体积单位",
    },
]

def run_evaluation():
    storage = DualEngineStorageManager(
        db_path="data/db/tutoring_knowledge.duckdb",
        chroma_dir="data/chroma",
        embedding_backend="deterministic",
    )
    retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic")
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=retriever,
        generator_contract_version="v2",
    )

    results = []
    output_dir = Path("reports/eval/manual_test_12cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "results.json"

    print("🚀 开始执行 12 道教研基准问题真实端到端评测...")
    for idx, case in enumerate(TEST_CASES, start=1):
        q = case["query"]
        print(f"\n[{idx}/12] 提问: {q}")
        t0 = time.perf_counter()
        try:
            resp = pipeline.ask(q, mode="auto")
            elapsed = round(time.perf_counter() - t0, 3)
            
            citations_data = [
                {
                    "session_id": c.session_id,
                    "turn_id": c.turn_id,
                    "speaker": c.speaker,
                    "quote_text": c.quote_text,
                    "verifiable_in_duckdb": c.verifiable_in_duckdb,
                }
                for c in resp.dialogue_citations
            ]

            case_result = {
                "id": case["id"],
                "query": q,
                "expected": case["expected"],
                "elapsed_seconds": elapsed,
                "status": "SUCCESS",
                "subject_path": resp.subject_path,
                "session_id": resp.session_id,
                "audit_status": resp.audit_status,
                "core_answer": resp.__dict__.get("generator_core_answer", ""),
                "misconception_diagnosis": resp.misconception_diagnosis,
                "key_aha_question": resp.key_aha_question,
                "recommended_talk_moves": resp.recommended_talk_moves,
                "scaffolding_steps": resp.scaffolding_steps,
                "answer_content": resp.answer_content,
                "citations": citations_data,
                "profiling": resp._profiling,
            }
            print(f"   ✅ 完成 ({elapsed}s) | 审计状态: {resp.audit_status} | 引用轮次: {len(citations_data)}")
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            print(f"   ❌ 失败 ({elapsed}s): {exc}")
            case_result = {
                "id": case["id"],
                "query": q,
                "expected": case["expected"],
                "elapsed_seconds": elapsed,
                "status": "ERROR",
                "error": str(exc),
            }
        results.append(case_result)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    storage.close()
    print(f"\n🎉 评测完成！全部结果已保存至 {out_file}")

if __name__ == "__main__":
    run_evaluation()
