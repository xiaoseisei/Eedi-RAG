"""
================================================================================
脚本名称: scripts/run_demo_query.py
业务定位: Step 5 - 端到端 RAG 权威教研备课指南与名师破局锦囊演示脚本
运行方式:
  python scripts/run_demo_query.py "你的问题"
================================================================================
"""

import sys
import logging
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from src.rag_pipeline import EndToEndPedagogicalRAGPipeline


def run_demo(query: str):
    storage = DualEngineStorageManager(
        db_path="data/db/tutoring_knowledge.duckdb",
        chroma_dir="data/chroma",
        embedding_backend="deterministic",
    )
    retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic")
    pipeline = EndToEndPedagogicalRAGPipeline(retriever=retriever)

    print("\n" + "=" * 80)
    print(f"🎯 正在执行端到端 RAG 教研分析与生成...")
    print(f"📝 教师输入: {query}")
    print("=" * 80 + "\n")

    result = pipeline.ask(query, mode="auto")

    print(result.rendered_markdown)
    print("\n" + "=" * 80)
    print(f"📊 结构化元数据快照:")
    print(f"  * 考纲路径: {result.subject_path}")
    print(f"  * 关联会话: Session #{result.session_id}")
    print(f"  * 破局一问: 「{result.key_aha_question}」")
    print(f"  * 教学动作: {result.recommended_talk_moves}")
    print(f"  * 真实对白凭据: {len(result.dialogue_citations)} 轮")
    print(f"  * 审计状态: {result.audit_status}")
    print("=" * 80)

    storage.close()


if __name__ == "__main__":
    default_q = "学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？名师如何用反例引导？"
    user_q = sys.argv[1] if len(sys.argv) > 1 else default_q
    run_demo(user_q)
