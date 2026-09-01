"""
================================================================================
模块名称: src/cli.py / scripts/interactive_cli.py
业务定位: Eedi-RAG 交互式名师教研备课助手控制台 (Interactive Pedagogical CLI)
核心功能:
  1. 连续交互式 REPL 循环 (Continuous Q&A Loop)
  2. 支持指令系统 (:help, :sample, :mode, :clear, :exit)
  3. 毫秒级呈现教研三段论排版与 100% 真实对白实录证据
================================================================================
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from src.reranker import PedagogicalGoldAssembler
from src.rag_pipeline import EndToEndPedagogicalRAGPipeline

# 关闭控制台冗长 INFO 日志，仅保留 WARNING/ERROR，使终端输出极其清爽
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


BANNER = r"""
========================================================================================
   ______          _ _         ____             _____                               
  |  ____|        | (_)       |  _ \     /\    / ____|                              
  | |__   ___   __| |_ ______ | |_) |   /  \  | |  __                               
  |  __| / _ \ / _` | |______||  _ <   / /\ \ | | |_ |                              
  | |___|  __/| (_| | |       | |_) | / ____ \| |__| |                              
  |______\___| \__,_|_|       |____/ /_/    \_\\_____|                              
                                                                                    
  🎓 Eedi-RAG 权威名师教研备课与启发辅导副驾驶 · 交互式控制台
  🔗 关系底表: DuckDB (真实原声对话) | 语义向量: ChromaDB | 检索融合: Multi-Query RRF
========================================================================================
"""

HELP_TEXT = """
【控制台可用指令清单】:
  - 直接输入教研问题: 执行端到端检索增强生成 (例如: '四舍五入到一位小数学生常犯什么错？')
  - :sample           : 列出 10 条典型的中英文教研测试问题供快速复制体验
  - :mode <auto|llm|deterministic> : 切换大模型/确定性保真生成模式 (当前默认: auto)
  - :clear            : 清理当前终端屏幕
  - :help             : 显示本帮助菜单
  - :exit / :q / quit : 退出控制台
"""

SAMPLE_QUERIES = [
    "四舍五入 5.4598 到 1 位小数时，学生为什么会误选 5.45？名师如何提问引导？",
    "学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？名师如何用反例引导？",
    "在频数统计表中计算极差时，学生为什么会混淆频数和数据本身？",
    "学生在做包含加法和除法的分式运算时容易犯什么运算顺序错误？",
    "将 3.153 四舍五入到最近的 0.02 时，学生面临的主要概念卡点是什么？",
    "学生对于 (-q)^2 和 -q^2 的负号位置与括号规则有什么深层认知盲区？",
    "学生判断 105 是否为质数时，为什么能背出定义却依然判断错误？",
    "解不等式两边同时除以负数（如 -12 ÷ -2）时，学生常漏掉哪个核心操作？",
    "在计算长方体体积时，学生为什么容易把体积公式和表面积/周长混在一起？",
    "学生为什么会把英语时间 '10 to 12' 误解为 10点12分（10:12）？"
]


class PedagogicalCLI:
    """
    交互式教研控制台应用。
    """

    def __init__(
        self,
        db_path: str = "data/db/tutoring_knowledge.duckdb",
        chroma_dir: str = "data/chroma",
        default_mode: str = "auto"
    ):
        self.db_path = db_path
        self.chroma_dir = chroma_dir
        self.mode = default_mode
        self.storage: Optional[DualEngineStorageManager] = None
        self.retriever: Optional[DualMetricRetriever] = None
        self.pipeline: Optional[EndToEndPedagogicalRAGPipeline] = None

    def initialize(self):
        """初始化底层双引擎与 RAG 管道。"""
        print("⏳ 正在初始化双引擎存储与教研生成管道...", end="", flush=True)
        self.storage = DualEngineStorageManager(
            db_path=self.db_path,
            chroma_dir=self.chroma_dir,
            embedding_backend="deterministic",
        )
        self.retriever = DualMetricRetriever(
            storage_manager=self.storage,
            query_rewrite_mode="deterministic",
            alpha=0.5,
        )
        assembler = PedagogicalGoldAssembler(lambda_diversity=0.7)
        self.pipeline = EndToEndPedagogicalRAGPipeline(
            retriever=self.retriever,
            assembler=assembler
        )
        print(" ✅ 就绪！\n")

    def run(self):
        """启动交互式 REPL 循环。"""
        self.initialize()
        print(BANNER)
        print("💡 提示: 输入问题直接提问，输入 `:help` 查看指令，输入 `:sample` 查看精选题库，输入 `:q` 退出。\n")

        query_count = 0

        while True:
            try:
                prompt_str = f"🎓 Eedi-RAG [{self.mode}] [{query_count + 1}] > "
                user_input = input(prompt_str).strip()

                if not user_input:
                    continue

                # 退出指令
                if user_input.lower() in [":q", ":exit", ":quit", "exit", "quit", "q"]:
                    print("\n👋 感谢使用 Eedi-RAG 教研助手，再见！")
                    break

                # 帮助指令
                if user_input.lower() in [":help", "help", ":h"]:
                    print(HELP_TEXT)
                    continue

                # 样本指令
                if user_input.lower() in [":sample", ":samples", "sample"]:
                    print("\n📋 【精选 10 条教研测试问题 (可复制提问)】:")
                    for idx, q in enumerate(SAMPLE_QUERIES, 1):
                        print(f"  [{idx:02d}] {q}")
                    print()
                    continue

                # 模式切换
                if user_input.startswith(":mode"):
                    parts = user_input.split()
                    if len(parts) > 1 and parts[1].lower() in ["auto", "llm", "deterministic"]:
                        self.mode = parts[1].lower()
                        print(f"⚙️ 生成模式已切换为: [{self.mode}]\n")
                    else:
                        print("⚠️ 模式参数无效，可选: :mode auto | :mode llm | :mode deterministic\n")
                    continue

                # 清屏指令
                if user_input.lower() in [":clear", "clear", "cls"]:
                    os.system("cls" if os.name == "nt" else "clear")
                    print(BANNER)
                    continue

                # 执行问答
                query_count += 1
                start_t = time.perf_counter()
                print("\n🧠 正在检索考纲知识库与真实辅导对白实录...", flush=True)

                response = self.pipeline.ask(user_input, mode=self.mode)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                print("\n" + "=" * 88)
                print(response.rendered_markdown)
                print("=" * 88)
                print(f"⏱️ 耗时: {elapsed_ms:.1f} ms | 考纲: {response.subject_path} | 真实对白凭据: {len(response.dialogue_citations)} 轮 | 审计: {response.audit_status}\n")

            except KeyboardInterrupt:
                print("\n\n👋 接收到退出信号，正在安全退出...")
                break
            except Exception as e:
                print(f"\n❌ 执行遇到异常: {e}\n")

        if self.storage:
            self.storage.close()


def main():
    cli = PedagogicalCLI()
    cli.run()


if __name__ == "__main__":
    main()
