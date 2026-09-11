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
import argparse
import hashlib
import json
import threading
import logging
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from src.reranker import PedagogicalGoldAssembler
from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
from src.business_models import BusinessQueryRequest

# 关闭控制台冗长 INFO 日志，仅保留 WARNING/ERROR，使终端输出极其清爽
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_BUSINESS_GRAPH_DB = (
    project_root
    / "reports"
    / "staging"
    / "session-card-graph-v1"
    / "graph-taxonomy-verify-2-20260908.duckdb"
)
APPROVED_BUSINESS_GRAPH_SHA256 = (
    "6efa0ee42a2b7a2004f194972bb7ffab75224dd69d17f190b96d58cdfefbac98"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


BANNER = r"""
========================================================================================
   ______          _ _         ____             _____                               
  |  ____|        | (_)       |  _ \     /\    / ____|                              
  | |__   ___   __| |_ ______ | |_) |   /  \  | |  __                               
  |  __| / _ \ / _` | |______||  _ <   / /\ \ | | |_ |                              
  | |___|  __/| (_| | |       | |_) | / ____ \| |__| |                              
  |______\___| \__,_|_|       |____/ /_/    \_\\_____|                              
                                                                                    
  🎓 Eedi-RAG 权威名师教研备课与启发辅导副驾驶 · 交互式控制台
  🔗 关系底表: DuckDB (真实原声对话) | 语义向量: ChromaDB | 检索: BM25 + Dense + Multi-Query RRF
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
        default_mode: str = "auto",
        embedding_backend: str = "deterministic",
        retrieval_mode: str = "bm25_dense",
        bm25_weight: float = 0.35,
        reranker_backend: str = "none",
        chunk_strategy: str = "card",
        rerank_unit: Optional[str] = None,
        evidence_selection_count: int = 7,
        reranker_pool_size: Optional[int] = None,
        parent_card_count: int = 5,
        generator_contract_version: str = "v1",
        pipeline_mode: str = "legacy",
        graph_db: str | Path = DEFAULT_BUSINESS_GRAPH_DB,
        embedding_mode: str = "disabled",
        debug_graph: bool = False,
        allow_candidate_graph: bool = False,
        intent_model: Optional[str] = None,
    ):
        """
        Args:
            db_path / chroma_dir: 双引擎存储路径。
            default_mode: 初始生成模式 (auto=有 Key 走 LLM / llm=强制 LLM / deterministic=确定性模板)。
            embedding_backend: 向量后端 ('deterministic' 哈希基线 或 'siliconflow' 真实语义 API)。
        """
        self.db_path = db_path
        self.chroma_dir = chroma_dir
        self.mode = default_mode
        if embedding_backend not in {"deterministic", "siliconflow"}:
            raise ValueError("embedding_backend must be deterministic or siliconflow")
        if retrieval_mode not in {"dense", "bm25_dense"}:
            raise ValueError("retrieval_mode must be dense or bm25_dense")
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be in [0, 1]")
        if reranker_backend not in {"none", "siliconflow"}:
            raise ValueError("reranker_backend must be none or siliconflow")
        if chunk_strategy not in {"card", "fallback"}:
            raise ValueError("chunk_strategy must be card or fallback")
        if rerank_unit is not None and rerank_unit not in {
            "card",
            "logical_evidence",
            "anchored_logical_window",
        }:
            raise ValueError(
                "rerank_unit must be card, logical_evidence, or anchored_logical_window"
            )
        if evidence_selection_count <= 0:
            raise ValueError("evidence_selection_count must be positive")
        if reranker_pool_size is not None and reranker_pool_size <= 0:
            raise ValueError("reranker_pool_size must be positive")
        # A real external reranker uses the production Anchored Parent-5/W7
        # route by default.  Local deterministic/no-reranker runs retain the
        # legacy card route unless the caller explicitly selects another unit.
        if rerank_unit is None:
            rerank_unit = (
                "anchored_logical_window"
                if reranker_backend == "siliconflow"
                else "card"
            )
        if reranker_pool_size is None:
            reranker_pool_size = 20 if rerank_unit == "anchored_logical_window" else 15
        if parent_card_count <= 0:
            raise ValueError("parent_card_count must be positive")
        if generator_contract_version not in {"v1", "v2"}:
            raise ValueError("generator_contract_version must be v1 or v2")
        if pipeline_mode not in {"legacy", "business-graph"}:
            raise ValueError("pipeline_mode must be legacy or business-graph")
        if embedding_mode not in {"disabled", "isolated-copy"}:
            raise ValueError("embedding_mode must be disabled or isolated-copy")
        self.embedding_backend = embedding_backend
        self.retrieval_mode = retrieval_mode
        self.bm25_weight = bm25_weight
        self.reranker_backend = reranker_backend
        self.chunk_strategy = chunk_strategy
        self.rerank_unit = rerank_unit
        self.evidence_selection_count = evidence_selection_count
        self.reranker_pool_size = reranker_pool_size
        self.parent_card_count = parent_card_count
        self.generator_contract_version = generator_contract_version
        self.pipeline_mode = pipeline_mode
        self.graph_db = Path(graph_db)
        self.embedding_mode = embedding_mode
        self.debug_graph = bool(debug_graph)
        self.allow_candidate_graph = bool(allow_candidate_graph)
        self.intent_model = intent_model
        self.storage: Optional[DualEngineStorageManager] = None
        self.retriever: Optional[DualMetricRetriever] = None
        self.pipeline: Optional[EndToEndPedagogicalRAGPipeline] = None
        self.business_pipeline = None

    def _validate_business_graph_artifact(self) -> dict[str, str]:
        """Require explicit opt-in before starting from an unapproved Graph."""

        graph_path = self.graph_db.resolve(strict=True)
        graph_hash = _sha256_file(graph_path)
        if graph_hash.lower() != APPROVED_BUSINESS_GRAPH_SHA256 and not self.allow_candidate_graph:
            raise ValueError(
                "graph artifact is not Gold-approved; pass --allow-candidate-graph explicitly"
            )
        return {
            "sha256": graph_hash,
            "approval_status": (
                "GOLD_APPROVED"
                if graph_hash.lower() == APPROVED_BUSINESS_GRAPH_SHA256
                else "PENDING_GOLD_REAPPROVAL"
            ),
        }

    def initialize(self):
        """初始化底层双引擎与 RAG 管道。"""
        print("⏳ 正在初始化双引擎存储与教研生成管道...", end="", flush=True)
        if self.pipeline_mode == "business-graph":
            from src.business_graph_pipeline import (
                BusinessGraphPipeline,
                BusinessGraphRuntimeConfig,
            )
            from src.intent_classifier import OpenAICompatibleIntentClassifier

            source_path = Path(self.db_path).resolve(strict=True)
            graph_path = self.graph_db.resolve(strict=True)
            chroma_path = Path(self.chroma_dir).resolve(strict=True)
            graph_provenance = self._validate_business_graph_artifact()
            embedding_retriever = None
            if self.embedding_mode == "isolated-copy":
                from src.business_graph_embedding import ChromaCardEmbeddingRetriever

                embedding_retriever = ChromaCardEmbeddingRetriever(
                    chroma_path,
                    embedding_backend=self.embedding_backend,
                    retrieval_mode=self.retrieval_mode,
                    bm25_weight=self.bm25_weight,
                )
            intent_classifier = OpenAICompatibleIntentClassifier.try_from_env(
                model=self.intent_model
            )
            try:
                self.business_pipeline = BusinessGraphPipeline(
                    source_db=source_path,
                    graph_db=graph_path,
                    chroma_dir=chroma_path,
                    runtime_config=BusinessGraphRuntimeConfig(
                        source_sha256=_sha256_file(source_path),
                        model_reranker_available=False,
                        generator_available=False,
                    ),
                    embedding_retriever=embedding_retriever,
                    intent_classifier=intent_classifier,
                )
            except Exception:
                close_embedding = getattr(embedding_retriever, "close", None)
                if callable(close_embedding):
                    close_embedding()
                raise
            classifier_info = (
                f"LLM 意图识别器={intent_classifier.model}"
                if intent_classifier is not None
                else "LLM 意图识别器=未配置(冲突/模糊问题诚实返回 UNMEASURED)"
            )
            print(
                " ✅ Graph Business 管道就绪 "
                f"(Graph={graph_provenance['approval_status']}；{classifier_info}；"
                "默认不启用 embedding；需 --embedding-mode isolated-copy)！\n"
            )
            return

        self.storage = DualEngineStorageManager(
            db_path=self.db_path,
            chroma_dir=self.chroma_dir,
            embedding_backend=self.embedding_backend,
        )
        self.retriever = DualMetricRetriever(
            storage_manager=self.storage,
            query_rewrite_mode="deterministic",
            alpha=0.5,
            retrieval_mode=self.retrieval_mode,
            bm25_weight=self.bm25_weight,
        )
        model_reranker = None
        if self.reranker_backend == "siliconflow":
            from src.reranker_provider import SiliconFlowQwen3Reranker

            model_reranker = SiliconFlowQwen3Reranker.from_env()
        assembler = PedagogicalGoldAssembler(
            lambda_diversity=0.7,
            max_prompt_tokens=4000,
            model_reranker=model_reranker,
            chunk_strategy=self.chunk_strategy,
            rerank_unit=self.rerank_unit,
            evidence_selection_count=self.evidence_selection_count,
            reranker_pool_size=self.reranker_pool_size,
            parent_card_count=self.parent_card_count,
        )
        self.pipeline = EndToEndPedagogicalRAGPipeline(
            retriever=self.retriever,
            assembler=assembler,
            chunk_strategy=self.chunk_strategy,
            generator_contract_version=self.generator_contract_version,
        )
        print(" ✅ 就绪！\n")

    def _generate_business_stream(
        self,
        query: str,
        assembled_prompt: str,
        on_chunk: Optional[Any] = None,
    ) -> str:
        """调用大模型基于业务证据与统计事实流式生成自然语言教研回答。"""
        import openai

        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL")
        if not api_key:
            key_ref = os.getenv(
                "GENERATOR_API_KEY_ENV",
                os.getenv("EMBEDDING_API_KEY_ENV", "SILICONFLOW_API_KEY"),
            )
            api_key = os.getenv(key_ref) if key_ref and not key_ref.startswith("sk-") else key_ref
            base_url = os.getenv(
                "GENERATOR_BASE_URL",
                os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
            )
            model = os.getenv("GENERATOR_MODEL", "Qwen/Qwen3.5-4B")

        if not api_key:
            raise RuntimeError("LLM API Key 未配置，无法执行生成")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.siliconflow.cn/v1",
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        )
        target_model = model or "Qwen/Qwen3.5-4B"

        system_msg = (
            "你是由 Eedi-RAG 驱动的权威名师教研备课与学情洞察专家。\n"
            "请严格基于系统提供的真实课堂对白证据与宏观图谱统计事实，专业、详实地解答用户的教研与学情问题。\n"
            "原则要求：\n"
            "1. 真实守信：只引用上下文中确实存在的误区卡、策略、数据或对白，绝不凭空捏造因果关系或虚假证据。\n"
            "2. 深度剖析：针对宏观问题提炼规律与考点分布；针对微观单题深入挖掘学生深层认知障碍并提供苏格拉底启发式引导话术。\n"
            "3. 清晰结构：采用教研报告排版，条理清晰，言之有据。"
        )
        user_msg = (
            f"【教研提问】\n{query}\n\n"
            f"【检索到的真实课堂证据与图谱上下文】\n{assembled_prompt}"
        )

        request_kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "stream": True,
            "temperature": 0.2,
        }
        if "qwen" in target_model.lower():
            request_kwargs["extra_body"] = {"enable_thinking": False}

        parts: list[str] = []
        stream = client.chat.completions.create(**request_kwargs)
        for event in stream:
            choices = getattr(event, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta is not None else None
            if not piece:
                continue
            parts.append(piece)
            if on_chunk is not None:
                on_chunk(piece)
        return "".join(parts).strip()

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

                if self.pipeline_mode == "business-graph":
                    if self.business_pipeline is None:
                        raise RuntimeError("business-graph pipeline 未初始化")
                    response = self.business_pipeline.ask_business(
                        BusinessQueryRequest(query=user_input, debug=self.debug_graph),
                        mode="deterministic",
                    )
                    elapsed_retrieval_ms = (time.perf_counter() - start_t) * 1000.0
                    intent_val = (
                        response.understanding.primary_intent.value
                        if response.understanding
                        else "UNKNOWN"
                    )
                    route_src = (
                        response.understanding.route_source
                        if response.understanding
                        else "UNKNOWN"
                    )
                    print("\n" + "=" * 88)
                    print(
                        f"🎯 业务意图: {intent_val} | 路由来源: {route_src} | "
                        f"状态: {response.status.value}"
                    )

                    # 越界拒答 (UNSUPPORTED)
                    if response.status.value == "UNSUPPORTED":
                        print("=" * 88)
                        print(f"🛑 {response.summary}")
                        if response.limitations:
                            print("📌 限制原因: " + "；".join(response.limitations))
                        print("=" * 88)
                        continue

                    # 无证据 (NO_EVIDENCE)
                    if response.status.value == "NO_EVIDENCE":
                        print("=" * 88)
                        print(f"⚠️ {response.summary}")
                        print("=" * 88)
                        continue

                    # deterministic 纯白盒模式：打印结构化 JSON 载荷
                    if self.mode == "deterministic":
                        print("=" * 88)
                        print(response.summary)
                        if response.result is not None:
                            print(json.dumps(response.result.payload, ensure_ascii=False, indent=2))
                        print(
                            f"状态: {response.status.value} | 审计: {response.audit_status} | "
                            f"证据: {len(response.evidence)} 条 | 检索耗时: {elapsed_retrieval_ms:.1f} ms"
                        )
                        if response.limitations:
                            print("限制: " + "；".join(response.limitations))
                        print("=" * 88)
                        continue

                    # auto / llm 模式：流式自然语言回答
                    assembled_prompt = ""
                    if response.result is not None:
                        assembled_prompt = response.result.payload.get("assembled_prompt", "")
                    if not assembled_prompt and response.result is not None:
                        assembled_prompt = json.dumps(response.result.payload, ensure_ascii=False)

                    print(
                        f"📊 课堂证据: {len(response.evidence)} 条 | "
                        f"检索分析耗时: {elapsed_retrieval_ms:.1f} ms"
                    )
                    print("=" * 88)
                    print("💬 【名师教研洞察解答】:\n")

                    answer_started = threading.Event()
                    heartbeat_stop = threading.Event()

                    def emit_business_feedback() -> None:
                        while not heartbeat_stop.wait(2.0):
                            if not answer_started.is_set():
                                print("⏳ 证据已就绪，大模型正在生成解答...", flush=True)

                    heartbeat_thread = threading.Thread(
                        target=emit_business_feedback,
                        name="eedi-rag-business-stream-feedback",
                        daemon=True,
                    )
                    heartbeat_thread.start()

                    def emit_business_chunk(chunk: str) -> None:
                        if chunk.strip():
                            answer_started.set()
                        print(chunk, end="", flush=True)

                    try:
                        self._generate_business_stream(
                            user_input,
                            assembled_prompt=assembled_prompt,
                            on_chunk=emit_business_chunk,
                        )
                    except Exception as exc:
                        print(f"\n❌ 模型生成遇到异常: {exc}")
                    finally:
                        heartbeat_stop.set()
                        heartbeat_thread.join(timeout=0.25)

                    print("\n\n" + "=" * 88)
                    total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                    print(
                        f"⏱️ 端到端总耗时: {total_elapsed_ms:.1f} ms | "
                        f"证据链条: {len(response.evidence)} 条 | "
                        f"审计状态: {response.audit_status}"
                    )
                    print("=" * 88)
                    continue

                if self.mode in {"auto", "llm"}:
                    answer_started = threading.Event()
                    heartbeat_stop = threading.Event()

                    def emit_waiting_feedback() -> None:
                        while not heartbeat_stop.wait(2.0):
                            if not answer_started.is_set():
                                print("\n⏳ 已完成检索，模型正在生成首段答案...", flush=True)

                    heartbeat_thread = threading.Thread(
                        target=emit_waiting_feedback,
                        name="eedi-rag-stream-feedback",
                        daemon=True,
                    )
                    heartbeat_thread.start()

                    def emit_answer_chunk(chunk: str) -> None:
                        if chunk.strip():
                            answer_started.set()
                        print(chunk, end="", flush=True)

                    try:
                        response = self.pipeline.ask_stream(
                            user_input,
                            mode=self.mode,
                            on_chunk=emit_answer_chunk,
                        )
                    finally:
                        heartbeat_stop.set()
                        heartbeat_thread.join(timeout=0.25)
                    print()
                else:
                    response = self.pipeline.ask(user_input, mode=self.mode)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                if self.mode in {"auto", "llm"}:
                    print("\n" + "=" * 88)
                else:
                    print("\n" + "=" * 88)
                    print(response.rendered_markdown)
                print("=" * 88)
                profiling = getattr(response, "_profiling", {})
                print(
                    f"⏱️ 总耗时: {elapsed_ms:.1f} ms | "
                    f"检索: {profiling.get('retrieval', 0.0) * 1000:.1f} ms | "
                    f"装配: {profiling.get('assembly', 0.0) * 1000:.1f} ms | "
                    f"生成: {profiling.get('generation', 0.0) * 1000:.1f} ms | "
                    f"首字: {profiling.get('output_ttft', 0.0) * 1000:.1f} ms | "
                    f"审计: {profiling.get('audit_render', 0.0) * 1000:.1f} ms\n"
                    f"考纲: {response.subject_path} | "
                    f"真实对白凭据: {len(response.dialogue_citations)} 轮 | "
                    f"审计: {response.audit_status}\n"
                )

            except KeyboardInterrupt:
                print("\n\n👋 接收到退出信号，正在安全退出...")
                break
            except Exception as e:
                print(f"\n❌ 执行遇到异常: {e}\n")

        if self.storage:
            self.storage.close()
        if self.business_pipeline:
            self.business_pipeline.close()


def main():
    """CLI 入口: 解析启动参数 (--db-path / --chroma-dir / --embedding-backend) 并进入 REPL。"""
    parser = argparse.ArgumentParser(description="Eedi-RAG interactive pedagogical CLI (BM25 + Dense + optional Qwen reranker)")
    parser.add_argument("--db-path", default="data/db/tutoring_knowledge.duckdb")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument(
        "--pipeline",
        "--route",
        dest="pipeline_mode",
        choices=("legacy", "business-graph"),
        default="legacy",
        help="explicitly select the legacy or business-graph route",
    )
    parser.add_argument("--graph-db", default=str(DEFAULT_BUSINESS_GRAPH_DB))
    parser.add_argument(
        "--embedding-mode",
        choices=("disabled", "isolated-copy"),
        default="disabled",
        help="business-graph dense Card lane; isolated-copy never opens production Chroma in place",
    )
    parser.add_argument("--debug-graph", action="store_true")
    parser.add_argument("--allow-candidate-graph", action="store_true")
    parser.add_argument("--embedding-backend", choices=("deterministic", "siliconflow"), default="deterministic")
    parser.add_argument("--retrieval-mode", choices=("dense", "bm25_dense"), default="bm25_dense")
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--reranker-backend", choices=("none", "siliconflow"), default="none")
    parser.add_argument("--chunk-strategy", choices=("card", "fallback"), default="card")
    parser.add_argument(
        "--rerank-unit",
        choices=("card", "logical_evidence", "anchored_logical_window"),
        default=None,
        help="rerank unit; with SiliconFlow defaults to anchored Parent-5/W7",
    )
    parser.add_argument("--evidence-selection-count", type=int, default=7)
    parser.add_argument("--reranker-pool-size", type=int, default=None)
    parser.add_argument("--parent-card-count", type=int, default=5)
    parser.add_argument("--generator-contract", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--intent-model",
        default=None,
        help="second-layer intent classifier model (e.g. Qwen/Qwen3.5-4B; defaults to INTENT_MODEL in .env)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "llm", "deterministic"),
        default="auto",
        help="generation mode: auto (default), llm, or deterministic",
    )
    args = parser.parse_args()
    cli = PedagogicalCLI(
        db_path=args.db_path,
        chroma_dir=args.chroma_dir,
        default_mode=args.mode,
        embedding_backend=args.embedding_backend,
        retrieval_mode=args.retrieval_mode,
        bm25_weight=args.bm25_weight,
        reranker_backend=args.reranker_backend,
        chunk_strategy=args.chunk_strategy,
        rerank_unit=args.rerank_unit,
        evidence_selection_count=args.evidence_selection_count,
        reranker_pool_size=args.reranker_pool_size,
        parent_card_count=args.parent_card_count,
        generator_contract_version=args.generator_contract,
        pipeline_mode=args.pipeline_mode,
        graph_db=args.graph_db,
        embedding_mode=args.embedding_mode,
        debug_graph=args.debug_graph,
        allow_candidate_graph=args.allow_candidate_graph,
        intent_model=args.intent_model,
    )
    cli.run()


if __name__ == "__main__":
    main()
