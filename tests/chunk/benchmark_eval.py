# -*- coding: utf-8 -*-
"""
tests/chunk/benchmark_eval.py
-----------------------------
长对话分块策略实证评测基准运行引擎 (Empirical Chunking Benchmark Evaluator)。
对比多种切块策略在构建时延、Token 膨胀比、检索命中率 (Hit@1, Hit@3) 及 MRR 上的表现，
输出量化评测报告与帕累托最优基座选型结论。
"""

import sys
import json
import time
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

# 动态确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.chunk.chunkers import (
    BaseChunker,
    Chunk,
    TurnSlidingWindowChunker,
    HeaderInjectedChunker,
    ParentChildChunker,
    KnowledgeDistilledChunker,
)


class LightweightRetriever:
    """
    确定性可复现的轻量化混合检索器 (Bilingual Character & Word N-Gram TF-IDF Vectorizer)
    无需依赖外部重型模型，实现毫秒级自包含向量化与余弦相似度排序。
    """
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_lengths: List[float] = []
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中英文双语分词与字符 n-gram 提取"""
        text = text.lower()
        # 英文单词
        words = re.findall(r"\b[a-z0-9_.]+\b", text)
        # 中文字符与连续 2-gram
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_bi = [chinese_chars[i] + chinese_chars[i+1] for i in range(len(chinese_chars) - 1)]
        return words + chinese_chars + chinese_bi

    def fit_and_index(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.doc_term_freqs = []
        self.doc_lengths = []
        doc_freq = Counter()
        n_docs = len(chunks)
        
        for c in chunks:
            tokens = self.tokenize(c.text)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for term in set(tokens):
                doc_freq[term] += 1
                
        # 计算平滑 IDF
        self.idf = {term: math.log((n_docs + 1) / (df + 1)) + 1.0 for term, df in doc_freq.items()}
        
        # 计算文档向量 L2 范数
        for tf in self.doc_term_freqs:
            norm_sq = sum((count * self.idf.get(term, 1.0)) ** 2 for term, count in tf.items())
            self.doc_lengths.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query_tokens = self.tokenize(query)
        query_tf = Counter(query_tokens)
        query_norm_sq = sum((count * self.idf.get(term, 1.0)) ** 2 for term, count in query_tf.items())
        query_len = math.sqrt(query_norm_sq) if query_norm_sq > 0 else 1.0
        
        scores: List[Tuple[int, float]] = []
        for doc_idx, doc_tf in enumerate(self.doc_term_freqs):
            dot_product = sum(
                q_count * doc_tf[term] * (self.idf.get(term, 1.0) ** 2)
                for term, q_count in query_tf.items()
                if term in doc_tf
            )
            similarity = dot_product / (query_len * self.doc_lengths[doc_idx])
            scores.append((doc_idx, similarity))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in scores[:top_k]]


def load_cleaned_sessions(sample_jsonl_path: Path) -> List[Dict[str, Any]]:
    sessions = []
    with open(sample_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def load_ground_truth_queries(gt_path: Path) -> List[Dict[str, Any]]:
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_strategy(
    chunker: BaseChunker,
    sessions: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    对单一分块策略进行全流程自动化评测
    """
    # 1. 评测切片耗时与产出量
    start_time = time.perf_counter()
    all_chunks: List[Chunk] = []
    for s in sessions:
        all_chunks.extend(chunker.chunk_session(s))
    build_time_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency_ms = build_time_ms / len(sessions) if sessions else 0.0

    # 2. 评测 Token 膨胀比
    total_raw_words = sum(
        sum(len(t.get("text", "").split()) for t in s.get("turns", []))
        for s in sessions
    )
    total_chunk_words = sum(c.token_count_approx for c in all_chunks)
    inflation_ratio = round(total_chunk_words / max(total_raw_words, 1), 2)

    # 3. 建立索引并检索评测
    retriever = LightweightRetriever()
    retriever.fit_and_index(all_chunks)

    session_hit_1 = 0
    session_hit_3 = 0
    session_rr_sum = 0.0
    turn_hit_1 = 0
    turn_hit_3 = 0
    turn_rr_sum = 0.0
    n_queries = len(queries)

    for q in queries:
        combined_query = f"{q.get('query', '')} {q.get('query_en', '')}"
        target_session_id = q.get("target_intervention_id")
        target_turn_ids = set(q.get("target_turn_ids", []))

        top_results = retriever.search(combined_query, top_k=5)

        # 检查 Session 命中与 MRR
        session_rank = None
        turn_rank = None

        for rank, (chunk, score) in enumerate(top_results, start=1):
            if chunk.intervention_id == target_session_id:
                if session_rank is None:
                    session_rank = rank
                # 检查 Turn 级精准重叠 (是否命中了关键教学轮次)
                chunk_turns = set(chunk.turn_ids)
                if chunk_turns.intersection(target_turn_ids) and turn_rank is None:
                    turn_rank = rank

        if session_rank is not None:
            if session_rank == 1:
                session_hit_1 += 1
            if session_rank <= 3:
                session_hit_3 += 1
            session_rr_sum += 1.0 / session_rank

        if turn_rank is not None:
            if turn_rank == 1:
                turn_hit_1 += 1
            if turn_rank <= 3:
                turn_hit_3 += 1
            turn_rr_sum += 1.0 / turn_rank

    session_mrr = round(session_rr_sum / n_queries, 4) if n_queries else 0.0
    turn_mrr = round(turn_rr_sum / n_queries, 4) if n_queries else 0.0
    
    # 4. 计算综合性价比得分
    # Composite Score = (Turn_MRR * 100) / (log(1 + Latency) * InflationRatio)
    cost_factor = max(math.log(1.0 + avg_latency_ms), 0.1) * max(inflation_ratio, 0.5)
    composite_score = round((turn_mrr * 100.0) / cost_factor, 2)

    return {
        "strategy_name": chunker.strategy_name,
        "num_chunks": len(all_chunks),
        "avg_latency_ms": round(avg_latency_ms, 3),
        "inflation_ratio": inflation_ratio,
        "session_hit_1": f"{session_hit_1}/{n_queries} ({session_hit_1/n_queries:.0%})",
        "session_hit_3": f"{session_hit_3}/{n_queries} ({session_hit_3/n_queries:.0%})",
        "session_mrr": session_mrr,
        "turn_hit_1": f"{turn_hit_1}/{n_queries} ({turn_hit_1/n_queries:.0%})",
        "turn_hit_3": f"{turn_hit_3}/{n_queries} ({turn_hit_3/n_queries:.0%})",
        "turn_mrr": turn_mrr,
        "composite_score": composite_score,
    }


def run_benchmark():
    project_root = Path(__file__).resolve().parent.parent.parent
    sample_jsonl = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    gt_json = project_root / "tests" / "chunk" / "ground_truth_queries.json"

    sessions = load_cleaned_sessions(sample_jsonl)
    queries = load_ground_truth_queries(gt_json)

    candidate_chunkers: List[BaseChunker] = [
        TurnSlidingWindowChunker(window_size=4, step_size=2),
        TurnSlidingWindowChunker(window_size=6, step_size=3),
        HeaderInjectedChunker(pair_size=2, step_size=2),
        ParentChildChunker(child_turn_size=2, child_step_size=2),
        KnowledgeDistilledChunker(),
    ]

    print("=" * 80)
    print("🚀 正在运行长对话分块策略实证基准评测 (Empirical Chunking Benchmark)...")
    print(f"📊 评测样本: {len(sessions)} 个真实会话 | 评测查询集: {len(queries)} 条黄金真值")
    print("=" * 80)

    results = []
    for chunker in candidate_chunkers:
        res = evaluate_strategy(chunker, sessions, queries)
        results.append(res)

    # 打印 Markdown 格式表格
    print("\n### 🏆 长对话切块策略综合评测基准榜单 (Benchmark Results Table)\n")
    headers = [
        "切块策略 (Strategy)",
        "总Chunk数",
        "单会话耗时(ms)",
        "Token膨胀比",
        "Session Hit@1",
        "Session Hit@3",
        "Session MRR",
        "Turn Hit@1",
        "Turn Hit@3",
        "Turn MRR",
        "性价比综合分",
    ]
    
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    print(header_line)
    print(sep_line)

    for r in results:
        row = [
            f"`{r['strategy_name']}`",
            str(r["num_chunks"]),
            f"{r['avg_latency_ms']:.2f}ms",
            f"{r['inflation_ratio']}x",
            r["session_hit_1"],
            r["session_hit_3"],
            f"{r['session_mrr']:.3f}",
            r["turn_hit_1"],
            r["turn_hit_3"],
            f"{r['turn_mrr']:.3f}",
            f"**{r['composite_score']}**",
        ]
        print("| " + " | ".join(row) + " |")

    # 保存评测结果至 markdown 文件
    output_md = project_root / "tests" / "chunk" / "benchmark_results.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# 长对话分块策略实证评测基准报告 (Benchmark Results Report)\n\n")
        f.write(f"> 评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"> 样本规模: {len(sessions)} 个清洗后真实会话，{len(queries)} 条标准教研真值问答对。\n\n")
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        for r in results:
            row = [
                f"`{r['strategy_name']}`",
                str(r["num_chunks"]),
                f"{r['avg_latency_ms']:.2f}ms",
                f"{r['inflation_ratio']}x",
                r["session_hit_1"],
                r["session_hit_3"],
                f"{r['session_mrr']:.3f}",
                r["turn_hit_1"],
                r["turn_hit_3"],
                f"{r['turn_mrr']:.3f}",
                f"**{r['composite_score']}**",
            ]
            f.write("| " + " | ".join(row) + " |\n")

    print(f"\n✅ 评测报告已持久化保存至: {output_md}")
    return results


if __name__ == "__main__":
    run_benchmark()
