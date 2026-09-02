"""
================================================================================
模块名称: src/retriever.py
业务定位: Step 4 - 双度量向量混合检索器与多视角 RRF 融合引擎 (Dual-Metric Hybrid Vector Retriever)
核心职责:
  1. 原始提问向量化 (Raw Query Embedding):
     - 接收用户原始提问，经由 EmbeddingFunction 计算为高维稠密浮点向量。
  2. 余弦相似度 (Cosine) + 欧氏距离相似度 (Euclidean L2) 联合打分:
     - Cosine Score: 衡量夹角方向 (消除字数模长偏置)
     - Euclidean L2 Score: 1 / (1 + L2_Distance)，衡量绝对空间邻近度
     - Hybrid Score: α * Cosine + (1 - α) * Euclidean (默认 α=0.5)
  3. 多视角子查询生成 + RRF 晚期融合 (Multi-Perspective RRF Fusion):
     - 调用 MultiPerspectiveQueryRewriter 动态注入考纲并派生 3 个视角
     - 并发检索多视角后按倒数排名融合: RRF(d) = ∑ 1 / (60 + Rank_p(d))
  4. 跨集合多路召回与重排序:
     - student_misconceptions (学情诊断与错因机理)
     - tutor_strategies (名师破局提问与启发脚手架)
     - fallback_windows (零信任纯原文滑动窗口)
  5. 瞬时跨引擎证据回溯 (Grounding Assembly):
     - 命中候选卡片后，自动提取 source_turn_ids 指针，从 DuckDB 中瞬时拉取真实对话对白作为不可篡改证据。
================================================================================
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import MultiPerspectiveQueries
from src.query_rewriter import MultiPerspectiveQueryRewriter
from src.storage_manager import DualEngineStorageManager, FastDeterministicEmbeddingFunction

logger = logging.getLogger(__name__)


def rrf_ranked_ids(rankings: list[list[Dict[str, Any]]], k_constant: int) -> list[str]:
    """Return a deterministic RRF ordering for trace/evaluation comparison."""
    if k_constant <= 0:
        raise ValueError("k_constant must be positive")
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    order = 0
    for ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            chunk_id = str(candidate["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_constant + rank)
            first_seen.setdefault(chunk_id, order)
            order += 1
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))
    ]


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    计算两个浮点向量之间的余弦相似度 (Cosine Similarity)，输出范围 [-1.0, 1.0]。
    """
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))


def compute_euclidean_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """
    计算两个浮点向量之间的欧氏几何距离 (L2 Distance)，输出范围 [0.0, +inf)。
    """
    if len(vec_a) != len(vec_b):
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def compute_euclidean_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    将欧氏几何距离映射为相似度评分 (Euclidean Similarity)，输出范围 (0.0, 1.0]。
    公式: S_L2 = 1.0 / (1.0 + Distance_L2)
    """
    dist = compute_euclidean_distance(vec_a, vec_b)
    return 1.0 / (1.0 + dist)


def compute_hybrid_score(
    vec_a: List[float],
    vec_b: List[float],
    alpha: float = 0.5
) -> Tuple[float, float, float]:
    """
    计算双度量联合评分:
    - 归一化余弦评分: norm_cos = (cosine_sim + 1.0) / 2.0  (映射至 [0.0, 1.0])
    - 欧氏相似度评分: l2_sim = 1.0 / (1.0 + l2_dist)       (映射至 (0.0, 1.0])
    - 综合加权评分: hybrid_score = alpha * norm_cos + (1.0 - alpha) * l2_sim
    
    返回: (hybrid_score, norm_cos, l2_sim)
    """
    raw_cos = compute_cosine_similarity(vec_a, vec_b)
    norm_cos = (raw_cos + 1.0) / 2.0
    l2_sim = compute_euclidean_similarity(vec_a, vec_b)
    hybrid_score = alpha * norm_cos + (1.0 - alpha) * l2_sim
    return hybrid_score, norm_cos, l2_sim


class DualMetricRetriever:
    """
    双度量混合向量检索器与多视角 RRF 融合引擎:
    - 结合余弦相似度与欧氏距离相似度对候选向量执行二次精细重排
    - 支持多视角子查询派生与倒数排名融合 (RRF)
    - 自动联动 DuckDB 完成 [Turn N] 原声证据回溯
    """

    def __init__(
        self,
        storage_manager: Optional[DualEngineStorageManager] = None,
        rewriter: Optional[MultiPerspectiveQueryRewriter] = None,
        query_rewrite_mode: Optional[str] = None,
        alpha: float = 0.5
    ):
        """
        初始化检索器。
        
        参数:
          storage_manager: 传入已初始化的双引擎存储管理器 (若为 None 则默认初始化本地底表)
          rewriter: 多视角查询改写器
          alpha: 余弦相似度权重 (1-alpha 为欧氏相似度权重，默认 0.5)
        """
        self.alpha = max(0.0, min(1.0, alpha))
        if storage_manager is None:
            raise ValueError("必须显式传入已配置 embedding 后端的 storage_manager")
        self.storage = storage_manager
            
        self.embedding_fn = self.storage.embedding_function
        if rewriter is not None and query_rewrite_mode is not None:
            raise ValueError("rewriter 与 query_rewrite_mode 只能指定一个")
        if rewriter is not None:
            self.rewriter = rewriter
        elif query_rewrite_mode == "deterministic":
            self.rewriter = MultiPerspectiveQueryRewriter(mode="deterministic")
        else:
            raise ValueError(
                "必须显式传入 rewriter，或设置 query_rewrite_mode='deterministic'"
            )

    def _rank_collection_candidates(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 3,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        从指定 ChromaDB 集合中获取所有/初筛候选，并按 (Cosine + Euclidean L2) 双度量重新精确打分排序。
        """
        coll = self.storage.chroma_client.get_collection(collection_name)
        total_count = coll.count()
        if total_count == 0:
            return []
            
        candidate_k = min(total_count, max(top_k * 5, 20))
        
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": candidate_k,
            "include": ["documents", "metadatas", "embeddings"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter
            
        raw_res = coll.query(**query_kwargs)
        if not raw_res or not raw_res.get("ids") or len(raw_res["ids"][0]) == 0:
            return []
            
        rescored_candidates = []
        ids = raw_res["ids"][0]
        docs = raw_res["documents"][0] if raw_res.get("documents") else [""] * len(ids)
        metas = raw_res["metadatas"][0] if raw_res.get("metadatas") else [{}] * len(ids)
        embeddings = raw_res["embeddings"][0] if raw_res.get("embeddings") else []
        
        for i in range(len(ids)):
            doc_vec = embeddings[i] if len(embeddings) > i else None
            if doc_vec is not None and len(doc_vec) > 0:
                hybrid_s, cos_s, l2_s = compute_hybrid_score(query_vector, doc_vec, alpha=self.alpha)
            else:
                hybrid_s, cos_s, l2_s = 0.0, 0.0, 0.0
                
            rescored_candidates.append({
                "chunk_id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "hybrid_score": round(hybrid_s, 5),
                "cosine_score": round(cos_s, 5),
                "euclidean_score": round(l2_s, 5),
                "collection": collection_name
            })
            
        # 按综合混合得分降序排列
        rescored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return rescored_candidates[:top_k]

    def retrieve_misconceptions(
        self,
        query_text: str,
        top_k: int = 3,
        fetch_evidence: bool = True,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        检索学生认知误区卡片 (student_misconceptions)。
        """
        query_vector = self.embedding_fn([query_text])[0]
        hits = self._rank_collection_candidates("student_misconceptions", query_vector, top_k=top_k, where_filter=where_filter)
        
        if fetch_evidence:
            for item in hits:
                sess_id = item["metadata"].get("session_id")
                turn_ids_raw = item["metadata"].get("source_turn_ids")
                turn_ids = json.loads(turn_ids_raw) if isinstance(turn_ids_raw, str) else (turn_ids_raw or [])
                if sess_id:
                    item["evidence_turns"] = self.storage.get_dialogue_turns(sess_id, turn_ids)
                    
        return hits

    def retrieve_strategies(
        self,
        query_text: str,
        top_k: int = 3,
        fetch_evidence: bool = True,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        检索名师启发式策略卡片 (tutor_strategies)。
        """
        query_vector = self.embedding_fn([query_text])[0]
        hits = self._rank_collection_candidates("tutor_strategies", query_vector, top_k=top_k, where_filter=where_filter)
        
        if fetch_evidence:
            for item in hits:
                sess_id = item["metadata"].get("session_id")
                turn_ids_raw = item["metadata"].get("source_turn_ids")
                turn_ids = json.loads(turn_ids_raw) if isinstance(turn_ids_raw, str) else (turn_ids_raw or [])
                if sess_id:
                    item["evidence_turns"] = self.storage.get_dialogue_turns(sess_id, turn_ids)
                    
        return hits

    def retrieve_fallback_windows(
        self,
        query_text: str,
        top_k: int = 3,
        where_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        零信任模式：纯原文滑动窗口直接检索 (fallback_windows)。
        """
        query_vector = self.embedding_fn([query_text])[0]
        return self._rank_collection_candidates("fallback_windows", query_vector, top_k=top_k, where_filter=where_filter)

    def retrieve_hybrid(
        self,
        query_text: str,
        top_k_each: int = 2,
        fetch_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        双度量并行检索错因与策略库（直接输入原始提问）。
        """
        misc_hits = self.retrieve_misconceptions(query_text, top_k=top_k_each, fetch_evidence=fetch_evidence)
        strat_hits = self.retrieve_strategies(query_text, top_k=top_k_each, fetch_evidence=fetch_evidence)
        
        return {
            "query": query_text,
            "metric_config": {
                "metric": "Cosine + Euclidean (L2) Hybrid",
                "alpha": self.alpha,
                "formula": "Score = alpha * Cosine_Norm + (1 - alpha) * (1 / (1 + L2_Dist))"
            },
            "misconceptions": misc_hits,
            "strategies": strat_hits,
            "total_retrieved": len(misc_hits) + len(strat_hits)
        }

    def retrieve_multi_perspective_rrf(
        self,
        raw_query: str,
        top_k_each: int = 3,
        fetch_evidence: bool = True,
        k_constant: int = 60
    ) -> Dict[str, Any]:
        """
        【黄金组合入口】：多视角子查询派生 + 专业信息动态注入 + RRF 倒数排名融合。
        
        执行步骤:
          1. 调用 MultiPerspectiveQueryRewriter 将原始提问转换为 3 个正交视角的专业 Query:
             - misconception_query (学情错因视角)
             - strategy_query (名师教法视角)
             - curriculum_query (考纲考点视角)
          2. 多视角并发检索 student_misconceptions 与 tutor_strategies;
          3. 实施 RRF 倒数排名融合: RRF(d) = ∑ 1 / (k + Rank_p(d));
          4. 瞬时回溯 DuckDB [Turn N] 原声证据。
        """
        # 1. 多视角派生与专业注入
        rewritten: MultiPerspectiveQueries = self.rewriter.rewrite(raw_query)

        # 2. 错因库多视角检索与 RRF 融合 (3-Way: misconception + curriculum + raw_query)
        misc_ranks_p1 = self.retrieve_misconceptions(rewritten.misconception_query, top_k=15, fetch_evidence=False)
        misc_ranks_p2 = self.retrieve_misconceptions(rewritten.curriculum_query, top_k=15, fetch_evidence=False)
        misc_ranks_p3 = self.retrieve_misconceptions(raw_query, top_k=15, fetch_evidence=False)
        
        misc_rrf_map: Dict[str, Dict[str, Any]] = {}
        for rank_idx, cand in enumerate(misc_ranks_p1, start=1):
            cid = cand["chunk_id"]
            if cid not in misc_rrf_map:
                misc_rrf_map[cid] = {"candidate": cand, "rrf_score": 0.0, "perspective_hits": []}
            misc_rrf_map[cid]["rrf_score"] += 1.0 / (k_constant + rank_idx)
            misc_rrf_map[cid]["perspective_hits"].append(f"misconception_perspective(#Rank{rank_idx})")

        for rank_idx, cand in enumerate(misc_ranks_p2, start=1):
            cid = cand["chunk_id"]
            if cid not in misc_rrf_map:
                misc_rrf_map[cid] = {"candidate": cand, "rrf_score": 0.0, "perspective_hits": []}
            misc_rrf_map[cid]["rrf_score"] += 1.0 / (k_constant + rank_idx)
            misc_rrf_map[cid]["perspective_hits"].append(f"curriculum_perspective(#Rank{rank_idx})")

        for rank_idx, cand in enumerate(misc_ranks_p3, start=1):
            cid = cand["chunk_id"]
            if cid not in misc_rrf_map:
                misc_rrf_map[cid] = {"candidate": cand, "rrf_score": 0.0, "perspective_hits": []}
            misc_rrf_map[cid]["rrf_score"] += 1.0 / (k_constant + rank_idx)
            misc_rrf_map[cid]["perspective_hits"].append(f"raw_query_perspective(#Rank{rank_idx})")

        fused_misc = sorted(misc_rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        top_misc_results = []
        for item in fused_misc[:top_k_each]:
            c = item["candidate"]
            c["rrf_score"] = round(item["rrf_score"], 6)
            c["perspective_hits"] = item["perspective_hits"]
            if fetch_evidence:
                sess_id = c["metadata"].get("session_id")
                turn_ids_raw = c["metadata"].get("source_turn_ids")
                turn_ids = json.loads(turn_ids_raw) if isinstance(turn_ids_raw, str) else (turn_ids_raw or [])
                if sess_id:
                    c["evidence_turns"] = self.storage.get_dialogue_turns(sess_id, turn_ids)
            top_misc_results.append(c)

        # 3. 策略库多视角检索与 RRF 融合 (3-Way: strategy + curriculum + raw_query)
        strat_ranks_p1 = self.retrieve_strategies(rewritten.strategy_query, top_k=15, fetch_evidence=False)
        strat_ranks_p2 = self.retrieve_strategies(rewritten.curriculum_query, top_k=15, fetch_evidence=False)
        strat_ranks_p3 = self.retrieve_strategies(raw_query, top_k=15, fetch_evidence=False)

        lane_trace = {
            "misconception": [
                {"chunk_id": str(c["chunk_id"]), "rank": rank, "score": c.get("hybrid_score")}
                for rank, c in enumerate(misc_ranks_p1, start=1)
            ],
            "misconception_curriculum": [
                {"chunk_id": str(c["chunk_id"]), "rank": rank, "score": c.get("hybrid_score")}
                for rank, c in enumerate(misc_ranks_p2, start=1)
            ],
            "misconception_raw": [
                {"chunk_id": str(c["chunk_id"]), "rank": rank, "score": c.get("hybrid_score")}
                for rank, c in enumerate(misc_ranks_p3, start=1)
            ],
            "strategy": [
                {"chunk_id": str(c["chunk_id"]), "rank": rank, "score": c.get("hybrid_score")}
                for rank, c in enumerate(strat_ranks_p1, start=1)
            ],
            "strategy_curriculum": [
                {"chunk_id": str(c["chunk_id"]), "rank": rank, "score": c.get("hybrid_score")}
                for rank, c in enumerate(strat_ranks_p2, start=1)
            ],
            "strategy_raw": [
                {"chunk_id": str(c["chunk_id"]), "rank": rank, "score": c.get("hybrid_score")}
                for rank, c in enumerate(strat_ranks_p3, start=1)
            ],
        }
        fused_trace = {
            "misconception_rewrite_only": rrf_ranked_ids([misc_ranks_p1, misc_ranks_p2], k_constant),
            "misconception_raw_inclusive": rrf_ranked_ids([misc_ranks_p1, misc_ranks_p2, misc_ranks_p3], k_constant),
            "strategy_rewrite_only": rrf_ranked_ids([strat_ranks_p1, strat_ranks_p2], k_constant),
            "strategy_raw_inclusive": rrf_ranked_ids([strat_ranks_p1, strat_ranks_p2, strat_ranks_p3], k_constant),
        }
        
        strat_rrf_map: Dict[str, Dict[str, Any]] = {}
        for rank_idx, cand in enumerate(strat_ranks_p1, start=1):
            cid = cand["chunk_id"]
            if cid not in strat_rrf_map:
                strat_rrf_map[cid] = {"candidate": cand, "rrf_score": 0.0, "perspective_hits": []}
            strat_rrf_map[cid]["rrf_score"] += 1.0 / (k_constant + rank_idx)
            strat_rrf_map[cid]["perspective_hits"].append(f"strategy_perspective(#Rank{rank_idx})")

        for rank_idx, cand in enumerate(strat_ranks_p2, start=1):
            cid = cand["chunk_id"]
            if cid not in strat_rrf_map:
                strat_rrf_map[cid] = {"candidate": cand, "rrf_score": 0.0, "perspective_hits": []}
            strat_rrf_map[cid]["rrf_score"] += 1.0 / (k_constant + rank_idx)
            strat_rrf_map[cid]["perspective_hits"].append(f"curriculum_perspective(#Rank{rank_idx})")

        for rank_idx, cand in enumerate(strat_ranks_p3, start=1):
            cid = cand["chunk_id"]
            if cid not in strat_rrf_map:
                strat_rrf_map[cid] = {"candidate": cand, "rrf_score": 0.0, "perspective_hits": []}
            strat_rrf_map[cid]["rrf_score"] += 1.0 / (k_constant + rank_idx)
            strat_rrf_map[cid]["perspective_hits"].append(f"raw_query_perspective(#Rank{rank_idx})")

        fused_strat = sorted(strat_rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        top_strat_results = []
        for item in fused_strat[:top_k_each]:
            c = item["candidate"]
            c["rrf_score"] = round(item["rrf_score"], 6)
            c["perspective_hits"] = item["perspective_hits"]
            if fetch_evidence:
                sess_id = c["metadata"].get("session_id")
                turn_ids_raw = c["metadata"].get("source_turn_ids")
                turn_ids = json.loads(turn_ids_raw) if isinstance(turn_ids_raw, str) else (turn_ids_raw or [])
                if sess_id:
                    c["evidence_turns"] = self.storage.get_dialogue_turns(sess_id, turn_ids)
            top_strat_results.append(c)

        return {
            "raw_query": raw_query,
            "execution_metadata": {
                "query_rewrite_backend": self.rewriter.backend_name,
                "embedding_backend": self.storage.embedding_backend,
            },
            "rewritten_queries": rewritten.model_dump(),
            "trace": {
                "expanded_queries": rewritten.model_dump(),
                "lanes": lane_trace,
                "fused_rankings": fused_trace,
                "fusion": "raw-inclusive-rrf",
                "rrf_k": k_constant,
                "candidate_count": len(top_misc_results) + len(top_strat_results),
            },
            "misconceptions": top_misc_results,
            "strategies": top_strat_results,
            "total_retrieved": len(top_misc_results) + len(top_strat_results)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("🚀 测试 Multi-Query + Domain Knowledge + RRF 融合检索器...")
    
    storage = DualEngineStorageManager(
        db_path="data/db/tutoring_knowledge.duckdb",
        chroma_dir="data/chroma",
        embedding_backend="deterministic",
    )
    
    retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic", alpha=0.5)
    
    raw_query = "学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？"
    res = retriever.retrieve_multi_perspective_rrf(raw_query, top_k_each=1, fetch_evidence=True)
    
    print(f"\n🔍 原始输入: {raw_query}")
    print(f"📝 注入专业术语: {res['rewritten_queries']['extracted_keywords']}")
    print(f"🎯 错因视角 Query: {res['rewritten_queries']['misconception_query']}")
    
    if res["misconceptions"]:
        top_m = res["misconceptions"][0]
        print(f"\n【RRF 命中错因卡】: {top_m['metadata'].get('misconception_name')}")
        print(f"  - RRF 得分: {top_m['rrf_score']} | 命中视角: {top_m['perspective_hits']}")
        print(f"  - 真实对白证据: {len(top_m.get('evidence_turns', []))} 轮")
        
    storage.close()
    print("\n✅ Multi-Perspective RRF 检索器测试通过！")
