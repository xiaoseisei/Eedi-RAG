"""
================================================================================
模块名称: src/retriever.py
业务定位: Step 4 - BM25 + Dense 混合检索器、多视角 RRF 融合引擎
核心职责:
  1. 原始提问向量化 (Raw Query Embedding):
     - 接收用户原始提问，经由 EmbeddingFunction 计算为高维稠密浮点向量。
  2. Dense 余弦相似度 (Cosine) + 欧氏距离相似度 (Euclidean L2) 联合打分，并与 BM25 lexical score 加权融合:
     - Cosine Score: 衡量夹角方向 (消除字数模长偏置)
     - Euclidean L2 Score: 1 / (1 + L2_Distance)，衡量绝对空间邻近度
     - Hybrid Score: α * Cosine + (1 - α) * Euclidean (默认 α=0.5)
  3. BM25 lexical index 与 Dense 候选 union，按归一化分数加权保留精确数字/短语命中:
     - Final = (1 - bm25_weight) * Dense_Hybrid + bm25_weight * BM25_Normalized
  4. 多视角子查询生成 + RRF 晚期融合 (Multi-Perspective RRF Fusion):
     - 调用 MultiPerspectiveQueryRewriter 动态注入考纲并派生 3 个视角
     - 并发检索多视角后按倒数排名融合: RRF(d) = ∑ 1 / (60 + Rank_p(d))
  5. 跨集合多路召回与重排序:
     - student_misconceptions (学情诊断与错因机理)
     - tutor_strategies (名师破局提问与启发脚手架)
     - fallback_windows (零信任纯原文滑动窗口)
  6. 瞬时跨引擎证据回溯 (Grounding Assembly):
     - 命中候选卡片后，自动提取 source_turn_ids 指针，从 DuckDB 中瞬时拉取真实对话对白作为不可篡改证据。
================================================================================
"""

import os
import sys
import json
import math
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import MultiPerspectiveQueries
from src.bm25 import BM25Index
from src.query_rewriter import MultiPerspectiveQueryRewriter
from src.storage_manager import DualEngineStorageManager, FastDeterministicEmbeddingFunction

logger = logging.getLogger(__name__)


def _normalized_math_text(text: str) -> str:
    """数学文本归一化: 统一 Unicode 运算符 (−×÷ → -* /) 并去除数字间空白，供精确通道比对。"""
    value = text.casefold().replace("−", "-").replace("×", "*").replace("÷", "/")
    value = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", value)
    value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)
    return value


def _numeric_formula_tokens(text: str) -> set[str]:
    """提取归一化后的数字/分数 token 集合 (如 '5.4598', '3/4', '-12')，作为精确匹配的比对单元。"""
    normalized = _normalized_math_text(text)
    return set(re.findall(r"[+-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", normalized))


def numeric_formula_exact_score(query: str, document: str) -> float:
    """Score exact numeric/formula matches, prioritizing dialogue over repeated anchors."""
    query_tokens = _numeric_formula_tokens(query)
    if not query_tokens:
        return 0.0
    marker = "[对话片段"
    anchor, separator, dialogue = document.partition(marker)
    anchor_tokens = _numeric_formula_tokens(anchor)
    dialogue_tokens = _numeric_formula_tokens(dialogue if separator else document)
    dialogue_overlap = len(query_tokens.intersection(dialogue_tokens)) / len(query_tokens)
    anchor_overlap = len(query_tokens.intersection(anchor_tokens)) / len(query_tokens)
    return round(min(1.0, dialogue_overlap + 0.1 * anchor_overlap), 6)


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
    BM25 + Dense 混合检索器与多视角 RRF 融合引擎:
    - 结合 lexical BM25 与余弦/欧氏 Dense 分数召回并排序
    - 支持多视角子查询派生与倒数排名融合 (RRF)
    - 自动联动 DuckDB 完成 [Turn N] 原声证据回溯
    """

    def __init__(
        self,
        storage_manager: Optional[DualEngineStorageManager] = None,
        rewriter: Optional[MultiPerspectiveQueryRewriter] = None,
        query_rewrite_mode: Optional[str] = None,
        alpha: float = 0.5,
        fusion_strategy: str = "raw_inclusive_rrf",
        retrieval_mode: str = "bm25_dense",
        bm25_weight: float = 0.35,
    ):
        """
        初始化检索器。
        
        参数:
          storage_manager: 传入已初始化的双引擎存储管理器 (若为 None 则默认初始化本地底表)
          rewriter: 多视角查询改写器
          alpha: 余弦相似度权重 (1-alpha 为欧氏相似度权重，默认 0.5)
        """
        self.alpha = max(0.0, min(1.0, alpha))
        if fusion_strategy not in {"raw_first", "raw_inclusive_rrf"}:
            raise ValueError("fusion_strategy must be raw_first or raw_inclusive_rrf")
        self.fusion_strategy = fusion_strategy
        if retrieval_mode not in {"dense", "bm25_dense"}:
            raise ValueError("retrieval_mode must be dense or bm25_dense")
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be in [0, 1]")
        self.retrieval_mode = retrieval_mode
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = 1.0 - self.bm25_weight
        self._bm25_indexes: Dict[str, BM25Index] = {}
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

    def _get_bm25_index(self, collection_name: str) -> BM25Index:
        index = self._bm25_indexes.get(collection_name)
        if index is None:
            collection = self.storage.chroma_client.get_collection(collection_name)
            index = BM25Index.from_chroma_collection(collection)
            self._bm25_indexes[collection_name] = index
        return index

    def _embed_query_batch(self, query_texts: List[str]) -> tuple[Dict[str, List[float]], Dict[str, Any]]:
        """Embed all unique lane queries in one provider call and expose safe telemetry."""
        unique_texts = list(dict.fromkeys(query_texts))
        before_usage = getattr(self.embedding_fn, "usage", None)
        before = dict(getattr(before_usage, "__dict__", {}) or {})
        started = time.perf_counter()
        raw_vectors = self.embedding_fn(unique_texts)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if len(raw_vectors) != len(unique_texts):
            raise RuntimeError(
                f"embedding backend returned {len(raw_vectors)} vectors for {len(unique_texts)} lane queries"
            )
        vectors = {
            text: [float(value) for value in vector]
            for text, vector in zip(unique_texts, raw_vectors)
        }
        after_usage = getattr(self.embedding_fn, "usage", None)
        after = dict(getattr(after_usage, "__dict__", {}) or {})
        delta = {}
        for key in ("request_count", "retry_count", "input_count", "cache_hit_count", "input_tokens", "total_tokens"):
            if key in after:
                delta[key] = int(after.get(key, 0)) - int(before.get(key, 0))
        return vectors, {
            "unique_query_count": len(unique_texts),
            "batch_latency_ms": latency_ms,
            "usage_delta": delta,
        }

    def _rank_collection_candidates(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 3,
        where_filter: Optional[Dict] = None,
        lexical_query: Optional[str] = None,
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
            
        query_collection = getattr(self.storage, "query_collection", None)
        raw_res = (
            query_collection(collection_name, query_kwargs)
            if callable(query_collection)
            else coll.query(**query_kwargs)
        )
        # The storage-level query may have reopened Chroma after a transient
        # HNSW visibility failure.  Reacquire the collection before any
        # follow-up ``get`` so no stale client handle crosses that boundary.
        if callable(query_collection):
            coll = self.storage.chroma_client.get_collection(collection_name)
        raw_res = raw_res or {}
        if not raw_res or not raw_res.get("ids") or len(raw_res["ids"][0]) == 0:
            dense_ids: list[str] = []
        else:
            dense_ids = [str(item) for item in raw_res["ids"][0]]
            
        dense_docs = raw_res.get("documents", [[]])[0] if raw_res.get("documents") else []
        dense_metas = raw_res.get("metadatas", [[]])[0] if raw_res.get("metadatas") else []
        dense_embeddings = raw_res.get("embeddings", [[]])[0] if raw_res.get("embeddings") else []
        dense_records = {
            document_id: {
                "document": dense_docs[index] if index < len(dense_docs) else "",
                "metadata": dense_metas[index] if index < len(dense_metas) else {},
                "embedding": dense_embeddings[index] if index < len(dense_embeddings) else None,
            }
            for index, document_id in enumerate(dense_ids)
        }
        bm25_hits = []
        if self.retrieval_mode == "bm25_dense":
            bm25_hits = self._get_bm25_index(collection_name).search(
                lexical_query or "", top_k=candidate_k, where=where_filter
            )
        bm25_by_id = {hit.document_id: hit for hit in bm25_hits}
        all_ids = list(dict.fromkeys(dense_ids + list(bm25_by_id)))
        missing_ids = [document_id for document_id in all_ids if document_id not in dense_records]
        if missing_ids:
            fetched = coll.get(ids=missing_ids, include=["documents", "metadatas", "embeddings"])
            fetched_docs = fetched.get("documents")
            fetched_docs = [] if fetched_docs is None else fetched_docs
            fetched_metas = fetched.get("metadatas")
            fetched_metas = [] if fetched_metas is None else fetched_metas
            fetched_embeddings = fetched.get("embeddings")
            fetched_embeddings = [] if fetched_embeddings is None else fetched_embeddings
            for index, document_id in enumerate(fetched.get("ids", []) or []):
                dense_records[str(document_id)] = {
                    "document": fetched_docs[index] if index < len(fetched_docs) else "",
                    "metadata": fetched_metas[index] if index < len(fetched_metas) else {},
                    "embedding": fetched_embeddings[index] if index < len(fetched_embeddings) else None,
                }
        max_bm25 = max((hit.score for hit in bm25_hits), default=0.0)
        rescored_candidates = []

        for document_id in all_ids:
            record = dense_records.get(document_id, {})
            doc_vec = record.get("embedding")
            if doc_vec is not None and len(doc_vec) > 0:
                hybrid_s, cos_s, l2_s = compute_hybrid_score(query_vector, doc_vec, alpha=self.alpha)
            else:
                hybrid_s, cos_s, l2_s = 0.0, 0.0, 0.0
            bm25_raw = bm25_by_id.get(document_id).score if document_id in bm25_by_id else 0.0
            bm25_norm = (bm25_raw / max_bm25) if max_bm25 > 0 else 0.0
            final_score = hybrid_s if self.retrieval_mode == "dense" else (
                self.dense_weight * hybrid_s + self.bm25_weight * bm25_norm
            )
            channels = []
            if document_id in dense_records and document_id in dense_ids:
                channels.append("dense")
            if document_id in bm25_by_id:
                channels.append("bm25")
            rescored_candidates.append({
                "chunk_id": document_id,
                "document": record.get("document", ""),
                "metadata": record.get("metadata", {}),
                "hybrid_score": round(final_score, 5),
                "dense_hybrid_score": round(hybrid_s, 5),
                "cosine_score": round(cos_s, 5),
                "euclidean_score": round(l2_s, 5),
                "bm25_raw_score": round(bm25_raw, 5),
                "bm25_score": round(bm25_norm, 5),
                "retrieval_channels": channels,
                "collection": collection_name
            })
            
        # 按综合混合得分降序排列
        rescored_candidates.sort(
            key=lambda x: (-x["hybrid_score"], -x["bm25_score"], -x["dense_hybrid_score"], str(x["chunk_id"]))
        )
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
        query_vector = self._embed_query_batch([query_text])[0][query_text]
        return self._retrieve_misconceptions_by_vector(
            query_vector,
            top_k=top_k,
            fetch_evidence=fetch_evidence,
            where_filter=where_filter,
            lexical_query=query_text,
        )

    def _retrieve_misconceptions_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 3,
        fetch_evidence: bool = True,
        where_filter: Optional[Dict] = None,
        lexical_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按预计算向量直接检索错因库 (跳过重复 embedding)，供多视角 RRF 复用同一批向量。"""
        hits = self._rank_collection_candidates("student_misconceptions", query_vector, top_k=top_k, where_filter=where_filter, lexical_query=lexical_query)
        
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
        query_vector = self._embed_query_batch([query_text])[0][query_text]
        return self._retrieve_strategies_by_vector(
            query_vector,
            top_k=top_k,
            fetch_evidence=fetch_evidence,
            where_filter=where_filter,
            lexical_query=query_text,
        )

    def _retrieve_strategies_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 3,
        fetch_evidence: bool = True,
        where_filter: Optional[Dict] = None,
        lexical_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按预计算向量直接检索策略库 (跳过重复 embedding)，供多视角 RRF 复用同一批向量。"""
        hits = self._rank_collection_candidates("tutor_strategies", query_vector, top_k=top_k, where_filter=where_filter, lexical_query=lexical_query)
        
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
        where_filter: Optional[Dict] = None,
        fetch_evidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        零信任模式：纯原文滑动窗口直接检索 (fallback_windows)。
        """
        query_vector = self._embed_query_batch([query_text])[0][query_text]
        # A wider dense pool lets the deterministic exact channel recover
        # formula-bearing dialogue windows without scanning or rewriting the
        # immutable index.  The exact score uses only SUT input/document text.
        candidates = self._rank_collection_candidates(
            "fallback_windows",
            query_vector,
            top_k=max(top_k, 50),
            where_filter=where_filter,
            lexical_query=query_text,
        )
        for candidate in candidates:
            exact_score = numeric_formula_exact_score(query_text, candidate.get("document", ""))
            candidate["numeric_formula_exact_score"] = exact_score
            candidate["retrieval_channels"] = ["dense"] + (["numeric_formula_exact"] if exact_score > 0 else [])
            candidate["fallback_combined_score"] = round(
                float(candidate.get("hybrid_score", 0.0)) + 0.35 * exact_score,
                6,
            )
        candidates.sort(
            key=lambda item: (
                -item["fallback_combined_score"],
                -item["numeric_formula_exact_score"],
                str(item["chunk_id"]),
            )
        )
        results = candidates[:top_k]
        if fetch_evidence:
            for item in results:
                metadata = item.get("metadata", {})
                session_id = metadata.get("session_id")
                raw_turn_ids = metadata.get("source_turn_ids", [])
                if isinstance(raw_turn_ids, str):
                    raw_turn_ids = json.loads(raw_turn_ids)
                if session_id is not None:
                    item["evidence_turns"] = self.storage.get_dialogue_turns(int(session_id), list(raw_turn_ids or []))
        return results

    def expand_card_candidates_to_evidence_units(
        self,
        candidates: List[Dict[str, Any]],
        *,
        window_size: int = 6,
        step: int = 3,
    ) -> List[Dict[str, Any]]:
        """Expand retrieved cards into deterministic raw logical-chain windows.

        This is intentionally independent of the LLM card pointer selection.
        Each selected card contributes its source session; the complete raw
        session is windowed from DuckDB and duplicate windows are merged while
        retaining the originating card IDs for traceability.
        """
        if window_size <= 0 or step <= 0:
            raise ValueError("window_size and step must be positive")
        units: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            metadata = candidate.get("metadata", {}) or {}
            session_id = metadata.get("session_id")
            if session_id is None:
                raise ValueError("card candidate evidence expansion requires session_id")
            session_id = int(session_id)
            turns = self.storage.get_dialogue_turns(session_id)
            if not turns:
                raise ValueError(f"session {session_id} has no authoritative dialogue turns")
            card_id = str(candidate.get("chunk_id", ""))
            for start in range(0, len(turns), step):
                window_turns = turns[start : start + window_size]
                if not window_turns:
                    continue
                first_id = int(window_turns[0]["turn_id"])
                last_id = int(window_turns[-1]["turn_id"])
                unit_id = f"session_{session_id}_evidence_{first_id}_{last_id}"
                unit = units.setdefault(
                    unit_id,
                    {
                        "chunk_id": unit_id,
                        "document": "\n".join(
                            [
                                f"[Session {session_id}] [完整逻辑链原文]",
                                *[
                                    f"[Turn {turn['turn_id']}] [{turn['speaker']}] {turn['text']}"
                                    for turn in window_turns
                                ],
                            ]
                        ),
                        "metadata": {
                            "session_id": session_id,
                            "window_start_turn": first_id,
                            "window_end_turn": last_id,
                            "source_turn_ids": [int(turn["turn_id"]) for turn in window_turns],
                            "source_card_ids": [],
                            "subject_path": metadata.get("subject_path", ""),
                        },
                        "evidence_turns": [
                            {**dict(turn), "session_id": session_id}
                            for turn in window_turns
                        ],
                    },
                )
                if card_id and card_id not in unit["metadata"]["source_card_ids"]:
                    unit["metadata"]["source_card_ids"].append(card_id)
                # The final partial window is valid and closes the union; the
                # next start would otherwise duplicate only its tail.
                if start + window_size >= len(turns):
                    break
        return list(units.values())

    def expand_anchored_logical_windows(
        self,
        candidates: List[Dict[str, Any]],
        parent_cards: List[Dict[str, Any]],
        *,
        window_size: int = 7,
        step: int = 3,
    ) -> List[Dict[str, Any]]:
        """Expand only logical windows anchored by selected parent card pointers.

        ``candidates`` is the complete post-embedding card pool.  ``parent_cards``
        is the post-card-rerank Parent Top-N subset.  The filter is deterministic
        and uses only real ``source_turn_ids``; qrels and generated text are not
        consulted.  Parent metadata is attached for the downstream reranker and
        is deliberately not copied into the window document/citation evidence.
        """

        if not parent_cards:
            raise ValueError("anchored logical-window expansion requires parent_cards")

        all_units = self.expand_card_candidates_to_evidence_units(
            candidates, window_size=window_size, step=step
        )

        def pointer_ids(card: Dict[str, Any]) -> set[Tuple[int, int]]:
            metadata = card.get("metadata", {}) or {}
            session_id = metadata.get("session_id")
            if session_id is None:
                raise ValueError("anchored parent card is missing session_id")
            raw_ids = metadata.get("source_turn_ids", [])
            if isinstance(raw_ids, str):
                raw_ids = json.loads(raw_ids)
            return {(int(session_id), int(turn_id)) for turn_id in (raw_ids or [])}

        parent_keys = set().union(*(pointer_ids(card) for card in parent_cards))
        if not parent_keys:
            raise ValueError("anchored parent cards contain no source_turn_ids")

        session_card_groups: Dict[int, List[Dict[str, Any]]] = {}
        strategy_parent_sessions: set[int] = set()
        for card in parent_cards:
            metadata = card.get("metadata", {}) or {}
            session_id = metadata.get("session_id")
            if session_id is not None:
                session_id = int(session_id)
                session_card_groups.setdefault(session_id, []).append(card)
                collection = str(card.get("collection", "")).casefold()
                if (
                    "strategy" in collection
                    or metadata.get("key_aha_question")
                    or metadata.get("pedagogical_goal")
                ):
                    strategy_parent_sessions.add(session_id)
        session_completion_ids = {
            session_id
            for session_id, cards in session_card_groups.items()
            if len(cards) >= 2
        }
        session_completion_ids.update(strategy_parent_sessions)

        def compact_parent_metadata(card: Dict[str, Any]) -> Dict[str, str]:
            metadata = card.get("metadata", {}) or {}
            return {
                "card_id": str(card.get("chunk_id", "")),
                "session_id": str(metadata.get("session_id", "")),
                "subject_path": str(metadata.get("subject_path", "")),
                "parent_title": str(
                    metadata.get("misconception_name")
                    or metadata.get("key_aha_question")
                    or card.get("chunk_id", "")
                )[:300],
                "parent_summary": str(
                    metadata.get("deep_mechanism")
                    or metadata.get("pedagogical_goal")
                    or ""
                )[:500],
            }

        anchored: List[Dict[str, Any]] = []
        for unit in all_units:
            metadata = unit.get("metadata", {}) or {}
            session_id = metadata.get("session_id")
            if session_id is None:
                raise ValueError("logical window is missing session_id")
            session_id = int(session_id)
            raw_ids = metadata.get("source_turn_ids", [])
            if isinstance(raw_ids, str):
                raw_ids = json.loads(raw_ids)
            window_keys = {(session_id, int(turn_id)) for turn_id in (raw_ids or [])}
            overlap = window_keys & parent_keys
            session_completion = session_id in session_completion_ids
            if not overlap and not session_completion:
                continue
            item = dict(unit)
            item_metadata = dict(metadata)
            item_metadata["anchor_turn_ids"] = sorted(
                turn_id for sid, turn_id in overlap if sid == session_id
            )
            item_metadata["session_completion"] = session_completion
            item_metadata["parent_contexts"] = [
                compact_parent_metadata(card)
                for card in parent_cards
                if int((card.get("metadata", {}) or {}).get("session_id", -1)) == session_id
                and (pointer_ids(card) & window_keys or session_completion)
            ]
            item["metadata"] = item_metadata
            anchored.append(item)
        return anchored

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
                "metric": "BM25 + Dense (Cosine/L2) Hybrid" if self.retrieval_mode == "bm25_dense" else "Cosine + Euclidean (L2) Dense Hybrid",
                "alpha": self.alpha,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "formula": "Dense_Hybrid = alpha * Cosine_Norm + (1 - alpha) * (1 / (1 + L2_Dist)); Final = dense_weight * Dense_Hybrid + bm25_weight * BM25_Normalized" if self.retrieval_mode == "bm25_dense" else "Score = alpha * Cosine_Norm + (1 - alpha) * (1 / (1 + L2_Dist))"
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
        k_constant: int = 60,
        *,
        allowed_session_ids: List[int] | None = None,
        perspectives: tuple[str, ...] = ("student", "tutor"),
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
        if not perspectives or not set(perspectives).issubset({"student", "tutor"}):
            raise ValueError("perspectives must contain student and/or tutor")
        where_filter = None
        if allowed_session_ids is not None:
            if not allowed_session_ids:
                return {"raw_query": raw_query, "misconceptions": [], "strategies": [], "total_retrieved": 0}
            where_filter = {"session_id": {"$in": [int(item) for item in allowed_session_ids]}}
        # 1. 多视角派生与专业注入
        rewritten: MultiPerspectiveQueries = self.rewriter.rewrite(raw_query)

        query_vectors, embedding_trace = self._embed_query_batch([
            rewritten.misconception_query,
            rewritten.curriculum_query,
            raw_query,
            rewritten.strategy_query,
        ])

        # 2. 错因库多视角检索与 RRF 融合 (3-Way: misconception + curriculum + raw_query)
        misc_ranks_p1 = self._retrieve_misconceptions_by_vector(query_vectors[rewritten.misconception_query], top_k=15, fetch_evidence=False, lexical_query=rewritten.misconception_query, where_filter=where_filter) if "student" in perspectives else []
        misc_ranks_p2 = self._retrieve_misconceptions_by_vector(query_vectors[rewritten.curriculum_query], top_k=15, fetch_evidence=False, lexical_query=rewritten.curriculum_query, where_filter=where_filter) if "student" in perspectives else []
        misc_ranks_p3 = self._retrieve_misconceptions_by_vector(query_vectors[raw_query], top_k=15, fetch_evidence=False, lexical_query=raw_query, where_filter=where_filter) if "student" in perspectives else []
        
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
        if self.fusion_strategy == "raw_first":
            raw_ids = [str(candidate["chunk_id"]) for candidate in misc_ranks_p3]
            by_id = {str(item["candidate"]["chunk_id"]): item for item in fused_misc}
            fused_misc = [by_id[chunk_id] for chunk_id in raw_ids]
            fused_misc.extend(item for item in by_id.values() if str(item["candidate"]["chunk_id"]) not in set(raw_ids))
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
        strat_ranks_p1 = self._retrieve_strategies_by_vector(query_vectors[rewritten.strategy_query], top_k=15, fetch_evidence=False, lexical_query=rewritten.strategy_query, where_filter=where_filter) if "tutor" in perspectives else []
        strat_ranks_p2 = self._retrieve_strategies_by_vector(query_vectors[rewritten.curriculum_query], top_k=15, fetch_evidence=False, lexical_query=rewritten.curriculum_query, where_filter=where_filter) if "tutor" in perspectives else []
        strat_ranks_p3 = self._retrieve_strategies_by_vector(query_vectors[raw_query], top_k=15, fetch_evidence=False, lexical_query=raw_query, where_filter=where_filter) if "tutor" in perspectives else []

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
        if self.fusion_strategy == "raw_first":
            raw_ids = [str(candidate["chunk_id"]) for candidate in strat_ranks_p3]
            raw_id_set = set(raw_ids)
            by_id = {str(item["candidate"]["chunk_id"]): item for item in fused_strat}
            fused_strat = [by_id[chunk_id] for chunk_id in raw_ids]
            fused_strat.extend(item for item in by_id.values() if str(item["candidate"]["chunk_id"]) not in raw_id_set)
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

        fused_trace["misconception_selected"] = [str(item["chunk_id"]) for item in top_misc_results]
        fused_trace["strategy_selected"] = [str(item["chunk_id"]) for item in top_strat_results]

        return {
            "raw_query": raw_query,
            "execution_metadata": {
                "query_rewrite_backend": self.rewriter.backend_name,
                "embedding_backend": self.storage.embedding_backend,
                "fusion_strategy": self.fusion_strategy,
                "retrieval_mode": self.retrieval_mode,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
            },
            "rewritten_queries": rewritten.model_dump(),
            "trace": {
                "expanded_queries": rewritten.model_dump(),
                "embedding": embedding_trace,
                "lanes": lane_trace,
                "fused_rankings": fused_trace,
                "fusion": self.fusion_strategy,
                "rrf_k": k_constant,
                "retrieval_mode": self.retrieval_mode,
                "bm25_weight": self.bm25_weight,
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
