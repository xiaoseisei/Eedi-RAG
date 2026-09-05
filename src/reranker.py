"""
================================================================================
模块名称: src/reranker.py
业务定位: Step 4 - Qwen Cross-Encoder 可选重排 + MMR 压缩与教研多样性黄金装配器
核心职责:
  1. 可选 Qwen3-Reranker-0.6B 对 BM25+Dense 候选进行语义重排；未配置时保持确定性 MMR 回归模式。
  2. 业务置信度与证据加权 (Business Quality & Evidence Boost):
     - 真实 DuckDB 原声证据覆盖 (evidence_turns > 0): 得分 +0.10
     - 破局一问完备度 (key_aha_question 非空): 得分 +0.05
     - 核心数值与题型关键词精确重合: 得分 +0.15
  3. MMR 多样性贪心去重 (MMR Diversity Greedy Selection):
     - MMR(d) = λ * Score_boosted(d) - (1 - λ) * max(Sim(d, S))
     - 平衡因子 λ=0.7，从初筛候选池中提纯出黄金 Top-1 错因卡 + Top-1 策略卡，彻底消除同质化冗余。
  4. 教研四槽位黄金装配 (The 4-Slot Gold Context Assembly):
     - [槽位 1: 学情认知误区诊断] -> 揭示深层思维障碍与错误选项
     - [槽位 2: 名师破局启发策略] -> 提纯核心破局一问与脚手架链
     - [槽位 3: 不可篡改原声实录] -> DuckDB 真实 [Turn N] 证据链
     - [槽位 4: 考纲考点与原题]   -> 题目题干与标准答案
  5. Token 预算压缩与防迷失 (Lost-in-the-Middle Mitigation):
     - 将上下文严格压缩至 2,000 Tokens，并通过独立节点和首尾锚点降低大模型注意力迷失。
================================================================================
"""

import os
import re
import sys
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def estimate_text_tokens(text: str) -> int:
    """Use the project's deterministic tokenizer-independent estimate."""

    return max(1, int(len(str(text or "")) * 0.5 + len(re.findall(r"\w+", str(text or ""))) * 0.5))


class GoldAssembledContext(BaseModel):
    """
    教研黄金上下文装配契约 (Gold Assembled Context Contract)。
    """
    raw_query: str = Field(description="用户原始输入提问")
    prompt_context_markdown: str = Field(description="渲染完成的高密度标准 Markdown 上下文 (可直接注入 LLM System/User Prompt)")
    selected_misconception: Optional[Dict[str, Any]] = Field(default=None, description="经 MMR 提纯出的 Top-1 学情认知误区卡")
    selected_strategy: Optional[Dict[str, Any]] = Field(default=None, description="经 MMR 提纯出的 Top-1 名师启发策略卡")
    selected_windows: List[Dict[str, Any]] = Field(default_factory=list, description="content-first 模式下选定的真实滑动窗口")
    selected_evidence_units: List[Dict[str, Any]] = Field(default_factory=list, description="card evidence-level 模式下按 Reranker 顺序选定的逻辑链单元")
    chunk_strategy: str = Field(default="card", description="card 或 fallback")
    rerank_unit: str = Field(default="card", description="card、logical_evidence 或 anchored_logical_window")
    evidence_selection_count: int = Field(default=0, ge=0, description="evidence-level 模式选定的逻辑链单元数")
    evidence_turns: List[Dict[str, Any]] = Field(default_factory=list, description="不可篡改的 DuckDB 真实师生对白证据列表")
    deepeval_context_nodes: List[str] = Field(
        default_factory=list,
        description="评测用的独立、去重、可单独评分的上下文节点",
    )
    estimated_token_count: int = Field(default=0, description="装配后的预估 Token 消耗数")
    compression_ratio: float = Field(default=0.0, description="相比全量候选的 Token 压缩率")
    budget_violation: bool = False
    truncation_loss: int = 0
    token_count_source: str = "estimated"
    context_budget_tokens: int = Field(default=0, ge=0)
    catalog_token_count: int = Field(default=0, ge=0)
    generator_context_token_count: int = Field(default=0, ge=0)


def compute_text_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    计算两段文本之间的字/词 Jaccard 相似度，用于衡量内容冗余度。
    """
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


class PedagogicalGoldAssembler:
    """
    教研多样性黄金装配器与 MMR 压缩重排引擎。
    """

    def __init__(
        self,
        lambda_diversity: float = 0.7,
        max_prompt_tokens: int = 2000,
        model_reranker: Any | None = None,
        reranker_pool_size: int = 15,
        chunk_strategy: str = "card",
        rerank_unit: str = "card",
        evidence_selection_count: int = 5,
        parent_card_count: int = 3,
    ):
        """
        初始化装配器。
        
        参数:
          lambda_diversity: MMR 平衡参数 (0.0~1.0, 越大越侧重相关性，越小越侧重多样性去重，默认 0.7)
          max_prompt_tokens: 黄金上下文最大 Token 预算上限 (默认 2000)
        """
        self.lambda_param = max(0.0, min(1.0, lambda_diversity))
        self.max_prompt_tokens = max_prompt_tokens
        self.model_reranker = model_reranker
        if chunk_strategy not in {"card", "fallback"}:
            raise ValueError("chunk_strategy must be card or fallback")
        self.chunk_strategy = chunk_strategy
        if rerank_unit not in {"card", "logical_evidence", "anchored_logical_window"}:
            raise ValueError("rerank_unit must be card, logical_evidence, or anchored_logical_window")
        if evidence_selection_count <= 0:
            raise ValueError("evidence_selection_count must be positive")
        self.rerank_unit = rerank_unit
        self.evidence_selection_count = evidence_selection_count
        if parent_card_count <= 0:
            raise ValueError("parent_card_count must be positive")
        self.parent_card_count = parent_card_count
        if reranker_pool_size <= 0:
            raise ValueError("reranker_pool_size must be positive")
        self.reranker_pool_size = reranker_pool_size

    def rerank_candidates(
        self,
        raw_query: str,
        candidates: List[Dict[str, Any]],
        *,
        unit_name: str,
    ) -> List[Dict[str, Any]]:
        """Public model-rerank stage used by multi-stage anchored retrieval.

        Anchored mode needs one model pass for Parent cards and another pass for
        anchored logical windows.  Requiring a configured model prevents a
        deterministic MMR result from being mislabeled as model reranking.
        """

        if self.model_reranker is None:
            raise ValueError(
                f"{unit_name} rerank requires an explicitly configured model_reranker"
            )
        if not candidates:
            return []
        if unit_name == "anchored_logical_window":
            documents: List[str] = []
            for candidate in candidates:
                metadata = candidate.get("metadata", {}) or {}
                lines = ["[Parent metadata: ranking context only]"]
                for parent in metadata.get("parent_contexts", []) or []:
                    lines.extend(
                        [
                            f"[Parent Session {parent.get('session_id', '?')}]",
                            f"[Topic] {parent.get('subject_path', '')}",
                            f"[Parent Card] {parent.get('parent_title', '')}",
                            f"[Parent Summary] {parent.get('parent_summary', '')}",
                        ]
                    )
                lines.extend(["[Logical Window]", str(candidate.get("document", ""))])
                documents.append("\n".join(lines))
            results = self.model_reranker.rerank(raw_query, documents, top_n=len(documents))
            reranked: List[Dict[str, Any]] = []
            for rank, result in enumerate(results, start=1):
                if result.index < 0 or result.index >= len(candidates):
                    raise ValueError(f"reranker returned invalid candidate index {result.index}")
                candidate = dict(candidates[result.index])
                candidate["reranker_score"] = float(result.relevance_score)
                candidate["reranker_rank"] = rank
                candidate["reranker_model"] = self.model_reranker.name()
                candidate["reranking_applied"] = True
                candidate["reranker_unit"] = unit_name
                reranked.append(candidate)
            if len(reranked) != len(candidates):
                raise ValueError("reranker returned an incomplete candidate permutation")
            return reranked
        return self._apply_model_reranker(raw_query, candidates)

    def _apply_model_reranker(
        self,
        raw_query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply an explicitly configured cross-encoder reranker to candidates.

        The model stage is separate from MMR assembly. If configured, provider
        errors are propagated; silently falling back would make the audit claim
        a model rerank that never happened.
        """
        if self.model_reranker is None or not candidates:
            return candidates
        documents = []
        for candidate in candidates:
            document = str(candidate.get("document", ""))
            evidence = candidate.get("evidence_turns", []) or []
            evidence_text = "\n".join(
                f"[Turn {turn.get('turn_id')}] [{turn.get('speaker')}] {turn.get('text', '')}"
                for turn in evidence
            )
            documents.append(f"{document}\n【直接证据】\n{evidence_text}" if evidence_text else document)
        results = self.model_reranker.rerank(raw_query, documents, top_n=len(documents))
        reranked: List[Dict[str, Any]] = []
        for rank, result in enumerate(results, start=1):
            candidate = dict(candidates[result.index])
            candidate["reranker_score"] = float(result.relevance_score)
            candidate["reranker_rank"] = rank
            candidate["reranker_model"] = self.model_reranker.name()
            candidate["reranking_applied"] = True
            reranked.append(candidate)
        if len(reranked) != len(candidates):
            raise ValueError("reranker returned an incomplete candidate permutation")
        return reranked

    def _calculate_boosted_score(
        self,
        candidate: Dict[str, Any],
        raw_query: str,
        keywords: List[str]
    ) -> float:
        """
        计算业务置信度与证据加权后的综合相关性得分。
        """
        if candidate.get("reranker_score") is not None:
            base_score = float(candidate["reranker_score"])
            norm_base = max(0.0, min(1.0, base_score))
        else:
            base_score = candidate.get("rrf_score") or candidate.get("hybrid_score", 0.5)
            # 归一化 RRF 分数至 0~1 区间 (RRF 理论最大值约 0.033)
            if candidate.get("rrf_score") is not None:
                norm_base = min(1.0, base_score * 30.0)
            else:
                norm_base = base_score

        bonus = 0.0
        meta = candidate.get("metadata", {})
        doc_text = candidate.get("document", "")

        # 1. 真实对白证据加分 (+0.10)
        evidence_turns = candidate.get("evidence_turns", [])
        if evidence_turns and len(evidence_turns) > 0:
            bonus += 0.10

        # 2. 名师核心破局一问加分 (+0.05)
        aha_q = meta.get("key_aha_question", "")
        if aha_q and len(aha_q.strip()) > 5:
            bonus += 0.05

        # 3. 核心关键词与数值覆盖加分 (+0.15)
        if keywords:
            matched_kws = [kw for kw in keywords if kw.lower() in doc_text.lower()]
            bonus += 0.15 * (len(matched_kws) / len(keywords))

        # Preserve a strong raw-query signal when RRF scores saturate after
        # normalization.  This is derived solely from the retriever trace and
        # prevents a lower-ranked card with incidental glossary overlap from
        # displacing the direct query match.
        for hit in candidate.get("perspective_hits", []):
            match = re.search(r"raw_query_perspective\(#Rank(\d+)\)", str(hit))
            if match:
                bonus += 0.50 / int(match.group(1))
                break

        return round(norm_base + bonus, 5)

    def _select_mmr_best(
        self,
        candidates: List[Dict[str, Any]],
        raw_query: str,
        keywords: List[str],
        already_selected_texts: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        使用 MMR 贪心算法挑选 1 张最佳非冗余卡片。
        """
        if not candidates:
            return None

        best_cand = None
        best_mmr_score = -float("inf")

        for cand in candidates:
            rel_score = self._calculate_boosted_score(cand, raw_query, keywords)
            doc_text = cand.get("document", "")

            # 计算与已选集合的最大相似度 (Redundancy Penalty)
            if already_selected_texts:
                max_sim = max(compute_text_jaccard_similarity(doc_text, s_text) for s_text in already_selected_texts)
            else:
                max_sim = 0.0

            mmr_score = self.lambda_param * rel_score - (1.0 - self.lambda_param) * max_sim
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_cand = cand

        return best_cand

    @staticmethod
    def _ordered_unit_evidence(selected_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate Turn facts while preserving logical-unit/reranker order."""
        evidence: List[Dict[str, Any]] = []
        seen: Set[Tuple[int, int]] = set()
        for unit in selected_units:
            metadata = unit.get("metadata", {}) or {}
            session_id = metadata.get("session_id")
            if session_id is None:
                raise ValueError("logical evidence unit missing session_id")
            session_id = int(session_id)
            for turn in unit.get("evidence_turns", []) or []:
                turn_id = int(turn["turn_id"])
                key = (session_id, turn_id)
                if key in seen:
                    continue
                item = dict(turn)
                item["session_id"] = session_id
                evidence.append(item)
                seen.add(key)
        return evidence

    @staticmethod
    def _window_context_prefix(document: str) -> str:
        """Keep non-Turn window headers while removing repeated dialogue lines."""

        lines = []
        for line in str(document or "").splitlines():
            if re.match(r"^\s*\[Turn\s+\d+\]", line):
                continue
            if line.strip():
                lines.append(line)
        return "\n".join(lines).strip()

    @classmethod
    def _unique_window_payload(
        cls,
        units: List[Dict[str, Any]],
    ) -> List[Tuple[int, Dict[str, Any], str, List[Dict[str, Any]]]]:
        """Return rank, unit, one-time header, and newly introduced Turns.

        Window documents are overlapping by design.  Packing uses authoritative
        ``evidence_turns`` as the deduplication key and never infers or creates
        Turn IDs from generated text.  A session header is emitted once, while
        each window retains its original reranker rank in the caller's heading.
        """

        seen_turns: Set[Tuple[int, int]] = set()
        seen_sessions: Set[int] = set()
        payload: List[Tuple[int, Dict[str, Any], str, List[Dict[str, Any]]]] = []
        for rank, unit in enumerate(units, start=1):
            metadata = unit.get("metadata", {}) or {}
            raw_session_id = metadata.get("session_id")
            if raw_session_id is None:
                raise ValueError("logical evidence unit missing session_id")
            session_id = int(raw_session_id)
            new_turns: List[Dict[str, Any]] = []
            for turn in unit.get("evidence_turns", []) or []:
                if turn.get("turn_id") is None:
                    raise ValueError("logical evidence Turn is missing turn_id")
                key = (session_id, int(turn["turn_id"]))
                if key in seen_turns:
                    continue
                seen_turns.add(key)
                item = dict(turn)
                item["session_id"] = session_id
                new_turns.append(item)
            prefix = ""
            if session_id not in seen_sessions:
                prefix = cls._window_context_prefix(str(unit.get("document", "")))
                seen_sessions.add(session_id)
            payload.append((rank, unit, prefix, new_turns))
        return payload

    @classmethod
    def _render_unique_window_block(
        cls,
        units: List[Dict[str, Any]],
        *,
        anchored: bool,
    ) -> List[str]:
        """Render selected windows with overlap-aware, provenance-safe text."""

        return "\n\n".join(
            cls._render_unique_window_nodes(units, anchored=anchored)
        ).splitlines()

    @classmethod
    def _render_unique_window_nodes(
        cls,
        units: List[Dict[str, Any]],
        *,
        anchored: bool,
        evidence_ids: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> List[str]:
        """Render each selected window as an independent evaluation node."""

        lines: List[str] = []
        nodes: List[str] = []
        for rank, unit, prefix, new_turns in cls._unique_window_payload(units):
            metadata = unit.get("metadata", {}) or {}
            label = "Anchored Reranker" if anchored else "Reranker"
            node_lines = [
                f"### {rank}. Session #{metadata.get('session_id', 'N/A')} · "
                f"Turn {metadata.get('window_start_turn', '?')}~{metadata.get('window_end_turn', '?')}"
            ]
            if prefix:
                node_lines.append(prefix)
            if new_turns:
                node_lines.extend(
                    (
                        f"[{evidence_ids.get((int(turn['session_id']), int(turn['turn_id'])))}] "
                        if evidence_ids and (int(turn['session_id']), int(turn['turn_id'])) in evidence_ids
                        else ""
                    )
                    + f"[Turn {turn['turn_id']}] [{turn.get('speaker', '')}] {turn.get('text', '')}"
                    for turn in new_turns
                )
            else:
                node_lines.append("[Window dialogue overlaps already selected Turns; omitted]")
            nodes.append("\n".join(node_lines))
        return nodes

    @staticmethod
    def _metadata_turn_ids(card: Dict[str, Any]) -> List[int]:
        metadata = card.get("metadata", {}) or {}
        raw_ids = metadata.get("source_turn_ids", [])
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except json.JSONDecodeError:
                raw_ids = []
        if not raw_ids:
            raw_ids = [
                turn.get("turn_id")
                for turn in card.get("evidence_turns", []) or []
                if turn.get("turn_id") is not None
            ]
        result: List[int] = []
        for value in raw_ids or []:
            try:
                turn_id = int(value)
            except (TypeError, ValueError):
                continue
            if turn_id > 0 and turn_id not in result:
                result.append(turn_id)
        return sorted(result)

    @staticmethod
    def _display_metadata_value(value: Any, max_chars: int) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith(("[", "{")):
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    pass
            else:
                value = text
        if isinstance(value, (list, tuple, set)):
            value = "；".join(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        text = str(value).strip()
        if len(text) <= max_chars:
            return text
        return text[: max(1, max_chars - 1)].rstrip() + "…"

    @classmethod
    def _render_parent_fact_nodes(
        cls,
        parent_cards: List[Dict[str, Any]],
    ) -> List[str]:
        """Render source-linked Parent Card abstractions as non-citation nodes."""

        nodes: List[str] = []
        seen_card_ids: Set[str] = set()
        for card in parent_cards:
            metadata = card.get("metadata", {}) or {}
            card_id = str(card.get("chunk_id", "")).strip()
            if not card_id or card_id in seen_card_ids:
                continue
            source_turn_ids = cls._metadata_turn_ids(card)
            if not source_turn_ids:
                continue
            seen_card_ids.add(card_id)
            collection = str(card.get("collection", "")).casefold()
            is_strategy = (
                "strategy" in collection
                or bool(metadata.get("key_aha_question"))
                or bool(metadata.get("pedagogical_goal"))
            )
            card_type = "tutor_strategy" if is_strategy else "student_misconception"
            lines = [
                f"## [DERIVED_FACT] Parent Card {card_id}",
                "Stored card abstraction grounded by the listed source Turns; not verbatim dialogue.",
                f"source_card_id={card_id}",
                f"source_session_id={metadata.get('session_id', '')}",
                f"source_turn_ids={source_turn_ids}",
                f"card_type={card_type}",
            ]
            fields = (
                (
                    ("misconception_name", "misconception", 100),
                    ("deep_mechanism", "deep_mechanism", 260),
                    ("confusion_triggers", "confusion_triggers", 100),
                    ("error_choice", "error_choice", 20),
                )
                if not is_strategy
                else (
                    ("strategy_category", "strategy_category", 80),
                    ("pedagogical_goal", "pedagogical_goal", 180),
                    ("key_aha_question", "key_aha_question", 140),
                    ("resolution_outcome", "resolution_outcome", 120),
                )
            )
            for source_key, display_key, max_chars in fields:
                value = cls._display_metadata_value(metadata.get(source_key), max_chars)
                if value:
                    lines.append(f"{display_key}={value}")
            nodes.append("\n".join(lines))
        return nodes

    @staticmethod
    def _render_compact_evidence_catalog(
        evidence: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[Tuple[int, int], str]]:
        """Build the compact v2 catalog used in the final Generator Context."""

        catalog_lines = ["## Evidence catalog (cite IDs only)"]
        evidence_ids: Dict[Tuple[int, int], str] = {}
        for index, turn in enumerate(evidence, start=1):
            key = (int(turn["session_id"]), int(turn["turn_id"]))
            evidence_id = f"E{index:03d}"
            evidence_ids[key] = evidence_id
        catalog_lines.append("Allowed IDs: " + ", ".join(evidence_ids.values()))
        if not evidence_ids:
            raise ValueError("final evidence catalog cannot be empty")
        return "\n".join(catalog_lines), evidence_ids

    def _assemble_card_logical_evidence(
        self,
        raw_query: str,
        retrieval_results: Dict[str, Any],
    ) -> GoldAssembledContext:
        """Use reranked raw logical chains as the final card-route context."""
        misc_candidates = retrieval_results.get("misconceptions", []) or []
        strat_candidates = retrieval_results.get("strategies", []) or []
        rewritten_meta = retrieval_results.get("rewritten_queries", {}) or {}
        keywords = rewritten_meta.get("extracted_keywords", [])

        # Card semantics remain available as anchors, but cards are not the
        # evidence ranking unit in this mode.
        selected_misc = self._select_mmr_best(misc_candidates, raw_query, keywords, [])
        selected_strat = self._select_mmr_best(strat_candidates, raw_query, keywords, [])
        evidence_units = retrieval_results.get("evidence_units", []) or []
        if not evidence_units:
            raise ValueError(
                "logical_evidence rerank requires retrieval_results['evidence_units']; "
                "refusing to silently fall back to card evidence order"
            )
        ranked_units = self._apply_model_reranker(raw_query, evidence_units)
        def render(units: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], int]:
            ordered = self._ordered_unit_evidence(units)
            lines = [
                "# 【权威教研参考知识基座 (Logical-evidence Reranked Context)】",
                "> 卡片仅提供语义锚点；最终对白按逻辑链 Reranker 顺序进入上下文。",
                "",
                "## 一、 卡片语义锚点",
            ]
            for label, card in (("学生错因卡", selected_misc), ("名师策略卡", selected_strat)):
                if not card:
                    lines.append(f"- *(未检索到{label})*")
                    continue
                meta = card.get("metadata", {}) or {}
                lines.append(
                    f"- **{label}** Session #{meta.get('session_id', 'N/A')}："
                    f"{meta.get('misconception_name') or meta.get('key_aha_question') or card.get('chunk_id', '')}"
                )
                # Keep only compact semantic anchors here.  The full evidence
                # text, not the card document, is the reranked context unit.
                if meta.get("deep_mechanism"):
                    lines.append(f"  - 错因机理：{meta['deep_mechanism']}")
                if meta.get("pedagogical_goal"):
                    lines.append(f"  - 教学目标：{meta['pedagogical_goal']}")
                if meta.get("key_aha_question"):
                    lines.append(f"  - 破局问题：{meta['key_aha_question']}")
            lines.extend(["", "## 二、 Reranker 排序后的真实逻辑链证据"])
            lines.extend(self._render_unique_window_block(units, anchored=False))
            # Window documents already contain the verbatim [Turn N] text.
            # Keep an authorization marker without duplicating every Turn in
            # the prompt; duplication would consume the budget twice.
            lines.extend([
                "",
                "## 三、 可引用的真实 Turn 证据",
                "- 仅允许引用上方逻辑链窗口中逐字出现的 `[Turn N]` 原文。",
            ])
            prompt = "\n".join(lines)
            return prompt, ordered, int(len(prompt) * 0.5 + len(re.findall(r"\w+", prompt)) * 0.5)

        # The configured count is an upper bound.  Preserve Reranker order,
        # but stop before adding a unit that would exceed the final budget.
        selected_units: List[Dict[str, Any]] = []
        prompt_markdown, ordered_evidence, estimated_tokens = render(selected_units)
        for unit in ranked_units[: self.evidence_selection_count]:
            trial_prompt, trial_evidence, trial_tokens = render(selected_units + [unit])
            if selected_units and trial_tokens > self.max_prompt_tokens:
                break
            selected_units.append(unit)
            prompt_markdown, ordered_evidence, estimated_tokens = trial_prompt, trial_evidence, trial_tokens

        untruncated_chars = len(prompt_markdown)
        budget_violation = estimated_tokens > self.max_prompt_tokens
        truncation_loss = 0
        if budget_violation:
            target_chars = max(1, self.max_prompt_tokens * 2)
            truncation_loss = max(0, len(prompt_markdown) - target_chars)
            prompt_markdown = prompt_markdown[:target_chars].rstrip() + "\n[Context truncated to token budget]"
            estimated_tokens = min(
                self.max_prompt_tokens,
                int(len(prompt_markdown) * 0.5 + len(re.findall(r"\w+", prompt_markdown)) * 0.5),
            )
        raw_chars = sum(len(str(unit.get("document", ""))) for unit in ranked_units)
        compression_ratio = max(0.0, round(1.0 - untruncated_chars / max(1, raw_chars), 2))
        return GoldAssembledContext(
            raw_query=raw_query,
            prompt_context_markdown=prompt_markdown,
            selected_misconception=selected_misc,
            selected_strategy=selected_strat,
            selected_windows=[],
            selected_evidence_units=selected_units,
            chunk_strategy="card",
            rerank_unit="logical_evidence",
            evidence_selection_count=len(selected_units),
            evidence_turns=ordered_evidence,
            estimated_token_count=estimated_tokens,
            compression_ratio=compression_ratio,
            budget_violation=budget_violation,
            truncation_loss=truncation_loss,
        )

    def _assemble_anchored_logical_windows(
        self,
        raw_query: str,
        retrieval_results: Dict[str, Any],
    ) -> GoldAssembledContext:
        """Assemble windows that were already card- and window-reranked.

        The pipeline performs the two model passes before calling the
        Assembler.  This method intentionally does not rerank again: it only
        preserves the supplied window rank, deduplicates authoritative Turns,
        and enforces the final token budget.
        """

        misc_candidates = retrieval_results.get("misconceptions", []) or []
        strat_candidates = retrieval_results.get("strategies", []) or []
        rewritten_meta = retrieval_results.get("rewritten_queries", {}) or {}
        keywords = rewritten_meta.get("extracted_keywords", [])
        selected_misc = self._select_mmr_best(misc_candidates, raw_query, keywords, [])
        selected_strat = self._select_mmr_best(strat_candidates, raw_query, keywords, [])
        evidence_units = retrieval_results.get("evidence_units", []) or []
        if not evidence_units:
            raise ValueError(
                "anchored logical-window rerank requires retrieval_results['evidence_units']"
            )
        if any(unit.get("reranker_rank") is None for unit in evidence_units):
            raise ValueError(
                "anchored logical-window Assembler requires pre-ranked evidence units"
            )
        ranked_units = sorted(
            (dict(unit) for unit in evidence_units),
            key=lambda unit: (
                int(unit.get("reranker_rank", 10**9)),
                -float(unit.get("reranker_score", 0.0)),
                str(unit.get("chunk_id", "")),
            ),
        )

        # Preserve coverage of the selected Parent card pointers before using
        # the remaining rank budget.  This is deterministic and uses only the
        # parent cards already selected by the production pipeline; it does not
        # consult evaluation qrels or infer missing evidence.
        parent_cards = retrieval_results.get("anchored_parent_cards") or []
        if not parent_cards:
            parent_cards = misc_candidates + strat_candidates

        parent_cards_by_session: Dict[int, List[Dict[str, Any]]] = {}
        parent_card_ids = {
            str(card.get("chunk_id", "")).strip()
            for card in parent_cards
            if str(card.get("chunk_id", "")).strip()
        }
        preferred_parent_sessions = {
            int((card.get("metadata", {}) or {}).get("session_id", -1))
            for card in (selected_misc, selected_strat)
            if card
            and str(card.get("chunk_id", "")).strip() in parent_card_ids
            and (card.get("metadata", {}) or {}).get("session_id") is not None
        }
        for card in parent_cards:
            metadata = card.get("metadata", {}) or {}
            session_id = metadata.get("session_id")
            if session_id is not None:
                parent_cards_by_session.setdefault(int(session_id), []).append(card)

        def card_kind(card: Dict[str, Any]) -> str:
            metadata = card.get("metadata", {}) or {}
            collection = str(card.get("collection", "")).casefold()
            if (
                "strategy" in collection
                or metadata.get("key_aha_question")
                or metadata.get("pedagogical_goal")
            ):
                return "strategy"
            return "misconception"

        session_kind_counts = {
            session_id: {card_kind(card) for card in cards}
            for session_id, cards in parent_cards_by_session.items()
        }
        query_value = str(raw_query).casefold()
        asks_student = any(term in query_value for term in ("学生", "student", "learner", "错因", "误区"))
        asks_tutor = any(term in query_value for term in ("导师", "老师", "名师", "tutor", "teacher", "引导", "策略", "教法"))
        if asks_tutor and not asks_student:
            strategy_sessions = {
                session_id
                for session_id, cards in parent_cards_by_session.items()
                if any(
                    "strategy" in str(card.get("collection", "")).casefold()
                    or (card.get("metadata", {}) or {}).get("key_aha_question")
                    or (card.get("metadata", {}) or {}).get("pedagogical_goal")
                    for card in cards
                )
            }
            selected_strategy_session = (
                int((selected_strat.get("metadata", {}) or {}).get("session_id"))
                if selected_strat
                and (selected_strat.get("metadata", {}) or {}).get("session_id") is not None
                and str(selected_strat.get("chunk_id", "")).strip() in parent_card_ids
                else None
            )
            focused_parent_sessions = (
                {selected_strategy_session}
                if selected_strategy_session is not None
                else strategy_sessions or preferred_parent_sessions
            )
        elif asks_student and not asks_tutor:
            misconception_sessions = {
                session_id
                for session_id, cards in parent_cards_by_session.items()
                if any(
                    "misconception" in str(card.get("collection", "")).casefold()
                    or (card.get("metadata", {}) or {}).get("misconception_name")
                    or (card.get("metadata", {}) or {}).get("deep_mechanism")
                    for card in cards
                )
            }
            selected_misc_session = (
                int((selected_misc.get("metadata", {}) or {}).get("session_id"))
                if selected_misc
                and (selected_misc.get("metadata", {}) or {}).get("session_id") is not None
                and str(selected_misc.get("chunk_id", "")).strip() in parent_card_ids
                else None
            )
            focused_parent_sessions = (
                {selected_misc_session}
                if selected_misc_session is not None
                else misconception_sessions or preferred_parent_sessions
            )
        elif asks_student and asks_tutor:
            max_kind_count = max(
                (len(kinds) for kinds in session_kind_counts.values()),
                default=0,
            )
            paired_sessions = {
                session_id
                for session_id, kinds in session_kind_counts.items()
                if len(kinds) == max_kind_count and max_kind_count >= 2
            }
            focused_parent_sessions = paired_sessions or preferred_parent_sessions
        elif preferred_parent_sessions:
            focused_parent_sessions = set(preferred_parent_sessions)
        else:
            max_cards_per_session = max(
                (len(cards) for cards in parent_cards_by_session.values()),
                default=0,
            )
            focused_parent_sessions = {
                session_id
                for session_id, cards in parent_cards_by_session.items()
                if len(cards) == max_cards_per_session
            }
        coverage_parent_cards = [
            card
            for card in parent_cards
            if int((card.get("metadata", {}) or {}).get("session_id", -1))
            in focused_parent_sessions
        ]
        if not coverage_parent_cards:
            coverage_parent_cards = list(parent_cards)

        def _source_keys(card_or_unit: Dict[str, Any]) -> set[tuple[int, int]]:
            metadata = card_or_unit.get("metadata", {}) or {}
            session_id = metadata.get("session_id")
            raw_ids = metadata.get("source_turn_ids", [])
            if isinstance(raw_ids, str):
                try:
                    raw_ids = json.loads(raw_ids)
                except json.JSONDecodeError:
                    raw_ids = []
            if session_id is None:
                evidence_turns = card_or_unit.get("evidence_turns", []) or []
                return {
                    (int(turn.get("session_id", -1)), int(turn["turn_id"]))
                    for turn in evidence_turns
                    if turn.get("session_id") is not None and turn.get("turn_id") is not None
                }
            if not raw_ids:
                evidence_turns = card_or_unit.get("evidence_turns", []) or []
                raw_ids = [turn["turn_id"] for turn in evidence_turns if turn.get("turn_id") is not None]
            return {(int(session_id), int(turn_id)) for turn_id in (raw_ids or [])}

        parent_pointer_keys = (
            set().union(*(_source_keys(card) for card in coverage_parent_cards))
            if coverage_parent_cards
            else set()
        )
        for unit in ranked_units:
            unit_keys = _source_keys(unit)
            unit["anchor_coverage_count"] = len(unit_keys & parent_pointer_keys)
        semantic_parent_cards = list(coverage_parent_cards)
        parent_fact_nodes = self._render_parent_fact_nodes(semantic_parent_cards)

        def render(
            units: List[Dict[str, Any]],
        ) -> Tuple[str, List[Dict[str, Any]], int, List[str], int]:
            ordered_units = sorted(
                units,
                key=lambda unit: (
                    int(unit.get("reranker_rank", 10**9)),
                    -float(unit.get("reranker_score", 0.0)),
                    str(unit.get("chunk_id", "")),
                ),
            )
            ordered = self._ordered_unit_evidence(ordered_units)
            if ordered:
                catalog_text, evidence_ids = self._render_compact_evidence_catalog(ordered)
            else:
                catalog_text, evidence_ids = "", {}
            window_nodes = self._render_unique_window_nodes(
                ordered_units,
                anchored=True,
                evidence_ids=evidence_ids,
            )
            context_nodes = [*parent_fact_nodes, *window_nodes]
            lines = [
                "# 【权威教研参考知识基座 (Anchored Logical-Window Context)】",
                "> Parent Card 提供有来源的派生事实；真实 Turn 仍是唯一可引用的原始证据。",
            ]
            if parent_fact_nodes:
                lines.extend(["", "## 一、 Parent Card 派生事实", *parent_fact_nodes])
            lines.extend(["", "## 二、 Anchored Reranker 排序后的真实逻辑链证据", *window_nodes])
            lines.extend(
                [
                    "",
                    "## 三、 可引用的真实 Turn 证据",
                    "- 仅允许引用上方逻辑链窗口中逐字出现的 `[Turn N]` 原文。",
                ]
            )
            prompt = "\n\n".join(lines)
            final_context = "\n\n".join(part for part in (prompt, catalog_text) if part)
            estimated = estimate_text_tokens(final_context)
            return prompt, ordered, estimated, context_nodes, estimate_text_tokens(catalog_text) if catalog_text else 0

        remaining_units = list(ranked_units)
        if focused_parent_sessions:
            focused_units = [
                unit
                for unit in remaining_units
                if int((unit.get("metadata", {}) or {}).get("session_id", -1))
                in focused_parent_sessions
            ]
            if focused_units:
                remaining_units = focused_units
        parent_card_keys = {
            str(card.get("chunk_id", "")): _source_keys(card)
            for card in coverage_parent_cards
            if str(card.get("chunk_id", "")).strip()
        }

        def parent_card_ids_for_unit(unit: Dict[str, Any]) -> Set[str]:
            metadata = unit.get("metadata", {}) or {}
            contexts = metadata.get("parent_contexts", []) or []
            context_ids = {
                str(item.get("card_id", "")).strip()
                for item in contexts
                if str(item.get("card_id", "")).strip()
            }
            if context_ids:
                return context_ids & set(parent_card_keys)
            unit_keys = _source_keys(unit)
            return {
                card_id
                for card_id, card_keys in parent_card_keys.items()
                if unit_keys & card_keys
            }

        def unit_identity(unit: Dict[str, Any]) -> str:
            metadata = unit.get("metadata", {}) or {}
            session_id = int(metadata.get("session_id", -1))
            source_ids = _source_keys(unit)
            start = metadata.get("window_start_turn")
            end = metadata.get("window_end_turn")
            if start is None and source_ids:
                start = min(turn_id for _, turn_id in source_ids)
            if end is None and source_ids:
                end = max(turn_id for _, turn_id in source_ids)
            return f"{unit.get('chunk_id', '')}|{session_id}|{start}|{end}"

        def unit_range(unit: Dict[str, Any]) -> Tuple[int, int, int]:
            metadata = unit.get("metadata", {}) or {}
            session_id = int(metadata.get("session_id", -1))
            source_ids = _source_keys(unit)
            if not source_ids:
                return session_id, 10**9, -1
            start = int(metadata.get("window_start_turn", min(turn_id for _, turn_id in source_ids)))
            end = int(metadata.get("window_end_turn", max(turn_id for _, turn_id in source_ids)))
            return session_id, start, end

        endpoint_markers_by_unit: Dict[str, Set[Tuple[int, str]]] = {}
        by_session: Dict[int, List[Dict[str, Any]]] = {}
        for unit in remaining_units:
            session_id, _, _ = unit_range(unit)
            by_session.setdefault(session_id, []).append(unit)
        for session_id, session_units in by_session.items():
            first = min(session_units, key=lambda unit: (unit_range(unit)[1], int(unit.get("reranker_rank", 10**9))))
            last = max(session_units, key=lambda unit: (unit_range(unit)[2], -int(unit.get("reranker_rank", 10**9))))
            endpoint_markers_by_unit.setdefault(unit_identity(first), set()).add((session_id, "start"))
            endpoint_markers_by_unit.setdefault(unit_identity(last), set()).add((session_id, "end"))
        endpoint_target = set().union(*endpoint_markers_by_unit.values())

        def temporal_coverage_score(turn_keys: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
            """Prefer broader, less fragmented real dialogue coverage."""

            by_session: Dict[int, Set[int]] = {}
            session_bounds: Dict[int, Tuple[int, int]] = {}
            for unit in remaining_units:
                session_id, start, end = unit_range(unit)
                session_bounds[session_id] = (
                    min(start, session_bounds.get(session_id, (start, end))[0]),
                    max(end, session_bounds.get(session_id, (start, end))[1]),
                )
            for session_id, turn_id in turn_keys:
                by_session.setdefault(session_id, set()).add(turn_id)

            longest_run = 0
            prefix_run = 0
            suffix_run = 0
            for session_id, covered_ids in by_session.items():
                ordered_ids = sorted(covered_ids)
                current_run = 0
                session_longest = 0
                previous_id: int | None = None
                for turn_id in ordered_ids:
                    current_run = current_run + 1 if previous_id == turn_id - 1 else 1
                    session_longest = max(session_longest, current_run)
                    previous_id = turn_id
                longest_run += session_longest
                lower_bound, upper_bound = session_bounds.get(
                    session_id,
                    (ordered_ids[0], ordered_ids[-1]),
                )
                session_prefix = 0
                while lower_bound + session_prefix <= upper_bound and lower_bound + session_prefix in covered_ids:
                    session_prefix += 1
                prefix_run += session_prefix
                suffix_length = 0
                while upper_bound - suffix_length >= lower_bound and upper_bound - suffix_length in covered_ids:
                    suffix_length += 1
                suffix_run += suffix_length
            return len(turn_keys), longest_run, prefix_run, suffix_run

        candidate_pool = remaining_units[: max(20, self.evidence_selection_count * 4)]
        max_selection = min(self.evidence_selection_count, len(candidate_pool))
        best: Tuple[Tuple[Any, ...], List[Dict[str, Any]], str, List[str], int, int] | None = None
        best_with_endpoints: Tuple[Tuple[Any, ...], List[Dict[str, Any]], str, List[str], int, int] | None = None
        for size in range(1, max_selection + 1):
            for indexes in combinations(range(len(candidate_pool)), size):
                units = [candidate_pool[index] for index in indexes]
                trial_prompt, trial_evidence, trial_tokens, trial_nodes, trial_catalog_tokens = render(units)
                if trial_tokens > self.max_prompt_tokens:
                    continue
                keys = {
                    (int(item["session_id"]), int(item["turn_id"]))
                    for item in trial_evidence
                }
                covered_parent = len(keys & parent_pointer_keys)
                card_ids = set().union(*(parent_card_ids_for_unit(unit) for unit in units))
                roles = {
                    str(item.get("speaker", "")).strip()
                    for item in trial_evidence
                    if str(item.get("speaker", "")).strip()
                }
                covered_endpoints = set().union(
                    *(endpoint_markers_by_unit.get(unit_identity(unit), set()) for unit in units)
                )
                sessions = {unit_range(unit)[0] for unit in units}
                score = (
                    covered_parent,
                    len(card_ids),
                    len(keys),
                    len(covered_endpoints),
                    len(roles),
                    -len(sessions),
                    -sum(int(unit.get("reranker_rank", 10**9)) for unit in units),
                    tuple(-int(unit.get("reranker_rank", 10**9)) for unit in units),
                )
                if best is None or score > best[0]:
                    best = (score, units, trial_prompt, trial_nodes, trial_tokens, trial_catalog_tokens)
                if endpoint_target and endpoint_target <= covered_endpoints:
                    temporal_score = temporal_coverage_score(keys)
                    endpoint_score = (
                        temporal_score[0],
                        temporal_score[2],
                        temporal_score[3],
                        temporal_score[1],
                        covered_parent,
                        len(card_ids),
                        len(covered_endpoints),
                        len(roles),
                        -len(sessions),
                        -sum(int(unit.get("reranker_rank", 10**9)) for unit in units),
                        tuple(-int(unit.get("reranker_rank", 10**9)) for unit in units),
                    )
                    if best_with_endpoints is None or endpoint_score > best_with_endpoints[0]:
                        best_with_endpoints = (
                            endpoint_score,
                            units,
                            trial_prompt,
                            trial_nodes,
                            trial_tokens,
                            trial_catalog_tokens,
                        )
        if best is None:
            raise ValueError("anchored logical-window assembly cannot fit any candidate within the token budget")
        chosen = best_with_endpoints or best
        _, selected_units, prompt_markdown, context_nodes, estimated_tokens, catalog_tokens = chosen
        prompt_markdown, ordered_evidence, estimated_tokens, context_nodes, catalog_tokens = render(selected_units)
        selected_units.sort(
            key=lambda unit: (
                int(unit.get("reranker_rank", 10**9)),
                -float(unit.get("reranker_score", 0.0)),
                str(unit.get("chunk_id", "")),
            )
        )
        prompt_markdown, ordered_evidence, estimated_tokens, context_nodes, catalog_tokens = render(selected_units)
        budget_violation = estimated_tokens > self.max_prompt_tokens
        truncation_loss = 0
        if budget_violation:
            raise ValueError("anchored logical-window assembly exceeded the token budget")
        raw_chars = sum(len(str(unit.get("document", ""))) for unit in ranked_units)
        compression_ratio = max(0.0, round(1.0 - len(prompt_markdown) / max(1, raw_chars), 2))
        return GoldAssembledContext(
            raw_query=raw_query,
            prompt_context_markdown=prompt_markdown,
            selected_misconception=selected_misc,
            selected_strategy=selected_strat,
            selected_windows=[],
            selected_evidence_units=selected_units,
            chunk_strategy="card",
            rerank_unit="anchored_logical_window",
            evidence_selection_count=len(selected_units),
            evidence_turns=ordered_evidence,
            deepeval_context_nodes=context_nodes,
            estimated_token_count=estimated_tokens,
            compression_ratio=compression_ratio,
            budget_violation=budget_violation,
            truncation_loss=truncation_loss,
            context_budget_tokens=self.max_prompt_tokens,
            catalog_token_count=catalog_tokens,
            generator_context_token_count=estimated_tokens,
            token_count_source="estimated_final_generator_context",
        )

    def assemble(
        self,
        raw_query: str,
        retrieval_results: Dict[str, Any]
    ) -> GoldAssembledContext:
        """
        执行 MMR 提纯与教研四槽位黄金装配。
        """
        effective_strategy = retrieval_results.get("chunk_strategy", self.chunk_strategy)
        if effective_strategy not in {"card", "fallback"}:
            raise ValueError("retrieval_results chunk_strategy must be card or fallback")
        if effective_strategy == "card" and self.rerank_unit == "anchored_logical_window":
            return self._assemble_anchored_logical_windows(raw_query, retrieval_results)
        if effective_strategy == "card" and self.rerank_unit == "logical_evidence":
            return self._assemble_card_logical_evidence(raw_query, retrieval_results)
        if effective_strategy == "fallback":
            windows = self._apply_model_reranker(raw_query, retrieval_results.get("windows", []))
            selected_windows = windows[: min(5, len(windows))]
            evidence_dict: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for window in selected_windows:
                metadata = window.get("metadata", {})
                session_id = metadata.get("session_id")
                if isinstance(session_id, str):
                    session_id = int(session_id)
                raw_turn_ids = metadata.get("source_turn_ids", [])
                if isinstance(raw_turn_ids, str):
                    raw_turn_ids = json.loads(raw_turn_ids)
                for turn in window.get("evidence_turns", []) or []:
                    evidence = dict(turn)
                    evidence["session_id"] = session_id
                    evidence_dict[(int(session_id), int(turn["turn_id"]))] = evidence
                window["metadata"] = {**metadata, "source_turn_ids": list(raw_turn_ids or [])}
            sorted_evidence = [evidence_dict[key] for key in sorted(evidence_dict)]
            lines = [
                "# 【真实对话窗口知识基座 (Content-first Grounding Context)】",
                "> 本模式直接以 fallback sliding windows 作为证据入口；卡片不是必要条件。",
                "",
                "## 一、 检索到的真实对话窗口",
            ]
            for index, window in enumerate(selected_windows, start=1):
                metadata = window.get("metadata", {})
                lines.append(
                    f"### {index}. Session #{metadata.get('session_id')} · "
                    f"Turn {metadata.get('window_start_turn', '?')}~{metadata.get('window_end_turn', '?')}"
                )
                lines.append(f"- **窗口检索得分**: `{window.get('hybrid_score', 0.0)}`; BM25=`{window.get('bm25_score', 0.0)}`")
                lines.append("```text")
                lines.append(str(window.get("document", "")))
                lines.append("```")
            lines.extend(["", "## 二、 可引用的真实 Turn 证据"])
            for turn in sorted_evidence:
                speaker_tag = "🎓 [学生]" if turn["speaker"] == "student" else "👩‍🏫 [导师]"
                lines.append(f"- **[Turn {turn['turn_id']} · Session {turn['session_id']}] {speaker_tag}**: \"{turn['text']}\"")
            prompt_markdown = "\n".join(lines)
            char_count = len(prompt_markdown)
            word_count = len(re.findall(r"\w+", prompt_markdown))
            est_tokens = int(char_count * 0.5 + word_count * 0.5)
            budget_violation = est_tokens > self.max_prompt_tokens
            truncation_loss = 0
            if budget_violation:
                target_chars = max(1, self.max_prompt_tokens * 2)
                truncation_loss = max(0, len(prompt_markdown) - target_chars)
                prompt_markdown = prompt_markdown[:target_chars].rstrip() + "\n[Context truncated to token budget]"
                est_tokens = min(self.max_prompt_tokens, int(len(prompt_markdown) * 0.5 + len(re.findall(r"\w+", prompt_markdown)) * 0.5))
            return GoldAssembledContext(
                raw_query=raw_query,
                prompt_context_markdown=prompt_markdown,
                selected_misconception=None,
                selected_strategy=None,
                selected_windows=selected_windows,
                chunk_strategy="fallback",
                evidence_turns=sorted_evidence,
                estimated_token_count=est_tokens,
                compression_ratio=max(0.0, round(1.0 - len(prompt_markdown) / max(1, sum(len(str(w.get("document", ""))) for w in windows)), 2)),
                budget_violation=budget_violation,
                truncation_loss=truncation_loss,
            )

        misc_candidates = self._apply_model_reranker(
            raw_query, retrieval_results.get("misconceptions", [])
        )
        strat_candidates = self._apply_model_reranker(
            raw_query, retrieval_results.get("strategies", [])
        )
        
        rewritten_meta = retrieval_results.get("rewritten_queries", {})
        keywords = rewritten_meta.get("extracted_keywords", [])

        # 1. MMR 挑选 Top-1 学情错因卡
        selected_misc = self._select_mmr_best(
            candidates=misc_candidates,
            raw_query=raw_query,
            keywords=keywords,
            already_selected_texts=[]
        )

        # 2. 策略槽位独立选择。错因卡与策略卡是互补信息，不允许跨槽位
        # Jaccard 惩罚把同一教学事件的正确策略压低。
        selected_strat = self._select_mmr_best(
            candidates=strat_candidates,
            raw_query=raw_query,
            keywords=keywords,
            already_selected_texts=[]
        )

        # 3. 聚合并去重真实对白证据 (来自 DuckDB)
        evidence_dict: Dict[Tuple[int, int], Dict[str, Any]] = {}
        if selected_misc and selected_misc.get("evidence_turns"):
            session_id = selected_misc.get("metadata", {}).get("session_id")
            for t in selected_misc["evidence_turns"]:
                if session_id is None:
                    raise ValueError("错因卡证据缺少 session_id，无法建立可审计引用")
                evidence = dict(t)
                evidence["session_id"] = session_id
                evidence_dict[(session_id, t["turn_id"])] = evidence
        if selected_strat and selected_strat.get("evidence_turns"):
            session_id = selected_strat.get("metadata", {}).get("session_id")
            for t in selected_strat["evidence_turns"]:
                if session_id is None:
                    raise ValueError("策略卡证据缺少 session_id，无法建立可审计引用")
                evidence = dict(t)
                evidence["session_id"] = session_id
                evidence_dict[(session_id, t["turn_id"])] = evidence

        sorted_evidence = [evidence_dict[k] for k in sorted(evidence_dict)]

        # 4. 渲染四槽位标准 Markdown
        lines = [
            "# 【权威教研参考知识基座 (Pedagogical Grounding Context)】\n",
            "> ℹ️ 本基座经由多视角混合检索与 MMR 多样性精排提纯；引用仍须经过逐字段审计。\n"
        ]

        # 槽位 1: 学情认知误区诊断
        lines.append("## 一、 学情认知误区诊断 (Student Misconception Profile)")
        if selected_misc:
            m_meta = selected_misc.get("metadata", {})
            if m_meta.get("session_id") is not None:
                lines.append(f"- **关联会话**: Session #{m_meta['session_id']}")
            if m_meta.get("question_id") is not None:
                lines.append(f"- **题目 ID**: {m_meta['question_id']}")
            if m_meta.get("subject_path"):
                lines.append(f"- **学科考纲**: `{m_meta['subject_path']}`")
            if m_meta.get("misconception_name"):
                lines.append(f"- **标准错因命名**: **{m_meta['misconception_name']}**")
            if m_meta.get("error_choice"):
                lines.append(f"- **学生典型误选**: 选项 `{m_meta.get('error_choice')}`")
            if m_meta.get("deep_mechanism"):
                lines.append(f"- **深层认知机理**: {m_meta['deep_mechanism']}")
            if m_meta.get("confusion_triggers"):
                lines.append(f"- **混淆触发点**: {m_meta.get('confusion_triggers')}")
        else:
            lines.append("- *(未检索到强匹配的学情错因卡)*")
        lines.append("")

        # 槽位 2: 名师破局启发策略
        lines.append("## 二、 名师破局启发策略 (Tutor Pedagogical Strategy)")
        if selected_strat:
            s_meta = selected_strat.get("metadata", {})
            if s_meta.get("session_id") is not None:
                lines.append(f"- **关联会话**: Session #{s_meta['session_id']}")
            if s_meta.get("strategy_category"):
                lines.append(f"- **策略分类**: `{s_meta['strategy_category']}`")
            if s_meta.get("key_aha_question"):
                lines.append(f"- **名师破局一问 (Key Aha Question)**: 💡 **「{s_meta['key_aha_question']}」**")
            if s_meta.get("talk_moves"):
                lines.append(f"- **教学动作标签 (Talk Moves)**: {s_meta.get('talk_moves')}")
            if s_meta.get("pedagogical_goal"):
                lines.append(f"- **引导教学目标**: {s_meta['pedagogical_goal']}")
            if s_meta.get("scaffolding_steps"):
                lines.append(f"- **启发脚手架步骤链**: {s_meta.get('scaffolding_steps')}")
        else:
            lines.append("- *(未检索到强匹配的名师启发卡)*")
        lines.append("")

        # 槽位 3: 真实师生对白实录证据 (不可篡改)
        lines.append("## 三、 真实师生对白实录证据 (Verbatim Dialogue Evidence)")
        lines.append("> ⚠️ 以下对话直接提取自底层关系事实表 (DuckDB session_dialogue_turns)，具有最高事实权威性：")
        if sorted_evidence:
            for t in sorted_evidence:
                speaker_tag = "🎓 [学生]" if t["speaker"] == "student" else "👩‍🏫 [导师]"
                moves = t.get("talk_moves")
                if moves is not None and len(moves) > 0:
                    moves_str = ", ".join(moves) if isinstance(moves, (list, tuple)) else str(moves)
                    moves_tag = f" `({moves_str})`"
                else:
                    moves_tag = ""
                lines.append(
                    f"- **[Turn {t['turn_id']} · Session {t['session_id']}] {speaker_tag}**: "
                    f"\"{t['text']}\"{moves_tag}"
                )
        else:
            lines.append("- *(无可引证的历史对白轮次)*")
        lines.append("")

        # 槽位 4: 考点原题锚点
        lines.append("## 四、 考纲考点与原题锚点")
        sample_cand = selected_misc or selected_strat
        if sample_cand:
            p_meta = sample_cand.get("metadata", {})
            if p_meta.get("question_text"):
                lines.append(f"- **考题原题**: {p_meta['question_text']}")
            if p_meta.get("subject_path"):
                lines.append(f"- **考纲知识树**: `{p_meta['subject_path']}`")

        prompt_markdown = "\n".join(lines)

        # 5. Token 统计与压缩率计算
        char_count = len(prompt_markdown)
        word_count = len(re.findall(r"\w+", prompt_markdown))
        est_tokens = int(char_count * 0.5 + word_count * 0.5)
        budget_violation = est_tokens > self.max_prompt_tokens
        truncation_loss = 0
        if budget_violation:
            # Deterministic tokenizer-independent enforcement.  The exact
            # model tokenizer can be supplied by the caller later; until then
            # we conservatively cap the emitted context and expose the loss.
            target_chars = max(1, int(self.max_prompt_tokens / max(0.5, (char_count + word_count) / max(1, char_count))))
            target_chars = min(target_chars, self.max_prompt_tokens * 2)
            truncation_loss = max(0, len(prompt_markdown) - target_chars)
            prompt_markdown = prompt_markdown[:target_chars].rstrip() + "\n[Context truncated to token budget]"
            new_chars = len(prompt_markdown)
            new_words = len(re.findall(r"\w+", prompt_markdown))
            est_tokens = min(self.max_prompt_tokens, int(new_chars * 0.5 + new_words * 0.5))

        raw_candidates_chars = sum(len(c.get("document", "")) for c in misc_candidates + strat_candidates)
        compression_ratio = round(1.0 - (char_count / max(1, raw_candidates_chars)), 2)

        return GoldAssembledContext(
            raw_query=raw_query,
            prompt_context_markdown=prompt_markdown,
            selected_misconception=selected_misc,
            selected_strategy=selected_strat,
            selected_windows=[],
            chunk_strategy="card",
            evidence_turns=sorted_evidence,
            estimated_token_count=est_tokens,
            compression_ratio=max(0.0, compression_ratio),
            budget_violation=budget_violation,
            truncation_loss=truncation_loss,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("🚀 测试轻量级 MMR 压缩与教研多样性黄金装配器...")
    
    from src.storage_manager import DualEngineStorageManager
    from src.retriever import DualMetricRetriever
    
    storage = DualEngineStorageManager(
        db_path="data/db/tutoring_knowledge.duckdb",
        chroma_dir="data/chroma",
        embedding_backend="deterministic",
    )
    retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic")
    assembler = PedagogicalGoldAssembler(lambda_diversity=0.7)
    
    raw_q = "四舍五入 5.4598 到 1 位小数时，学生为什么会误选 5.45？名师怎么引导？"
    ret_res = retriever.retrieve_multi_perspective_rrf(raw_q, top_k_each=3, fetch_evidence=True)
    
    gold_res = assembler.assemble(raw_q, ret_res)
    
    print(f"\n📝 原始提问: {gold_res.raw_query}")
    print(f"📊 预估 Token 数: {gold_res.estimated_token_count} (Token 压缩率: {gold_res.compression_ratio*100:.1f}%)")
    print(f"🎯 选定错因卡: {gold_res.selected_misconception['metadata'].get('misconception_name') if gold_res.selected_misconception else 'None'}")
    print(f"🎯 选定策略卡: {gold_res.selected_strategy['metadata'].get('key_aha_question') if gold_res.selected_strategy else 'None'}")
    print(f"🔍 回溯对白证据轮数: {len(gold_res.evidence_turns)} 轮")
    
    print("\n" + "=" * 80)
    print(gold_res.prompt_context_markdown)
    print("=" * 80)
    
    storage.close()
    print("\n✅ MMR 黄金装配器测试通过！")
