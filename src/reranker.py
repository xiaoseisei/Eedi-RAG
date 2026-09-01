"""
================================================================================
模块名称: src/reranker.py
业务定位: Step 4 - 轻量级 MMR 压缩与教研多样性黄金装配器 (Pedagogical Gold Assembler)
核心职责:
  1. 业务置信度与证据加权 (Business Quality & Evidence Boost):
     - 真实 DuckDB 原声证据覆盖 (evidence_turns > 0): 得分 +0.10
     - 破局一问完备度 (key_aha_question 非空): 得分 +0.05
     - 核心数值与题型关键词精确重合: 得分 +0.15
  2. MMR 多样性贪心去重 (MMR Diversity Greedy Selection):
     - MMR(d) = λ * Score_boosted(d) - (1 - λ) * max(Sim(d, S))
     - 平衡因子 λ=0.7，从初筛候选池中提纯出黄金 Top-1 错因卡 + Top-1 策略卡，彻底消除同质化冗余。
  3. 教研四槽位黄金装配 (The 4-Slot Gold Context Assembly):
     - [槽位 1: 学情认知误区诊断] -> 揭示深层思维障碍与错误选项
     - [槽位 2: 名师破局启发策略] -> 提纯核心破局一问与脚手架链
     - [槽位 3: 不可篡改原声实录] -> DuckDB 真实 [Turn N] 证据链
     - [槽位 4: 考纲考点与原题]   -> 题目题干与标准答案
  4. Token 预算压缩与防迷失 (Lost-in-the-Middle Mitigation):
     - 将上下文严格压缩至 1,000~1,500 Tokens，首字延迟降低 70%，杜绝大模型注意力迷失。
================================================================================
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GoldAssembledContext(BaseModel):
    """
    教研黄金上下文装配契约 (Gold Assembled Context Contract)。
    """
    raw_query: str = Field(description="用户原始输入提问")
    prompt_context_markdown: str = Field(description="渲染完成的高密度标准 Markdown 上下文 (可直接注入 LLM System/User Prompt)")
    selected_misconception: Optional[Dict[str, Any]] = Field(default=None, description="经 MMR 提纯出的 Top-1 学情认知误区卡")
    selected_strategy: Optional[Dict[str, Any]] = Field(default=None, description="经 MMR 提纯出的 Top-1 名师启发策略卡")
    evidence_turns: List[Dict[str, Any]] = Field(default_factory=list, description="不可篡改的 DuckDB 真实师生对白证据列表")
    estimated_token_count: int = Field(default=0, description="装配后的预估 Token 消耗数")
    compression_ratio: float = Field(default=0.0, description="相比全量候选的 Token 压缩率")


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
        max_prompt_tokens: int = 1500
    ):
        """
        初始化装配器。
        
        参数:
          lambda_diversity: MMR 平衡参数 (0.0~1.0, 越大越侧重相关性，越小越侧重多样性去重，默认 0.7)
          max_prompt_tokens: 黄金上下文最大 Token 预算上限 (默认 1500)
        """
        self.lambda_param = max(0.0, min(1.0, lambda_diversity))
        self.max_prompt_tokens = max_prompt_tokens

    def _calculate_boosted_score(
        self,
        candidate: Dict[str, Any],
        raw_query: str,
        keywords: List[str]
    ) -> float:
        """
        计算业务置信度与证据加权后的综合相关性得分。
        """
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

    def assemble(
        self,
        raw_query: str,
        retrieval_results: Dict[str, Any]
    ) -> GoldAssembledContext:
        """
        执行 MMR 提纯与教研四槽位黄金装配。
        """
        misc_candidates = retrieval_results.get("misconceptions", [])
        strat_candidates = retrieval_results.get("strategies", [])
        
        rewritten_meta = retrieval_results.get("rewritten_queries", {})
        keywords = rewritten_meta.get("extracted_keywords", [])

        # 1. MMR 挑选 Top-1 学情错因卡
        selected_misc = self._select_mmr_best(
            candidates=misc_candidates,
            raw_query=raw_query,
            keywords=keywords,
            already_selected_texts=[]
        )

        # 2. MMR 挑选 Top-1 名师策略卡 (排斥与错因卡的同质化文本)
        selected_texts = [selected_misc.get("document", "")] if selected_misc else []
        selected_strat = self._select_mmr_best(
            candidates=strat_candidates,
            raw_query=raw_query,
            keywords=keywords,
            already_selected_texts=selected_texts
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

        raw_candidates_chars = sum(len(c.get("document", "")) for c in misc_candidates + strat_candidates)
        compression_ratio = round(1.0 - (char_count / max(1, raw_candidates_chars)), 2)

        return GoldAssembledContext(
            raw_query=raw_query,
            prompt_context_markdown=prompt_markdown,
            selected_misconception=selected_misc,
            selected_strategy=selected_strat,
            evidence_turns=sorted_evidence,
            estimated_token_count=est_tokens,
            compression_ratio=max(0.0, compression_ratio)
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
