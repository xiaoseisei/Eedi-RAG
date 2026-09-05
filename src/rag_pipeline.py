"""
================================================================================
模块名称: src/rag_pipeline.py
业务定位: Step 5 - 基于真实课堂证据的 RAG 问答生成管道
核心流程:
  1. 多视角检索与 MMR 黄金装配
  2. 结构化受限生成 (LLM 模式 / 调用方显式选择的确定性保真模态)
  3. 真实对白防伪审计与全透明调试溯源区展示
================================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    DialogueCitation,
    PedagogicalGuidanceResponse,
)
from src.generator_contract import (
    GeneratorGuidancePayloadV2,
    build_evidence_catalog,
    materialize_generator_citations,
    render_evidence_catalog,
    normalize_v2_payload_data,
    validate_claim_citation_closure,
    validate_role_coverage,
)
from src.retriever import DualMetricRetriever
from src.reranker import PedagogicalGoldAssembler, GoldAssembledContext

logger = logging.getLogger(__name__)


class RAGGenerationError(RuntimeError):
    """生成依赖或 LLM 输出不可用，且未获调用方降级授权。"""


class CitationAuditError(RAGGenerationError):
    """回答中的任一引用无法逐字段映射到真实会话证据。"""


class LLMDialogueCitation(BaseModel):
    """Citation fields the model must copy from the final assembled evidence."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    session_id: int = Field(ge=1)
    turn_id: int = Field(ge=1)
    speaker: Literal["student", "tutor"]
    quote_text: str = Field(min_length=1)


class LLMGuidancePayload(BaseModel):
    """
    LLM 轻量输出契约：短回答 + 完整引用四元组。
    引用必须逐字复制自本次最终 Assembler evidence。
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    subject_path: str = Field(min_length=1, description="学科考纲路径")
    answer: str = Field(min_length=1, max_length=200, description="核心回答（80字左右）")
    dialogue_citations: List[LLMDialogueCitation] = Field(
        min_length=1,
        description="来自最终上下文的完整 (session_id, turn_id, speaker, quote_text) 引用",
    )
    transfer_question: Optional[str] = Field(default=None, description="变式巩固题（如有）")


SYSTEM_PEDAGOGICAL_PROMPT = """你是资深中学数学教研专家。根据【知识基座】回答用户提问。

【回答风格】：
- 直接、精炼，像给同事的口头答复
- 学情类问题 → 一句话说出核心误区与成因（80字以内）
- 教法类问题 → 一句话说出核心引导策略（80字以内）
- 不要展开长篇分析，不要分章节，不要建议步骤

【输出格式】严格输出以下 JSON：

{
  "subject_path": "学科考纲路径",
  "answer": "核心回答（80字以内）",
  "dialogue_citations": [
    {
      "session_id": 10,
      "turn_id": 9,
      "speaker": "tutor",
      "quote_text": "逐字复制【真实师生对白实录证据】中的完整原文"
    }
  ],
  "transfer_question": null
}

【引用硬约束】：
- 每条引用必须完整填写 session_id、turn_id、speaker、quote_text；
- 只能逐字复制本次【真实师生对白实录证据】中可见的条目；
- 禁止改写 quote_text、猜测 Turn、引用未进入最终上下文的 Session。
"""


SYSTEM_PEDAGOGICAL_PROMPT_V2 = """你是资深中学数学教研专家。根据最终知识基座回答用户提问。

【最高优先级红线】
1. 先回答用户实际问题；不得把相邻问题当成用户问题回答。
2. 如果用户要求“一句话”“核心原因”或“如何提问”，`answer` 只保留对应核心内容，最多 1～2 句；不要自动展开完整诊断、证据审计和教学方案。
3. 不得把卡片摘要、模型常识或未经证据支持的数学推导写成历史对白事实。
4. 不得编造 evidence_id；不得改写、拼接或复制 `[Turn N] [speaker]` 到任何引用字段。
5. 引用采用最小充分集合：只引用实际支撑回答主张的证据，不要为了“看起来完整”罗列无关 Turn。

【回答目标】
- 直接回答用户的问题，不要回答另一个相关但未被询问的问题。
- 根据问题范围决定详细程度：窄问题优先简洁回答；明确要求诊断、证据和干预时，才展开完整教研答复。
- 结构化字段仍必须完整输出，但与问题无关的字段使用空数组、`null` 或“未观察到”，不要在 `answer` 中重复所有字段。
- `evidence_explanation`（证据解释）只解释本题实际使用的 evidence_id；不要把全部上下文逐条复述。
- 保持清晰、具体、可供教研员直接使用；不能为了增加细节而引入未被证据支持的新事实。

【事实与推断】
- claim_type=fact：只能陈述最终知识基座中直接支持的事实，至少绑定一个 evidence_id。
- claim_type=inference：允许基于对白做明确推断，但必须至少绑定一个 evidence_id，并填写 qualification，说明这是基于证据的解释而不是原话事实。
- claim_type=recommendation：给出可执行教学动作；如果引用了历史事实，也要绑定 evidence_id。
- 不要把卡片摘要当作真实对白；真实对白只通过 Evidence catalog 的 evidence_id 引用。
- 如果上下文不足以支持某个事实，明确写“未观察到充分证据”，不要猜测。

【引用硬约束】
- dialogue_citations 只能输出 evidence_id，禁止输出 session_id、turn_id 或 quote_text。
- 每个事实 claim 的 evidence_ids 至少一个，并且该 claim 至少有一个 evidence_id 出现在 dialogue_citations。
- 每个 inference claim 也必须至少绑定一个 evidence_id，并且该 ID 至少有一个出现在 dialogue_citations；recommendation 可不引用历史证据。
- 如果问题同时询问学生错因和导师引导，dialogue_citations 必须同时覆盖 student 和 tutor 的 evidence_id。
- 多角色问题还必须分别填写 `student_evidence_ids` 和 `tutor_evidence_ids`，两组 ID 都必须出现在 dialogue_citations。
- 不要创造 Evidence catalog 中不存在的 ID，不要复制 `[Turn N]` 等展示标记到任何 citation 字段。

【正例与反例（仅示范格式，不是本题证据）】
正例：用户问“导师如何用一句话区分体积和表面积？”时，`answer` 应只回答“体积用长×宽×高计算，表面积是各个面的面积总和。”；若该事实由 E001 支持，则 claims 和 dialogue_citations 都引用 E001。
多角色正例：用户同时问“学生错在哪里，导师如何引导？”时，分别填写 `student_evidence_ids=["E010"]`、`tutor_evidence_ids=["E011"]`，并让 `dialogue_citations` 同时包含 `{"evidence_id":"E010"}` 和 `{"evidence_id":"E011"}`。
反例：用户只问一句话，却在 `answer` 中继续展开完整错因、全部 Turn、证据审计和三步教学方案；或引用 E001，却把 E002 的内容写进答案；或把“可能是位值混淆”写成“学生一定缺乏位值概念”。

【输出格式】严格输出 JSON，不要输出 Markdown 或额外文字：
{
  "subject_path": "学科考纲路径",
  "answer": "完整、直接回答问题的核心答复（600 字符以内）",
  "misconception_diagnosis": "学生错因诊断；没有学生证据时明确说明未观察到",
  "evidence_explanation": "逐条说明哪些结论由哪些 evidence_id 支持，哪些是有限定的推断",
  "key_aha_question": "可直接用于课堂点拨的核心问题",
  "scaffolding_steps": ["按顺序给出可执行的引导步骤"],
  "pedagogical_intervention": ["给教研员的可执行教学动作"],
  "recommended_talk_moves": ["<Press for Accuracy>"],
  "claims": [
    {
      "claim_id": "C1",
      "claim_text": "一个事实、推断或教学建议",
      "claim_type": "fact",
      "evidence_ids": ["E001"],
      "qualification": null
    }
  ],
  "student_evidence_ids": ["E001"],
  "tutor_evidence_ids": ["E002"],
  "dialogue_citations": [{"evidence_id": "E001"}, {"evidence_id": "E002"}],
  "transfer_question": null
}
"""


class EndToEndPedagogicalRAGPipeline:
    """
    端到端 RAG 问答与意图精准路由教研生成管道。
    """

    def __init__(
        self,
        retriever: Optional[DualMetricRetriever] = None,
        assembler: Optional[PedagogicalGoldAssembler] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        chunk_strategy: str = "card",
        llm_timeout_seconds: float | None = None,
        generator_contract_version: str = "v1",
    ):
        self.retriever = retriever
        self.assembler = assembler or PedagogicalGoldAssembler(lambda_diversity=0.7)
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model_name = model_name or os.getenv("LLM_MODEL")
        if llm_timeout_seconds is None:
            raw_timeout = os.getenv("LLM_TIMEOUT_SECONDS", "60")
            try:
                llm_timeout_seconds = float(raw_timeout)
            except (TypeError, ValueError) as exc:
                raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number") from exc
        if llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be positive")
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        if generator_contract_version not in {"v1", "v2"}:
            raise ValueError("generator_contract_version must be v1 or v2")
        self.generator_contract_version = generator_contract_version
        if chunk_strategy not in {"card", "fallback"}:
            raise ValueError("chunk_strategy must be card or fallback")
        self.chunk_strategy = chunk_strategy

    def _prepare_context_with_timings(
        self,
        query: str,
        *,
        top_k_each: int = 3,
        fetch_evidence: bool = True,
    ) -> tuple[Dict[str, Any], GoldAssembledContext, Dict[str, float]]:
        """Run retrieval and final context assembly without Generator calls."""

        import time as _time

        if self.retriever is None:
            raise RuntimeError("DualMetricRetriever 未初始化，无法执行检索！")
        if top_k_each <= 0:
            raise ValueError("top_k_each must be positive")
        timings: Dict[str, float] = {}
        t0 = _time.time()
        configured_pool = (
            getattr(self.assembler, "reranker_pool_size", top_k_each)
            if getattr(self.assembler, "model_reranker", None) is not None
            else top_k_each
        )
        if self.chunk_strategy == "fallback":
            windows = self.retriever.retrieve_fallback_windows(
                query,
                top_k=max(top_k_each, configured_pool),
                fetch_evidence=fetch_evidence,
            )
            retrieval_res = {
                "chunk_strategy": "fallback",
                "raw_query": query,
                "windows": windows,
                "misconceptions": [],
                "strategies": [],
                "rewritten_queries": {"extracted_keywords": []},
                "trace": {
                    "retrieval_mode": self.retriever.retrieval_mode,
                    "chunk_strategy": "fallback",
                },
            }
        else:
            retrieval_res = self.retriever.retrieve_multi_perspective_rrf(
                raw_query=query,
                top_k_each=max(top_k_each, configured_pool),
                fetch_evidence=fetch_evidence,
            )
            rerank_unit = getattr(self.assembler, "rerank_unit", "card")
            if rerank_unit == "anchored_logical_window":
                if getattr(self.assembler, "model_reranker", None) is None:
                    raise RAGGenerationError(
                        "anchored_logical_window requires an explicitly configured model reranker"
                    )
                ranked_lanes: Dict[str, List[Dict[str, Any]]] = {}
                for lane in ("misconceptions", "strategies"):
                    ranked_cards = self.assembler.rerank_candidates(
                        query,
                        list(retrieval_res.get(lane, []) or []),
                        unit_name="card",
                    )
                    retrieval_res[lane] = ranked_cards
                    ranked_lanes[lane] = ranked_cards
                parent_limit = getattr(self.assembler, "parent_card_count", 3)
                parent_cards: List[Dict[str, Any]] = []
                for lane in ("misconceptions", "strategies"):
                    if ranked_lanes[lane] and len(parent_cards) < parent_limit:
                        parent_cards.append(ranked_lanes[lane][0])
                remaining_cards = sorted(
                    [
                        card
                        for cards in ranked_lanes.values()
                        for card in cards
                        if card not in parent_cards
                    ],
                    key=lambda card: (
                        -float(card.get("reranker_score", 0.0)),
                        int(card.get("reranker_rank", 10**9)),
                        str(card.get("chunk_id", "")),
                    ),
                )
                parent_cards.extend(
                    remaining_cards[: max(0, parent_limit - len(parent_cards))]
                )
                card_candidates = list(retrieval_res.get("misconceptions", []) or []) + list(
                    retrieval_res.get("strategies", []) or []
                )
                anchored_units = self.retriever.expand_anchored_logical_windows(
                    card_candidates,
                    parent_cards,
                    window_size=6,
                    step=3,
                )
                ranked_units = self.assembler.rerank_candidates(
                    query,
                    anchored_units,
                    unit_name="anchored_logical_window",
                )
                retrieval_res["evidence_units"] = ranked_units
                retrieval_res["anchored_parent_cards"] = parent_cards
                retrieval_res.setdefault("trace", {}).update(
                    {
                        "rerank_unit": "anchored_logical_window",
                        "parent_card_count": len(parent_cards),
                        "parent_card_limit": parent_limit,
                        "evidence_unit_count": len(ranked_units),
                        "evidence_window_size": 6,
                        "evidence_window_step": 3,
                    }
                )
            elif rerank_unit == "logical_evidence":
                card_candidates = list(retrieval_res.get("misconceptions", []) or []) + list(
                    retrieval_res.get("strategies", []) or []
                )
                retrieval_res["evidence_units"] = self.retriever.expand_card_candidates_to_evidence_units(
                    card_candidates,
                    window_size=6,
                    step=3,
                )
                retrieval_res.setdefault("trace", {})["rerank_unit"] = "logical_evidence"
                retrieval_res["trace"]["evidence_unit_count"] = len(
                    retrieval_res["evidence_units"]
                )
        timings["retrieval"] = round(_time.time() - t0, 3)
        t0 = _time.time()
        gold_ctx = self.assembler.assemble(
            raw_query=query,
            retrieval_results=retrieval_res,
        )
        timings["assembly"] = round(_time.time() - t0, 3)
        return retrieval_res, gold_ctx, timings

    def prepare_context(
        self,
        query: str,
        *,
        top_k_each: int = 3,
        fetch_evidence: bool = True,
    ) -> tuple[Dict[str, Any], GoldAssembledContext]:
        """Return the production final context without invoking the Generator."""

        retrieval_res, gold_ctx, _ = self._prepare_context_with_timings(
            query,
            top_k_each=top_k_each,
            fetch_evidence=fetch_evidence,
        )
        return retrieval_res, gold_ctx

    def _generator_context_text(self, gold_ctx: GoldAssembledContext) -> str:
        """Build the exact context string supplied to the selected Generator contract."""

        return self._generator_context_text_for_version(
            gold_ctx, self.generator_contract_version
        )

    @staticmethod
    def _generator_context_text_for_version(
        gold_ctx: GoldAssembledContext, contract_version: str
    ) -> str:
        if contract_version == "v1":
            return gold_ctx.prompt_context_markdown
        catalog = build_evidence_catalog(gold_ctx)
        return "\n\n".join(
            [gold_ctx.prompt_context_markdown, render_evidence_catalog(catalog)]
        )

    def _synthesize_deterministic_grounding(
        self,
        query: str,
        gold_ctx: GoldAssembledContext,
        retrieval_res: Dict[str, Any]
    ) -> PedagogicalGuidanceResponse:
        """
        确定性保真模态 (Deterministic Grounding Mode)：
        综合检索结果生成包含学情诊断与教法建议的完整回答。
        """
        misc_candidates = retrieval_res.get("misconceptions", [])
        strat_candidates = retrieval_res.get("strategies", [])

        if gold_ctx.chunk_strategy == "fallback":
            windows = gold_ctx.selected_windows
            if not windows or not gold_ctx.evidence_turns:
                raise RAGGenerationError("fallback content-first generation lacks real window evidence")
            first_meta = windows[0].get("metadata", {})
            session_id = first_meta.get("session_id")
            subject_path = first_meta.get("subject_path") or "Mathematics"
            if isinstance(session_id, str):
                session_id = int(session_id)
            if session_id is None:
                raise RAGGenerationError("fallback content-first generation lacks session_id")
            answer_content = (
                f"针对教研问题【{query}】，直接依据检索到的真实对话窗口总结："
                "先保留学生的原始困惑，再依据导师的连续追问和解释归纳教学处理方式。"
            )
            citations = [
                DialogueCitation(
                    session_id=int(turn["session_id"]),
                    turn_id=int(turn["turn_id"]),
                    speaker=turn["speaker"],
                    quote_text=turn["text"],
                    verifiable_in_duckdb=False,
                )
                for turn in gold_ctx.evidence_turns
            ]
            response = PedagogicalGuidanceResponse(
                query=query,
                subject_path=subject_path,
                session_id=int(session_id),
                answer_content=answer_content,
                misconception_diagnosis="本次为真实对话窗口直取，未依赖卡片摘要。",
                key_aha_question="请以窗口中的连续师生对白作为教学证据。",
                recommended_talk_moves=[],
                scaffolding_steps=[],
                dialogue_citations=citations,
                transfer_question=None,
                retrieved_sources_debug=self._extract_debug_sources(retrieval_res),
                audit_status="PENDING",
            )
            return response

        misc = gold_ctx.selected_misconception or (misc_candidates[0] if misc_candidates else {})
        strat = gold_ctx.selected_strategy or (strat_candidates[0] if strat_candidates else {})
        m_meta = misc.get("metadata", {})
        s_meta = strat.get("metadata", {})

        session_id = m_meta.get("session_id") or s_meta.get("session_id")
        subject_path = m_meta.get("subject_path") or s_meta.get("subject_path")
        if session_id is None or not subject_path:
            raise RAGGenerationError("确定性生成缺少 session_id 或 subject_path 真实来源")

        citations: List[DialogueCitation] = []

        # =========================================================================
        # 综合生成：学情诊断 + 名师教法
        # =========================================================================
        misc_name = m_meta.get("misconception_name")
        deep_mech = m_meta.get("deep_mechanism")
        key_aha = s_meta.get("key_aha_question") or ""
        err_choice = m_meta.get("error_choice", "")
        choice_str = f"（学生典型误选为选项 `{err_choice}`）" if err_choice else ""
        ped_goal = s_meta.get("pedagogical_goal") or ""
        talk_moves_raw = s_meta.get("talk_moves", [])
        recommended_moves = list(talk_moves_raw) if isinstance(talk_moves_raw, (list, tuple)) else []
        scaff_raw = s_meta.get("scaffolding_steps")
        scaffolding_steps = list(scaff_raw) if isinstance(scaff_raw, (list, tuple)) else []
        if not misc_name or not deep_mech:
            raise RAGGenerationError("确定性生成缺少真实 misconception_name 或 deep_mechanism")

        answer_paragraphs = [
            f"针对教研问题【{query}】，结合历史真实辅导数据分析如下：\n",
        ]

        # 学情错因诊断
        answer_paragraphs.append(f"\n### 🔍 二、 学情认知卡点诊断 (Misconception Diagnosis)")
        answer_paragraphs.append(f"- **误区命名**: **{misc_name}** {choice_str}")
        answer_paragraphs.append(f"- **深层机理**: {deep_mech}")
        answer_paragraphs.append(f"- **考纲知识点**: `{subject_path}`")

        # 名师教法策略
        answer_paragraphs.append(f"\n### 💡 三、 名师启发引导策略 (Tutor Intervention)")
        if key_aha:
            answer_paragraphs.append(f"- **核心破局一问**: 🎯 **「{key_aha}」**")
        if ped_goal:
            answer_paragraphs.append(f"- **教学目标**: {ped_goal}")
        if scaffolding_steps:
            answer_paragraphs.append("- **分步引导脚手架**:")
            for step in scaffolding_steps:
                answer_paragraphs.append(f"  • {step}")
        if recommended_moves:
            moves_str = ", ".join(f"`{m}`" for m in recommended_moves[:4])
            answer_paragraphs.append(f"- **教学动作 (Talk Moves)**: {moves_str}")

        answer_content = "\n".join(answer_paragraphs)

        # 引用证据
        for t in gold_ctx.evidence_turns:
            citations.append(DialogueCitation(
                session_id=t["session_id"],
                turn_id=t["turn_id"],
                speaker=t["speaker"],
                quote_text=t["text"],
                verifiable_in_duckdb=False,
            ))

        citations = list({
            (c.session_id, c.turn_id, c.speaker, c.quote_text): c
            for c in citations
        }.values())
        if not citations:
            raise CitationAuditError("确定性生成没有可引用的真实对白")

        # 构建调试原文来源明细
        debug_sources = self._extract_debug_sources(retrieval_res)

        transfer_q = s_meta.get("transfer_question") or m_meta.get("transfer_question")

        resp = PedagogicalGuidanceResponse(
            query=query,
            subject_path=subject_path,
            session_id=session_id,
            answer_content=answer_content,
            misconception_diagnosis=deep_mech,
            key_aha_question=key_aha,
            recommended_talk_moves=recommended_moves,
            scaffolding_steps=scaffolding_steps,
            dialogue_citations=citations,
            transfer_question=transfer_q,
            retrieved_sources_debug=debug_sources,
            audit_status="PENDING",
        )
        resp.rendered_markdown = self._render_pretty_markdown(resp, gold_ctx, debug_sources)
        return resp

    def _generate_with_llm(
        self,
        query: str,
        gold_ctx: GoldAssembledContext,
        retrieval_res: Dict[str, Any],
        *,
        contract_version: str | None = None,
    ) -> PedagogicalGuidanceResponse:
        """调用 LLM；两次尝试均失败时显式抛错，不在内部静默降级。"""
        import time as _time

        if not self.api_key:
            raise RAGGenerationError("缺少 LLM_API_KEY，无法执行 LLM 生成")
        if not self.model_name:
            raise RAGGenerationError("缺少 LLM_MODEL，无法执行 LLM 生成")

        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.llm_timeout_seconds,
            )
        except Exception as exc:
            raise RAGGenerationError(f"LLM 客户端初始化失败: {exc}") from exc

        effective_contract = contract_version or self.generator_contract_version
        if effective_contract not in {"v1", "v2"}:
            raise ValueError("contract_version must be v1 or v2")
        system_prompt = (
            SYSTEM_PEDAGOGICAL_PROMPT_V2
            if effective_contract == "v2"
            else SYSTEM_PEDAGOGICAL_PROMPT
        )
        # 探针: prompt 构建
        t_prompt = _time.time()
        generator_context = (
            self._generator_context_text(gold_ctx)
            if effective_contract == self.generator_contract_version
            else self._generator_context_text_for_version(gold_ctx, effective_contract)
        )
        base_user_prompt = (
            f"【用户教研提问】: {query}\n\n"
            f"{generator_context}\n\n"
            "仅输出 JSON，不要输出其他内容。"
        )
        user_prompt = base_user_prompt
        prompt_tokens_est = len(user_prompt) // 2  # 粗估: ~2 字符/token
        logger.info(f"  📏 [探针] Prompt 构建: {round(_time.time()-t_prompt, 3)}s | 预估输入 tokens: ~{prompt_tokens_est}")

        last_error: Optional[Exception] = None
        payload: Any = None
        catalog = build_evidence_catalog(gold_ctx) if effective_contract == "v2" else None
        contract_repair_attempted = False
        contract_auto_merged_ids: list[str] = []
        for attempt in range(2):
            try:
                # 探针: LLM API 调用 (核心瓶颈)
                t_llm = _time.time()
                llm_response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                llm_elapsed = round(_time.time() - t_llm, 3)

                raw_content = llm_response.choices[0].message.content
                if not raw_content:
                    raise ValueError("LLM 返回空内容")

                # 探针: 响应元信息
                usage = getattr(llm_response, "usage", None)
                if usage:
                    logger.info(
                        f"  🤖 [探针] LLM API 调用: {llm_elapsed}s | "
                        f"输入 tokens: {usage.prompt_tokens} | "
                        f"输出 tokens: {usage.completion_tokens} | "
                        f"总 tokens: {usage.total_tokens} | "
                        f"生成速度: {round(usage.completion_tokens / llm_elapsed, 1)} tokens/s"
                    )

                # 探针: JSON 解析 + Pydantic 校验
                t_parse = _time.time()
                payload_type = (
                    GeneratorGuidancePayloadV2
                    if effective_contract == "v2"
                    else LLMGuidancePayload
                )
                decoded_payload = json.loads(raw_content)
                if effective_contract == "v2":
                    decoded_payload, merged_ids = normalize_v2_payload_data(
                        decoded_payload,
                        catalog or {},
                    )
                    contract_auto_merged_ids.extend(merged_ids)
                payload = payload_type.model_validate(decoded_payload)
                if effective_contract == "v2":
                    validate_claim_citation_closure(payload)
                    validate_role_coverage(
                        payload.dialogue_citations,
                        catalog or {},
                        query=query,
                        student_evidence_ids=payload.student_evidence_ids,
                        tutor_evidence_ids=payload.tutor_evidence_ids,
                    )
                logger.info(f"  🔧 [探针] JSON 解析+校验: {round(_time.time()-t_parse, 3)}s")

                break
            except (json.JSONDecodeError, ValidationError, ValueError, AttributeError, IndexError) as exc:
                last_error = exc
                logger.warning(
                    "LLM 输出校验失败 (attempt %s/2): %s",
                    attempt + 1,
                    exc,
                )
                if attempt == 0 and effective_contract == "v2":
                    contract_repair_attempted = True
                    user_prompt = (
                        base_user_prompt
                        + "\n\n【契约修复反馈】上一次 JSON 未通过证据契约："
                        + str(exc)
                        + "。请只修正 claims/dialogue_citations/student_evidence_ids/tutor_evidence_ids 的 evidence_id、角色覆盖或推断限定，"
                        "然后重新输出完整 JSON；不要输出解释文字。"
                    )
            except Exception as exc:
                last_error = exc
                logger.warning("LLM 请求失败 (attempt %s/2): %s", attempt + 1, exc)

        if payload is None:
            raise RAGGenerationError(f"LLM 生成在 2 次尝试后失败: {last_error}") from last_error

        if effective_contract == "v2":
            catalog = catalog or build_evidence_catalog(gold_ctx)
            try:
                validate_role_coverage(
                    payload.dialogue_citations,
                    catalog,
                    query=query,
                    student_evidence_ids=payload.student_evidence_ids,
                    tutor_evidence_ids=payload.tutor_evidence_ids,
                )
                citations = materialize_generator_citations(
                    payload.dialogue_citations,
                    catalog,
                )
            except ValueError as exc:
                raise RAGGenerationError(f"Generator v2 evidence contract failed: {exc}") from exc
            referenced_ids = list(dict.fromkeys(citation.session_id for citation in citations))
            talk_moves = list(payload.recommended_talk_moves)[:4]
            debug_sources = self._extract_debug_sources(retrieval_res)
            answer_content = "\n\n".join(
                [
                    payload.answer,
                    f"诊断：{payload.misconception_diagnosis}",
                    f"证据解释：{payload.evidence_explanation}",
                    "教学干预：\n" + "\n".join(f"- {step}" for step in payload.pedagogical_intervention),
                ]
            )
            grounding_claim_lines = []
            for claim in payload.claims:
                evidence = ", ".join(claim.evidence_ids) or "无"
                qualification = (
                    f"；限定：{claim.qualification}"
                    if claim.qualification
                    else ""
                )
                grounding_claim_lines.append(
                    f"- [{claim.claim_type}] [{evidence}] {claim.claim_text}{qualification}"
                )
            grounding_text = "\n\n".join(
                [
                    payload.answer,
                    f"诊断：{payload.misconception_diagnosis}",
                    f"证据解释：{payload.evidence_explanation}",
                    "事实与推断主张：\n" + "\n".join(grounding_claim_lines),
                ]
            )
            response = PedagogicalGuidanceResponse(
                query=query,
                subject_path=payload.subject_path,
                session_id=referenced_ids[0] if referenced_ids else None,
                answer_content=answer_content,
                misconception_diagnosis=payload.misconception_diagnosis,
                key_aha_question=payload.key_aha_question,
                recommended_talk_moves=talk_moves,
                scaffolding_steps=list(payload.scaffolding_steps),
                dialogue_citations=citations,
                transfer_question=payload.transfer_question,
                retrieved_sources_debug=debug_sources,
                audit_status="PENDING",
            )
            response.__dict__["generator_contract_version"] = "v2"
            response.__dict__["generator_contract_repair_attempted"] = contract_repair_attempted
            response.__dict__["generator_contract_auto_merged_ids"] = list(
                dict.fromkeys(contract_auto_merged_ids)
            )
            response.__dict__["generator_core_answer"] = payload.answer
            response.__dict__["generator_grounding_text"] = grounding_text
            response.__dict__["generator_citation_evidence_ids"] = [
                citation.evidence_id for citation in payload.dialogue_citations
            ]
            response.__dict__["generator_claims"] = [
                claim.model_dump() for claim in payload.claims
            ]
            response.__dict__["generator_context_text"] = generator_context
            response.rendered_markdown = self._render_pretty_markdown(response, gold_ctx, debug_sources)
            return response

        # ── v1 后处理: 只接受模型从最终 Assembler evidence 复制的四元组 ──
        t_lookup = _time.time()
        citations = [
            DialogueCitation(**citation.model_dump(), verifiable_in_duckdb=False)
            for citation in payload.dialogue_citations
        ]
        referenced_ids = list(dict.fromkeys(citation.session_id for citation in citations))

        # 从检索结果中查找破局一问和教学动作（来自被引用的 session）
        key_aha = ""
        talk_moves: List[str] = []
        for src in retrieval_res.get("strategies", []):
            meta = src.get("metadata", {})
            if meta.get("session_id") in referenced_ids:
                if not key_aha and meta.get("key_aha_question"):
                    key_aha = meta["key_aha_question"]
                if meta.get("talk_moves"):
                    talk_moves = list(meta["talk_moves"])[:4]

        if not key_aha:
            key_aha = payload.answer

        # 从检索结果中提取调试信息
        debug_sources = self._extract_debug_sources(retrieval_res)

        response = PedagogicalGuidanceResponse(
            query=query,
            subject_path=payload.subject_path,
            session_id=referenced_ids[0] if referenced_ids else None,
            answer_content=payload.answer,
            misconception_diagnosis=payload.answer,
            key_aha_question=key_aha,
            recommended_talk_moves=talk_moves,
            scaffolding_steps=[],
            dialogue_citations=citations,
            transfer_question=payload.transfer_question,
            retrieved_sources_debug=debug_sources,
            audit_status="PENDING",
        )
        response.rendered_markdown = self._render_pretty_markdown(response, gold_ctx, debug_sources)
        return response

    def _extract_debug_sources(self, retrieval_res: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从检索结果中提取用于调试溯源展示的卡片明细。"""
        debug_sources = []

        for window in retrieval_res.get("windows", [])[:5]:
            metadata = window.get("metadata", {})
            debug_sources.append({
                "type": "fallback_window",
                "session_id": metadata.get("session_id"),
                "similarity_score": round(window.get("fallback_combined_score", window.get("hybrid_score", 0.0)), 4),
                "subject_path": metadata.get("subject_path"),
                "title": f"Turn {metadata.get('window_start_turn', '?')}~{metadata.get('window_end_turn', '?')}",
                "document_text": window.get("document", ""),
                "evidence_turns": window.get("evidence_turns", []),
            })
        
        # 错因卡 (Top 5)
        for c in retrieval_res.get("misconceptions", [])[:5]:
            c_meta = c.get("metadata", {})
            debug_sources.append({
                "type": "student_misconception",
                "session_id": c_meta.get("session_id"),
                "similarity_score": round(c.get("rrf_score") or c.get("hybrid_score", 0.0), 4),
                "subject_path": c_meta.get("subject_path"),
                "title": c_meta.get("misconception_name"),
                "document_text": c.get("document", ""),
                "evidence_turns": c.get("evidence_turns", [])
            })

        # 策略卡 (Top 5)
        for s in retrieval_res.get("strategies", [])[:5]:
            s_meta = s.get("metadata", {})
            debug_sources.append({
                "type": "tutor_strategy",
                "session_id": s_meta.get("session_id"),
                "similarity_score": round(s.get("rrf_score") or s.get("hybrid_score", 0.0), 4),
                "subject_path": s_meta.get("subject_path"),
                "title": s_meta.get("key_aha_question"),
                "document_text": s.get("document", ""),
                "evidence_turns": s.get("evidence_turns", [])
            })

        return debug_sources

    def _audit_citations(
        self,
        response: PedagogicalGuidanceResponse,
        gold_ctx: GoldAssembledContext,
        citation_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """逐项精确校验 session、turn、speaker 和原文；任一失败即拒绝回答。"""
        if not response.dialogue_citations:
            response.audit_status = "GROUNDING_DEGRADED"
            raise CitationAuditError("空引用不能通过审计")

        records = citation_records or [citation.model_dump() for citation in response.dialogue_citations]
        if len(records) != len(response.dialogue_citations):
            response.audit_status = "GROUNDING_DEGRADED"
            raise CitationAuditError("引用传输记录与响应引用数量不一致")

        evidence_facts = {
            (t.get("session_id"), t.get("turn_id"), t.get("speaker"), t.get("text"))
            for t in gold_ctx.evidence_turns
        }
        if any(None in fact for fact in evidence_facts):
            response.audit_status = "GROUNDING_DEGRADED"
            raise CitationAuditError("证据缺少 session_id、turn_id、speaker 或原文")

        invalid = []
        for citation, record in zip(response.dialogue_citations, records):
            # Verification is derived here only.  A caller/model supplied flag
            # must never bypass the final-evidence authorization check.
            citation.verifiable_in_duckdb = False
            fact = (
                record.get("session_id"),
                record.get("turn_id"),
                record.get("speaker"),
                record.get("quote_text"),
            )
            verified = fact[1] is not None and fact[1] > 0 and fact in evidence_facts
            citation.verifiable_in_duckdb = verified
            if not verified:
                invalid.append(fact)

        if invalid:
            response.audit_status = "GROUNDING_DEGRADED"
            raise CitationAuditError(f"引用未精确匹配真实会话证据: {invalid}")
        response.audit_status = "AUDITED_100_VERIFIED"

    def _render_pretty_markdown(
        self,
        resp: PedagogicalGuidanceResponse,
        gold_ctx: GoldAssembledContext,
        debug_sources: List[Dict[str, Any]]
    ) -> str:
        """
        渲染包含【自然解答正文】与【全透明调试溯源区】的美化 Markdown。
        """
        main_title = "教研洞察与深度分析"

        lines = [
            f"# 🎓 Eedi-RAG {main_title}\n",
            f"> **提问意图**: `{resp.query}`",
            f"> **防伪审计**: `[{resp.audit_status}]`\n",
            "---",
            "## 📝 【教研深度解答 (AI Synthesis Answer)】\n",
            resp.answer_content or resp.misconception_diagnosis,
            ""
        ]

        if resp.transfer_question:
            lines.append("---")
            lines.append("### 📝 【课后变式训练】")
            lines.append(f"{resp.transfer_question}\n")

        # 调试溯源区
        lines.append("---")
        lines.append("## 📚 【检索索引的知识原文与原声证据 (Retrieved Grounding Sources & Debug Trace)】")
        lines.append("> ℹ️ *以下为本次问答从底层 ChromaDB 向量库与 DuckDB 关系表检索命中的真实知识卡片原文与历史师生对话记录，供调试与教研白盒核验：*\n")

        for idx, src in enumerate(debug_sources, 1):
            type_tag = {
                "student_misconception": "🏷️ [学生错因卡]",
                "tutor_strategy": "💡 [名师策略卡]",
                "fallback_window": "🪟 [真实滑动窗口]",
            }.get(src.get("type"), "📚 [检索证据]")
            lines.append(f"### {idx}. {type_tag} 会话 Session #{src.get('session_id', 'N/A')} (相关度得分: {src.get('similarity_score', 0.0)})")
            lines.append(f"- **考纲路径**: `{src.get('subject_path', '数学考纲')}`")
            core_label = "窗口范围" if src.get("type") == "fallback_window" else "卡片核心"
            lines.append(f"- **{core_label}**: **{src.get('title', 'N/A')}**")
            
            text_label = "窗口原文切片" if src.get("type") == "fallback_window" else "卡片向量文本切片"
            lines.append(f"- **{text_label}**:")
            lines.append("```text")
            doc_lines = src.get("document_text", "").strip().split("\n")
            lines.extend(doc_lines[:6])
            if len(doc_lines) > 6:
                lines.append(f"... (共 {len(doc_lines)} 行)")
            lines.append("```")
            
            ev_turns = src.get("evidence_turns", [])
            if ev_turns:
                lines.append(f"- **DuckDB 关联历史对白实录 ({len(ev_turns)} 轮)**:")
                for t in ev_turns[:4]:
                    spk = "🎓 学生" if t["speaker"] == "student" else "👩‍🏫 导师"
                    lines.append(f"  - `[Turn {t['turn_id']}]` **{spk}**: \"{t['text']}\"")
                if len(ev_turns) > 4:
                    lines.append(f"  - *... (其余 {len(ev_turns)-4} 轮已略)*")
            lines.append("")

        return "\n".join(lines)

    def ask(
        self,
        query: str,
        mode: str = "auto",
        top_k_each: int = 3,
        fetch_evidence: bool = True
    ) -> PedagogicalGuidanceResponse:
        """端到端问答核心入口；deterministic 必须由调用方显式选择。"""
        import time as _time

        if self.retriever is None:
            raise RuntimeError("DualMetricRetriever 未初始化，无法执行检索！")
        if mode not in {"auto", "llm", "deterministic"}:
            raise ValueError("mode 必须是 auto、llm 或 deterministic")
        if mode == "auto" and not self.api_key:
            raise RAGGenerationError(
                "auto 模式缺少 LLM_API_KEY；如需真实确定性生成，请显式传 mode='deterministic'"
            )

        timings = {}
        logger.info(f"🚀 [Step5_RAG] 接收提问: '{query}' (mode={mode})")

        # 1. 检索与 MMR 黄金装配
        retrieval_res, gold_ctx, prepare_timings = self._prepare_context_with_timings(
            query,
            top_k_each=top_k_each,
            fetch_evidence=fetch_evidence,
        )
        timings.update(prepare_timings)

        # 3. 生成
        t0 = _time.time()
        if mode in {"llm", "auto"}:
            response = self._generate_with_llm(query, gold_ctx, retrieval_res)
        else:
            response = self._synthesize_deterministic_grounding(query, gold_ctx, retrieval_res)
        timings["generation"] = round(_time.time() - t0, 3)

        # 4. 引用防伪审计
        t0 = _time.time()
        self._audit_citations(response, gold_ctx)
        response.rendered_markdown = self._render_pretty_markdown(
            response,
            gold_ctx,
            response.retrieved_sources_debug,
        )
        timings["audit_render"] = round(_time.time() - t0, 3)

        timings["total"] = round(sum(timings.values()), 3)
        response._profiling = timings

        logger.info(
            f"✅ [Step5_RAG] 完成 | "
            f"检索={timings['retrieval']}s | "
            f"装配={timings['assembly']}s | "
            f"生成={timings['generation']}s | "
            f"审计渲染={timings['audit_render']}s | "
            f"总计={timings['total']}s"
        )
        return response
