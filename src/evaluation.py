"""
================================================================================
模块名称: src/evaluation.py
业务定位: Step 6 - 可信的 LLM-as-a-Judge 评测引擎 (Trustworthy RAG Evaluator)
核心原则:
  - 绝不伪造数据: 缺失指标标 None，不给默认值
  - 失败即报失败: LLM 调用失败重试 3 次后如实标记
  - 可选指标显式标注: 区分"实测"和"未测"
================================================================================
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import DialogueCitation

logger = logging.getLogger(__name__)

# ================================================================================
# 1. 结构化评测结果 Pydantic 契约
# ================================================================================

class AtomicClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim: str = Field(description="从模型回答中拆解出的最小原子事实陈述")
    status: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"] = Field(
        description="该断言在参考上下文中的支持状态"
    )
    reasoning: str = Field(description="判断该断言状态的依据与分析")


class JudgeLLMOutput(BaseModel):
    """
    LLM 裁判返回的原始 JSON schema。
    用于校验 LLM 输出是否符合预期格式，缺失字段会触发 ValidationError。
    """
    model_config = ConfigDict(extra="forbid", strict=True)

    context_recall: float = Field(ge=0.0, le=1.0, description="上下文召回率")
    context_precision: float = Field(ge=0.0, le=1.0, description="上下文精准度")
    faithfulness: float = Field(ge=0.0, le=1.0, description="事实忠实度 / 无幻觉率")
    answer_relevance: float = Field(ge=0.0, le=1.0, description="回答相关性")
    socratic_quality: float = Field(ge=0.0, le=1.0, description="苏格拉底启发度")
    root_cause_diagnosis: str = Field(min_length=1, description="白盒根因诊断")
    detailed_critique: str = Field(min_length=1, description="详细点评")


class EvaluationReport(BaseModel):
    """
    完整评测报告。所有指标均为实测值，无默认值伪造。
    """
    model_config = ConfigDict(extra="forbid", strict=True)

    # RAGAS 核心四大指标 (0.0 ~ 1.0)
    context_recall: float = Field(ge=0.0, le=1.0, description="上下文召回率")
    context_precision: float = Field(ge=0.0, le=1.0, description="上下文精准度")
    faithfulness: float = Field(ge=0.0, le=1.0, description="事实忠实度 / 无幻觉率")
    answer_relevance: float = Field(ge=0.0, le=1.0, description="回答相关性")
    # 垂直教研指标，可选 (None = 未测，不伪造)
    socratic_quality: Optional[float] = Field(ge=0.0, le=1.0, description="苏格拉底式启发教学质量")
    misconception_depth: Optional[float] = Field(ge=0.0, le=1.0, description="错因认知深度得分")
    citation_accuracy: Optional[float] = Field(ge=0.0, le=1.0, description="真实对白引用准确率")
    intent_match: Optional[float] = Field(ge=0.0, le=1.0, description="意图分类匹配度")
    # 综合评估
    ragas_score: float = Field(ge=0.0, le=1.0, description="调和平均总分")
    root_cause_diagnosis: str = Field(min_length=1, description="白盒根因诊断分析")
    critique: str = Field(description="LLM 裁判的详细点评与改进建议")
    eval_mode: Literal["LLM_JUDGE", "HEURISTIC_FALLBACK"] = Field(
        description="当前采用的评测模式"
    )
    eval_status: Literal["SUCCESS", "LLM_FAILED", "PARTIAL", "DEGRADED"] = Field(
        description="评测状态"
    )


# ================================================================================
# 2. LLM 裁判 Prompt
# ================================================================================

JUDGE_SYSTEM_PROMPT = """你是一名资深的教育大模型评测专家与中学数学教研裁判。
你的任务是对 RAG 系统生成的教研回答进行严谨、客观的量化打分。

你需要评估以下 4 个核心维度 (每个 0.0~1.0):
1. context_recall: 检索上下文覆盖了 Ground Truth 中多少核心事实
2. context_precision: 排在前面的检索 Chunk 是否是解答问题真正需要的
3. faithfulness: 回答中的每个事实是否都能在检索上下文中找到依据
4. answer_relevance: 回答是否直击用户提问的核心意图

你还需要给出:
- socratic_quality: 教法是否通过启发式提问引导，而非直接给答案 (0.0~1.0)
- root_cause_diagnosis: 一句话诊断系统的主要问题
- detailed_critique: 详细的点评与改进建议

必须输出纯 JSON 对象，包含上述所有字段，数值必须是 0.0~1.0 之间的浮点数。"""


# ================================================================================
# 3. 评测引擎
# ================================================================================

class RagasEvaluatorEngine:
    """
    可信的 RAG 评测引擎。
    核心原则: 绝不伪造数据，缺失即报缺失。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        allow_heuristic_fallback: bool = False,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model_name = model_name or os.getenv("JUDGE_LLM_MODEL") or os.getenv("LLM_MODEL")
        self.max_retries = max_retries
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.enable_llm = bool(self.api_key)

        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.enable_llm and not self.model_name:
            raise ValueError("JUDGE_LLM_MODEL (or explicit model_name) is required when LLM_API_KEY is set")
        if not self.enable_llm and not self.allow_heuristic_fallback:
            raise ValueError(
                "LLM_API_KEY is required for trustworthy evaluation; "
                "set allow_heuristic_fallback=True to explicitly request degraded heuristic metrics"
            )
        if not self.enable_llm:
            logger.warning("启发式评测已由调用方显式开启；结果将标记为 DEGRADED")

    def configuration(self) -> Dict[str, Any]:
        """Return non-secret configuration recorded alongside benchmark reports."""
        return {
            "backend": "openai-compatible" if self.enable_llm else "deterministic-keyword-heuristic",
            "model": self.model_name,
            "base_url_configured": bool(self.base_url),
            "max_retries": self.max_retries,
            "allow_heuristic_fallback": self.allow_heuristic_fallback,
        }

    # ─────────────────────────────────────────────
    # 公开入口
    # ─────────────────────────────────────────────

    def evaluate_case(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        verbatim_quotes: Optional[List[Dict[str, Any]]] = None,
        actual_citations: Optional[List[DialogueCitation]] = None,
        expected_category: Optional[str] = None,
        actual_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对单条用例执行评测。返回的 dict 包含所有必要字段，数值为 float (可为 None)。
        """
        # 引用审计 (确定性计算，不需要 LLM)
        citation_hit = self._audit_citations(
            contexts,
            verbatim_quotes,
            actual_citations,
        )

        # 尝试 LLM 评测
        if self.enable_llm:
            llm_result = self._evaluate_with_llm(question, answer, contexts, ground_truth)
            if llm_result is not None:
                llm_result["citation_hit"] = citation_hit
                # 意图匹配: 确定性计算
                if expected_category and actual_intent:
                    llm_result["intent_match"] = 1.0 if expected_category == actual_intent else 0.0
                else:
                    llm_result["intent_match"] = None
                return llm_result

        if not self.allow_heuristic_fallback:
            raise RuntimeError("LLM judge failed; heuristic fallback was not explicitly enabled")

        # 只有显式授权才允许启发式降级
        logger.warning("LLM 评测不可用，执行调用方显式授权的启发式降级")
        heuristic_result = self._evaluate_with_heuristics(question, answer, contexts, ground_truth)
        heuristic_result["citation_hit"] = citation_hit
        if expected_category and actual_intent:
            heuristic_result["intent_match"] = 1.0 if expected_category == actual_intent else 0.0
        else:
            heuristic_result["intent_match"] = None
        return heuristic_result

    # ─────────────────────────────────────────────
    # LLM 评测路径
    # ─────────────────────────────────────────────

    def _evaluate_with_llm(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Optional[Dict[str, Any]]:
        """调用 LLM 裁判评测。失败返回 None，绝不伪造数据。"""
        prompt = self._build_judge_prompt(question, answer, contexts, ground_truth)

        judge = self._call_llm_judge_with_retry(prompt)
        if judge is None:
            return None

        # Pydantic 校验通过，直接取值 — 字段存在、类型正确、范围合法
        ragas_score = self._compute_ragas_score(
            judge.context_recall,
            judge.context_precision,
            judge.faithfulness,
            judge.answer_relevance
        )

        return {
            "context_recall": judge.context_recall,
            "context_precision": judge.context_precision,
            "faithfulness": judge.faithfulness,
            "answer_relevance": judge.answer_relevance,
            "socratic_quality": judge.socratic_quality,  # None 或 float
            "ragas_score": ragas_score,
            "root_cause_diagnosis": judge.root_cause_diagnosis,
            "critique": judge.detailed_critique,
            "eval_mode": "LLM_JUDGE",
            "eval_status": "SUCCESS",
        }

    def _call_llm_judge_with_retry(self, prompt: str) -> Optional[JudgeLLMOutput]:
        """
        调用 LLM 裁判，用 Pydantic 做 schema 校验。
        校验失败自动重试，全部失败返回 None。
        """
        try:
            import openai
        except ImportError:
            logger.error("❌ openai 包未安装，无法调用 LLM 裁判")
            return None

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw_content = response.choices[0].message.content or "{}"
                parsed = json.loads(raw_content)

                # Pydantic schema 校验: 自动检查字段存在、类型、范围 (0.0~1.0)
                validated = JudgeLLMOutput.model_validate(parsed)
                return validated

            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ LLM 裁判返回非法 JSON (attempt {attempt}/{self.max_retries}): {e}")
            except Exception as e:
                # Pydantic ValidationError 也走这里
                logger.warning(f"⚠️ LLM 裁判输出校验失败 (attempt {attempt}/{self.max_retries}): {e}")

        logger.error(f"❌ LLM 裁判在 {self.max_retries} 次尝试后仍然失败")
        return None

    def _build_judge_prompt(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> str:
        """构建发给 LLM 裁判的评测 prompt。"""
        contexts_text = "\n---\n".join(
            f"[Context {i+1}]\n{ctx[:1500]}" for i, ctx in enumerate(contexts[:10])
        )

        return f"""请对以下 RAG 问答进行评测:

【用户提问】
{question}

【RAG 系统回答】
{answer[:3000]}

【检索到的上下文】
{contexts_text}

【黄金标准答案 (Ground Truth)】
{ground_truth}

请输出 JSON，包含以下字段:
- context_recall (0.0~1.0): 检索上下文覆盖了 Ground Truth 中多少核心事实
- context_precision (0.0~1.0): 排在前面的 Chunk 是否是解答问题真正需要的
- faithfulness (0.0~1.0): 回答中的每个事实是否都能在检索上下文中找到依据
- answer_relevance (0.0~1.0): 回答是否直击用户提问的核心意图
- socratic_quality (0.0~1.0): 教法是否通过启发式提问引导，而非直接给答案
- root_cause_diagnosis: 一句话诊断系统的主要问题
- detailed_critique: 详细的点评与改进建议"""

    # ─────────────────────────────────────────────
    # 启发式降级路径
    # ─────────────────────────────────────────────

    def _evaluate_with_heuristics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        确定性启发式评测。不依赖 LLM，基于规则计算。
        明确标记 eval_mode="HEURISTIC_FALLBACK"，提醒用户这是降级结果。
        """
        # 文本预处理
        answer_lower = answer.lower()
        gt_lower = ground_truth.lower()
        contexts_combined = " ".join(contexts).lower()

        # 分词 (简单空格 + 标点分割)
        answer_tokens = set(re.findall(r'\b\w+\b', answer_lower))
        gt_tokens = set(re.findall(r'\b\w+\b', gt_lower))
        context_tokens = set(re.findall(r'\b\w+\b', contexts_combined))

        # 过滤停用词 (长度 < 3 的词)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                       "being", "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "can", "shall",
                       "to", "of", "in", "for", "on", "with", "at", "by", "from",
                       "as", "into", "through", "during", "before", "after", "and",
                       "but", "or", "nor", "not", "so", "yet", "both", "either",
                       "neither", "each", "every", "all", "any", "few", "more",
                       "most", "other", "some", "such", "no", "only", "own", "same",
                       "than", "too", "very", "just", "because", "if", "when", "where",
                       "how", "what", "which", "who", "whom", "this", "that", "these",
                       "those", "i", "me", "my", "we", "our", "you", "your", "he",
                       "him", "his", "she", "her", "it", "its", "they", "them", "their"}

        answer_tokens -= stop_words
        gt_tokens -= stop_words
        context_tokens -= stop_words

        # Context Recall: Ground Truth 中有多少关键信息出现在检索上下文中
        if gt_tokens:
            recall = len(gt_tokens & context_tokens) / len(gt_tokens)
        else:
            recall = 0.0

        # Context Precision: 检索上下文中有多少内容与问题相关
        question_tokens = set(re.findall(r'\b\w+\b', question.lower())) - stop_words
        if context_tokens:
            precision = len(question_tokens & context_tokens) / len(context_tokens)
        else:
            precision = 0.0

        # Faithfulness: 回答中有多少内容能在检索上下文中找到依据
        if answer_tokens:
            faithfulness = len(answer_tokens & context_tokens) / len(answer_tokens)
        else:
            faithfulness = 0.0

        # Answer Relevance: 回答与问题的关键词重叠度
        if answer_tokens and question_tokens:
            relevance = len(answer_tokens & question_tokens) / len(question_tokens)
        else:
            relevance = 0.0

        # 调和平均
        ragas_score = self._compute_ragas_score(recall, precision, faithfulness, relevance)

        return {
            "context_recall": round(min(recall, 1.0), 4),
            "context_precision": round(min(precision, 1.0), 4),
            "faithfulness": round(min(faithfulness, 1.0), 4),
            "answer_relevance": round(min(relevance, 1.0), 4),
            "socratic_quality": None,  # 启发式无法评估此项
            "ragas_score": round(ragas_score, 4),
            "root_cause_diagnosis": "HEURISTIC_FALLBACK: 基于关键词重叠的粗略评估，精度有限",
            "critique": "",
            "eval_mode": "HEURISTIC_FALLBACK",
            "eval_status": "DEGRADED",
        }

    # ─────────────────────────────────────────────
    # 引用审计
    # ─────────────────────────────────────────────

    def _audit_citations(
        self,
        contexts: List[str],
        verbatim_quotes: Optional[List[Dict[str, Any]]],
        actual_citations: Optional[List[DialogueCitation]],
    ) -> float:
        """
        计算引用命中率: 实际引用中有多少能在 verbatim_quotes 中找到。
        返回 0.0~1.0。
        """
        if not actual_citations:
            return 0.0

        if not verbatim_quotes:
            return 0.0

        # 四元组必须逐字段精确一致；缺任一标识都不能审计为命中。
        valid_quotes = set()
        for q in verbatim_quotes:
            text = q.get("quote_text")
            session_id = q.get("session_id")
            turn_id = q.get("turn_id")
            speaker = q.get("speaker")
            if (
                isinstance(text, str)
                and text
                and session_id is not None
                and turn_id is not None
                and isinstance(speaker, str)
                and speaker
            ):
                valid_quotes.add((session_id, turn_id, speaker, text))

        if not valid_quotes:
            return 0.0

        # 计算命中率
        hits = 0
        for cite in actual_citations:
            candidate = (
                getattr(cite, "session_id", None),
                cite.turn_id,
                cite.speaker,
                cite.quote_text,
            )
            if candidate in valid_quotes:
                hits += 1

        return round(hits / len(actual_citations), 4) if actual_citations else 0.0

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────

    @staticmethod
    def _compute_ragas_score(
        recall: float,
        precision: float,
        faithfulness: float,
        relevance: float
    ) -> float:
        """计算 RAGAS 调和平均分。"""
        eps = 1e-6
        score = 4.0 / (
            (1.0 / (recall + eps)) +
            (1.0 / (precision + eps)) +
            (1.0 / (faithfulness + eps)) +
            (1.0 / (relevance + eps))
        )
        return round(min(max(score, 0.0), 1.0), 4)
