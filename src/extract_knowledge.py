"""
================================================================================
模块名称: src/extract_knowledge.py
业务定位: Step 2 - LLM 教学知识结构化抽取与双卡片蒸馏引擎 (Pedagogical Knowledge Distiller)
核心职责:
  1. 组装富含考纲树、题干选项与带时序/教学动作标记的完整会话上下文 Prompt。
  2. 运用 Few-Shot Chain-of-Thought (COT) 与 Pydantic Schema 强类型约束，
     从口语化长会话中提纯【学生认知误区卡片】与【名师启发式策略卡片】。
  3. 执行 100% 原声字面量子串校验门禁 (Verbatim Substring Validation)，杜绝大模型编造原话。
  4. 支持多厂商路由 (OpenAI / DeepSeek / SiliconFlow / 阿里千问 / 本地 Ollama / vLLM)。
  5. 实现【自纠补救机制 (Self-Correction Feedback Loop)】：当 JSON 解析或 Pydantic 校验失败时，
     将精准错误信息反馈回多轮对话上下文，让大模型自纠重试 (默认最多 3 次)。
  6. 严格遵守《反假可用与工程真实性守则》：依赖缺失或重试耗尽后显式拒绝执行，
     批处理在全部成功前不产出文件，绝不返回空卡、部分卡或模板伪造卡。
================================================================================
"""

import os
import sys
import json
import re
import argparse
import logging
import tempfile
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    CleanedSession,
    DialogueTurn,
    StudentMisconceptionProfile,
    TutorStrategyProfile,
    ExtractedPIU
)

logger = logging.getLogger(__name__)

# Bump this when evidence-selection instructions change so extraction reports
# can distinguish old representative-pointer runs from exhaustive evidence runs.
EXTRACTION_PROMPT_VERSION = "evidence-completeness-v2"
EVIDENCE_AUDIT_VERSION = "evidence-audit-v1"


class LLMUnavailableError(RuntimeError):
    """当未配置有效的 LLM API Key 或外部模型服务不可用时抛出 (Fail-Fast)。"""
    pass


class LLMExtractionError(RuntimeError):
    """当大模型抽取过程异常（如网络中断、响应格式非法、自纠重试耗尽仍然校验失败）时抛出。"""
    pass


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _StrictMisconceptionPayload(BaseModel):
    """LLM 边界专用契约；不能继承存储模型中的宽松默认值。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    misconception_name: NonEmptyStr
    error_choice: Optional[NonEmptyStr] = None
    deep_mechanism: NonEmptyStr
    confusion_triggers: List[NonEmptyStr] = Field(min_length=1)
    verbatim_student_quotes: List[NonEmptyStr] = Field(min_length=1)
    source_turn_ids: List[int] = Field(min_length=1)


class _StrictTutorStrategyPayload(BaseModel):
    """LLM 边界专用契约，拒绝缺失、空值和未知字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    pedagogical_goal: NonEmptyStr
    strategy_category: Literal[
        "Socratic_Questioning", "Scaffolding", "Counter_Example", "Analogy", "Revoicing"
    ]
    key_aha_question: NonEmptyStr
    scaffolding_steps: List[NonEmptyStr] = Field(min_length=1)
    analogy_or_metaphor: Optional[NonEmptyStr] = None
    talk_moves: List[NonEmptyStr] = Field(min_length=1)
    resolution_outcome: NonEmptyStr
    source_turn_ids: List[int] = Field(min_length=1)


class _StrictExtractionPayload(BaseModel):
    """LLM 原始 JSON 的顶层严格契约: 双卡片必须齐全，多余字段与空值一律拒绝。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    misconception: _StrictMisconceptionPayload
    tutor_strategy: _StrictTutorStrategyPayload


class _StrictEvidenceAuditPayload(BaseModel):
    """Second-pass contract for a minimal, role-correct evidence set."""

    model_config = ConfigDict(extra="forbid", strict=True)

    misconception_source_turn_ids: List[int] = Field(min_length=1)
    tutor_strategy_source_turn_ids: List[int] = Field(min_length=1)


class _SemanticMisconceptionPayload(BaseModel):
    """Semantic-only payload; model pointers are advisory and ignored."""

    model_config = ConfigDict(extra="forbid", strict=True)

    misconception_name: NonEmptyStr
    error_choice: Optional[NonEmptyStr] = None
    deep_mechanism: NonEmptyStr
    confusion_triggers: List[NonEmptyStr] = Field(min_length=1)
    verbatim_student_quotes: List[NonEmptyStr] = Field(min_length=1)
    source_turn_ids: List[int] = Field(default_factory=list)


class _SemanticTutorStrategyPayload(BaseModel):
    """Semantic-only payload; model pointers are advisory and ignored."""

    model_config = ConfigDict(extra="forbid", strict=True)

    pedagogical_goal: NonEmptyStr
    strategy_category: Literal[
        "Socratic_Questioning", "Scaffolding", "Counter_Example", "Analogy", "Revoicing"
    ]
    key_aha_question: NonEmptyStr
    scaffolding_steps: List[NonEmptyStr] = Field(min_length=1)
    analogy_or_metaphor: Optional[NonEmptyStr] = None
    talk_moves: List[NonEmptyStr] = Field(min_length=1)
    resolution_outcome: NonEmptyStr
    source_turn_ids: List[int] = Field(default_factory=list)


class _SemanticExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    misconception: _SemanticMisconceptionPayload
    tutor_strategy: _SemanticTutorStrategyPayload


def build_extraction_prompt(session: CleanedSession, semantic_only: bool = False) -> str:
    """
    组装供 LLM 消费的教育知识蒸馏 Prompt 上下文。
    
    参数:
      session (CleanedSession): 清洗融合后的完整会话 PIU 对象
      
    返回:
      str: 格式化后的完整 Prompt 文本
    """
    taxonomy_str = "\n".join(f"- {p}" for p in session.subjects.paths) if session.subjects.paths else "通用数学考点"
    options_str = "\n".join(f"  选项 {k}: {v}" for k, v in sorted(session.question.options.items())) if session.question.options else "  (无固定选项)"
    
    # 格式化时序师生交互流
    turns_formatted = []
    for t in session.turns:
        speaker_tag = "[TUTOR]" if t.is_tutor else "[STUDENT]"
        moves_tag = f" (Moves: {', '.join(t.talk_moves)})" if t.talk_moves else ""
        turns_formatted.append(f"[Turn {t.turn_id}] {speaker_tag}{moves_tag}: {t.text}")
    
    turns_str = "\n".join(turns_formatted)

    semantic_instruction = """
【Semantic-only 抽取模式】
本次只抽取卡片语义，不要把 source_turn_ids 当作可信证据指针。source_turn_ids 仅为兼容字段，
可以留空，后续由外部 evidence index 基于完整原文逻辑链重新绑定。每条学生原声只需能在全量学生轮次中
找到；核心提问只需能在全量导师轮次中找到。不要因为模型自选的 source_turn_ids 不包含对应原文而修改语义。
""" if semantic_only else ""
    
    prompt = f"""你是一名资深的教育测量与教学法专家（Educational Measurement & Pedagogy Expert）。
请你仔细阅读以下这场真实的 1v1 在线数学辅导多轮交互实录，并为该辅导提炼两大物理隔离的【原子知识卡片】：
1. 【学生认知误区卡片 (StudentMisconceptionProfile)】：诊断学生暴露出的深层认知障碍机理，提取学生原声。
2. 【名师启发式策略卡片 (TutorStrategyProfile)】：提炼导师采用的破局灵魂提问 (Aha Moment)、引导脚手架与教学法分类。

================================================================================
【辅导基本事实】
会话ID: {session.intervention_id}
题目ID: {session.question_id}
导师ID: {session.tutor_id}

【学科考纲层级路径】
{taxonomy_str}

【考题原题与选项分布】
题干内容: {session.question.question_text}
选项分布:
{options_str}

【师生时序对话实录】
{turns_str}
================================================================================

【输出约束与规范】
1. 必须输出严格符合 JSON 语法的结构化对象，包含 "misconception" 和 "tutor_strategy" 两大根对象。
2. "misconception.verbatim_student_quotes" 中的每一句话，必须是上方实录中学生实际发言的精确字面子串 (Exact Substring)，严禁任何美化或编造！
3. "tutor_strategy.strategy_category" 必须从以下标准分类中选取一个: ['Socratic_Questioning', 'Scaffolding', 'Counter_Example', 'Analogy', 'Revoicing']。
4. "tutor_strategy.key_aha_question" 必须是导师在实录中提出的最核心破局提问。
5. `source_turn_ids` 是**证据覆盖集合**，不是代表性样本：必须穷举所有直接支撑该卡片结论的对话轮次，按时序列出且不得重复。学生卡片至少检查“首次暴露错误、解释/坚持错误、修正或确认”各类学生发言；导师卡片至少检查“诊断/追问、关键提问、脚手架/验证、结果确认”各类导师发言。若某类在实录中不存在，不要臆造轮次。
6. 在输出 JSON 前先进行一次“证据完整性自检”：逐轮扫描完整实录，分别建立学生和导师的直接证据清单，再将清单中的全部 Turn ID 写入对应 `source_turn_ids`。禁止只返回一条最能代表主题的 Turn，也禁止为了减少字段而省略同一论证链上的其他直接证据。
7. `misconception.verbatim_student_quotes` 应覆盖所选学生证据中的关键原声；每条引用必须能在对应学生 Turn 中逐字找到。`tutor_strategy.key_aha_question` 必须逐字来自所选导师 Turn；其他导师证据通过 `source_turn_ids` 保留，不要把它们改写成不存在的引用。
8. `key_aha_question` 必须是**单个导师 Turn 内的短精确子串**（建议只保留核心问题句）；不得拼接多个 Turn，不得添加 `Turn N` 标签、引号或实录中不存在的表扬/emoji 文本。对应 `source_turn_ids` 至少包含该问题所在的 Tutor Turn。
{semantic_instruction}

请严格按照以下 JSON Schema 键名结构输出:
```json
{{
  "misconception": {{
    "misconception_name": "学术化标准错因命名 (如: '四舍五入数位保留规则与进位混淆')",
    "error_choice": "学生误选的选项字母或名称 (如 'A', 'B', 'Sophie')",
    "deep_mechanism": "深层认知障碍机理剖析 (解释学生为什么会这么想)",
    "confusion_triggers": ["触发困惑的核心概念或数值1", "概念2"],
    "verbatim_student_quotes": ["学生暴露错误的原声直接引用1", "原声引用2"],
    "source_turn_ids": [10, 12]
  }},
  "tutor_strategy": {{
    "pedagogical_goal": "阶段性教学引导目标",
    "strategy_category": "Socratic_Questioning",
    "key_aha_question": "名师破局核心提问 (Aha Moment Prompt)",
    "scaffolding_steps": [
      "Step 1: 引导学生重新审视题干中的关键条件与目标",
      "Step 2: 分步拆解计算过程",
      "Step 3: 验证答案是否符合题目全部要求"
    ],
    "analogy_or_metaphor": null,
    "talk_moves": ["<Press for Accuracy>"],
    "resolution_outcome": "最终辅导成效 (如: '学生在导师启发下修正了思路，推导出正确解答')",
    "source_turn_ids": [9, 11]
  }}
}}
```
请输出结构化 JSON:
"""
    return prompt


def build_evidence_audit_prompt(session: CleanedSession, extracted: ExtractedPIU) -> str:
    """Build a second-pass prompt that audits pointer completeness and noise.

    The auditor is not asked to rewrite the cards.  It only returns the
    minimal, role-correct set of Turn IDs that directly supports each draft
    card, allowing the caller to keep card semantics stable while expanding
    missing evidence pointers.
    """
    turns_formatted = []
    for turn in session.turns:
        speaker_tag = "[TUTOR]" if turn.is_tutor else "[STUDENT]"
        turns_formatted.append(f"[Turn {turn.turn_id}] {speaker_tag}: {turn.text}")
    draft_json = json.dumps(extracted.model_dump(), ensure_ascii=False, indent=2)
    return f"""你是严格的证据审计员，不要重写卡片语义，只审查 source_turn_ids。

会话 ID: {session.intervention_id}

【完整师生时序对话】
{chr(10).join(turns_formatted)}

【第一遍生成的卡片草稿】
{draft_json}

请分别为 misconception 和 tutor_strategy 返回**最小充分且完整**的直接证据 Turn ID 集合：
1. 学生集合只允许 [STUDENT] Turn；导师集合只允许 [TUTOR] Turn；ID 必须存在于上方实录。
2. “完整”表示覆盖卡片结论所依赖的全部直接证据链（错误暴露、关键解释/坚持、修正确认；或诊断追问、Aha 提问、脚手架、结果确认），不能只留一条代表性 Turn。
3. “最小充分”表示排除问候、纯语气词、泛化过渡、与卡片结论无直接关系的计算或上下文；对每个候选做反事实检查：删掉它后若其他证据仍足以支持同一结论，就不要纳入。
4. 不要根据题目常识臆造证据，也不要把相邻但无直接支撑作用的 Turn 当作证据。

只输出严格 JSON，不要输出解释：
{{
  "misconception_source_turn_ids": [1, 2],
  "tutor_strategy_source_turn_ids": [3, 4]
}}
"""


def _canonicalize_evidence_text(text: str) -> str:
    """\u5f15\u7528\u6bd4\u5bf9\u524d\u7684\u6587\u672c\u89c4\u8303\u5316: NFKC \u5f52\u4e00 + \u96f6\u5bbd\u5b57\u7b26\u5254\u9664 + \u7a7a\u767d\u538b\u7f29 + \u53bb\u9996\u5c3e emoji\u3002

    \u7528\u4e8e\u5438\u6536 LLM \u8f6c\u5f55\u5f15\u7528\u4e0e\u539f\u6587\u4e4b\u95f4\u7684\u65e0\u5bb3\u5dee\u5f02 (\u6487\u53f7/\u7a7a\u683c/\u8868\u60c5\u7b26\u53f7)\uff0c
    \u53ea\u4fdd\u7559\u6709\u610f\u4e49\u7684\u6587\u5b57\u5dee\u5f02\u4f5c\u4e3a Grounding Gate \u7684\u5931\u8d25\u4f9d\u636e\u3002
    """
    value = unicodedata.normalize("NFKC", str(text)).replace("\u200b", "")
    # Normalize typographic punctuation emitted by transcription/LLM output;
    # this is formatting normalization, not fuzzy matching.
    value = value.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...",
    }))
    value = re.sub(r"\s+", " ", value).strip()
    while value and unicodedata.category(value[0]) in {"So", "Sk"}:
        value = value[1:].lstrip()
    while value and unicodedata.category(value[-1]) in {"So", "Sk"}:
        value = value[:-1].rstrip()
    return value


def validate_verbatim_grounding(quote: str, turns: List[DialogueTurn]) -> Tuple[bool, Optional[int]]:
    """
    校验给定的原声引用是否真实存在于对话轮次中 (Exact Substring Validation)。
    
    参数:
      quote (str): 抽取的引用文本
      turns (List[DialogueTurn]): 会话全量轮次列表
      
    返回:
      Tuple[bool, Optional[int]]: (是否合法, 命中的 turn_id)
    """
    if not quote or not quote.strip():
        return False, None
        
    cleaned_quote = _canonicalize_evidence_text(quote)
    for t in turns:
        if cleaned_quote in _canonicalize_evidence_text(t.text):
            return True, t.turn_id
            
    return False, None


def _call_structured_once(
    client: Any,
    *,
    effective_model: Optional[str],
    messages: List[Dict[str, str]],
    prompt: str,
    temperature: float,
) -> Any:
    """Call an injected structured client or OpenAI-compatible client once."""
    if hasattr(client, "generate_structured"):
        request = messages if getattr(client, "supports_messages", False) else prompt
        return client.generate_structured(request)
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        if not effective_model:
            raise LLMUnavailableError("使用 OpenAI 兼容客户端时必须显式配置 LLM_MODEL。")
        response = client.chat.completions.create(
            model=effective_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM 返回空响应")
        return content
    raise LLMUnavailableError(
        "llm_client 必须实现 generate_structured() 或 OpenAI 兼容 chat.completions 接口。"
    )


def _validate_audit_turn_ids(
    *,
    turn_ids: List[int],
    session: CleanedSession,
    expected_tutor: bool,
    label: str,
) -> None:
    """审计轮次 ID 门禁: 不允许重复，且每个 ID 必须存在且说话人角色与预期一致。"""
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError(f"{label} source_turn_ids 不允许重复")
    turns_by_id = {turn.turn_id: turn for turn in session.turns}
    for turn_id in turn_ids:
        turn = turns_by_id.get(turn_id)
        if turn is None or turn.is_tutor != expected_tutor:
            role = "导师" if expected_tutor else "学生"
            raise ValueError(f"{label} source_turn_ids 包含不存在或非{role}轮次: {turn_id}")


def _merge_evidence_audit(
    extracted: ExtractedPIU,
    audit: _StrictEvidenceAuditPayload,
    session: CleanedSession,
) -> ExtractedPIU:
    """Union audited IDs with draft IDs and ground any newly added student quotes."""
    _validate_audit_turn_ids(
        turn_ids=audit.misconception_source_turn_ids,
        session=session,
        expected_tutor=False,
        label="misconception audit",
    )
    _validate_audit_turn_ids(
        turn_ids=audit.tutor_strategy_source_turn_ids,
        session=session,
        expected_tutor=True,
        label="tutor_strategy audit",
    )
    if extracted.misconception is None or extracted.tutor_strategy is None:
        raise ValueError("evidence audit 要求两张已成功生成的卡片")
    turn_map = {turn.turn_id: turn for turn in session.turns}
    student_ids = sorted(set(extracted.misconception.source_turn_ids) | set(audit.misconception_source_turn_ids))
    tutor_ids = sorted(set(extracted.tutor_strategy.source_turn_ids) | set(audit.tutor_strategy_source_turn_ids))
    existing_quotes = list(extracted.misconception.verbatim_student_quotes)
    canonical_quotes = {_canonicalize_evidence_text(quote) for quote in existing_quotes}
    for turn_id in student_ids:
        text = turn_map[turn_id].text
        canonical = _canonicalize_evidence_text(text)
        if canonical not in canonical_quotes:
            existing_quotes.append(text)
            canonical_quotes.add(canonical)
    misconception = extracted.misconception.model_copy(
        update={"source_turn_ids": student_ids, "verbatim_student_quotes": existing_quotes}
    )
    tutor_strategy = extracted.tutor_strategy.model_copy(update={"source_turn_ids": tutor_ids})
    return extracted.model_copy(update={"misconception": misconception, "tutor_strategy": tutor_strategy})


def _run_evidence_audit(
    *,
    extracted: ExtractedPIU,
    session: CleanedSession,
    client: Any,
    effective_model: Optional[str],
    temperature: float,
    max_retries: int,
) -> ExtractedPIU:
    """二次证据审计: 让 LLM 在严格契约下补选角色正确的证据轮次，再与草稿并集合并。

    审计失败会带错误反馈自纠重试 (最多 max_retries 次)，耗尽后抛 LLMExtractionError，
    绝不静默采用未审计的卡片。
    """
    prompt = build_evidence_audit_prompt(session, extracted)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a strict evidence auditor. Output JSON only."},
        {"role": "user", "content": prompt},
    ]
    last_error: Optional[Exception] = None
    last_raw: Any = {}
    for attempt in range(1, max_retries + 1):
        try:
            last_raw = _call_structured_once(
                client,
                effective_model=effective_model,
                messages=messages,
                prompt=prompt,
                temperature=temperature,
            )
            if isinstance(last_raw, str):
                last_raw = json.loads(last_raw)
            audit = _StrictEvidenceAuditPayload.model_validate(last_raw)
            return _merge_evidence_audit(extracted, audit, session)
        except (LLMUnavailableError, LLMExtractionError):
            raise
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning(
                    "Session %s evidence audit failed (attempt %s/%s): %s; retrying",
                    session.intervention_id,
                    attempt,
                    max_retries,
                    exc,
                )
                serialized = last_raw if isinstance(last_raw, str) else json.dumps(last_raw, ensure_ascii=False)
                messages.extend(
                    [
                        {"role": "assistant", "content": serialized},
                        {"role": "user", "content": f"证据审计输出校验失败: {exc}。只修正 Turn ID 集合后重新输出严格 JSON。"},
                    ]
                )
    raise LLMExtractionError(
        f"证据审计在 {max_retries} 次重试后仍然失败 (Session {session.intervention_id}): {last_error}"
    ) from last_error


def _parse_and_ground_payload(
    raw: Any,
    session: CleanedSession,
    semantic_only: bool = False,
) -> ExtractedPIU:
    """解析 LLM 输出并执行角色/原文门禁。

    In ``semantic_only`` mode the model's source pointers are advisory only;
    quote and Aha validation run against all authoritative turns of the
    corresponding role.  The deterministic evidence index binds final
    pointers after this function returns.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)
    payload = (
        _SemanticExtractionPayload.model_validate(raw)
        if semantic_only
        else _StrictExtractionPayload.model_validate(raw)
    )
    turns_by_id = {turn.turn_id: turn for turn in session.turns}

    all_student_turns = [turn for turn in session.turns if not turn.is_tutor]
    all_tutor_turns = [turn for turn in session.turns if turn.is_tutor]
    if semantic_only:
        student_turns = all_student_turns
    else:
        student_turns = []
        for turn_id in payload.misconception.source_turn_ids:
            turn = turns_by_id.get(turn_id)
            if turn is None or turn.is_tutor:
                raise ValueError(f"学生 source_turn_ids 包含不存在或非学生轮次: {turn_id}")
            student_turns.append(turn)
    for quote in payload.misconception.verbatim_student_quotes:
        if not any(validate_verbatim_grounding(quote, [turn])[0] for turn in student_turns):
            raise ValueError(
                f"学生原声引用未匹配其 source_turn_ids 对应的学生原文 (Grounding Gate Failed): {quote!r}"
            )

    if semantic_only:
        tutor_turns = all_tutor_turns
    else:
        tutor_turns = []
        for turn_id in payload.tutor_strategy.source_turn_ids:
            turn = turns_by_id.get(turn_id)
            if turn is None or not turn.is_tutor:
                raise ValueError(f"导师 source_turn_ids 包含不存在或非导师轮次: {turn_id}")
            tutor_turns.append(turn)
    if not any(
        _canonicalize_evidence_text(payload.tutor_strategy.key_aha_question)
        in _canonicalize_evidence_text(turn.text)
        for turn in tutor_turns
    ):
        raise ValueError(
            "导师 key_aha_question 未匹配其 source_turn_ids 对应的导师原文 "
            f"(Grounding Gate Failed): {payload.tutor_strategy.key_aha_question!r}"
        )

    misc_data = payload.misconception.model_dump()
    if semantic_only:
        # Compatibility values are complete role sets until the external
        # evidence index replaces them with its logical-chain binding.
        misc_data["source_turn_ids"] = [turn.turn_id for turn in all_student_turns]
    misc_data.update(
        session_id=session.intervention_id,
        question_id=session.question_id,
        subject_path=session.subjects.paths[0] if session.subjects.paths else "",
    )
    strategy_data = payload.tutor_strategy.model_dump()
    if semantic_only:
        strategy_data["source_turn_ids"] = [turn.turn_id for turn in all_tutor_turns]
    strategy_data.update(session_id=session.intervention_id, question_id=session.question_id)
    return ExtractedPIU(
        session_id=session.intervention_id,
        question_id=session.question_id,
        misconception=StudentMisconceptionProfile.model_validate(misc_data),
        tutor_strategy=TutorStrategyProfile.model_validate(strategy_data),
        extraction_status="success",
    )


def extract_knowledge_from_session(
    session: CleanedSession,
    llm_client: Any = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    evidence_audit: bool = False,
    allow_none_on_failure: bool = False,
    semantic_only: bool = False,
) -> ExtractedPIU:
    """
    对单个 CleanedSession 执行知识蒸馏抽取，内置“错误反馈自纠重试循环 (Self-Correction Loop)”。
    
    真实性与补救闭环:
      1. 若模型输出遇到 JSON 语法错误或 Pydantic Schema 校验失败，将精准错误原因追加至对话历史，
         触发模型自我修正 (最多重试 max_retries 次)；
      2. 若重试耗尽或未配置 API Key / 模型 / 依赖，始终抛出明确异常 (Fail-Fast)。
    
    参数:
      session (CleanedSession): 输入会话对象
      llm_client (Any): 可选的 LLM 客户端包装类 (需提供 generate_structured(messages/prompt))
      api_key (Optional[str]): API 密钥 (默认只读取项目契约 LLM_API_KEY)
      base_url (Optional[str]): API 基础端点地址 (默认读取 LLM_BASE_URL)
      model (Optional[str]): 模型名称 (默认读取 LLM_MODEL；不存在隐式默认值)
      temperature (float): 采样温度 (默认 0.1)
      max_retries (int): 校验失败时的最大自纠重试次数 (默认 3 次)
      evidence_audit (bool): 是否在首轮抽取后执行第二遍最小充分证据审计 (默认关闭，需显式开启)
      allow_none_on_failure (bool): 已废弃兼容参数；任何取值都始终 Fail-Fast
      semantic_only (bool): 只抽取卡片语义；忽略 LLM source_turn_ids，由外部 evidence index 绑定证据
      
    返回:
      ExtractedPIU: 仅返回两张卡均通过严格 schema 与 grounding 的真实成功对象
    """
    if max_retries < 1:
        raise ValueError("max_retries 必须至少为 1")
    if allow_none_on_failure:
        logger.warning("allow_none_on_failure 已废弃；真实性门禁始终 Fail-Fast")

    prompt = build_extraction_prompt(session, semantic_only=semantic_only)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are an expert pedagogical knowledge extractor. Output structured JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    effective_model = model or os.environ.get("LLM_MODEL")
    client = llm_client
    if client is None:
        effective_api_key = api_key or os.environ.get("LLM_API_KEY")
        effective_base_url = base_url or os.environ.get("LLM_BASE_URL")
        if not effective_api_key:
            raise LLMUnavailableError(
                f"未检测到有效的 LLM API Key (Session {session.intervention_id})；"
                "请配置 LLM_API_KEY 或显式传入 api_key，严禁伪造执行。"
            )
        if not effective_model:
            raise LLMUnavailableError(
                f"未配置 LLM_MODEL (Session {session.intervention_id})；禁止使用隐式模型默认值。"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailableError("缺少 openai 依赖，无法执行真实 LLM 抽取。") from exc
        kwargs = {"api_key": effective_api_key}
        if effective_base_url:
            kwargs["base_url"] = effective_base_url
        client = OpenAI(**kwargs)

    last_error: Optional[Exception] = None
    last_raw: Any = {}
    for attempt in range(1, max_retries + 1):
        try:
            last_raw = _call_structured_once(
                client,
                effective_model=effective_model,
                messages=messages,
                prompt=prompt,
                temperature=temperature,
            )
            parsed = _parse_and_ground_payload(last_raw, session, semantic_only=semantic_only)
            if evidence_audit:
                return _run_evidence_audit(
                    extracted=parsed,
                    session=session,
                    client=client,
                    effective_model=effective_model,
                    temperature=temperature,
                    max_retries=max_retries,
                )
            return parsed
        except (LLMUnavailableError, LLMExtractionError):
            raise
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning(
                    "Session %s 抽取校验失败 (尝试 %s/%s): %s；正在请求模型自纠",
                    session.intervention_id,
                    attempt,
                    max_retries,
                    exc,
                )
                serialized = last_raw if isinstance(last_raw, str) else json.dumps(last_raw, ensure_ascii=False)
                messages.extend(
                    [
                        {"role": "assistant", "content": serialized},
                        {
                            "role": "user",
                            "content": f"上次输出校验失败: {exc}。请按完整 schema 修正后重新输出。",
                        },
                    ]
                )

    raise LLMExtractionError(
        f"LLM 在 {max_retries} 次自纠尝试后仍然失败 (Session {session.intervention_id}): {last_error}"
    ) from last_error


def extract_semantic_cards_from_session(
    session: CleanedSession,
    llm_client: Any = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> ExtractedPIU:
    """Explicit semantic-only extraction entry point.

    The caller must bind evidence through ``src.evidence_index`` after this
    function returns; model-selected source pointers are never authoritative.
    """
    return extract_knowledge_from_session(
        session,
        llm_client=llm_client,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        semantic_only=True,
    )


def batch_extract_knowledge(
    input_path: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
    llm_client: Any = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    allow_skip_on_error: bool = False
) -> List[ExtractedPIU]:
    """
    批量从 CleanedSession JSONL 中抽取教学知识，并持久化输出为 ExtractedPIU JSONL 文件。
    
    参数:
      input_path (Path): cleaned_sessions.jsonl 输入路径
      output_path (Path): extracted_pius.jsonl 输出保存路径
      max_samples (Optional[int]): 最大处理会话数
      llm_client (Any): 可选的自定义 LLM 客户端
      api_key (Optional[str]): API 密钥
      base_url (Optional[str]): API Base URL
      model (Optional[str]): 模型名称
      max_retries (int): 失败自纠重试上限
      allow_skip_on_error (bool): 已废弃；传 True 会显式拒绝，批量任务必须原子失败
      
    返回:
      List[ExtractedPIU]: 真实抽取成功的对象列表
    """
    if allow_skip_on_error:
        raise ValueError("allow_skip_on_error 已禁用：批量抽取必须原子失败，不能跳过坏数据后伪装完整")
    extracted_results: List[ExtractedPIU] = []
    
    with open(input_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if max_samples and idx >= max_samples:
                break
            line_str = line.strip()
            if not line_str:
                continue
            session_data = json.loads(line_str)
            session = CleanedSession.model_validate(session_data)
            
            extracted = extract_knowledge_from_session(
                session,
                llm_client=llm_client,
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_retries=max_retries,
                allow_none_on_failure=False
            )
            if extracted.extraction_status != "success" or extracted.misconception is None or extracted.tutor_strategy is None:
                raise LLMExtractionError(
                    f"Session {session.intervention_id} 未生成两张真实成功卡，拒绝批量产出"
                )
            extracted_results.append(extracted)

    if not extracted_results:
        raise ValueError("输入中没有可抽取的会话，拒绝覆盖输出为空结果")
            
    # 全部成功后才原子替换输出；任一失败时保留旧文件且不留下部分 JSONL。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fout:
            temp_name = fout.name
            for item in extracted_results:
                fout.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
        os.replace(temp_name, output_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
            
    return extracted_results


if __name__ == "__main__":
    if load_dotenv is not None:
        load_dotenv()
    parser = argparse.ArgumentParser(description="Eedi-RAG LLM 教学知识结构化蒸馏抽取脚本 (严格 Fail-Fast)")
    parser.add_argument("--input", type=str, default="data/sample/cleaned_sessions_sample.jsonl", help="输入 CleanedSession JSONL 路径")
    parser.add_argument("--output", type=str, default="data/sample/extracted_pius_sample.jsonl", help="输出 ExtractedPIU JSONL 路径")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API Key (也可在 .env 中设置 LLM_API_KEY)")
    parser.add_argument("--base-url", type=str, default=None, help="LLM Base URL (例如 https://api.deepseek.com)")
    parser.add_argument("--model", type=str, default=None, help="模型名称（必填，或设置 LLM_MODEL）")
    parser.add_argument("--max-retries", type=int, default=3, help="校验失败时的最大自纠重试次数 (默认 3)")
    parser.add_argument("--max-samples", type=int, default=None, help="最大抽样处理会话数")
    
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    print(f"🚀 开始执行知识蒸馏抽取: {in_path} -> {out_path} (最大自纠重试: {args.max_retries})")
    try:
        results = batch_extract_knowledge(
            input_path=in_path,
            output_path=out_path,
            max_samples=args.max_samples,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            max_retries=args.max_retries,
        )
        print(f"✅ 抽取完成! 真实成功提取: {len(results)} 场会话的知识卡片。")
    except Exception as err:
        print(f"❌ 知识蒸馏执行中断 (Fail-Fast): {err}", file=sys.stderr)
        sys.exit(1)
