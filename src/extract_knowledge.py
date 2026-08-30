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
  6. 严格遵守《反假可用与工程真实性守则》：重试耗尽后显式拒绝执行 (Fail-Fast 抛出 LLMExtractionError)
     或返回受检 None 触发上层调度器切换为纯规则滑动窗口切块，绝不编写或返回任何形式的假数据！
================================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

# 自动载入 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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


class LLMUnavailableError(RuntimeError):
    """当未配置有效的 LLM API Key 或外部模型服务不可用时抛出 (Fail-Fast)。"""
    pass


class LLMExtractionError(RuntimeError):
    """当大模型抽取过程异常（如网络中断、响应格式非法、自纠重试耗尽仍然校验失败）时抛出。"""
    pass


def build_extraction_prompt(session: CleanedSession) -> str:
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
        
    cleaned_quote = quote.strip().lower()
    for t in turns:
        if cleaned_quote in t.text.lower():
            return True, t.turn_id
            
    return False, None


def extract_knowledge_from_session(
    session: CleanedSession,
    llm_client: Any = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    allow_none_on_failure: bool = False
) -> Optional[ExtractedPIU]:
    """
    对单个 CleanedSession 执行知识蒸馏抽取，内置“错误反馈自纠重试循环 (Self-Correction Loop)”。
    
    真实性与补救闭环:
      1. 若模型输出遇到 JSON 语法错误或 Pydantic Schema 校验失败，将精准错误原因追加至对话历史，
         触发模型自我修正 (最多重试 max_retries 次)；
      2. 若重试耗尽或未配置 API Key，按 allow_none_on_failure 决定抛出异常 (Fail-Fast)
         或返回受检 None (触发调度层切换纯规则滑动窗口)，绝不编造任何假卡片。
    
    参数:
      session (CleanedSession): 输入会话对象
      llm_client (Any): 可选的 LLM 客户端包装类 (需提供 generate_structured(messages/prompt))
      api_key (Optional[str]): API 密钥 (默认读取 LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY)
      base_url (Optional[str]): API 基础端点地址 (默认读取 LLM_BASE_URL / OPENAI_BASE_URL)
      model (Optional[str]): 模型名称 (默认读取 LLM_MODEL，缺省为 'gpt-4o-mini')
      temperature (float): 采样温度 (默认 0.1)
      max_retries (int): 校验失败时的最大自纠重试次数 (默认 3 次)
      allow_none_on_failure (bool): 为 True 时失败返回 None；为 False 时失败直接抛出异常
      
    返回:
      Optional[ExtractedPIU]: 真实抽取校验成功的教学资产对象，或在允许软失败时返回 None
    """
    prompt = build_extraction_prompt(session)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are an expert pedagogical knowledge extractor. Output structured JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    # 1. 若提供了自定义 LLM Client (例如单元测试 Mock 或外部包装 Client)
    if llm_client is not None and hasattr(llm_client, "generate_structured"):
        for attempt in range(max_retries):
            try:
                # 兼容支持 messages 或 prompt 传参的 client
                resp = llm_client.generate_structured(messages if hasattr(llm_client, "supports_messages") and llm_client.supports_messages else prompt)
                resp_dict = resp if isinstance(resp, dict) else json.loads(resp)
                
                misc_data = resp_dict.get("misconception", {})
                misc_data["session_id"] = session.intervention_id
                misc_data["question_id"] = session.question_id
                misc_data["subject_path"] = session.subjects.paths[0] if session.subjects.paths else ""
                
                strat_data = resp_dict.get("tutor_strategy", {})
                strat_data["session_id"] = session.intervention_id
                strat_data["question_id"] = session.question_id
                
                misconception_card = StudentMisconceptionProfile.model_validate(misc_data)
                tutor_strategy_card = TutorStrategyProfile.model_validate(strat_data)
                
                # Grounding Gate: 校验原声引用是否真实存在于对话实录中
                invalid_quotes = [q for q in misconception_card.verbatim_student_quotes if not validate_verbatim_grounding(q, session.turns)[0]]
                if invalid_quotes:
                    raise ValueError(
                        f"原声引用未通过字面量真实性门禁 (Grounding Gate Failed): {invalid_quotes} 在会话实录中不存在精确子串匹配！"
                        f"严禁杜撰或美化学生原声，必须摘录会话实录中学生实际发言的字面量子串。"
                    )
                
                return ExtractedPIU(
                    session_id=session.intervention_id,
                    question_id=session.question_id,
                    misconception=misconception_card,
                    tutor_strategy=tutor_strategy_card,
                    extraction_status="success"
                )
            except Exception as e:
                err_msg = str(e)
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Session {session.intervention_id} 自定义 Client 抽取校验失败 (尝试 {attempt+1}/{max_retries}): {err_msg}。正在重试..."
                    )
                    messages.append({"role": "assistant", "content": json.dumps(resp_dict if 'resp_dict' in locals() else {}, ensure_ascii=False)})
                    messages.append({
                        "role": "user",
                        "content": f"上次输出校验失败: {err_msg}。请严格按照 Schema 修正错误并重新输出完整合法 JSON。"
                    })
                else:
                    final_msg = f"自定义 Client 在 {max_retries} 次自纠尝试后仍然失败 (Session {session.intervention_id}): {err_msg}"
                    if allow_none_on_failure:
                        logger.warning(final_msg)
                        return None
                    raise LLMExtractionError(final_msg) from e
                    
    # 2. 检查环境变量或入参中的 API 配置
    effective_api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    effective_base_url = base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
    effective_model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    
    if not effective_api_key:
        msg = (
            f"未检测到有效的 LLM API Key (Session {session.intervention_id})。"
            f"根据工程真实性守则，知识蒸馏层严禁伪造执行。"
            f"请在 .env 中配置 LLM_API_KEY，或通过 CLI 参数 --api-key 传入。"
        )
        if allow_none_on_failure:
            logger.warning(msg)
            return None
        raise LLMUnavailableError(msg)
        
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": effective_api_key}
        if effective_base_url:
            client_kwargs["base_url"] = effective_base_url
            
        client = OpenAI(**client_kwargs)
        
        # 3. 带自纠反馈的多轮调用重试循环 (Self-Correction Loop)
        for attempt in range(max_retries):
            raw_content = "{}"
            try:
                response = client.chat.completions.create(
                    model=effective_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature
                )
                raw_content = response.choices[0].message.content or "{}"
                resp_dict = json.loads(raw_content)
                
                misc_data = resp_dict.get("misconception", {})
                misc_data["session_id"] = session.intervention_id
                misc_data["question_id"] = session.question_id
                misc_data["subject_path"] = session.subjects.paths[0] if session.subjects.paths else ""
                
                strat_data = resp_dict.get("tutor_strategy", {})
                strat_data["session_id"] = session.intervention_id
                strat_data["question_id"] = session.question_id
                
                # 严格 Schema 校验
                misconception_card = StudentMisconceptionProfile.model_validate(misc_data)
                tutor_strategy_card = TutorStrategyProfile.model_validate(strat_data)
                
                # Grounding Gate: 校验原声引用是否真实存在于对话实录中
                invalid_quotes = [q for q in misconception_card.verbatim_student_quotes if not validate_verbatim_grounding(q, session.turns)[0]]
                if invalid_quotes:
                    raise ValueError(
                        f"原声引用未通过字面量真实性门禁 (Grounding Gate Failed): {invalid_quotes} 在会话实录中不存在精确子串匹配！"
                        f"严禁杜撰或美化学生原声，必须摘录会话实录中学生实际发言的字面量子串。"
                    )
                
                return ExtractedPIU(
                    session_id=session.intervention_id,
                    question_id=session.question_id,
                    misconception=misconception_card,
                    tutor_strategy=tutor_strategy_card,
                    extraction_status="success"
                )
            except (ValidationError, json.JSONDecodeError, KeyError, Exception) as e:
                err_msg = str(e)
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Session {session.intervention_id} 抽取校验失败 (尝试 {attempt+1}/{max_retries}): {err_msg}。正在反馈错误给大模型进行自纠重试..."
                    )
                    # 将失败的响应与具体错误反馈追加至对话历史，触发模型自纠
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"你的上一轮输出无法通过严格的 Pydantic 数据契约校验，错误明细如下:\n{err_msg}\n"
                            f"请务必修正上述字段缺失或格式错误，严格按照规定的字段名称和数据类型，重新输出完整的合法 JSON。"
                        )
                    })
                else:
                    final_msg = f"大模型在 {max_retries} 次反馈自纠尝试后仍然抽取校验失败 (Session {session.intervention_id}): {err_msg}"
                    if allow_none_on_failure:
                        logger.error(final_msg)
                        return None
                    raise LLMExtractionError(final_msg) from e
                    
    except LLMUnavailableError:
        raise
    except LLMExtractionError:
        raise
    except Exception as general_err:
        msg = f"大模型 API 网络或运行时严重异常 ({effective_model} @ {effective_base_url or 'default'}): {general_err}"
        if allow_none_on_failure:
            logger.error(msg)
            return None
        raise LLMExtractionError(msg) from general_err


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
      allow_skip_on_error (bool): 为 True 时跳过抽取失败的会话；为 False 时遇错立即中断 (Fail-Fast)
      
    返回:
      List[ExtractedPIU]: 真实抽取成功的对象列表
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_results = []
    
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
                allow_none_on_failure=allow_skip_on_error
            )
            if extracted is not None:
                extracted_results.append(extracted)
            
    # 写入输出 JSONL (只写入真实成功的卡片)
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in extracted_results:
            fout.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
            
    return extracted_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eedi-RAG LLM 教学知识结构化蒸馏抽取脚本 (带自纠补救与真实降级)")
    parser.add_argument("--input", type=str, default="data/sample/cleaned_sessions_sample.jsonl", help="输入 CleanedSession JSONL 路径")
    parser.add_argument("--output", type=str, default="data/sample/extracted_pius_sample.jsonl", help="输出 ExtractedPIU JSONL 路径")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API Key (也可在 .env 中设置 LLM_API_KEY)")
    parser.add_argument("--base-url", type=str, default=None, help="LLM Base URL (例如 https://api.deepseek.com)")
    parser.add_argument("--model", type=str, default=None, help="模型名称 (例如 deepseek-chat 或 gpt-4o-mini)")
    parser.add_argument("--max-retries", type=int, default=3, help="校验失败时的最大自纠重试次数 (默认 3)")
    parser.add_argument("--max-samples", type=int, default=None, help="最大抽样处理会话数")
    parser.add_argument("--allow-skip", action="store_true", help="是否允许跳过失败/无Key的单条会话")
    
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
            allow_skip_on_error=args.allow_skip
        )
        print(f"✅ 抽取完成! 真实成功提取: {len(results)} 场会话的知识卡片。")
    except Exception as err:
        print(f"❌ 知识蒸馏执行中断 (Fail-Fast): {err}", file=sys.stderr)
        sys.exit(1)
