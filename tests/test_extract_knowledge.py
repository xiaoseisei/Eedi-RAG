"""
================================================================================
测试模块: tests/test_extract_knowledge.py
测试定位: 验证 Step 2 教学知识卡片结构化抽取引擎 (src/extract_knowledge.py)
测试覆盖:
  1. Prompt 模板构造与上下文拼接 (build_extraction_prompt)
  2. 事实引证子串校验与防幻觉修正 (validate_verbatim_grounding)
  3. 无 LLM Key 时严格拒绝执行与 Fail-Fast 异常抛出 (LLMUnavailableError)
  4. 允许软降级时返回 None 并记录警告日志
  5. 注入 Mock LLM Client 时的结构化输出解析与 Pydantic 模型强类型校验
  6. 错误反馈驱动的自纠补救重试机制 (Self-Correction Retry Loop on ValidationError)
  7. 自纠重试耗尽后的 Fail-Fast 异常抛出与软降级行为
  8. 批量抽取流水线在 Mock LLM 下的端到端执行与持久化
================================================================================
"""

import os
import sys
import json
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    CleanedSession,
    StudentMisconceptionProfile,
    TutorStrategyProfile,
    ExtractedPIU
)
from src.extract_knowledge import (
    build_extraction_prompt,
    validate_verbatim_grounding,
    extract_knowledge_from_session,
    batch_extract_knowledge,
    LLMUnavailableError,
    LLMExtractionError
)


@pytest.fixture
def sample_session() -> CleanedSession:
    """载入一个真实清洗后的 CleanedSession 样本用于测试。"""
    sample_file = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    with open(sample_file, "r", encoding="utf-8") as f:
        line = f.readline()
        data = json.loads(line)
        return CleanedSession.model_validate(data)


@pytest.fixture
def mock_llm_response() -> dict:
    """标准测试 Mock LLM 结构化输出字典。"""
    return {
        "misconception": {
            "misconception_name": "四舍五入到一位小数时保留两位",
            "error_choice": "Sophie",
            "deep_mechanism": "学生误认为保留一位小数等于取小数点后两位数，未理解一位小数定义。",
            "confusion_triggers": ["5.45", "1 decimal place"],
            "verbatim_student_quotes": ["5.45"],
            "source_turn_ids": [17]
        },
        "tutor_strategy": {
            "pedagogical_goal": "引导学生明确一位小数的定义并观察第二位数字决定进位",
            "strategy_category": "Socratic_Questioning",
            "key_aha_question": "what do you think 5.4598 rounds to to 1dp?",
            "scaffolding_steps": ["Step 1: 圈出第一位小数", "Step 2: 判断第二位小数是舍还是入"],
            "analogy_or_metaphor": None,
            "talk_moves": ["<Press for Accuracy>"],
            "resolution_outcome": "学生自主推导并回答出正确答案 5.5",
            "source_turn_ids": [16, 18]
        }
    }


class MockTestLLMClient:
    def __init__(self, response_dict: dict):
        self.response_dict = response_dict
        
    def generate_structured(self, prompt: Any) -> dict:
        resp = json.loads(json.dumps(self.response_dict))
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        # 根据不同会话的 Prompt 文本动态提供真实存在的子串引用
        if "54+58" in prompt_str:
            resp["misconception"]["verbatim_student_quotes"] = ["the value of 54+58/2"]
        elif "Alex and Sophie" in prompt_str:
            resp["misconception"]["verbatim_student_quotes"] = ["5.45"]
        else:
            import re
            m = re.search(r"\[STUDENT\][^:]*:\s*([^\n]+)", prompt_str)
            if m:
                resp["misconception"]["verbatim_student_quotes"] = [m.group(1)[:15].strip()]
        return resp


class MockSelfCorrectingLLMClient:
    """
    模拟大模型自纠行为的测试客户端:
    - 第 1 次调用: 返回缺少必填字段的残缺 JSON (触发 Pydantic ValidationError)
    - 第 2 次调用 (收到反馈后): 返回完整合法的 JSON (自纠成功)
    """
    def __init__(self, valid_response: dict):
        self.valid_response = valid_response
        self.call_count = 0
        
    def generate_structured(self, prompt_or_messages: Any) -> dict:
        self.call_count += 1
        if self.call_count == 1:
            # 故意漏掉 tutor_strategy 中的 key_aha_question 和 scaffolding_steps
            return {
                "misconception": {
                    "misconception_name": "残缺错因",
                    "deep_mechanism": "...",
                    "verbatim_student_quotes": ["5.45"],
                    "source_turn_ids": [17]
                },
                "tutor_strategy": {
                    "pedagogical_goal": "残缺目标",
                    "strategy_category": "Socratic_Questioning",
                    "resolution_outcome": "...",
                    "source_turn_ids": [16]
                    # 缺少必填的 key_aha_question 字段
                }
            }
        # 第 2 次返回正确数据
        return self.valid_response


class MockAlwaysFailingLLMClient:
    """模拟无论怎么反馈重试都始终输出非法数据的测试客户端。"""
    def __init__(self):
        self.call_count = 0
        
    def generate_structured(self, prompt_or_messages: Any) -> dict:
        self.call_count += 1
        return {"invalid_key": "totally_broken_data"}


def test_build_extraction_prompt(sample_session: CleanedSession):
    """测试 Prompt 模板生成是否包含学科树、题干、选项以及编号对话轮次。"""
    prompt = build_extraction_prompt(sample_session)
    
    assert f"会话ID: {sample_session.intervention_id}" in prompt
    assert f"题目ID: {sample_session.question_id}" in prompt
    assert "【学科考纲层级路径】" in prompt
    assert "【考题原题与选项分布】" in prompt
    assert "【师生时序对话实录】" in prompt
    assert "[Turn 1]" in prompt
    assert "[TUTOR]" in prompt or "[STUDENT]" in prompt


def test_validate_verbatim_grounding(sample_session: CleanedSession):
    """测试原声字面量子串校验门禁。"""
    # 构造一个合法的引用 (必须是真实 Turn 的子串)
    first_student_turn = next(t for t in sample_session.turns if not t.is_tutor)
    valid_quote = first_student_turn.text[:10]  # 前10个字符
    
    # 构造一个非法幻觉引用
    hallucinated_quote = "This is a completely made up hallucination from LLM"
    
    # 验证合法引用通过
    is_valid, matched_turn = validate_verbatim_grounding(valid_quote, sample_session.turns)
    assert is_valid is True
    assert matched_turn == first_student_turn.turn_id
    
    # 验证幻觉引用被拦截
    is_valid_fake, _ = validate_verbatim_grounding(hallucinated_quote, sample_session.turns)
    assert is_valid_fake is False


def test_extract_knowledge_fails_fast_when_no_llm(sample_session: CleanedSession, monkeypatch):
    """测试当未配置 LLM API Key 且未传入 Mock Client 时，严格抛出 LLMUnavailableError (Fail-Fast)。"""
    # 确保环境变量清空
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    
    with pytest.raises(LLMUnavailableError) as exc_info:
        extract_knowledge_from_session(sample_session, llm_client=None, api_key=None, allow_none_on_failure=False)
        
    assert "未检测到有效的 LLM API Key" in str(exc_info.value)
    assert "严禁伪造执行" in str(exc_info.value)


def test_extract_knowledge_returns_none_when_allowed(sample_session: CleanedSession, monkeypatch):
    """测试当允许软失败 (allow_none_on_failure=True) 时，无 Key 返回 None 而非抛出或伪造数据。"""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    
    result = extract_knowledge_from_session(sample_session, llm_client=None, api_key=None, allow_none_on_failure=True)
    assert result is None


def test_extract_knowledge_with_mock_llm(sample_session: CleanedSession, mock_llm_response: dict):
    """测试当注入 Mock LLM Client 时，抽取器能否正确解析 JSON 并做模型校验。"""
    client = MockTestLLMClient(mock_llm_response)
    extracted = extract_knowledge_from_session(sample_session, llm_client=client)
    
    assert extracted is not None
    assert extracted.extraction_status == "success"
    assert extracted.misconception.misconception_name == "四舍五入到一位小数时保留两位"
    assert extracted.tutor_strategy.strategy_category == "Socratic_Questioning"
    assert "what do you think" in extracted.tutor_strategy.key_aha_question


def test_extract_knowledge_self_correction_remediation(sample_session: CleanedSession, mock_llm_response: dict):
    """测试自纠补救机制：第1次输出残缺触发重试，第2次自纠成功后产出合法卡片。"""
    client = MockSelfCorrectingLLMClient(mock_llm_response)
    extracted = extract_knowledge_from_session(sample_session, llm_client=client, max_retries=3)
    
    assert client.call_count == 2  # 验证确实发生了第 2 次自纠重试
    assert extracted is not None
    assert extracted.extraction_status == "success"
    assert extracted.tutor_strategy.key_aha_question == "what do you think 5.4598 rounds to to 1dp?"


def test_extract_knowledge_self_correction_exhausted(sample_session: CleanedSession):
    """测试自纠重试耗尽后的真降级与 Fail-Fast：重试3次仍失败时抛出 LLMExtractionError。"""
    failing_client = MockAlwaysFailingLLMClient()
    
    # 1. 严格模式 ➔ 抛出 LLMExtractionError
    with pytest.raises(LLMExtractionError) as exc_info:
        extract_knowledge_from_session(sample_session, llm_client=failing_client, max_retries=3, allow_none_on_failure=False)
        
    assert failing_client.call_count == 3  # 验证重试了满 3 次
    assert "3 次自纠尝试后仍然失败" in str(exc_info.value)
    
    # 2. 软降级模式 ➔ 返回 None (供调度层切换纯规则滑动窗口)
    failing_client_2 = MockAlwaysFailingLLMClient()
    res = extract_knowledge_from_session(sample_session, llm_client=failing_client_2, max_retries=3, allow_none_on_failure=True)
    assert failing_client_2.call_count == 3
    assert res is None


def test_extract_knowledge_grounding_gate_rejects_hallucinated_quotes(sample_session: CleanedSession, mock_llm_response: dict):
    """测试 Grounding Gate：若大模型编造了不存在的原声引用，抽取器必须拦截并触发自纠/失败。"""
    # 构造带有编造幻觉原声的响应
    bad_response = json.loads(json.dumps(mock_llm_response))
    bad_response["misconception"]["verbatim_student_quotes"] = [
        "This is an absolute hallucination quote that never appeared in turns"
    ]
    
    class MockHallucinatingLLMClient:
        def generate_structured(self, prompt: Any) -> dict:
            return bad_response
            
    hallucinating_client = MockHallucinatingLLMClient()
    
    # 严格模式下应因 3 次重试失败抛出 LLMExtractionError
    with pytest.raises(LLMExtractionError) as exc_info:
        extract_knowledge_from_session(sample_session, llm_client=hallucinating_client, max_retries=3, allow_none_on_failure=False)
        
    assert "Grounding Gate Failed" in str(exc_info.value)
    
    # 软降级模式下应返回 None
    res = extract_knowledge_from_session(sample_session, llm_client=hallucinating_client, max_retries=3, allow_none_on_failure=True)
    assert res is None


def test_batch_extract_knowledge_with_mock_llm(tmp_path: Path, mock_llm_response: dict):
    """测试批量抽取并持久化至 JSONL 文件。"""
    sample_file = project_root / "data" / "sample" / "cleaned_sessions_sample.jsonl"
    output_file = tmp_path / "extracted_pius_test.jsonl"
    
    client = MockTestLLMClient(mock_llm_response)
    extracted_list = batch_extract_knowledge(
        input_path=sample_file,
        output_path=output_file,
        max_samples=3,
        llm_client=client
    )
    
    assert len(extracted_list) == 3
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 3
        data = json.loads(lines[0])
        assert data["extraction_status"] == "success"
        assert "misconception" in data
        assert "tutor_strategy" in data

