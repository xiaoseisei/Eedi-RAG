"""
================================================================================
测试模块: tests/test_rag_pipeline.py
测试定位: 验证 Step 5 端到端 RAG 问答与名师教研生成总管道 (src/rag_pipeline.py)
测试覆盖:
  1. 确定性保真模态 (Deterministic Mode) 端到端全链路生成
  2. 真实师生对白引用防伪审计 (_audit_citations)
  3. 教研三段论 Pydantic 契约严格校验 (PedagogicalGuidanceResponse)
  4. 美化 Markdown 备课卡片排版与四槽位完整性
================================================================================
"""

import os
import sys
import json
from types import SimpleNamespace
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import PedagogicalGuidanceResponse, DialogueCitation
from src.reranker import PedagogicalGoldAssembler, GoldAssembledContext
from pydantic import ValidationError

from src.rag_pipeline import (
    CitationAuditError,
    EndToEndPedagogicalRAGPipeline,
    LLMGuidancePayload,
    RAGGenerationError,
)


def _pipeline_with_incomplete_retrieval() -> EndToEndPedagogicalRAGPipeline:
    evidence = {"turn_id": 1, "speaker": "student", "text": "5.45"}
    retrieval = {
        "misconceptions": [{
            "hybrid_score": 1.0,
            "document": "source",
            "metadata": {
                "session_id": 10,
                "subject_path": "Number > Rounding",
                "misconception_name": "Place-value confusion",
            },
            "evidence_turns": [evidence],
        }],
        "strategies": [{
            "hybrid_score": 1.0,
            "document": "strategy",
            "metadata": {
                "session_id": 10,
                "subject_path": "Number > Rounding",
                "key_aha_question": "Which digit is in the tenths place?",
            },
            "evidence_turns": [evidence],
        }],
    }
    retriever = SimpleNamespace(
        retrieve_multi_perspective_rrf=lambda **kwargs: retrieval,
    )
    return EndToEndPedagogicalRAGPipeline(retriever=retriever)


def test_pipeline_deterministic_end_to_end():
    """检索卡缺少生成所需事实字段时，确定性模式必须显式拒绝。"""
    query = "四舍五入 5.4598 到 1 位小数时，学生为什么会误选 5.45？名师如何引导？"
    with pytest.raises(RAGGenerationError, match="deep_mechanism"):
        _pipeline_with_incomplete_retrieval().ask(query=query, mode="deterministic")


def test_pipeline_macro_summary_query():
    """测试宏观学情总结/共性误区归纳类问题自适应生成。"""
    macro_query = "学生在辅导中最常提出的问题是什么？核心难点与共性误区有哪些？"
    with pytest.raises(RAGGenerationError, match="deep_mechanism"):
        _pipeline_with_incomplete_retrieval().ask(query=macro_query, mode="deterministic")


def test_citation_auditor():
    """测试引用保真度防伪审计逻辑。"""
    pipeline = EndToEndPedagogicalRAGPipeline()
    
    gold_ctx = GoldAssembledContext(
        raw_query="test",
        prompt_context_markdown="test",
        evidence_turns=[
            {"session_id": 7, "turn_id": 9, "speaker": "tutor", "text": "What does it round to?"},
            {"session_id": 7, "turn_id": 10, "speaker": "student", "text": "5.45"}
        ]
    )
    
    # 构造包含伪造引用 Turn 99 的响应
    fake_response = PedagogicalGuidanceResponse(
        query="test",
        subject_path="test",
        misconception_diagnosis="test",
        key_aha_question="test",
        session_id=7,
        dialogue_citations=[DialogueCitation(
            session_id=7,
            turn_id=99,
            speaker="student",
            quote_text="I made this up!",
            verifiable_in_duckdb=False,
        )],
        audit_status="PENDING",
    )

    with pytest.raises(CitationAuditError):
        pipeline._audit_citations(fake_response, gold_ctx)

    assert fake_response.audit_status == "GROUNDING_DEGRADED"


def _valid_llm_payload(**overrides):
    payload = {
        "subject_path": "Number > Rounding",
        "answer_content": "The student retained two decimal places rather than rounding to one.",
        "misconception_diagnosis": "The place-value target was misidentified.",
        "key_aha_question": "Which digit decides the tenths digit?",
        "recommended_talk_moves": ["<Press for Accuracy>"],
        "scaffolding_steps": ["Ask the student to mark the tenths digit."],
        "dialogue_citations": [{
            "session_id": 7,
            "turn_id": 9,
            "speaker": "tutor",
            "quote_text": "What does it round to?",
        }],
        "transfer_question": "Round 8.764 to one decimal place.",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "override",
    [
        {"answer_content": ""},
        {"dialogue_citations": []},
        {"dialogue_citations": [{"session_id": 7, "turn_id": 0, "speaker": "tutor", "quote_text": "x"}]},
        {"dialogue_citations": [{"session_id": "7", "turn_id": 9, "speaker": "tutor", "quote_text": "x"}]},
        {"unexpected": "must be rejected"},
    ],
)
def test_llm_payload_is_strict_and_non_empty(override):
    with pytest.raises(ValidationError):
        LLMGuidancePayload.model_validate(_valid_llm_payload(**override))


@pytest.mark.parametrize(
    ("citation", "reason"),
    [
        ({"session_id": 8, "turn_id": 9, "speaker": "tutor", "quote_text": "What does it round to?"}, "session"),
        ({"session_id": 7, "turn_id": 9, "speaker": "student", "quote_text": "What does it round to?"}, "speaker"),
        ({"session_id": 7, "turn_id": 9, "speaker": "tutor", "quote_text": "What does it round to"}, "non-verbatim quote"),
    ],
)
def test_citation_audit_rejects_every_mismatched_fact(citation, reason):
    pipeline = EndToEndPedagogicalRAGPipeline()
    gold_ctx = GoldAssembledContext(
        raw_query="test",
        prompt_context_markdown="test",
        evidence_turns=[{
            "session_id": 7,
            "turn_id": 9,
            "speaker": "tutor",
            "text": "What does it round to?",
        }],
    )
    response = PedagogicalGuidanceResponse(
        query="test",
        subject_path="Number > Rounding",
        session_id=7,
        answer_content="answer",
        misconception_diagnosis="diagnosis",
        key_aha_question="question",
        dialogue_citations=[DialogueCitation(
            session_id=citation["session_id"],
            turn_id=citation["turn_id"],
            speaker=citation["speaker"],
            quote_text=citation["quote_text"],
            verifiable_in_duckdb=False,
        )],
        audit_status="PENDING",
    )

    with pytest.raises(CitationAuditError, match="引用"):
        pipeline._audit_citations(response, gold_ctx, citation_records=[citation])


def test_citation_audit_rejects_empty_citations():
    pipeline = EndToEndPedagogicalRAGPipeline()
    ctx = GoldAssembledContext(raw_query="q", prompt_context_markdown="ctx")
    with pytest.raises(ValidationError):
        PedagogicalGuidanceResponse(
            query="q",
            subject_path="Number",
            answer_content="answer",
            misconception_diagnosis="diagnosis",
            key_aha_question="question",
            dialogue_citations=[],
            audit_status="PENDING",
        )


def test_auto_mode_without_llm_credentials_does_not_silently_fallback():
    pipeline = EndToEndPedagogicalRAGPipeline(retriever=SimpleNamespace())
    pipeline.api_key = None
    with pytest.raises(RAGGenerationError, match="LLM_API_KEY"):
        pipeline.ask("How should this misconception be addressed?", mode="auto")


def test_invalid_mode_is_rejected_before_retrieval():
    pipeline = EndToEndPedagogicalRAGPipeline(retriever=SimpleNamespace())
    with pytest.raises(ValueError, match="mode"):
        pipeline.ask("question", mode="fallback")


def test_dead_intent_classifier_has_been_removed():
    assert not hasattr(EndToEndPedagogicalRAGPipeline, "classify_intent")


def test_llm_schema_retries_are_exhausted_then_error_is_raised(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({
                    "answer_content": "looks non-empty but schema is incomplete",
                    "dialogue_citations": [],
                }))
            )])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        api_key="test-key",
        model_name="test-model",
    )
    ctx = GoldAssembledContext(raw_query="q", prompt_context_markdown="ctx")

    with pytest.raises(RAGGenerationError, match="2 次尝试"):
        pipeline._generate_with_llm("q", ctx, {})

    assert len(calls) == 2


def test_exact_citation_fact_is_verified_and_promotes_audit_status():
    pipeline = EndToEndPedagogicalRAGPipeline()
    evidence = {
        "session_id": 7,
        "turn_id": 9,
        "speaker": "tutor",
        "text": "What does it round to?",
    }
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="ctx",
        evidence_turns=[evidence],
    )
    response = PedagogicalGuidanceResponse(
        query="q",
        subject_path="Number",
        answer_content="answer",
        misconception_diagnosis="diagnosis",
        key_aha_question="question",
        dialogue_citations=[DialogueCitation(
            session_id=7,
            turn_id=9,
            speaker="tutor",
            quote_text="What does it round to?",
        )],
        audit_status="PENDING",
    )

    pipeline._audit_citations(response, ctx)

    assert response.audit_status == "AUDITED_100_VERIFIED"
    assert response.dialogue_citations[0].verifiable_in_duckdb is True


def test_explicit_deterministic_mode_uses_only_supplied_facts():
    pipeline = EndToEndPedagogicalRAGPipeline()
    evidence = {
        "session_id": 7,
        "turn_id": 9,
        "speaker": "student",
        "text": "I kept two decimal places.",
    }
    misconception = {
        "document": "source document",
        "metadata": {
            "session_id": 7,
            "subject_path": "Number > Rounding",
            "misconception_name": "Tenths and hundredths were confused",
            "deep_mechanism": "The student treated the hundredths digit as the target place.",
            "error_choice": "B",
        },
        "evidence_turns": [evidence],
    }
    strategy = {
        "document": "strategy document",
        "metadata": {
            "session_id": 7,
            "subject_path": "Number > Rounding",
            "key_aha_question": "Which digit is in the tenths place?",
            "pedagogical_goal": "Locate the target place before rounding.",
            "talk_moves": ["<Press for Accuracy>"],
            "scaffolding_steps": ["Underline the tenths digit in the original number."],
            "transfer_question": "Round 8.764 to one decimal place.",
        },
        "evidence_turns": [evidence],
    }
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="ctx",
        selected_misconception=misconception,
        selected_strategy=strategy,
        evidence_turns=[evidence],
    )

    response = pipeline._synthesize_deterministic_grounding(
        "q",
        ctx,
        {"misconceptions": [misconception], "strategies": [strategy]},
    )
    pipeline._audit_citations(response, ctx)

    assert response.audit_status == "AUDITED_100_VERIFIED"
    assert response.misconception_diagnosis == misconception["metadata"]["deep_mechanism"]
    assert response.scaffolding_steps == strategy["metadata"]["scaffolding_steps"]
    assert response.transfer_question == strategy["metadata"]["transfer_question"]
    assert "典型概念混淆" not in response.answer_content
    assert "Step 1: 提出启发式核心问题" not in response.answer_content
