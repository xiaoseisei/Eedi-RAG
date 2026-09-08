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
    SYSTEM_PEDAGOGICAL_PROMPT_V2,
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


def test_citation_auditor_never_trusts_preverified_flag():
    """A caller-provided verification flag must not bypass quadruple auditing."""
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
        misconception_diagnosis="diagnosis",
        key_aha_question="question",
        session_id=7,
        dialogue_citations=[DialogueCitation(
            session_id=7,
            turn_id=99,
            speaker="student",
            quote_text="fabricated",
            verifiable_in_duckdb=True,
        )],
        audit_status="PENDING",
    )

    with pytest.raises(CitationAuditError):
        pipeline._audit_citations(response, gold_ctx)

    assert response.audit_status == "GROUNDING_DEGRADED"
    assert response.dialogue_citations[0].verifiable_in_duckdb is False


def _valid_llm_payload(**overrides):
    payload = {
        "subject_path": "Number > Rounding",
        "answer": "The student retained two decimal places rather than rounding to one.",
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
        {"answer": ""},
        {"dialogue_citations": []},
        {"dialogue_citations": [{"session_id": 7, "turn_id": 0, "speaker": "tutor", "quote_text": "x"}]},
        {"dialogue_citations": [{"session_id": "7", "turn_id": 9, "speaker": "tutor", "quote_text": "x"}]},
        {"unexpected": "must be rejected"},
    ],
)
def test_llm_payload_is_strict_and_non_empty(override):
    with pytest.raises(ValidationError):
        LLMGuidancePayload.model_validate(_valid_llm_payload(**override))


def test_llm_payload_accepts_only_complete_citation_quadruples():
    payload = LLMGuidancePayload.model_validate(_valid_llm_payload())

    assert payload.dialogue_citations[0].model_dump() == {
        "session_id": 7,
        "turn_id": 9,
        "speaker": "tutor",
        "quote_text": "What does it round to?",
    }


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


def test_pipeline_exposes_bounded_llm_timeout() -> None:
    pipeline = EndToEndPedagogicalRAGPipeline(llm_timeout_seconds=60.0)

    assert pipeline.llm_timeout_seconds == 60.0


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


def test_llm_generation_uses_only_final_assembler_evidence(monkeypatch):
    evidence = {
        "session_id": 7,
        "turn_id": 9,
        "speaker": "tutor",
        "text": "What does it round to?",
    }
    content = json.dumps(_valid_llm_payload(), ensure_ascii=False)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                    usage=None,
                )
            )
        )
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    # No storage handle is deliberately exposed: generation must not fetch
    # arbitrary turns by session id after the assembler has fixed the evidence.
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(),
        api_key="test-key",
        model_name="test-model",
    )
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="ctx",
        evidence_turns=[evidence],
    )

    response = pipeline._generate_with_llm("q", ctx, {})
    pipeline._audit_citations(response, ctx)

    assert response.dialogue_citations[0].model_dump(exclude={"verifiable_in_duckdb"}) == {
        "session_id": 7,
        "turn_id": 9,
        "speaker": "tutor",
        "quote_text": "What does it round to?",
    }
    assert response.audit_status == "AUDITED_100_VERIFIED"


def test_generator_v2_uses_evidence_ids_and_materializes_exact_quotes(monkeypatch):
    sent = []
    content = json.dumps(
        {
            "subject_path": "Number > Rounding",
            "answer": "The student confused decimal places.",
            "misconception_diagnosis": "The student treated 1dp as retaining two decimal places.",
            "evidence_explanation": "The dialogue records the student's answer exactly.",
            "key_aha_question": "Which digit decides?",
            "scaffolding_steps": ["Ask the student to identify the target place."],
            "pedagogical_intervention": ["Ask the student to inspect the next digit before rounding."],
            "recommended_talk_moves": ["<Press for Accuracy>"],
            "claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "The student gave 5.45.",
                    "claim_type": "fact",
                    "evidence_ids": ["E001"],
                }
            ],
            "dialogue_citations": [{"evidence_id": "E001"}],
            "transfer_question": None,
        },
        ensure_ascii=False,
    )

    class Completions:
        def create(self, **kwargs):
            sent.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=None,
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(),
        api_key="test-key",
        model_name="test-model",
        generator_contract_version="v2",
    )
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="ctx",
        evidence_turns=[
            {"session_id": 7, "turn_id": 9, "speaker": "tutor", "text": "What does it round to?"}
        ],
    )

    response = pipeline._generate_with_llm("q", ctx, {})

    assert response.dialogue_citations[0].quote_text == "What does it round to?"
    assert response.misconception_diagnosis.startswith("The student treated")
    assert response.key_aha_question == "Which digit decides?"
    assert response.scaffolding_steps == ["Ask the student to identify the target place."]
    assert response.__dict__["generator_contract_version"] == "v2"
    assert response.__dict__["generator_core_answer"] == "The student confused decimal places."
    assert "教学干预" not in response.__dict__["generator_grounding_text"]
    assert "[fact] [E001]" in response.__dict__["generator_grounding_text"]
    assert "Allowed IDs: E001" in sent[0]["messages"][1]["content"]
    assert "Evidence ID 白名单" in sent[0]["messages"][1]["content"]
    assert "quote_text=" not in sent[0]["messages"][1]["content"]


def test_streaming_api_forwards_plain_text_without_json_validation(monkeypatch):
    content = "The student confused decimal places."

    class Stream:
        def __iter__(self):
            midpoint = len(content) // 2
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content[:midpoint]))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content[midpoint:]))]),
            ])

    class Completions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            kwargs_seen.append(kwargs)
            return Stream()
    kwargs_seen = []

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(),
        api_key="test-key",
        model_name="test-model",
        generator_contract_version="v2",
    )
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="ctx",
        evidence_turns=[
            {"session_id": 7, "turn_id": 9, "speaker": "tutor", "text": "What does it round to?"}
        ],
        selected_strategy={"metadata": {"subject_path": "Number"}},
    )

    response_chunks = []
    response = pipeline._generate_with_llm_streaming(
        "q", ctx, {}, on_chunk=response_chunks.append
    )

    assert "".join(response_chunks) == content
    assert response.answer_content == content
    assert response.__dict__["generator_grounding_text"] == content
    assert response.audit_status == "PENDING"
    assert response.__dict__["generation_stream"]["stream_requested"] is True
    assert response.__dict__["generation_stream"]["chunk_count"] == 2
    assert response.__dict__["generation_stream"]["contract_validation_seconds"] == 0.0
    assert "response_format" not in kwargs_seen[0]
    streaming_prompt = kwargs_seen[0]["messages"][0]["content"]
    assert "【最高优先级红线】" in streaming_prompt
    assert "严格输出 JSON" not in streaming_prompt
    assert "直接输出给用户的回答正文" in streaming_prompt


def test_streaming_accepts_text_that_is_not_json(monkeypatch):
    invalid = "This is a valid plain-text answer, even though it is not JSON."
    class Stream:
        def __iter__(self):
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=invalid))])])

    class Completions:
        def create(self, **kwargs):
            assert "response_format" not in kwargs
            return Stream()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(),
        api_key="test-key",
        model_name="test-model",
        generator_contract_version="v2",
    )
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="ctx",
        evidence_turns=[
            {"session_id": 7, "turn_id": 9, "speaker": "tutor", "text": "What does it round to?"}
        ],
        selected_strategy={"metadata": {"subject_path": "Number"}},
    )

    response = pipeline._generate_with_llm_streaming("q", ctx, {})
    assert response.answer_content == invalid


def test_ask_stream_emits_before_final_audit(monkeypatch):
    content = "Direct answer."

    class Stream:
        def __iter__(self):
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]),
            ])

    class Completions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return Stream()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(),
        api_key="test-key",
        model_name="test-model",
        generator_contract_version="v2",
    )
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="ctx",
        evidence_turns=[
            {"session_id": 7, "turn_id": 9, "speaker": "tutor", "text": "Observed evidence."}
        ],
        selected_strategy={"metadata": {"subject_path": "Number"}},
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_context_with_timings",
        lambda query, top_k_each, fetch_evidence: ({}, ctx, {"retrieval": 0.01, "assembly": 0.02}),
    )
    events = []

    def audit(response, context):
        events.append(("audit", response.audit_status))
        response.audit_status = "AUDITED_100_VERIFIED"

    monkeypatch.setattr(pipeline, "_audit_citations", audit)
    chunks = []

    response = pipeline.ask_stream(
        "q", mode="llm", on_chunk=lambda chunk: (chunks.append(chunk), events.append(("chunk", chunk)))
    )

    assert events[0][0] == "chunk"
    assert events[-1] == ("audit", "PENDING")
    assert response.audit_status == "AUDITED_100_VERIFIED"
    assert chunks
    assert "Direct answer." in "".join(chunks)
    assert response._profiling["ttft"] >= 0
    assert response._profiling["stream_complete"] >= response._profiling["ttft"]


def test_generator_v2_rejects_context_over_total_budget(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            raise AssertionError("LLM must not be called for an over-budget context")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(),
        api_key="test-key",
        model_name="test-model",
        generator_contract_version="v2",
    )
    ctx = GoldAssembledContext(
        raw_query="q",
        prompt_context_markdown="x" * 5000,
        evidence_turns=[
            {"session_id": 7, "turn_id": 9, "speaker": "tutor", "text": "What does it round to?"}
        ],
        context_budget_tokens=100,
    )

    with pytest.raises(RAGGenerationError, match="token budget"):
        pipeline._generate_with_llm("q", ctx, {})


def test_generator_v2_prompt_has_red_lines_and_scope_examples():
    prompt = SYSTEM_PEDAGOGICAL_PROMPT_V2

    assert "最高优先级红线" in prompt
    assert "最小充分集合" in prompt
    assert "一句话" in prompt
    assert "正例与反例" in prompt
    assert "不得编造 evidence_id" in prompt
    assert "inference claim 也必须至少绑定一个 evidence_id" in prompt
    assert "学生一定缺乏位值概念" in prompt
    assert "student_evidence_ids=[\"E010\"]" in prompt
    assert "tutor_evidence_ids=[\"E011\"]" in prompt


def test_generator_v2_retries_with_contract_feedback_for_missing_role(monkeypatch):
    responses = []
    base = {
        "subject_path": "Number",
        "answer": "诊断学生并说明导师引导。",
        "misconception_diagnosis": "学生存在相关误区。",
        "evidence_explanation": "事实由证据支持。",
        "key_aha_question": "Which digit decides?",
        "scaffolding_steps": ["Ask the student to identify the next digit."],
        "pedagogical_intervention": ["Use a worked example."],
        "recommended_talk_moves": [],
        "claims": [
            {"claim_id": "C1", "claim_text": "学生给出错误答案。", "claim_type": "fact", "evidence_ids": ["E001"]}
        ],
        "transfer_question": None,
    }
    first = dict(base, dialogue_citations=[{"evidence_id": "E001"}])
    second = dict(
        base,
        student_evidence_ids=["E001"],
        tutor_evidence_ids=["E002"],
        dialogue_citations=[{"evidence_id": "E001"}, {"evidence_id": "E002"}],
    )

    class Completions:
        def create(self, **kwargs):
            responses.append(kwargs)
            content = first if len(responses) == 1 else second
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))],
                usage=None,
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(),
        api_key="test-key",
        model_name="test-model",
        generator_contract_version="v2",
    )
    ctx = GoldAssembledContext(
        raw_query="学生为什么错，导师如何引导？",
        prompt_context_markdown="ctx",
        evidence_turns=[
            {"session_id": 7, "turn_id": 1, "speaker": "student", "text": "I chose A"},
            {"session_id": 7, "turn_id": 2, "speaker": "tutor", "text": "Which digit decides?"},
        ],
    )

    response = pipeline._generate_with_llm(ctx.raw_query, ctx, {})

    assert len(responses) == 2
    assert "契约修复反馈" in responses[1]["messages"][1]["content"]
    assert len(response.dialogue_citations) == 2
    assert response.__dict__["generator_contract_repair_attempted"] is True


def test_generator_v2_auto_closes_explicit_role_ids_in_citations(monkeypatch):
    content = {
        "subject_path": "Number",
        "answer": "直接回答。",
        "misconception_diagnosis": "诊断。",
        "evidence_explanation": "证据。",
        "key_aha_question": "问题。",
        "scaffolding_steps": ["步骤"],
        "pedagogical_intervention": ["动作"],
        "recommended_talk_moves": [],
        "claims": [
            {"claim_id": "C1", "claim_text": "事实", "claim_type": "fact", "evidence_ids": ["E001"]}
        ],
        "student_evidence_ids": ["E001"],
        "tutor_evidence_ids": ["E002"],
        "dialogue_citations": [{"evidence_id": "E001"}],
        "transfer_question": None,
    }

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))],
                usage=None,
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: fake_client))
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=SimpleNamespace(), api_key="test-key", model_name="test-model", generator_contract_version="v2"
    )
    ctx = GoldAssembledContext(
        raw_query="学生为什么错，导师如何引导？",
        prompt_context_markdown="ctx",
        evidence_turns=[
            {"session_id": 7, "turn_id": 1, "speaker": "student", "text": "student fact"},
            {"session_id": 7, "turn_id": 2, "speaker": "tutor", "text": "tutor fact"},
        ],
    )

    response = pipeline._generate_with_llm(ctx.raw_query, ctx, {})

    assert [citation.turn_id for citation in response.dialogue_citations] == [1, 2]
    assert response.__dict__["generator_contract_auto_merged_ids"] == ["E002"]


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


def test_card_logical_evidence_path_expands_and_audits_ranked_context():
    evidence = {"session_id": 10, "turn_id": 2, "speaker": "student", "text": "I chose 5.45"}
    misconception = {
        "chunk_id": "session_10_misconception",
        "hybrid_score": 1.0,
        "document": "card",
        "metadata": {
            "session_id": 10,
            "subject_path": "Number > Rounding",
            "misconception_name": "Place-value confusion",
            "deep_mechanism": "The target place is confused.",
            "error_choice": "B",
        },
        "evidence_turns": [evidence],
    }
    strategy = {
        "chunk_id": "session_10_tutor_strategy",
        "hybrid_score": 1.0,
        "document": "strategy",
        "metadata": {
            "session_id": 10,
            "subject_path": "Number > Rounding",
            "key_aha_question": "Which digit decides?",
            "pedagogical_goal": "Locate the target place.",
            "talk_moves": [],
            "scaffolding_steps": ["Check the next digit."],
        },
        "evidence_turns": [{"session_id": 10, "turn_id": 1, "speaker": "tutor", "text": "Which digit decides?"}],
    }

    class Reranker:
        def name(self):
            return "fake"

        def rerank(self, query, documents, *, top_n):
            from src.reranker_provider import RerankResult

            return [RerankResult(index=1, relevance_score=0.9, document=documents[1]), RerankResult(index=0, relevance_score=0.1, document=documents[0])]

    class CardRetriever:
        retrieval_mode = "bm25_dense"

        def __init__(self):
            self.expanded = False

        def retrieve_multi_perspective_rrf(self, *, raw_query, top_k_each, fetch_evidence):
            return {"chunk_strategy": "card", "misconceptions": [misconception], "strategies": [strategy], "rewritten_queries": {"extracted_keywords": []}}

        def expand_card_candidates_to_evidence_units(self, candidates, *, window_size, step):
            self.expanded = True
            return [
                {"chunk_id": "low", "document": "low chain", "metadata": {"session_id": 10, "window_start_turn": 1, "window_end_turn": 2}, "evidence_turns": [evidence]},
                {"chunk_id": "high", "document": "high chain", "metadata": {"session_id": 10, "window_start_turn": 3, "window_end_turn": 4}, "evidence_turns": [{"session_id": 10, "turn_id": 3, "speaker": "tutor", "text": "Which digit decides?"}]},
            ]

    retriever = CardRetriever()
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=retriever,
        assembler=PedagogicalGoldAssembler(model_reranker=Reranker(), rerank_unit="logical_evidence", evidence_selection_count=1),
        chunk_strategy="card",
    )
    response = pipeline.ask("学生为什么选 5.45？", mode="deterministic")
    assert retriever.expanded is True
    assert response.audit_status == "AUDITED_100_VERIFIED"
    assert [citation.turn_id for citation in response.dialogue_citations] == [3]


def test_fallback_chunk_strategy_uses_real_windows_as_evidence():
    evidence = {"turn_id": 2, "speaker": "student", "text": "I chose 5.45"}
    retrieval = {
        "chunk_strategy": "fallback",
        "windows": [{
            "chunk_id": "session_10_win_1_6",
            "hybrid_score": 0.9,
            "bm25_score": 1.0,
            "document": "[Turn 1] [Tutor]: What does it round to?\n[Turn 2] [Student]: I chose 5.45",
            "metadata": {
                "session_id": 10,
                "subject_path": "Number > Rounding",
                "window_start_turn": 1,
                "window_end_turn": 6,
                "source_turn_ids": "[1, 2]",
            },
            "evidence_turns": [evidence],
        }],
        "misconceptions": [],
        "strategies": [],
        "rewritten_queries": {"extracted_keywords": []},
    }

    class FallbackRetriever:
        retrieval_mode = "bm25_dense"

        def retrieve_fallback_windows(self, query, *, top_k, fetch_evidence):
            assert top_k >= 1
            assert fetch_evidence is True
            return retrieval["windows"]

    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=FallbackRetriever(),
        assembler=PedagogicalGoldAssembler(chunk_strategy="fallback"),
        chunk_strategy="fallback",
    )
    response = pipeline.ask("学生为什么选 5.45？", mode="deterministic")
    assert response.audit_status == "AUDITED_100_VERIFIED"
    assert response.dialogue_citations[0].turn_id == 2
