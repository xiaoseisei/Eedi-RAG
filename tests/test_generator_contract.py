from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.generator_contract import (
    GeneratorClaim,
    GeneratorCitationRef,
    GeneratorGuidancePayloadV2,
    build_evidence_catalog,
    materialize_generator_citations,
    render_evidence_catalog,
    summarize_claim_coverage,
    normalize_v2_payload_data,
    validate_claim_citation_closure,
    validate_role_coverage,
)
from src.reranker import GoldAssembledContext


def _context() -> GoldAssembledContext:
    return GoldAssembledContext(
        raw_query="why?",
        prompt_context_markdown="final context",
        evidence_turns=[
            {"session_id": 10, "turn_id": 2, "speaker": "tutor", "text": "Which digit decides?"},
            {"session_id": 10, "turn_id": 3, "speaker": "student", "text": "The hundredths digit"},
        ],
        estimated_token_count=10,
    )


def test_evidence_catalog_ids_are_deterministic_and_preserve_exact_provenance() -> None:
    catalog = build_evidence_catalog(_context())

    assert list(catalog) == ["E001", "E002"]
    assert catalog["E001"].session_id == 10
    assert catalog["E001"].turn_id == 2
    assert catalog["E001"].quote_text == "Which digit decides?"
    rendered = render_evidence_catalog(catalog)
    assert "evidence_id=E001" in rendered
    assert "quote_text=Which digit decides?" in rendered


def test_compact_evidence_catalog_preserves_id_provenance_without_quote_duplication() -> None:
    catalog = build_evidence_catalog(_context())

    rendered = render_evidence_catalog(catalog, include_quote_text=False)

    assert "Allowed IDs: E001, E002" in rendered
    assert "quote_text=" not in rendered


def test_materialize_citations_uses_catalog_text_and_rejects_unknown_id() -> None:
    catalog = build_evidence_catalog(_context())

    citations = materialize_generator_citations([GeneratorCitationRef(evidence_id="E002")], catalog)

    assert citations[0].session_id == 10
    assert citations[0].turn_id == 3
    assert citations[0].speaker == "student"
    assert citations[0].quote_text == "The hundredths digit"
    with pytest.raises(ValueError, match="unknown evidence_id"):
        materialize_generator_citations([GeneratorCitationRef(evidence_id="E999")], catalog)


def test_factual_claim_requires_evidence_and_inference_requires_qualification() -> None:
    with pytest.raises(ValidationError):
        GeneratorClaim(claim_id="C1", claim_text="fact", claim_type="fact", evidence_ids=[])
    with pytest.raises(ValidationError):
        GeneratorClaim(claim_id="C2", claim_text="inference", claim_type="inference", evidence_ids=["E001"])

    claim = GeneratorClaim(
        claim_id="C2",
        claim_text="inference",
        claim_type="inference",
        evidence_ids=["E001"],
        qualification="This is a cautious interpretation of the dialogue.",
    )
    assert claim.qualification
    with pytest.raises(ValidationError, match="inference claims require at least one evidence_id"):
        GeneratorClaim(
            claim_id="C3",
            claim_text="unsupported inference",
            claim_type="inference",
            evidence_ids=[],
            qualification="This is a cautious interpretation.",
        )


def test_v2_payload_contains_claims_and_evidence_ids_only() -> None:
    payload = GeneratorGuidancePayloadV2(
        subject_path="Number > Rounding",
        answer="The student confused decimal places.",
        misconception_diagnosis="The student treated 1dp as retaining two decimal places.",
        evidence_explanation="The student explicitly gave 5.45 in the dialogue.",
        key_aha_question="Which digit decides the rounding?",
        scaffolding_steps=["Ask the student to identify the target decimal place."],
        pedagogical_intervention=["Ask the student to identify the next digit before rounding."],
        claims=[
            GeneratorClaim(
                claim_id="C1",
                claim_text="The student gave 5.45.",
                claim_type="fact",
                evidence_ids=["E001"],
            )
        ],
        dialogue_citations=[GeneratorCitationRef(evidence_id="E001")],
    )

    assert payload.dialogue_citations[0].evidence_id == "E001"
    assert "quote_text" not in payload.model_dump_json()


def test_v2_payload_allows_empty_optional_teaching_lists_for_narrow_questions() -> None:
    payload = GeneratorGuidancePayloadV2.model_validate(
        {
            "subject_path": "Number",
            "answer": "核心答案",
            "misconception_diagnosis": "未观察到",
            "evidence_explanation": "E001 支持核心答案。",
            "key_aha_question": "未要求",
            "scaffolding_steps": [],
            "pedagogical_intervention": [],
            "claims": [{"claim_id": "C1", "claim_text": "事实", "claim_type": "fact", "evidence_ids": ["E001"]}],
            "dialogue_citations": [{"evidence_id": "E001"}],
        }
    )

    assert payload.scaffolding_steps == []
    assert payload.pedagogical_intervention == []


def test_multi_role_citations_must_cover_student_and_tutor_evidence() -> None:
    context = GoldAssembledContext(
        raw_query="学生为什么会错，导师如何引导？",
        prompt_context_markdown="final context",
        evidence_turns=[
            {"session_id": 10, "turn_id": 1, "speaker": "student", "text": "I chose 5.45"},
            {"session_id": 10, "turn_id": 2, "speaker": "tutor", "text": "Which digit decides?"},
        ],
        estimated_token_count=10,
    )
    catalog = build_evidence_catalog(context)

    with pytest.raises(ValueError, match="both student and tutor"):
        validate_role_coverage(
            [GeneratorCitationRef(evidence_id="E001")], catalog, query=context.raw_query
        )
    validate_role_coverage(
        [GeneratorCitationRef(evidence_id="E001"), GeneratorCitationRef(evidence_id="E002")],
        catalog,
        query=context.raw_query,
    )


def test_summarize_claim_coverage_distinguishes_fact_support_and_citations() -> None:
    payload = GeneratorGuidancePayloadV2.model_validate(
        {
            "subject_path": "Number",
            "answer": "核心答案",
            "misconception_diagnosis": "诊断",
            "evidence_explanation": "解释",
            "key_aha_question": "问题",
            "scaffolding_steps": ["步骤"],
            "pedagogical_intervention": ["动作"],
            "claims": [
                {"claim_id": "C1", "claim_text": "事实", "claim_type": "fact", "evidence_ids": ["E001"]},
                {"claim_id": "C2", "claim_text": "推断", "claim_type": "inference", "evidence_ids": ["E001"], "qualification": "基于对白推断"},
                {"claim_id": "C3", "claim_text": "建议", "claim_type": "recommendation", "evidence_ids": []},
            ],
            "dialogue_citations": [{"evidence_id": "E001"}],
        }
    )

    assert summarize_claim_coverage(payload) == {
        "claim_count": 3,
        "fact_claim_count": 1,
        "fact_claims_with_evidence": 1,
        "fact_claim_coverage": 1.0,
        "cited_evidence_count": 1,
    }


def test_claim_citation_closure_requires_inference_evidence_to_be_cited() -> None:
    payload = GeneratorGuidancePayloadV2.model_construct(
        subject_path="Number",
        answer="核心答案",
        misconception_diagnosis="诊断",
        evidence_explanation="解释",
        key_aha_question="问题",
        scaffolding_steps=["步骤"],
        pedagogical_intervention=["动作"],
        claims=[GeneratorClaim(claim_id="C1", claim_text="推断", claim_type="inference", evidence_ids=["E002"], qualification="基于对白推断")],
        dialogue_citations=[GeneratorCitationRef(evidence_id="E001")],
        recommended_talk_moves=[],
        transfer_question=None,
    )
    with pytest.raises(ValueError, match="C1"):
        validate_claim_citation_closure(payload)


def test_claim_citation_closure_requires_all_declared_evidence_ids() -> None:
    payload = GeneratorGuidancePayloadV2.model_construct(
        subject_path="Number",
        answer="核心答案",
        misconception_diagnosis="诊断",
        evidence_explanation="解释",
        key_aha_question="问题",
        scaffolding_steps=["步骤"],
        pedagogical_intervention=["动作"],
        claims=[GeneratorClaim(claim_id="C1", claim_text="事实", claim_type="fact", evidence_ids=["E001", "E002"])],
        dialogue_citations=[GeneratorCitationRef(evidence_id="E001")],
        recommended_talk_moves=[],
        student_evidence_ids=[],
        tutor_evidence_ids=[],
        transfer_question=None,
    )

    with pytest.raises(ValueError, match="C1"):
        validate_claim_citation_closure(payload)


def test_normalize_v2_payload_closes_declared_claim_and_role_ids_only() -> None:
    catalog = build_evidence_catalog(_context())
    raw = {
        "claims": [{"evidence_ids": ["E001", "E002"]}],
        "student_evidence_ids": ["E002"],
        "tutor_evidence_ids": ["E001"],
        "dialogue_citations": [{"evidence_id": "E001"}],
    }

    normalized, merged = normalize_v2_payload_data(raw, catalog)

    assert merged == ["E002"]
    assert [item["evidence_id"] for item in normalized["dialogue_citations"]] == ["E001", "E002"]
    assert normalized["student_evidence_ids"] == ["E002"]
    assert normalized["tutor_evidence_ids"] == ["E001"]


def test_normalize_v2_payload_zero_pads_existing_short_ids_only() -> None:
    catalog = build_evidence_catalog(_context())
    normalized, merged = normalize_v2_payload_data(
        {
            "claims": [{"evidence_ids": ["E1", "E002"]}],
            "dialogue_citations": [{"evidence_id": "E1"}],
        },
        catalog,
    )

    assert merged == ["E002"]
    assert normalized["claims"][0]["evidence_ids"] == ["E001", "E002"]
    assert [item["evidence_id"] for item in normalized["dialogue_citations"]] == ["E001", "E002"]
