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
