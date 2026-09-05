"""Strict Generator v2 evidence-id and claim-level contract."""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models import DialogueCitation


EvidenceId = str


class EvidenceCatalogItem(BaseModel):
    """One authoritative Turn exposed to the Generator by stable ID."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    session_id: int = Field(ge=1)
    turn_id: int = Field(ge=1)
    speaker: Literal["student", "tutor"]
    quote_text: str = Field(min_length=1)


class GeneratorCitationRef(BaseModel):
    """Generator-supplied citation reference; no free-form quote is accepted."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")


class GeneratorClaim(BaseModel):
    """A response claim with explicit fact/inference/recommendation semantics."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    claim_id: str = Field(pattern=r"^C[0-9]{1,}$")
    claim_text: str = Field(min_length=1, max_length=1000)
    claim_type: Literal["fact", "inference", "recommendation"]
    evidence_ids: List[str] = Field(default_factory=list)
    qualification: str | None = Field(default=None, max_length=500)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: List[str]) -> List[str]:
        if len(set(value)) != len(value):
            raise ValueError("claim evidence_ids must be unique")
        for evidence_id in value:
            if not __import__("re").fullmatch(r"E[0-9]{3,}", evidence_id):
                raise ValueError("claim evidence_ids must use E### identifiers")
        return value

    @model_validator(mode="after")
    def validate_support(self) -> "GeneratorClaim":
        if self.claim_type == "fact" and not self.evidence_ids:
            raise ValueError("factual claims require at least one evidence_id")
        if self.claim_type == "inference":
            if not self.evidence_ids:
                raise ValueError("inference claims require at least one evidence_id")
            if not self.qualification:
                raise ValueError("inference claims require an explicit qualification")
        return self


class GeneratorGuidancePayloadV2(BaseModel):
    """Strict structured Generator output for the v2 prompt."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    subject_path: str = Field(min_length=1)
    answer: str = Field(min_length=1, max_length=600)
    misconception_diagnosis: str = Field(min_length=1, max_length=1200)
    evidence_explanation: str = Field(min_length=1, max_length=1200)
    key_aha_question: str = Field(min_length=1, max_length=600)
    scaffolding_steps: List[str] = Field(min_length=1, max_length=8)
    pedagogical_intervention: List[str] = Field(min_length=1, max_length=8)
    recommended_talk_moves: List[str] = Field(default_factory=list, max_length=8)
    claims: List[GeneratorClaim] = Field(min_length=1)
    dialogue_citations: List[GeneratorCitationRef] = Field(min_length=1)
    student_evidence_ids: List[str] = Field(default_factory=list, max_length=16)
    tutor_evidence_ids: List[str] = Field(default_factory=list, max_length=16)
    transfer_question: str | None = Field(default=None, max_length=600)

    @field_validator("student_evidence_ids", "tutor_evidence_ids")
    @classmethod
    def role_evidence_ids_are_valid(cls, value: List[str]) -> List[str]:
        for evidence_id in value:
            if not __import__("re").fullmatch(r"E[0-9]{3,}", evidence_id):
                raise ValueError("role evidence IDs must use E### identifiers")
        return value

    @model_validator(mode="after")
    def validate_claim_ids_and_citations(self) -> "GeneratorGuidancePayloadV2":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim_id values must be unique")
        citation_ids = [citation.evidence_id for citation in self.dialogue_citations]
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("dialogue_citations evidence_id values must be unique")
        cited = set(citation_ids)
        missing_claim_citations = [
            claim.claim_id
            for claim in self.claims
            if claim.claim_type in {"fact", "inference"}
            and not set(claim.evidence_ids) <= cited
        ]
        if missing_claim_citations:
            raise ValueError(
                "fact/inference claims must be represented in dialogue_citations: "
                + ", ".join(missing_claim_citations)
            )
        for role, evidence_ids in (
            ("student", self.student_evidence_ids),
            ("tutor", self.tutor_evidence_ids),
        ):
            if len(set(evidence_ids)) != len(evidence_ids):
                raise ValueError(f"{role}_evidence_ids must be unique")
            if not set(evidence_ids) <= cited:
                raise ValueError(f"{role}_evidence_ids must be included in dialogue_citations")
        return self


def validate_claim_citation_closure(payload: GeneratorGuidancePayloadV2) -> None:
    """Require every fact/inference claim's evidence to appear in citations."""

    cited = {citation.evidence_id for citation in payload.dialogue_citations}
    missing = [
        claim.claim_id
        for claim in payload.claims
        if claim.claim_type in {"fact", "inference"}
        and not set(claim.evidence_ids) <= cited
    ]
    if missing:
        raise ValueError(
            "fact/inference claims must be represented in dialogue_citations: "
            + ", ".join(missing)
        )


def summarize_claim_coverage(payload_or_response) -> dict[str, int | float]:
    """Summarize validated fact-claim support without trusting free-form text."""

    claims = getattr(payload_or_response, "claims", None)
    if claims is None:
        claims = getattr(payload_or_response, "__dict__", {}).get("generator_claims", [])
    normalized = [claim if isinstance(claim, dict) else claim.model_dump() for claim in claims]
    fact_claims = [claim for claim in normalized if claim.get("claim_type") == "fact"]
    cited_ids = set(
        getattr(payload_or_response, "generator_citation_evidence_ids", [])
    )
    for citation in getattr(payload_or_response, "dialogue_citations", []):
        if hasattr(citation, "evidence_id"):
            cited_ids.add(citation.evidence_id)
        elif isinstance(citation, dict):
            cited_ids.add(citation.get("evidence_id"))
    covered = [
        claim for claim in fact_claims
        if set(claim.get("evidence_ids", [])) & cited_ids
    ]
    fact_count = len(fact_claims)
    return {
        "claim_count": len(normalized),
        "fact_claim_count": fact_count,
        "fact_claims_with_evidence": len(covered),
        "fact_claim_coverage": (len(covered) / fact_count) if fact_count else 1.0,
        "cited_evidence_count": len(cited_ids),
    }


def build_evidence_catalog(context) -> Dict[str, EvidenceCatalogItem]:
    """Build deterministic IDs from final Assembler evidence order."""

    catalog: Dict[str, EvidenceCatalogItem] = {}
    by_key: dict[tuple[int, int], str] = {}
    for turn in context.evidence_turns:
        try:
            session_id = int(turn["session_id"])
            turn_id = int(turn["turn_id"])
            speaker = str(turn["speaker"])
            quote_text = str(turn["text"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("final evidence contains incomplete Turn provenance") from exc
        key = (session_id, turn_id)
        if key in by_key:
            existing_id = by_key[key]
            if catalog[existing_id].quote_text != quote_text or catalog[existing_id].speaker != speaker:
                raise ValueError(f"duplicate Turn has conflicting provenance: {key}")
            continue
        evidence_id = f"E{len(catalog) + 1:03d}"
        item = EvidenceCatalogItem(
            evidence_id=evidence_id,
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            quote_text=quote_text,
        )
        catalog[evidence_id] = item
        by_key[key] = evidence_id
    if not catalog:
        raise ValueError("final evidence catalog cannot be empty")
    return catalog


def render_evidence_catalog(catalog: Dict[str, EvidenceCatalogItem]) -> str:
    """Render IDs and fields separately so quote text cannot absorb Turn tags."""

    if not catalog:
        raise ValueError("evidence catalog cannot be empty")
    lines = ["## Evidence catalog (cite IDs only)"]
    for evidence_id, item in catalog.items():
        if evidence_id != item.evidence_id:
            raise ValueError("evidence catalog key does not match item evidence_id")
        lines.extend(
            [
                f"[evidence_id={item.evidence_id}]",
                f"session_id={item.session_id}",
                f"turn_id={item.turn_id}",
                f"speaker={item.speaker}",
                f"quote_text={item.quote_text}",
            ]
        )
    return "\n".join(lines)


def materialize_generator_citations(
    citation_refs: List[GeneratorCitationRef],
    catalog: Dict[str, EvidenceCatalogItem],
) -> List[DialogueCitation]:
    """Resolve model IDs to canonical citation quadruples from real evidence."""

    citations: List[DialogueCitation] = []
    seen: set[str] = set()
    for ref in citation_refs:
        if ref.evidence_id in seen:
            raise ValueError(f"duplicate evidence_id citation: {ref.evidence_id}")
        item = catalog.get(ref.evidence_id)
        if item is None:
            raise ValueError(f"unknown evidence_id: {ref.evidence_id}")
        seen.add(ref.evidence_id)
        citations.append(
            DialogueCitation(
                session_id=item.session_id,
                turn_id=item.turn_id,
                speaker=item.speaker,
                quote_text=item.quote_text,
                verifiable_in_duckdb=False,
            )
        )
    if not citations:
        raise ValueError("at least one citation reference is required")
    return citations


def query_requires_both_roles(query: str) -> bool:
    """Detect explicit two-role teaching questions without reading qrels."""

    value = str(query).casefold()
    student_terms = ("学生", "student", "learner", "错因", "误区")
    tutor_terms = ("导师", "老师", "名师", "tutor", "teacher", "引导", "策略", "教法")
    return any(term in value for term in student_terms) and any(
        term in value for term in tutor_terms
    )


def validate_role_coverage(
    citation_refs: List[GeneratorCitationRef],
    catalog: Dict[str, EvidenceCatalogItem],
    *,
    query: str,
    student_evidence_ids: List[str] | None = None,
    tutor_evidence_ids: List[str] | None = None,
) -> None:
    """Require both speakers when the query explicitly asks about both roles."""

    if not query_requires_both_roles(query):
        return
    if student_evidence_ids is not None and not student_evidence_ids:
        raise ValueError("multi-role query requires student_evidence_ids")
    if tutor_evidence_ids is not None and not tutor_evidence_ids:
        raise ValueError("multi-role query requires tutor_evidence_ids")
    speakers = set()
    for ref in citation_refs:
        item = catalog.get(ref.evidence_id)
        if item is None:
            raise ValueError(f"unknown evidence_id: {ref.evidence_id}")
        speakers.add(item.speaker)
    if {"student", "tutor"} - speakers:
        raise ValueError("multi-role query citations must cover both student and tutor evidence")
    for role, evidence_ids in (
        ("student", student_evidence_ids),
        ("tutor", tutor_evidence_ids),
    ):
        if evidence_ids is None:
            continue
        for evidence_id in evidence_ids:
            item = catalog.get(evidence_id)
            if item is None:
                raise ValueError(f"{role}_evidence_ids contains unknown evidence_id: {evidence_id}")
            if item.speaker != role:
                raise ValueError(f"{role}_evidence_ids must identify {role} evidence")
