from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from evals.contracts import EvidenceRef, normalize_whitespace


@dataclass(frozen=True)
class GroundingAudit:
    citation_precision: float | None
    citation_recall: float
    role_accuracy: float | None
    quote_grounding_precision: float | None
    valid_citation_count: int
    required_evidence_count: int
    covered_required_count: int
    invalid_citation_indexes: list[int]
    candidate_authorization_rate: float | None = None
    unauthorized_citation_indexes: list[int] = field(default_factory=list)


def _quote_matches(candidate: str, source: str, mode: Literal["exact", "substring"]) -> bool:
    candidate_normalized = normalize_whitespace(candidate)
    source_normalized = normalize_whitespace(source)
    if mode == "exact":
        return candidate_normalized == source_normalized
    return candidate_normalized in source_normalized


def audit_grounding(
    output_citations: list[EvidenceRef],
    authoritative_turns: list[EvidenceRef],
    required_evidence: list[EvidenceRef],
    expected_speaker: Literal["student", "tutor"] | None = None,
    quote_match_mode: Literal["exact", "substring"] = "substring",
    retrieved_evidence: list[EvidenceRef] | None = None,
) -> GroundingAudit:
    if not authoritative_turns:
        raise ValueError("authoritative_turns must not be empty")
    if not required_evidence:
        raise ValueError("required_evidence must not be empty")

    by_turn: dict[tuple[int, int], EvidenceRef] = {}
    for source in authoritative_turns:
        if source.turn_key in by_turn:
            raise ValueError(f"duplicate authoritative turn: {source.turn_key}")
        by_turn[source.turn_key] = source

    role_hits = 0
    quote_hits = 0
    valid_indexes: set[int] = set()
    invalid_indexes: list[int] = []
    unauthorized_indexes: list[int] = []
    retrieved_keys = None
    if retrieved_evidence is not None:
        retrieved_keys = {
            (item.session_id, item.turn_id, item.speaker, normalize_whitespace(item.quote_text))
            for item in retrieved_evidence
        }
    for index, citation in enumerate(output_citations):
        source = by_turn.get(citation.turn_key)
        role_matches = bool(source and citation.speaker == source.speaker)
        if expected_speaker is not None:
            role_matches = role_matches and citation.speaker == expected_speaker
        quote_matches = bool(source and _quote_matches(citation.quote_text, source.quote_text, quote_match_mode))
        authorized = retrieved_keys is None or any(
            citation.turn_key == item.turn_key
            and citation.speaker == item.speaker
            and _quote_matches(citation.quote_text, item.quote_text, quote_match_mode)
            for item in (retrieved_evidence or [])
        )
        if not authorized:
            unauthorized_indexes.append(index)
        role_hits += int(role_matches)
        quote_hits += int(quote_matches)
        if role_matches and quote_matches and authorized:
            valid_indexes.add(index)
        else:
            invalid_indexes.append(index)

    covered_required = 0
    for required in required_evidence:
        covered = any(
            index in valid_indexes
            and citation.turn_key == required.turn_key
            and _quote_matches(citation.quote_text, required.quote_text, quote_match_mode)
            for index, citation in enumerate(output_citations)
        )
        covered_required += int(covered)

    output_count = len(output_citations)
    return GroundingAudit(
        citation_precision=(len(valid_indexes) / output_count) if output_count else None,
        citation_recall=covered_required / len(required_evidence),
        role_accuracy=(role_hits / output_count) if output_count else None,
        quote_grounding_precision=(quote_hits / output_count) if output_count else None,
        valid_citation_count=len(valid_indexes),
        required_evidence_count=len(required_evidence),
        covered_required_count=covered_required,
        invalid_citation_indexes=invalid_indexes,
        candidate_authorization_rate=(
            (output_count - len(unauthorized_indexes)) / output_count
            if retrieved_evidence is not None and output_count
            else None
        ),
        unauthorized_citation_indexes=unauthorized_indexes,
    )
