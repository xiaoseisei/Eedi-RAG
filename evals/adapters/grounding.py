from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from evals.contracts import EvidenceRef, GroundingEvalCase


def _to_evidence_ref(item: Any) -> EvidenceRef:
    if isinstance(item, EvidenceRef):
        return item
    if not isinstance(item, dict):
        raise TypeError("citation/evidence records must be mappings")
    quote = item.get("quote_text", item.get("text"))
    return EvidenceRef(
        session_id=item["session_id"],
        turn_id=item["turn_id"],
        speaker=item["speaker"],
        quote_text=quote,
    )


def build_grounding_case(
    *,
    case_id: str,
    required_evidence: Iterable[EvidenceRef | dict[str, Any]],
    authoritative_turns: Iterable[EvidenceRef | dict[str, Any]],
    output_citations: Iterable[EvidenceRef | dict[str, Any]],
    retrieved_evidence: Iterable[EvidenceRef | dict[str, Any]] | None = None,
    expected_speaker: str | None = None,
    slices: dict[str, str] | None = None,
) -> GroundingEvalCase:
    """Adapt a real final response and retrieval trace into the strict v2 case."""
    return GroundingEvalCase(
        case_id=case_id,
        output_citations=[_to_evidence_ref(item) for item in output_citations],
        authoritative_turns=[_to_evidence_ref(item) for item in authoritative_turns],
        required_evidence=[_to_evidence_ref(item) for item in required_evidence],
        retrieved_evidence=(
            [_to_evidence_ref(item) for item in retrieved_evidence]
            if retrieved_evidence is not None
            else None
        ),
        expected_speaker=expected_speaker,  # type: ignore[arg-type]
        slices=slices or {},
    )


def build_case_from_response(
    *,
    case_id: str,
    response: Any,
    required_evidence: Iterable[EvidenceRef | dict[str, Any]],
    authoritative_turns: Iterable[EvidenceRef | dict[str, Any]],
    retrieval_trace: dict[str, Any],
    slices: dict[str, str] | None = None,
) -> GroundingEvalCase:
    citations = getattr(response, "dialogue_citations", None)
    if citations is None:
        raise ValueError("response must expose dialogue_citations")
    retrieved = retrieval_trace.get("retrieved_evidence")
    if retrieved is None:
        retrieved = retrieval_trace.get("evidence_turns", [])
    return build_grounding_case(
        case_id=case_id,
        required_evidence=required_evidence,
        authoritative_turns=authoritative_turns,
        output_citations=citations,
        retrieved_evidence=retrieved,
        slices=slices,
    )


def capture_system_grounding_trace(pipeline: Any, query: str) -> dict[str, list[dict[str, Any]]]:
    """Run the explicit deterministic system mode and retain its real trace."""
    response = pipeline.ask(query=query, mode="deterministic", fetch_evidence=True)
    citations = []
    for citation in response.dialogue_citations:
        item = citation.model_dump() if hasattr(citation, "model_dump") else dict(citation)
        citations.append({key: item[key] for key in ("session_id", "turn_id", "speaker", "quote_text")})
    retrieved_evidence = []
    for source in getattr(response, "retrieved_sources_debug", []):
        for turn in source.get("evidence_turns", []):
            item = dict(turn)
            if "session_id" not in item and source.get("session_id") is not None:
                item["session_id"] = source["session_id"]
            if "quote_text" not in item and "text" in item:
                item["quote_text"] = item["text"]
            retrieved_evidence.append({
                key: item[key]
                for key in ("session_id", "turn_id", "speaker", "quote_text")
            })
    return {
        "output_citations": citations,
        "retrieved_evidence": retrieved_evidence,
    }
