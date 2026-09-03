from __future__ import annotations

"""Provisional multi-relevant graded qrels for Turn/window diagnostics.

The labels are deterministic derivations from human-selected evidence turns and
stored source metadata. They are deliberately marked provisional and must not
be presented as independent human relevance judgments.
"""

from typing import Any, Iterable


def _required_turns(case: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(item["session_id"]), int(item["turn_id"]))
        for item in case.get("verbatim_grounding_quotes", [])
    }


def build_window_graded_qrels(
    case: dict[str, Any],
    windows: Iterable[dict[str, Any]],
    *,
    label_source: str = "provisional_rule_from_human_turns_v1",
) -> list[dict[str, Any]]:
    """Return one graded qrel row for every indexed window.

    Relevance levels:
      3 = target window covers every required Turn;
      2 = target window covers at least one required Turn;
      1 = same subject/session neighbourhood but no required Turn;
      0 = unrelated window.

    The rule is intentionally conservative about direct evidence (only exact
    required Turn overlap earns level 2/3) and exposes ``needs_human_review``
    so downstream reports cannot silently promote these labels to ground truth.
    """
    required = _required_turns(case)
    if not required:
        raise ValueError("case must contain at least one verbatim grounding quote")
    target_sessions = {session_id for session_id, _ in required}
    target_subject = str(case.get("subject_path") or "").strip()
    query_id = str(case.get("query_id") or case.get("case_id") or "").strip()
    if not query_id:
        raise ValueError("case must contain query_id or case_id")

    rows: list[dict[str, Any]] = []
    for window in windows:
        document_id = str(window.get("chunk_id") or window.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("window must contain chunk_id/document_id")
        session_id = int(window["session_id"])
        source_turn_ids = {
            (session_id, int(turn_id))
            for turn_id in (window.get("source_turn_ids") or [])
        }
        overlap = len(required.intersection(source_turn_ids))
        same_subject = target_subject and str(window.get("subject_path") or "").strip() == target_subject
        same_session = session_id in target_sessions
        if overlap == len(required):
            relevance = 3
            rationale = "window covers all required evidence turns"
        elif overlap > 0:
            relevance = 2
            rationale = "window covers a subset of required evidence turns"
        elif same_session or same_subject:
            relevance = 1
            rationale = "topically related session/subject without direct required turn"
        else:
            relevance = 0
            rationale = "no direct turn overlap or subject/session relation"
        rows.append({
            "query_id": query_id,
            "document_id": document_id,
            "relevance": relevance,
            "slot": "fallback_window",
            "required_turn_overlap": overlap,
            "required_turn_count": len(required),
            "label_source": label_source,
            "relevance_source": "human_verbatim_grounding_quote_plus_metadata_rule",
            "needs_human_review": True,
            "rationale": rationale,
            "evidence_turns": [
                {"session_id": session_id, "turn_id": turn_id}
                for _, turn_id in sorted(required.intersection(source_turn_ids))
            ],
        })
    if not rows:
        raise ValueError("windows must contain at least one document")
    return rows


def build_card_graded_qrels(
    case: dict[str, Any],
    cards: Iterable[dict[str, Any]],
    *,
    target_slot: str,
    label_source: str = "provisional_rule_from_human_turns_v1",
) -> list[dict[str, Any]]:
    """Grade misconception/strategy cards against role-aware evidence.

    The target slot is derived from the query intent by the evaluation control
    plane, never by the system under test.  A card earns level 3 when its
    role-correct source pointers cover every required quote for the slot, level
    2 for a subset, level 1 for same-session/subject topical support, and level
    0 otherwise.  All rows require human review before production claims.
    """
    if target_slot not in {"misconception", "strategy"}:
        raise ValueError("target_slot must be misconception or strategy")
    required = [
        (int(item["session_id"]), int(item["turn_id"]), str(item["speaker"]))
        for item in case.get("verbatim_grounding_quotes", [])
        if (target_slot == "misconception" and item["speaker"] == "student")
        or (target_slot == "strategy" and item["speaker"] == "tutor")
    ]
    if not required:
        raise ValueError(f"case has no required {target_slot} evidence")
    required_keys = {(sid, tid) for sid, tid, _ in required}
    target_sessions = {sid for sid, _, _ in required}
    target_subject = str(case.get("subject_path") or "").strip()
    query_id = str(case.get("query_id") or case.get("case_id") or "").strip()
    if not query_id:
        raise ValueError("case must contain query_id or case_id")

    rows: list[dict[str, Any]] = []
    for card in cards:
        document_id = str(card.get("chunk_id") or card.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("card must contain chunk_id/document_id")
        session_id = int(card["session_id"])
        source_turn_ids = {
            (session_id, int(turn_id))
            for turn_id in (card.get("source_turn_ids") or [])
        }
        overlap = len(required_keys.intersection(source_turn_ids))
        same_session = session_id in target_sessions
        same_subject = target_subject and str(card.get("subject_path") or "").strip() == target_subject
        if overlap == len(required_keys):
            relevance = 3
            rationale = "card role-correct pointers cover all required evidence turns"
        elif overlap > 0:
            relevance = 2
            rationale = "card role-correct pointers cover a subset of required evidence turns"
        elif same_session or same_subject:
            relevance = 1
            rationale = "topically/session related card without direct required role-correct turn"
        else:
            relevance = 0
            rationale = "unrelated card"
        rows.append({
            "query_id": query_id,
            "document_id": document_id,
            "slot": target_slot,
            "relevance": relevance,
            "required_turn_overlap": overlap,
            "required_turn_count": len(required_keys),
            "label_source": label_source,
            "relevance_source": "human_verbatim_grounding_quote_plus_role_aware_pointer_rule",
            "needs_human_review": True,
            "rationale": rationale,
            "evidence_turns": [
                {"session_id": sid, "turn_id": tid}
                for sid, tid in sorted(required_keys.intersection(source_turn_ids))
            ],
        })
    if not rows:
        raise ValueError("cards must contain at least one document")
    return rows


def turn_coverage_at_k(
    ranked_windows: list[dict[str, Any]],
    required_turns: set[tuple[int, int]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not required_turns:
        raise ValueError("required_turns must not be empty")
    covered = {
        (int(window["session_id"]), int(turn_id))
        for window in ranked_windows[:k]
        for turn_id in (window.get("source_turn_ids") or [])
    }
    return len(covered.intersection(required_turns)) / len(required_turns)


def graded_recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int, *, minimum_relevance: int = 2) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    positives = {doc_id for doc_id, relevance in qrels.items() if relevance >= minimum_relevance}
    if not positives:
        raise ValueError("qrels must contain positive graded relevance")
    return len(positives.intersection(ranked_ids[:k])) / len(positives)


def graded_precision_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int, *, minimum_relevance: int = 1) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not qrels:
        raise ValueError("qrels must not be empty")
    return sum(qrels.get(doc_id, 0) >= minimum_relevance for doc_id in ranked_ids[:k]) / min(k, len(ranked_ids))
