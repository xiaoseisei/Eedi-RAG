from __future__ import annotations

import math

import pytest

from evals.metrics.citation import audit_grounding
from evals.metrics.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evals.contracts import EvidenceRef


def test_retrieval_metrics_use_all_positive_graded_qrels() -> None:
    qrels = {"direct": 3, "supporting": 2, "irrelevant": 0}
    ranked = ["noise", "supporting", "direct"]

    assert recall_at_k(ranked, qrels, 1) == 0.0
    assert recall_at_k(ranked, qrels, 2) == 0.5
    assert precision_at_k(ranked, qrels, 2) == 0.5
    assert mean_reciprocal_rank(ranked, qrels) == 0.5

    expected_dcg = 3 / math.log2(3) + 7 / math.log2(4)
    ideal_dcg = 7 + 3 / math.log2(3)
    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(expected_dcg / ideal_dcg)


def test_retrieval_metrics_reject_invalid_inputs_instead_of_defaulting() -> None:
    with pytest.raises(ValueError, match="positive relevance"):
        recall_at_k(["a"], {"a": 0}, 1)
    with pytest.raises(ValueError, match="k"):
        precision_at_k(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError, match="duplicate"):
        mean_reciprocal_rank(["a", "a"], {"a": 1})


def test_grounding_audit_checks_session_turn_role_and_verbatim_quote() -> None:
    authoritative = [
        EvidenceRef(
            session_id=10,
            turn_id=4,
            speaker="student",
            quote_text="I think the answer is 5.45 maybe",
        ),
        EvidenceRef(
            session_id=10,
            turn_id=5,
            speaker="tutor",
            quote_text="Which digit decides the rounding?",
        ),
    ]
    output = [
        EvidenceRef(
            session_id=10,
            turn_id=4,
            speaker="student",
            quote_text="the answer is 5.45",
        ),
        EvidenceRef(
            session_id=10,
            turn_id=5,
            speaker="student",
            quote_text="Which digit decides the rounding?",
        ),
    ]

    audit = audit_grounding(
        output_citations=output,
        authoritative_turns=authoritative,
        required_evidence=authoritative,
        expected_speaker=None,
        quote_match_mode="substring",
    )

    assert audit.citation_precision == 0.5
    assert audit.citation_recall == 0.5
    assert audit.role_accuracy == 0.5
    assert audit.invalid_citation_indexes == [1]


def test_grounding_audit_does_not_treat_empty_output_as_perfect() -> None:
    source = [EvidenceRef(session_id=1, turn_id=1, speaker="student", quote_text="real evidence")]
    audit = audit_grounding([], source, source, quote_match_mode="exact")

    assert audit.citation_precision is None
    assert audit.citation_recall == 0.0
    assert audit.role_accuracy is None

