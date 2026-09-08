"""Tests for deterministic raw-evidence indexing and card binding."""

from src.evidence_index import bind_card_to_evidence_index, build_evidence_index
from src.models import (
    CleanedSession,
    DialogueTurn,
    Question,
    StudentMisconceptionProfile,
    SubjectHierarchy,
    TutorStrategyProfile,
)


def _session() -> CleanedSession:
    turns = [
        DialogueTurn(
            turn_id=1,
            speaker="tutor",
            is_tutor=True,
            raw_messages=["Let us begin"],
            text="Let us begin",
        ),
        DialogueTurn(
            turn_id=2,
            speaker="student",
            is_tutor=False,
            raw_messages=["I think the answer is B"],
            text="I think the answer is B",
        ),
        DialogueTurn(
            turn_id=3,
            speaker="student",
            is_tutor=False,
            raw_messages=["yes"],
            text="yes",
            is_greeting_or_noise=True,
        ),
        DialogueTurn(
            turn_id=4,
            speaker="tutor",
            is_tutor=True,
            raw_messages=["What makes you think that?"],
            text="What makes you think that?",
        ),
        DialogueTurn(
            turn_id=5,
            speaker="student",
            is_tutor=False,
            raw_messages=["I divided before adding"],
            text="I divided before adding",
        ),
        DialogueTurn(
            turn_id=6,
            speaker="tutor",
            is_tutor=True,
            raw_messages=["Check the order of operations"],
            text="Check the order of operations",
        ),
        DialogueTurn(
            turn_id=7,
            speaker="tutor",
            is_tutor=True,
            raw_messages=["bye"],
            text="bye",
            is_greeting_or_noise=True,
        ),
    ]
    return CleanedSession(
        intervention_id=42,
        question_id=99,
        tutor_id=7,
        subjects=SubjectHierarchy(paths=["Number > Arithmetic"]),
        question=Question(question_id=99, question_text="Evaluate 2 + 3 * 4", options={}),
        turns=turns,
        total_turns=len(turns),
        student_turn_count=3,
        tutor_turn_count=4,
        has_valid_tutoring=True,
    )


def test_evidence_index_uses_overlapping_windows_and_covers_every_turn():
    evidence_index = build_evidence_index(_session(), window_size=3, step=2)

    assert len(evidence_index.windows) >= 3
    assert any(
        set(left.source_turn_ids) & set(right.source_turn_ids)
        for left, right in zip(evidence_index.windows, evidence_index.windows[1:])
    )
    covered = {
        turn_id
        for window in evidence_index.windows
        for turn_id in window.source_turn_ids
    }
    assert covered == {1, 2, 3, 4, 5, 6, 7}
    assert all("I think the answer is B" not in window.content or "[Turn 2]" in window.content for window in evidence_index.windows)
    assert evidence_index.evidence_index_hash == build_evidence_index(
        _session(), window_size=3, step=2
    ).evidence_index_hash


def test_bind_card_replaces_llm_pointers_with_non_noise_role_turns():
    evidence_index = build_evidence_index(_session(), window_size=3, step=2)
    card = StudentMisconceptionProfile(
        session_id=42,
        question_id=99,
        subject_path="Number > Arithmetic",
        misconception_name="Order of operations confusion",
        deep_mechanism="The student applies division before multiplication incorrectly.",
        verbatim_student_quotes=["I divided before adding"],
        source_turn_ids=[2],
    )

    bound = bind_card_to_evidence_index(card, evidence_index)

    assert bound.source_turn_ids == [2, 5]
    assert 3 not in bound.source_turn_ids
    assert bound.source_index_ids
    assert bound.evidence_binding_policy == "deterministic_role_non_noise_v1"
    assert bound.evidence_index_version == evidence_index.evidence_index_version
    assert bound.evidence_index_hash == evidence_index.evidence_index_hash
    assert bound.source_turn_ids != [2]


def test_bind_card_removes_quotes_from_noise_turns_and_keeps_real_text():
    evidence_index = build_evidence_index(_session(), window_size=3, step=2)
    card = StudentMisconceptionProfile(
        session_id=42,
        question_id=99,
        subject_path="Number > Arithmetic",
        misconception_name="Order of operations confusion",
        deep_mechanism="The student applies division before multiplication incorrectly.",
        verbatim_student_quotes=["yes"],
        source_turn_ids=[3],
    )

    bound = bind_card_to_evidence_index(card, evidence_index)

    assert bound.verbatim_student_quotes == ["I think the answer is B"]


def test_bind_strategy_uses_non_noise_tutor_turns_and_is_deterministic():
    evidence_index = build_evidence_index(_session(), window_size=3, step=2)
    card = TutorStrategyProfile(
        session_id=42,
        question_id=99,
        pedagogical_goal="Surface the order-of-operations rule",
        strategy_category="Socratic_Questioning",
        key_aha_question="What makes you think that?",
        scaffolding_steps=["Ask for the student's reason"],
        talk_moves=["<Press for Accuracy>"],
        resolution_outcome="The student can inspect operation order.",
        source_turn_ids=[4],
    )

    first = bind_card_to_evidence_index(card, evidence_index)
    second = bind_card_to_evidence_index(card, evidence_index)

    assert first.source_turn_ids == [1, 4, 6]
    assert 7 not in first.source_turn_ids
    assert first.source_index_ids == second.source_index_ids
    assert first.source_turn_ids == second.source_turn_ids
    assert first.model_dump() == second.model_dump()
