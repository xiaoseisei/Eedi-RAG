"""Deterministic raw-turn evidence indexing.

The LLM extracts the semantic card fields, but it does not choose the raw
evidence pointers.  This module builds an immutable, content-addressed index
from :class:`~src.models.CleanedSession` and binds cards to all non-noise turns
for the card's role.  Consequently, changing an extraction prompt does not
require asking an LLM to rediscover the same source turns.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models import (
    CleanedSession,
    StudentMisconceptionProfile,
    TutorStrategyProfile,
)


EVIDENCE_INDEX_VERSION = "evidence-index-v1"
EVIDENCE_BINDING_POLICY = "deterministic_role_non_noise_v1"

EvidenceRole = Literal["student", "tutor"]
Card = Union[StudentMisconceptionProfile, TutorStrategyProfile]


class EvidenceIndexWindow(BaseModel):
    """One chronological, overlapping raw-dialogue evidence window."""

    model_config = ConfigDict(extra="forbid")

    index_id: str = Field(min_length=1, description="稳定的原文索引窗口 ID")
    chain_id: str = Field(min_length=1, description="所属会话逻辑链 ID")
    session_id: int = Field(ge=1)
    window_ordinal: int = Field(ge=0, description="按会话时间顺序排列的窗口序号")
    window_start_turn: int = Field(ge=1)
    window_end_turn: int = Field(ge=1)
    source_turn_ids: List[int] = Field(min_length=1)
    content: str = Field(min_length=1, description="由真实 Turn 拼接的原文索引内容")

    @model_validator(mode="after")
    def validate_window(self) -> "EvidenceIndexWindow":
        if self.window_start_turn > self.window_end_turn:
            raise ValueError("window_start_turn 不得大于 window_end_turn")
        if any(turn_id < 1 for turn_id in self.source_turn_ids):
            raise ValueError("source_turn_ids 必须全部为正整数")
        if self.source_turn_ids[0] != self.window_start_turn:
            raise ValueError("窗口首个 source_turn_id 必须等于 window_start_turn")
        if self.source_turn_ids[-1] != self.window_end_turn:
            raise ValueError("窗口末个 source_turn_id 必须等于 window_end_turn")
        return self

    @property
    def window_id(self) -> str:
        """兼容调用方使用 ``window_id`` 表示索引 ID。"""

        return self.index_id


class EvidenceIndex(BaseModel):
    """A versioned evidence index for one complete CleanedSession."""

    model_config = ConfigDict(extra="forbid")

    session_id: int = Field(ge=1)
    question_id: int = Field(ge=1)
    window_size: int = Field(ge=1)
    step: int = Field(ge=1)
    evidence_index_version: str = Field(min_length=1)
    evidence_index_hash: str = Field(min_length=1)
    chain_id: str = Field(min_length=1, description="会话级完整逻辑链 ID")
    windows: List[EvidenceIndexWindow] = Field(default_factory=list)
    turn_ids: List[int] = Field(default_factory=list)
    turn_roles: Dict[int, EvidenceRole] = Field(default_factory=dict)
    noise_turn_ids: List[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_turn_metadata(self) -> "EvidenceIndex":
        if len(set(self.turn_ids)) != len(self.turn_ids):
            raise ValueError("EvidenceIndex.turn_ids 不得重复")
        turn_set = set(self.turn_ids)
        if set(self.turn_roles) != turn_set:
            raise ValueError("turn_roles 必须覆盖且仅覆盖所有 turn_ids")
        if not set(self.noise_turn_ids).issubset(turn_set):
            raise ValueError("noise_turn_ids 必须属于 turn_ids")
        covered = {
            turn_id
            for window in self.windows
            for turn_id in window.source_turn_ids
        }
        if covered != turn_set:
            raise ValueError("EvidenceIndex windows 必须 union 完整覆盖会话所有 Turn")
        return self

    @property
    def index_ids(self) -> List[str]:
        """稳定的窗口 ID 顺序，便于 provenance 序列化。"""

        return [window.index_id for window in self.windows]

    @property
    def all_source_turn_ids(self) -> List[int]:
        """会话 Turn 的时间顺序列表。"""

        return list(self.turn_ids)

    def role_turn_ids(self, role: EvidenceRole) -> List[int]:
        """返回指定角色且非 noise 的真实 Turn ID。"""

        noise = set(self.noise_turn_ids)
        return [
            turn_id
            for turn_id in self.turn_ids
            if self.turn_roles[turn_id] == role and turn_id not in noise
        ]


def _window_starts(total: int, window_size: int, step: int) -> List[int]:
    """Return deterministic starts while repairing any uncovered gaps.

    The regular stride is retained for reproducibility.  If a caller selects a
    stride larger than the window, additional contiguous windows are inserted
    only where needed so that no real Turn disappears from the index.
    """

    if total == 0:
        return []

    starts: List[int] = []
    start = 0
    while True:
        starts.append(start)
        if start + window_size >= total:
            break
        start += step
        if start >= total:
            break

    def ids_for(window_start: int) -> range:
        return range(window_start, min(window_start + window_size, total))

    covered = {position for candidate in starts for position in ids_for(candidate)}
    while len(covered) < total:
        missing = next(position for position in range(total) if position not in covered)
        repair_start = min(missing, max(0, total - window_size))
        if repair_start not in starts:
            starts.append(repair_start)
        covered.update(ids_for(repair_start))

    return sorted(set(starts))


def _render_window(session: CleanedSession, turn_slice: list) -> str:
    subject_path = session.subjects.paths[0] if session.subjects.paths else "Mathematics"
    lines = [
        f"[学科考纲]: {subject_path}",
        f"[考题原题]: {session.question.question_text}",
        "[完整逻辑链原文]:",
    ]
    for turn in turn_slice:
        speaker_tag = "[Tutor]" if turn.is_tutor else "[Student]"
        lines.append(f"[Turn {turn.turn_id}] {speaker_tag}: {turn.text}")
    return "\n".join(lines)


def _hash_index_payload(index: EvidenceIndex) -> str:
    payload = {
        "session_id": index.session_id,
        "question_id": index.question_id,
        "window_size": index.window_size,
        "step": index.step,
        "evidence_index_version": index.evidence_index_version,
        "chain_id": index.chain_id,
        "turn_ids": index.turn_ids,
        "turn_roles": {str(key): value for key, value in sorted(index.turn_roles.items())},
        "noise_turn_ids": index.noise_turn_ids,
        "windows": [
            {
                "index_id": window.index_id,
                "chain_id": window.chain_id,
                "window_ordinal": window.window_ordinal,
                "window_start_turn": window.window_start_turn,
                "window_end_turn": window.window_end_turn,
                "source_turn_ids": window.source_turn_ids,
                "content": window.content,
            }
            for window in index.windows
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_evidence_index(
    session: CleanedSession,
    window_size: int = 6,
    step: int = 3,
    evidence_index_version: str = EVIDENCE_INDEX_VERSION,
) -> EvidenceIndex:
    """Build overlapping raw windows with complete Turn union coverage.

    No model call is made.  Every line in a window is rendered directly from
    the corresponding ``DialogueTurn.text`` and every source pointer is taken
    from the input session's actual Turn IDs.
    """

    if window_size < 1:
        raise ValueError("window_size 必须为正整数")
    if step < 1:
        raise ValueError("step 必须为正整数")
    if not evidence_index_version.strip():
        raise ValueError("evidence_index_version 不得为空")

    turns = list(session.turns)
    turn_ids = [turn.turn_id for turn in turns]
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError(f"Session {session.intervention_id} 存在重复 turn_id，拒绝建立索引")
    if any(turn_id < 1 for turn_id in turn_ids):
        raise ValueError("CleanedSession.turn_id 必须全部为正整数")

    turn_roles: Dict[int, EvidenceRole] = {
        turn.turn_id: ("tutor" if turn.is_tutor else "student") for turn in turns
    }
    noise_turn_ids = [turn.turn_id for turn in turns if turn.is_greeting_or_noise]
    chain_id = f"session_{session.intervention_id}_logic_{evidence_index_version}"
    starts = _window_starts(len(turns), window_size, step)
    windows: List[EvidenceIndexWindow] = []
    for ordinal, start in enumerate(starts):
        turn_slice = turns[start : start + window_size]
        source_turn_ids = [turn.turn_id for turn in turn_slice]
        windows.append(
            EvidenceIndexWindow(
                index_id=(
                    f"session_{session.intervention_id}_evidence_"
                    f"{source_turn_ids[0]}_{source_turn_ids[-1]}"
                ),
                chain_id=chain_id,
                session_id=session.intervention_id,
                window_ordinal=ordinal,
                window_start_turn=source_turn_ids[0],
                window_end_turn=source_turn_ids[-1],
                source_turn_ids=source_turn_ids,
                content=_render_window(session, turn_slice),
            )
        )

    index = EvidenceIndex(
        session_id=session.intervention_id,
        question_id=session.question_id,
        window_size=window_size,
        step=step,
        evidence_index_version=evidence_index_version,
        evidence_index_hash="pending",
        chain_id=chain_id,
        windows=windows,
        turn_ids=turn_ids,
        turn_roles=turn_roles,
        noise_turn_ids=noise_turn_ids,
    )
    return index.model_copy(update={"evidence_index_hash": _hash_index_payload(index)})


def bind_card_to_evidence_index(card: Card, evidence_index: EvidenceIndex) -> Card:
    """Bind a card to all non-noise raw evidence for its semantic role.

    Existing LLM-produced ``source_turn_ids`` are deliberately replaced.  The
    returned card's pointers and provenance are derived solely from the index,
    making this operation deterministic and independent of an LLM response.
    """

    if card.session_id != evidence_index.session_id:
        raise ValueError(
            f"卡片 session_id={card.session_id} 与 evidence index "
            f"session_id={evidence_index.session_id} 不一致"
        )
    if card.question_id != evidence_index.question_id:
        raise ValueError(
            f"卡片 question_id={card.question_id} 与 evidence index "
            f"question_id={evidence_index.question_id} 不一致"
        )

    role: EvidenceRole = "student" if isinstance(card, StudentMisconceptionProfile) else "tutor"
    eligible_turn_ids = evidence_index.role_turn_ids(role)
    if not eligible_turn_ids:
        raise ValueError(
            f"Session {evidence_index.session_id} 没有可绑定的非 noise {role} Turn"
        )
    eligible_set = set(eligible_turn_ids)
    source_index_ids = [
        window.index_id
        for window in evidence_index.windows
        if any(turn_id in eligible_set for turn_id in window.source_turn_ids)
    ]
    if not source_index_ids:
        raise ValueError("证据索引没有覆盖可绑定的角色 Turn，拒绝生成空指针")

    update = {
        "source_turn_ids": eligible_turn_ids,
        "source_index_ids": source_index_ids,
        "evidence_binding_policy": EVIDENCE_BINDING_POLICY,
        "evidence_index_version": evidence_index.evidence_index_version,
        "evidence_index_hash": evidence_index.evidence_index_hash,
    }
    return type(card).model_validate({**card.model_dump(), **update})


# Descriptive alias used by the incremental ingestion command.  Keeping the
# shorter name preserves compatibility with the first implementation.
build_logical_evidence_index = build_evidence_index


__all__ = [
    "EVIDENCE_BINDING_POLICY",
    "EVIDENCE_INDEX_VERSION",
    "EvidenceIndex",
    "EvidenceIndexWindow",
    "bind_card_to_evidence_index",
    "build_evidence_index",
    "build_logical_evidence_index",
]
