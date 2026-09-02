from __future__ import annotations

import time

from evals.contracts import ChunkRankedRecord, ChunkStructuralEvalCase
from src.chunker import SlidingWindowChunker, estimate_tokens
from src.models import CleanedSession


def capture_production_chunks(
    session: CleanedSession,
    *,
    window_size: int = 6,
    step: int = 3,
) -> list[ChunkRankedRecord]:
    """Adapt the production SlidingWindowChunker without reimplementing it."""
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    chunks = SlidingWindowChunker(window_size=window_size, step=step).chunk(session)
    return [
        ChunkRankedRecord(
            chunk_id=chunk.chunk_id,
            session_id=chunk.session_id,
            source_turn_ids=set(chunk.source_turn_ids),
            token_count=chunk.token_count or estimate_tokens(chunk.content),
        )
        for chunk in chunks
    ]


def capture_production_chunk_trace(
    session: CleanedSession,
    *,
    window_size: int = 6,
    step: int = 3,
) -> tuple[list[ChunkRankedRecord], float]:
    started = time.perf_counter()
    chunks = capture_production_chunks(session, window_size=window_size, step=step)
    return chunks, (time.perf_counter() - started) * 1000.0


def audit_production_chunk_structure(
    session: CleanedSession,
    *,
    window_size: int = 6,
    step: int = 3,
) -> ChunkStructuralEvalCase:
    """Audit the actual production window text and pointers without ranking it."""
    emitted = SlidingWindowChunker(window_size=window_size, step=step).chunk(session)
    turns_by_id = {turn.turn_id: turn for turn in session.turns}
    declared_turns = sum(len(chunk.source_turn_ids) for chunk in emitted)
    valid_pointers = 0
    verbatim_lines = 0
    valid_boundaries = 0
    covered: set[int] = set()
    out_of_bounds = 0
    for chunk in emitted:
        source_ids = list(chunk.source_turn_ids)
        covered.update(source_ids)
        for turn_id in source_ids:
            turn = turns_by_id.get(turn_id)
            if turn is None:
                out_of_bounds += 1
                continue
            valid_pointers += 1
            speaker = "[Tutor]" if turn.is_tutor else "[Student]"
            verbatim_lines += int(f"[Turn {turn_id}] {speaker}: {turn.text}" in chunk.content)
        start = chunk.metadata.get("window_start_turn")
        end = chunk.metadata.get("window_end_turn")
        valid_boundaries += int(bool(source_ids) and start == source_ids[0] and end == source_ids[-1])
    source_tokens = max(1, sum(estimate_tokens(turn.text) for turn in session.turns))
    emitted_tokens = sum(chunk.token_count or estimate_tokens(chunk.content) for chunk in emitted)
    denominator = max(1, declared_turns)
    return ChunkStructuralEvalCase(
        case_id=f"session-{session.intervention_id}-chunk-structure",
        pointer_validity=valid_pointers / denominator,
        verbatim_integrity=verbatim_lines / denominator,
        turn_coverage=len(covered.intersection(turns_by_id)) / max(1, len(turns_by_id)),
        boundary_integrity=valid_boundaries / max(1, len(emitted)),
        inflation_ratio=emitted_tokens / source_tokens,
        orphan_count=0,
        out_of_bounds_turn_count=out_of_bounds,
        slices={"strategy": "sliding-window-production"},
    )
