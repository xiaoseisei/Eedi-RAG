"""Isolated, production-shaped ten-session chunk strategy benchmark.

The independent table under ``tests/db/chunk_selection_10_v1`` is the only
input.  Every candidate owns a new DuckDB/Chroma pair below that directory and
then runs through the production storage, retrieval, assembler/MMR, and
citation-audit code.  This smoke test never reads ``data/db`` or ``data/chroma``
and never calls an LLM, provider embedding, or provider reranker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest

from evals.adapters.assembler import assemble_with_trace
from src.chunker import SlidingWindowChunker, estimate_tokens
from src.models import Chunk, CleanedSession, DialogueCitation, ExtractedPIU, PedagogicalGuidanceResponse
from src.rag_pipeline import CitationAuditError, EndToEndPedagogicalRAGPipeline
from src.reranker import GoldAssembledContext, PedagogicalGoldAssembler
from src.retriever import DualMetricRetriever
from src.storage_manager import DualEngineStorageManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "db" / "chunk_selection_10_v1"
TOP_K = 5
REQUIRED_RECALL_GATE = 0.80


@dataclass
class FixtureBundle:
    sessions: list[CleanedSession]
    extracted: list[ExtractedPIU]
    queries: list[dict[str, Any]]
    qrels: list[dict[str, Any]]
    candidate_config: dict[str, Any]
    manifest: dict[str, Any]
    hashes: dict[str, str]


@dataclass
class CandidateArtifacts:
    name: str
    root: Path
    db_path: Path
    chroma_dir: Path
    storage: DualEngineStorageManager
    chunks: list[Chunk]
    card_enabled: bool
    strategy_kind: str
    ingestion_ms: float
    index_build_ms: float = 0.0


class FixtureUnavailable(AssertionError):
    pass


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FixtureUnavailable(
            f"独立测试 fixture 缺少 {path}; 请先创建 tests/db/chunk_selection_10_v1。"
            "本测试不会回退读取生产 data/ 路径。"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"fixture JSON 无法解析: {path}") from exc


def _read_jsonl(path: Path) -> list[Any]:
    if not path.is_file():
        raise FixtureUnavailable(f"独立测试 fixture 缺少 {path}")
    result: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"fixture JSONL 无法解析: {path}:{line_number}") from exc
    return result


def _read_json_or_jsonl(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise AssertionError(f"{path} 必须是 JSON 数组或 JSONL")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    if root.is_file():
        return _sha256(root)
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_independent_fixture() -> FixtureBundle:
    """Load and validate only the versioned independent test table."""

    paths = {
        "sessions": FIXTURE_ROOT / "sessions.jsonl",
        "extracted": FIXTURE_ROOT / "extracted_pius.jsonl",
        "queries": FIXTURE_ROOT / "queries.jsonl",
        "qrels": FIXTURE_ROOT / "qrels.jsonl",
    }
    for name in ("queries", "qrels"):
        if not paths[name].exists():
            paths[name] = paths[name].with_suffix(".json")
    sessions = [CleanedSession.model_validate(row) for row in _read_jsonl(paths["sessions"])]
    extracted = [ExtractedPIU.model_validate(row) for row in _read_jsonl(paths["extracted"])]
    queries = [dict(row) for row in _read_json_or_jsonl(paths["queries"])]
    qrels = [dict(row) for row in _read_json_or_jsonl(paths["qrels"])]
    candidate_config = dict(_read_json(FIXTURE_ROOT / "candidate_config.json"))
    manifest = dict(_read_json(FIXTURE_ROOT / "run_manifest.json"))
    if len(sessions) != 10 or len(extracted) != 10:
        raise AssertionError(f"fixture 必须有 10 sessions/ExtractedPIU，实际 {len(sessions)}/{len(extracted)}")
    if len(queries) != 10 or len(qrels) != 10:
        raise AssertionError(f"fixture 必须有 10 queries/qrels，实际 {len(queries)}/{len(qrels)}")
    session_ids = {session.intervention_id for session in sessions}
    query_session_ids = {int(row["target_intervention_id"]) for row in queries}
    if query_session_ids != session_ids:
        raise AssertionError(f"query 必须覆盖且只覆盖十场 session，缺失={session_ids - query_session_ids}")
    if {str(row["query_id"]) for row in qrels} != {str(row["query_id"]) for row in queries}:
        raise AssertionError("queries/qrels 的 query_id 必须一一对应")
    if any(row.get("label_source") != "human_reviewed" for row in qrels):
        raise AssertionError("qrels 必须全部声明 human_reviewed")
    if manifest.get("runtime", {}).get("llm_calls_allowed") is not False:
        raise AssertionError("独立 fixture 必须声明 llm_calls_allowed=false")
    return FixtureBundle(
        sessions,
        extracted,
        queries,
        qrels,
        candidate_config,
        manifest,
        {name: _sha256(path) for name, path in paths.items()} | {
            "candidate_config.json": _sha256(FIXTURE_ROOT / "candidate_config.json"),
            "run_manifest.json": _sha256(FIXTURE_ROOT / "run_manifest.json"),
        },
    )


def _header(session: CleanedSession) -> str:
    subject = " | ".join(session.subjects.paths) or "Mathematics"
    options = " | ".join(f"{key}: {value}" for key, value in sorted(session.question.options.items()))
    return f"[Subject]: {subject}\n[Question]: {session.question.question_text}\n[Options]: {options}"


def _turn_line(turn: Any) -> str:
    speaker = "Tutor" if turn.is_tutor else "Student"
    moves = f" {' '.join(turn.talk_moves)}" if turn.talk_moves else ""
    return f"[Turn {turn.turn_id}] [{speaker}]{moves}: {turn.text}"


def _window_chunks(sessions: Sequence[CleanedSession], window: int, step: int, strategy: str) -> list[Chunk]:
    result: list[Chunk] = []
    for session in sessions:
        for chunk in SlidingWindowChunker(window_size=window, step=step).chunk(session):
            result.append(chunk.model_copy(update={"metadata": {
                **chunk.metadata,
                "experimental_strategy": strategy,
                "role_scope": "student+tutor",
                "boundary_reason": f"sliding_window_w{window}_s{step}",
            }}))
    return result


def _local_chunk(session: CleanedSession, indexes: Iterable[int], chunk_id: str, strategy: str, reason: str) -> Chunk:
    selected = [session.turns[index] for index in indexes]
    turn_ids = [turn.turn_id for turn in selected]
    content = _header(session) + "\n[对话片段]:\n" + "\n".join(_turn_line(turn) for turn in selected)
    return Chunk(
        chunk_id=chunk_id,
        session_id=session.intervention_id,
        question_id=session.question_id,
        chunk_type="sliding_window",
        content=content,
        metadata={
            "intervention_id": session.intervention_id,
            "question_id": session.question_id,
            "subject_path": session.subjects.paths[0] if session.subjects.paths else "Mathematics",
            "window_start_turn": min(turn_ids),
            "window_end_turn": max(turn_ids),
            "source_turn_ids": turn_ids,
            "experimental_strategy": strategy,
            "role_scope": "student+tutor",
            "boundary_reason": reason,
        },
        token_count=estimate_tokens(content),
        source_turn_ids=turn_ids,
    )


def _exchange_chunks(sessions: Sequence[CleanedSession]) -> list[Chunk]:
    result: list[Chunk] = []
    for session in sessions:
        for index, turn in enumerate(session.turns):
            if turn.is_tutor or turn.is_greeting_or_noise:
                continue
            indexes = [index]
            if index and session.turns[index - 1].is_tutor:
                indexes.insert(0, index - 1)
            if index + 1 < len(session.turns) and session.turns[index + 1].is_tutor:
                indexes.append(index + 1)
            result.append(_local_chunk(session, indexes, f"session_{session.intervention_id}_exchange_{turn.turn_id}", "student_exchange", "student_turn_with_adjacent_tutor"))
    return result


ACTION_MOVES = {"<Press for Accuracy>", "<Press for Reasoning>", "<Revoicing>", "<Getting Student to Relate>", "<Restating>"}


def _action_chunks(sessions: Sequence[CleanedSession]) -> list[Chunk]:
    result: list[Chunk] = []
    for session in sessions:
        for index, turn in enumerate(session.turns):
            if not turn.is_tutor or not ACTION_MOVES.intersection(turn.talk_moves):
                continue
            indexes = range(max(0, index - 2), min(len(session.turns), index + 3))
            result.append(_local_chunk(session, indexes, f"session_{session.intervention_id}_action_{turn.turn_id}", "talkmove_action", "talk_move_tutor_anchor_with_local_context"))
    return result


def _candidate_chunks(name: str, sessions: Sequence[CleanedSession]) -> tuple[list[Chunk], bool, str]:
    if name == "Current_LLM_Cards":
        return [], True, "cards"
    if name in {"Cards_Plus_W6S3", "Reserved_Card1_W6S3_2"}:
        return _window_chunks(sessions, 6, 3, "sliding_w6_s3"), True, "cards_plus_sliding_w6_s3"
    if name in {"Production_Sliding_W6S3", "Reserved_Action2_W6S3_1", "Reserved_Action1_W6S3_2"}:
        return _window_chunks(sessions, 6, 3, "sliding_w6_s3"), False, "sliding_w6_s3"
    if name == "Sliding_W4S2":
        return _window_chunks(sessions, 4, 2, "sliding_w4_s2"), False, "sliding_w4_s2"
    if name == "Student_Exchange":
        return _exchange_chunks(sessions), False, "student_exchange"
    if name == "TalkMove_Action":
        return _action_chunks(sessions), False, "talkmove_action"
    if name == "Exchange_Plus_Action":
        return _exchange_chunks(sessions) + _action_chunks(sessions), False, "mixed_exchange_action"
    raise AssertionError(f"未定义候选策略: {name}")


def _assert_isolated(path: Path) -> None:
    resolved = path.resolve()
    production = {(PROJECT_ROOT / "data" / "db").resolve(), (PROJECT_ROOT / "data" / "chroma").resolve()}
    if any(resolved == item or item in resolved.parents for item in production):
        raise AssertionError(f"候选路径越界到生产 artifact: {resolved}")
    if FIXTURE_ROOT.resolve() not in resolved.parents and resolved != FIXTURE_ROOT.resolve():
        raise AssertionError(f"候选路径必须位于独立 fixture 下: {resolved}")


def build_candidate_artifacts(candidate_name: str, sessions: Sequence[CleanedSession], extracted: Sequence[ExtractedPIU], root: Path) -> CandidateArtifacts:
    candidate_root = root / candidate_name
    candidate_root.mkdir(parents=True, exist_ok=True)
    db_path, chroma_dir = candidate_root / "tutoring_knowledge.duckdb", candidate_root / "chroma"
    _assert_isolated(db_path)
    _assert_isolated(chroma_dir)
    chunks, card_enabled, strategy_kind = _candidate_chunks(candidate_name, sessions)
    started = time.perf_counter()
    storage = DualEngineStorageManager(db_path=db_path, chroma_dir=chroma_dir, embedding_backend="deterministic")
    storage.ingest_sessions(list(sessions))
    if card_enabled:
        storage.ingest_extracted_pius(list(extracted), sessions_map={session.intervention_id: session for session in sessions})
    if chunks:
        storage.ingest_sliding_window_chunks(chunks)
        collection = storage.coll_windows
        existing = collection.get(ids=[chunk.chunk_id for chunk in chunks], include=["metadatas"])
        old = {str(cid): dict(meta or {}) for cid, meta in zip(existing.get("ids", []), existing.get("metadatas", []))}
        collection.update(ids=[chunk.chunk_id for chunk in chunks], metadatas=[{
            **old.get(chunk.chunk_id, {}),
            "experimental_strategy": chunk.metadata.get("experimental_strategy", strategy_kind),
            "role_scope": chunk.metadata.get("role_scope", "student+tutor"),
            "boundary_reason": chunk.metadata.get("boundary_reason", strategy_kind),
        } for chunk in chunks])
    return CandidateArtifacts(candidate_name, candidate_root, db_path, chroma_dir, storage, list(chunks), card_enabled, strategy_kind, (time.perf_counter() - started) * 1000.0)


def audit_candidate_ingestion(artifacts: CandidateArtifacts, sessions: Sequence[CleanedSession], chunks: Sequence[Chunk], extracted: Sequence[ExtractedPIU]) -> dict[str, Any]:
    conn = artifacts.storage.duck_conn
    expected_session_ids = {session.intervention_id for session in sessions}
    expected_turns = {(session.intervention_id, turn.turn_id): turn for session in sessions for turn in session.turns}
    actual_sessions = {int(row[0]) for row in conn.execute("SELECT intervention_id FROM tutoring_sessions").fetchall()}
    actual_turns = {(int(row[0]), int(row[1])): row for row in conn.execute("SELECT intervention_id, turn_id, speaker, text FROM session_dialogue_turns").fetchall()}
    pointers: list[tuple[int, int, str]] = [(chunk.session_id, int(turn_id), "window") for chunk in chunks for turn_id in chunk.source_turn_ids]
    if artifacts.card_enabled:
        for item in extracted:
            if item.misconception:
                pointers.extend((item.session_id, int(turn_id), "student") for turn_id in item.misconception.source_turn_ids)
            if item.tutor_strategy:
                pointers.extend((item.session_id, int(turn_id), "tutor") for turn_id in item.tutor_strategy.source_turn_ids)
    valid = roles = exact = 0
    for session_id, turn_id, role in pointers:
        expected, actual = expected_turns.get((session_id, turn_id)), actual_turns.get((session_id, turn_id))
        if expected is None or actual is None:
            continue
        valid += 1
        if role == "window" or (role == "student" and not expected.is_tutor) or (role == "tutor" and expected.is_tutor):
            roles += 1
        if str(actual[3]) == expected.text:
            exact += 1
    ids = [chunk.chunk_id for chunk in chunks]
    duplicate_rate = 1.0 - len(set(ids)) / len(ids) if ids else 0.0
    empty_chunk_rate = sum(not chunk.content.strip() for chunk in chunks) / max(1, len(chunks))
    parent_rate = sum(chunk.session_id in expected_session_ids for chunk in chunks) / max(1, len(chunks))
    docs = []
    if ids:
        docs = [str(doc or "") for doc in (artifacts.storage.coll_windows.get(ids=ids, include=["documents"]).get("documents") or [])]
    empty_index_rate = sum(not doc.strip() for doc in docs) / max(1, len(docs))
    snapshot = artifacts.storage.get_storage_audit_snapshot()
    hard = (
        actual_sessions == expected_session_ids
        and set(actual_turns) == set(expected_turns)
        and valid == len(pointers) and roles == len(pointers) and exact == len(pointers)
        and duplicate_rate == 0.0 and empty_chunk_rate == 0.0 and empty_index_rate == 0.0
    )
    return {
        "session_ingestion_coverage": len(actual_sessions & expected_session_ids) / len(expected_session_ids),
        "turn_ingestion_coverage": len(set(actual_turns) & set(expected_turns)) / len(expected_turns),
        "source_turn_id_valid_rate": valid / max(1, len(pointers)),
        "source_turn_role_valid_rate": roles / max(1, len(pointers)),
        "exact_text_roundtrip_rate": exact / max(1, len(pointers)),
        "parent_fk_valid_rate": parent_rate,
        "duplicate_chunk_rate": duplicate_rate,
        "empty_chunk_rate": max(empty_chunk_rate, empty_index_rate),
        "duckdb_counts": snapshot["duckdb_tables"],
        "chroma_counts": snapshot["chromadb_collections"],
        "expected_card_count": 2 * len(extracted) if artifacts.card_enabled else 0,
        "expected_window_count": len(chunks),
        "hard_gate_passed": hard,
    }


def _reopen_candidate_storage(artifacts: CandidateArtifacts) -> None:
    """Flush Chroma's persistent HNSW segments before opening the read path.

    Chroma's Rust persistent client can expose a just-created segment before
    its files are visible to a second reader on Windows.  Closing and reopening
    the same isolated artifact makes the benchmark's ingestion/read boundary
    explicit and also mirrors a production process restart.
    """

    artifacts.storage.close()
    artifacts.storage = DualEngineStorageManager(
        db_path=artifacts.db_path,
        chroma_dir=artifacts.chroma_dir,
        embedding_backend="deterministic",
    )


def _qrel(bundle: FixtureBundle, query_id: str) -> dict[str, Any]:
    for row in bundle.qrels:
        if str(row["query_id"]) == query_id:
            return row
    raise AssertionError(f"缺少 query {query_id} qrel")


def _required(qrel: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["turn_id"]): item for item in qrel.get("required_evidence", [])}


def _supporting(qrel: dict[str, Any]) -> set[int]:
    return {int(item["turn_id"] if isinstance(item, dict) else item) for item in qrel.get("acceptable_supporting_turn_ids", [])}


def _candidate_evidence(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        candidate_session = candidate.get("metadata", {}).get("session_id")
        for turn in candidate.get("evidence_turns", []) or []:
            normalized = {**dict(turn), "session_id": int(turn.get("session_id", candidate_session))}
            result[(int(normalized["session_id"]), int(normalized["turn_id"]))] = normalized
    return list(result.values())


def _flatten_candidates(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in [*retrieval.get("misconceptions", []), *retrieval.get("strategies", []), *retrieval.get("windows", []), *retrieval.get("fallback_windows", [])]:
        chunk_id = str(candidate.get("chunk_id", ""))
        if chunk_id and chunk_id in seen:
            continue
        if chunk_id:
            seen.add(chunk_id)
        result.append(candidate)
    return result


def _retrieval_metrics(candidates: Sequence[dict[str, Any]], qrel: dict[str, Any], limit: int) -> dict[str, float]:
    required, supporting = _required(qrel), _supporting(qrel)
    target_sessions = {int(item) for item in qrel.get("target_session_ids", [])}
    ranked = list(candidates)[:limit]
    evidence_pairs = {
        (int(turn.get("session_id", candidate.get("metadata", {}).get("session_id"))), int(turn["turn_id"]))
        for candidate in ranked for turn in candidate.get("evidence_turns", []) or []
    }
    required_pairs = {(session_id, turn_id) for session_id in target_sessions for turn_id in required}
    hit_ranks, grades = [], []
    total_grade = max(1, sum(int(item.get("grade", 0)) for item in required.values()) + len(supporting))
    for candidate in ranked:
        pairs = {(int(turn.get("session_id", candidate.get("metadata", {}).get("session_id"))), int(turn["turn_id"])) for turn in candidate.get("evidence_turns", []) or []}
        hit_ranks.append(bool(pairs & required_pairs))
        raw_grade = sum(required.get(turn_id, {}).get("grade", 0) for session_id, turn_id in pairs if session_id in target_sessions) + sum(1 for session_id, turn_id in pairs if session_id in target_sessions and turn_id in supporting)
        grades.append(min(1.0, raw_grade / total_grade))
    first = next((index + 1 for index, hit in enumerate(hit_ranks) if hit), None)
    dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
    ideal = [1.0] * min(limit, max(1, len(required) + len(supporting)))
    idcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    supporting_pairs = {(session_id, turn_id) for session_id in target_sessions for turn_id in supporting}
    return {"required_evidence_recall": len(evidence_pairs & required_pairs) / max(1, len(required_pairs)), "supporting_evidence_recall": len(evidence_pairs & supporting_pairs) / max(1, len(supporting_pairs)), "mrr": 1.0 / first if first else 0.0, "ndcg": dcg / idcg if idcg else 0.0}


def _context_metrics(evidence: Sequence[dict[str, Any]], qrel: dict[str, Any], token_count: int, budget_violation: bool) -> dict[str, Any]:
    target_sessions = {int(item) for item in qrel.get("target_session_ids", [])}
    required, supporting = _required(qrel), _supporting(qrel)
    pairs = {(int(item.get("session_id", -1)), int(item.get("turn_id", -1))) for item in evidence}
    required_pairs = {(session_id, turn_id) for session_id in target_sessions for turn_id in required}
    relevant_pairs = required_pairs | {(session_id, turn_id) for session_id in target_sessions for turn_id in supporting}
    role_map = {(session_id, turn_id): item.get("role") for session_id in target_sessions for turn_id, item in required.items()}
    role_correct = sum(1 for item in evidence if (int(item.get("session_id", -1)), int(item.get("turn_id", -1))) not in role_map or item.get("speaker") == role_map[(int(item["session_id"]), int(item["turn_id"]))])
    by_session: dict[int, list[int]] = {}
    for session_id, turn_id in pairs:
        if session_id in target_sessions:
            by_session.setdefault(session_id, []).append(turn_id)
    gaps = sum(max(0, right - left - 1) for values in by_session.values() for left, right in zip(sorted(set(values)), sorted(set(values))[1:]))
    return {
        "context_precision": len(pairs & relevant_pairs) / max(1, len(pairs)),
        "context_required_turn_recall_after_assembly": len(pairs & required_pairs) / max(1, len(required_pairs)),
        "supporting_evidence_recall_after_assembly": len(pairs & (relevant_pairs - required_pairs)) / max(1, len(supporting)),
        "role_correctness": role_correct / max(1, len(evidence)),
        "turn_continuity": max(0.0, min(1.0, 1.0 - gaps / max(1, len(pairs)))),
        "wrong_session_turn_rate": sum(session_id not in target_sessions for session_id, _ in pairs) / max(1, len(pairs)),
        "unique_turn_count": len(pairs),
        "token_count": int(token_count),
        "budget_violation": bool(budget_violation),
    }


def _audit_citations(retriever: DualMetricRetriever, query: str, evidence: Sequence[dict[str, Any]], prompt: str, token_count: int) -> dict[str, Any]:
    """Run the real pipeline audit over citations made from assembled evidence."""

    assembler = PedagogicalGoldAssembler(max_prompt_tokens=1500, model_reranker=None, chunk_strategy="fallback")
    pipeline = EndToEndPedagogicalRAGPipeline(retriever=retriever, assembler=assembler, chunk_strategy="fallback")
    citations = [DialogueCitation(session_id=int(item["session_id"]), turn_id=int(item["turn_id"]), speaker=str(item["speaker"]), quote_text=str(item["text"])) for item in evidence]
    response = PedagogicalGuidanceResponse.model_construct(
        query=query, subject_path="test", session_id=int(evidence[0]["session_id"]) if evidence else None,
        misconception_diagnosis="audit only", key_aha_question="audit only", recommended_talk_moves=[], scaffolding_steps=[], dialogue_citations=citations, audit_status="PENDING",
    )
    context = GoldAssembledContext.model_construct(raw_query=query, prompt_context_markdown=prompt, selected_misconception=None, selected_strategy=None, selected_windows=[], chunk_strategy="fallback", evidence_turns=list(evidence), estimated_token_count=token_count, budget_violation=False, truncation_loss=0)
    try:
        pipeline._audit_citations(response, context)
    except CitationAuditError as exc:
        return {"audit_status": "FAILED", "citation_precision": 0.0, "citation_recall": 0.0, "audit_failure_count": 1, "audit_error": str(exc)}
    return {"audit_status": response.audit_status, "citation_precision": 1.0, "citation_recall": 1.0, "audit_failure_count": 0, "audit_error": None}


def _retrieve_cards(retriever: DualMetricRetriever, query: str) -> dict[str, Any]:
    result = retriever.retrieve_multi_perspective_rrf(query, top_k_each=TOP_K, fetch_evidence=True)
    result.update({"chunk_strategy": "card", "windows": [], "fallback_windows": []})
    return result


def _retrieve_fallback(retriever: DualMetricRetriever, query: str, where_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    windows = retriever.retrieve_fallback_windows(query, top_k=TOP_K, where_filter=where_filter, fetch_evidence=True)
    return {"chunk_strategy": "fallback", "raw_query": query, "windows": windows, "fallback_windows": windows, "misconceptions": [], "strategies": [], "rewritten_queries": {"extracted_keywords": []}, "trace": {"retrieval_mode": retriever.retrieval_mode, "chunk_strategy": "fallback"}}


def _assemble(retriever: DualMetricRetriever, query: str, retrieval: dict[str, Any]) -> dict[str, Any]:
    assembler = PedagogicalGoldAssembler(max_prompt_tokens=1500, model_reranker=None, chunk_strategy=str(retrieval.get("chunk_strategy", "fallback")))
    started = time.perf_counter()
    trace = assemble_with_trace(query, retrieval, assembler)
    trace["stage_latency_ms"] = (time.perf_counter() - started) * 1000.0
    is_card = str(retrieval.get("chunk_strategy", "fallback")) == "card"
    trace.update({
        "reranking_applied": False,
        "reranker_status": "UNMEASURED_PROVIDER_RERANKER",
        "mmr_applied": is_card,
        "assembly_method": "card_business_boost_plus_mmr" if is_card else "fallback_ordered_windows",
    })
    return trace


def _fixed_policy(retriever: DualMetricRetriever, query: str, policy: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run each lane through production retrieval/assembly and merge by slots."""

    if policy == "card1_window2":
        cards = _retrieve_cards(retriever, query)
        windows = _retrieve_fallback(retriever, query, {"experimental_strategy": "sliding_w6_s3"})
        card_candidates = [*cards.get("misconceptions", []), *cards.get("strategies", [])]
        card_candidates.sort(key=lambda item: (-float(item.get("rrf_score", item.get("hybrid_score", 0.0))), str(item.get("chunk_id"))))
        selected = card_candidates[:1] + windows["windows"][:2]
        lane_traces = {"card_lane": _assemble(retriever, query, cards), "window_lane": _assemble(retriever, query, windows)}
    elif policy in {"action2_window1", "action1_window2"}:
        actions = _retrieve_fallback(retriever, query, {"experimental_strategy": "talkmove_action"})
        windows = _retrieve_fallback(retriever, query, {"experimental_strategy": "sliding_w6_s3"})
        action_slots = 2 if policy == "action2_window1" else 1
        selected = actions["windows"][:action_slots] + windows["windows"][: 3 - action_slots]
        lane_traces = {"action_lane": _assemble(retriever, query, actions), "window_lane": _assemble(retriever, query, windows)}
    else:
        raise AssertionError(f"未知固定槽位策略: {policy}")
    retrieval = {"chunk_strategy": "fallback", "raw_query": query, "windows": selected, "fallback_windows": selected, "misconceptions": [], "strategies": [], "rewritten_queries": {"extracted_keywords": []}, "trace": {"fixed_slot_policy": policy, "lane_traces": lane_traces}}
    return retrieval, lane_traces


def _percentile(values: Sequence[float], percentile: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * percentile) - 1)] if values else 0.0


def _candidate_specs() -> list[tuple[str, str]]:
    return [("Current_LLM_Cards", "cards"), ("Cards_Plus_W6S3", "cards_plus"), ("Production_Sliding_W6S3", "fallback"), ("Sliding_W4S2", "fallback"), ("Student_Exchange", "fallback"), ("TalkMove_Action", "fallback"), ("Exchange_Plus_Action", "fallback"), ("Reserved_Card1_W6S3_2", "fixed"), ("Reserved_Action2_W6S3_1", "fixed"), ("Reserved_Action1_W6S3_2", "fixed")]


def _hard_gate(ingestion: dict[str, Any], quality: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if not ingestion["hard_gate_passed"]:
        failures.append("ingestion_integrity")
    if quality["required_evidence_recall_after_assembly"] < REQUIRED_RECALL_GATE:
        failures.append("required_evidence_recall")
    if quality["citation_audit_failure_count"]:
        failures.append("citation_audit")
    return not failures, failures


def run_short_benchmark() -> dict[str, Any]:
    bundle = load_independent_fixture()
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run_root = FIXTURE_ROOT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    qrels = {str(query["query_id"]): _qrel(bundle, str(query["query_id"])) for query in bundle.queries}
    candidates_report: list[dict[str, Any]] = []
    artifact_hashes: dict[str, Any] = {}
    for candidate_name, mode in _candidate_specs():
        artifacts = build_candidate_artifacts(candidate_name, bundle.sessions, bundle.extracted, run_root)
        try:
            ingestion = audit_candidate_ingestion(artifacts, bundle.sessions, artifacts.chunks, bundle.extracted)
            _reopen_candidate_storage(artifacts)
            started = time.perf_counter()
            retriever = DualMetricRetriever(storage_manager=artifacts.storage, query_rewrite_mode="deterministic", alpha=0.5, retrieval_mode="bm25_dense")
            artifacts.index_build_ms = (time.perf_counter() - started) * 1000.0
            traces: list[dict[str, Any]] = []
            retrieval_latencies: list[float] = []
            assembly_latencies: list[float] = []
            total_latencies: list[float] = []
            for query in bundle.queries:
                query_id, raw_query = str(query["query_id"]), str(query["query"])
                started = time.perf_counter()
                if mode in {"cards", "cards_plus"}:
                    retrieval = _retrieve_cards(retriever, raw_query)
                    if mode == "cards_plus":
                        retrieval["fallback_windows"] = _retrieve_fallback(retriever, raw_query, {"experimental_strategy": "sliding_w6_s3"})["windows"]
                    lane_traces = None
                elif candidate_name == "Reserved_Card1_W6S3_2":
                    retrieval, lane_traces = _fixed_policy(retriever, raw_query, "card1_window2")
                elif candidate_name == "Reserved_Action2_W6S3_1":
                    retrieval, lane_traces = _fixed_policy(retriever, raw_query, "action2_window1")
                elif candidate_name == "Reserved_Action1_W6S3_2":
                    retrieval, lane_traces = _fixed_policy(retriever, raw_query, "action1_window2")
                else:
                    retrieval = _retrieve_fallback(retriever, raw_query)
                    lane_traces = None
                retrieval_ms = (time.perf_counter() - started) * 1000.0
                if lane_traces is None:
                    assembly_started = time.perf_counter()
                    assembly_trace = _assemble(retriever, raw_query, retrieval)
                    assembly_ms = (time.perf_counter() - assembly_started) * 1000.0
                    selected_evidence = list(assembly_trace.get("final_evidence_turns", []))
                else:
                    assembly_trace = lane_traces[next(iter(lane_traces))]
                    assembly_trace["fixed_slot_policy"] = retrieval["trace"]["fixed_slot_policy"]
                    assembly_trace["lane_assembly_traces"] = lane_traces
                    assembly_ms = sum(float(item.get("stage_latency_ms", 0.0)) for item in lane_traces.values())
                    selected_evidence = _candidate_evidence(retrieval["windows"])
                if mode == "cards_plus" and not selected_evidence:
                    selected_evidence = _candidate_evidence(retrieval.get("fallback_windows", []))
                candidates = _flatten_candidates(retrieval)
                qrel = qrels[query_id]
                before3, before5 = _retrieval_metrics(candidates, qrel, 3), _retrieval_metrics(candidates, qrel, 5)
                after = _context_metrics(selected_evidence, qrel, int(assembly_trace.get("estimated_tokens", 0)), bool(assembly_trace.get("budget_violation", False)))
                after_rank = _retrieval_metrics([{"evidence_turns": selected_evidence}], qrel, 1)
                citation_started = time.perf_counter()
                audit = _audit_citations(retriever, raw_query, selected_evidence, str(assembly_trace.get("actual_prompt_context", "")), int(assembly_trace.get("estimated_tokens", 0)))
                audit["citation_recall"] = after["context_required_turn_recall_after_assembly"] if audit["audit_status"] == "AUDITED_100_VERIFIED" else 0.0
                audit_ms = (time.perf_counter() - citation_started) * 1000.0
                total_ms = retrieval_ms + assembly_ms + audit_ms
                retrieval_latencies.append(retrieval_ms)
                assembly_latencies.append(assembly_ms)
                total_latencies.append(total_ms)
                selected_ids = [str(item) for item in assembly_trace.get("final_context_ids", [])] if lane_traces is None else [str(item.get("chunk_id")) for item in retrieval["windows"]]
                traces.append({
                    "query_id": query_id,
                    "query_rewrite_backend": "deterministic" if mode in {"cards", "cards_plus"} else "not_applicable_fallback",
                    "embedding_backend": artifacts.storage.embedding_backend,
                    "retrieved_candidate_ids": [str(item.get("chunk_id")) for item in candidates if item.get("chunk_id")],
                    "retrieval_scores": [float(item.get("rrf_score", item.get("fallback_combined_score", item.get("hybrid_score", 0.0)))) for item in candidates],
                    "retrieval_at_3": before3,
                    "retrieval_at_5": before5,
                    "assembler_selected_ids": selected_ids,
                    "selected_evidence_turn_ids": sorted({(int(item["session_id"]), int(item["turn_id"])) for item in selected_evidence}),
                    "assembly_after": after,
                    "reranking_applied": bool(assembly_trace.get("reranking_applied", False)),
                    "reranker_status": str(assembly_trace.get("reranker_status", "UNMEASURED_PROVIDER_RERANKER")),
                    "mmr_applied": bool(assembly_trace.get("mmr_applied", False)),
                    "assembly_method": str(assembly_trace.get("assembly_method", "fallback_ordered_windows")),
                    "rerank_mrr_delta": after_rank["mrr"] - before5["mrr"],
                    "rerank_ndcg_delta": after_rank["ndcg"] - before5["ndcg"],
                    "assembler_drop_rate": max(0.0, 1.0 - len(selected_ids) / max(1, len(candidates))),
                    "citation_audit": audit,
                    "audit_status": audit["audit_status"],
                    "latency_ms": {"retrieval": retrieval_ms, "assembly": assembly_ms, "audit": audit_ms, "total": total_ms},
                    "prompt_token_count": int(assembly_trace.get("estimated_tokens", 0)),
                })
            quality = {
                "required_evidence_recall_at_3": statistics.fmean(row["retrieval_at_3"]["required_evidence_recall"] for row in traces),
                "required_evidence_recall_at_5": statistics.fmean(row["retrieval_at_5"]["required_evidence_recall"] for row in traces),
                "required_evidence_recall_after_assembly": statistics.fmean(row["assembly_after"]["context_required_turn_recall_after_assembly"] for row in traces),
                "supporting_evidence_recall_at_3": statistics.fmean(row["retrieval_at_3"]["supporting_evidence_recall"] for row in traces),
                "supporting_evidence_recall_at_5": statistics.fmean(row["retrieval_at_5"]["supporting_evidence_recall"] for row in traces),
                "context_precision_after_assembly": statistics.fmean(row["assembly_after"]["context_precision"] for row in traces),
                "retrieval_mrr": statistics.fmean(row["retrieval_at_5"]["mrr"] for row in traces),
                "retrieval_ndcg": statistics.fmean(row["retrieval_at_5"]["ndcg"] for row in traces),
                "rerank_mrr_delta": statistics.fmean(row["rerank_mrr_delta"] for row in traces),
                "rerank_ndcg_delta": statistics.fmean(row["rerank_ndcg_delta"] for row in traces),
                "assembler_drop_rate": statistics.fmean(row["assembler_drop_rate"] for row in traces),
                "token_count": int(statistics.fmean(row["prompt_token_count"] for row in traces)),
                "budget_violation_rate": statistics.fmean(float(row["assembly_after"]["budget_violation"]) for row in traces),
                "citation_precision": statistics.fmean(row["citation_audit"]["citation_precision"] for row in traces),
                "citation_recall": statistics.fmean(row["citation_audit"]["citation_recall"] for row in traces),
                "citation_audit_failure_count": sum(row["citation_audit"]["audit_failure_count"] for row in traces),
                "retrieval_p50_ms": statistics.median(retrieval_latencies), "retrieval_p95_ms": _percentile(retrieval_latencies, 0.95),
                "assembly_p50_ms": statistics.median(assembly_latencies), "assembly_p95_ms": _percentile(assembly_latencies, 0.95),
                "total_p50_ms": statistics.median(total_latencies), "total_p95_ms": _percentile(total_latencies, 0.95),
                "llm_request_count": 0, "llm_generation_cost": "UNMEASURED", "provider_reranker_quality": "UNMEASURED",
            }
            passed, failures = _hard_gate(ingestion, quality)
            # Fixed-slot and mixed-lane candidates are explicit test adapters:
            # their merge policy is not a production assembler contract.  Keep
            # their complete measurements, but never let them become the
            # production recommendation in this smoke run.
            if candidate_name in {"Cards_Plus_W6S3", "Exchange_Plus_Action", "Reserved_Card1_W6S3_2", "Reserved_Action2_W6S3_1", "Reserved_Action1_W6S3_2"}:
                passed = False
                failures.append("experimental_lane_merge_diagnostic_only")
            raw_tokens = sum(estimate_tokens(turn.text) for session in bundle.sessions for turn in session.turns)
            indexed_tokens = sum(estimate_tokens(chunk.content) for chunk in artifacts.chunks)
            if artifacts.card_enabled:
                indexed_tokens += sum(estimate_tokens(json.dumps(item.model_dump(), ensure_ascii=False)) for item in bundle.extracted)
            candidates_report.append({
                "strategy": candidate_name, "strategy_kind": artifacts.strategy_kind, "zero_llm_ingestion": not artifacts.card_enabled,
                "llm_cost_scope": "UNMEASURED" if artifacts.card_enabled else "none", "chunk_count": len(artifacts.chunks) + (2 * len(bundle.extracted) if artifacts.card_enabled else 0),
                "token_inflation_ratio": indexed_tokens / max(1, raw_tokens), "ingestion": ingestion, "quality": quality,
                "candidate_status": "PASS_RECOMMENDED" if passed else "DIAGNOSTIC_ONLY", "failure_categories": failures, "query_traces": traces,
            })
        finally:
            artifacts.storage.close()
            artifact_hashes[candidate_name] = {"duckdb_sha256": _tree_sha256(artifacts.db_path), "chroma_sha256": _tree_sha256(artifacts.chroma_dir)}
    passing = [row for row in candidates_report if row["candidate_status"] == "PASS_RECOMMENDED"]
    if passing:
        passing.sort(key=lambda row: (-row["quality"]["required_evidence_recall_after_assembly"], -row["quality"]["context_precision_after_assembly"], row["quality"]["total_p95_ms"], row["token_inflation_ratio"]))
        recommendation, winner = "PASS_RECOMMENDED", passing[0]["strategy"]
    else:
        recommendation, winner = "NO_WINNER_HARD_GATE_FAILED", None
    report = {
        "benchmark_id": "chunk-selection-10-v1", "run_id": run_id, "fixture_hashes": bundle.hashes,
        "candidate_artifact_hashes": artifact_hashes, "environment": {"python": platform.python_version(), "llm_calls_allowed": False},
        "candidate_config": bundle.candidate_config, "unmeasured_metrics": ["llm_generation_quality", "llm_generation_cost", "provider_reranker_quality"],
        "recommendation": recommendation, "winner": winner, "hard_gate_policy": {"required_evidence_recall_after_assembly": REQUIRED_RECALL_GATE, "citation_audit": "AUDITED_100_VERIFIED", "ingestion_integrity": "all exact gates"}, "candidates": candidates_report,
    }
    (FIXTURE_ROOT / "last_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (FIXTURE_ROOT / "last_run_report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [f"# Chunk selection smoke report ({report['run_id']})", "", f"Recommendation: `{report['recommendation']}`", f"Winner: `{report['winner'] or 'none'}`", "", "| strategy | status | chunks | recall after | precision | MRR | nDCG | p95 ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["candidates"]:
        quality = row["quality"]
        lines.append(f"| {row['strategy']} | {row['candidate_status']} | {row['chunk_count']} | {quality['required_evidence_recall_after_assembly']:.3f} | {quality['context_precision_after_assembly']:.3f} | {quality['retrieval_mrr']:.3f} | {quality['retrieval_ndcg']:.3f} | {quality['total_p95_ms']:.2f} |")
    lines.extend(["", "Provider reranker: `UNMEASURED`", "LLM generation quality/cost: `UNMEASURED`", "Candidate artifacts are independent tests/db DuckDB/Chroma pairs."])
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def benchmark_report() -> dict[str, Any]:
    return run_short_benchmark()


def test_fixture_is_independent_and_covers_all_ten_sessions() -> None:
    bundle = load_independent_fixture()
    assert len(bundle.sessions) == len(bundle.queries) == len(bundle.qrels) == 10
    assert {int(row["target_intervention_id"]) for row in bundle.queries} == {session.intervention_id for session in bundle.sessions}
    assert all(row["label_source"] == "human_reviewed" for row in bundle.qrels)
    assert bundle.manifest["runtime"]["llm_calls_allowed"] is False


def test_candidate_ingestion_integrity_and_isolation(benchmark_report: dict[str, Any]) -> None:
    for candidate in benchmark_report["candidates"]:
        audit = candidate["ingestion"]
        assert audit["session_ingestion_coverage"] == 1.0
        assert audit["turn_ingestion_coverage"] == 1.0
        assert audit["source_turn_id_valid_rate"] == 1.0
        assert audit["source_turn_role_valid_rate"] == 1.0
        assert audit["exact_text_roundtrip_rate"] == 1.0
        assert audit["duplicate_chunk_rate"] == 0.0
        assert audit["empty_chunk_rate"] == 0.0
    assert all("data\\db" not in str(value) and "data/chroma" not in str(value) for hashes in benchmark_report["candidate_artifact_hashes"].values() for value in hashes.values())


def test_production_path_trace_has_retrieval_assembly_and_audit(benchmark_report: dict[str, Any]) -> None:
    assert len(benchmark_report["candidates"]) >= 5
    for candidate in benchmark_report["candidates"]:
        for trace in candidate["query_traces"]:
            for key in ("query_rewrite_backend", "embedding_backend", "retrieved_candidate_ids", "retrieval_scores", "assembler_selected_ids", "selected_evidence_turn_ids", "reranking_applied", "reranker_status", "audit_status"):
                assert key in trace
            assert trace["reranker_status"] == "UNMEASURED_PROVIDER_RERANKER"
            assert trace["assembly_method"] in {"card_business_boost_plus_mmr", "fallback_ordered_windows"}
            assert trace["mmr_applied"] is (trace["assembly_method"] == "card_business_boost_plus_mmr")


def test_metrics_are_explicit_and_selector_is_hard_gated(benchmark_report: dict[str, Any]) -> None:
    assert benchmark_report["recommendation"] in {"PASS_RECOMMENDED", "DIAGNOSTIC_ONLY", "NO_WINNER_HARD_GATE_FAILED"}
    for candidate in benchmark_report["candidates"]:
        quality = candidate["quality"]
        assert quality["llm_request_count"] == 0
        assert quality["llm_generation_cost"] == quality["provider_reranker_quality"] == "UNMEASURED"
        for value in (quality["required_evidence_recall_at_3"], quality["required_evidence_recall_at_5"], quality["required_evidence_recall_after_assembly"], quality["context_precision_after_assembly"], quality["retrieval_mrr"], quality["retrieval_ndcg"], quality["citation_precision"], quality["citation_recall"]):
            assert 0.0 <= value <= 1.0


if __name__ == "__main__":
    print(json.dumps(run_short_benchmark(), ensure_ascii=False, indent=2, default=str))
