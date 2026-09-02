"""
================================================================================
脚本名称: scripts/build_l1_component_datasets.py
业务定位: 从 30 题黄金集构建 L1 四组件评测数据集 (grounding / chunking / rewrite / assembler)
构建原则:
  1. 所有 ranked_ids / rewritten_query / final_context 均来自真实系统执行，不手写模拟
  2. qrels 只来自黄金集 verbatim_grounding_quotes (人工标注的相关 session)
  3. 原文引用一律从 cleaned_sessions.jsonl 逐字取回
输出:
  evals/datasets/grounding-v1.json
  evals/datasets/chunking-v1.json
  evals/datasets/rewrite-v1.json
  evals/datasets/assembler-v1.json
================================================================================
"""

from __future__ import annotations

import json
import re
import sys
import time
import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evals.contracts import (
    AssemblerEvalCase,
    ChunkRankedRecord,
    ChunkingEvalCase,
    EvidenceRef,
    FusionEvalCase,
    GroundingEvalCase,
    RetrievalEvalCase,
    RewriteEvalCase,
    TurnKey,
    canonical_system_config_hash,
)
from evals.adapters.chunking import (
    audit_production_chunk_structure,
    capture_production_chunks,
    capture_production_chunk_trace,
)
from evals.adapters.assembler import assemble_with_trace
from evals.adapters.grounding import capture_system_grounding_trace
from src.models import CleanedSession
from src.chunker import estimate_tokens
from src.storage_manager import DualEngineStorageManager
from evals.live_storage import hash_storage_artifacts

GOLDEN_PATH = Path("data/golden_test_set.json")
SESSIONS_PATH = Path("data/cleaned_sessions.jsonl")


def load_golden_cases() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def load_sessions() -> dict[int, CleanedSession]:
    sessions: dict[int, CleanedSession] = {}
    with SESSIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = CleanedSession.model_validate(json.loads(line))
                sessions[s.intervention_id] = s
    return sessions


def turn_to_ref(turn) -> EvidenceRef:
    return EvidenceRef(
        session_id=turn.session_ref if hasattr(turn, "session_ref") else 0,
        turn_id=turn.turn_id,
        speaker="tutor" if turn.is_tutor else "student",
        quote_text=turn.text,
    )


def order_ranked_chunks(records: list[ChunkRankedRecord]) -> list[ChunkRankedRecord]:
    """Preserve the production/ranker order; never move target hits to the front."""
    return list(records)


def query_lane_for_intent(intent: str) -> str:
    if intent == "STUDENT_INSIGHT":
        return "misconception"
    if intent == "TUTOR_INTERVENTION":
        return "strategy"
    return "curriculum"


def build_grounding_cases(golden: list[dict], sessions: dict[int, CleanedSession], pipeline) -> list[dict]:
    """Capture citations from an explicit real deterministic system execution."""
    cases = []
    for idx, case in enumerate(golden, 1):
        quotes = case.get("verbatim_grounding_quotes", [])
        if not quotes:
            continue
        session_ids = sorted({q["session_id"] for q in quotes})

        authoritative: list[EvidenceRef] = []
        for sid in session_ids:
            sess = sessions.get(sid)
            if sess is None:
                raise ValueError(f"golden case {idx} 引用的 session {sid} 不在 cleaned_sessions 中")
            for t in sess.turns:
                authoritative.append(EvidenceRef(
                    session_id=sid,
                    turn_id=t.turn_id,
                    speaker="tutor" if t.is_tutor else "student",
                    quote_text=t.text,
                ))

        required = [
            EvidenceRef(
                session_id=q["session_id"],
                turn_id=q["turn_id"],
                speaker=q["speaker"],
                quote_text=q["quote_text"],
            )
            for q in quotes
        ]
        execution_error = None
        try:
            trace = capture_system_grounding_trace(pipeline, case["question"])
        except Exception as exc:
            # A real retrieval/generation failure is an observable result, not
            # a reason to fabricate citations.  Keep an empty output so the
            # runner reports recall=0 and denominator-limited precision.
            execution_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING grounding case {idx} execution failed: {execution_error}")
            trace = {"output_citations": [], "retrieved_evidence": []}
        cases.append(json.loads(GroundingEvalCase(
            case_id=f"golden-{idx:02d}-grounding",
            output_citations=[EvidenceRef.model_validate(item) for item in trace["output_citations"]],
            authoritative_turns=authoritative,
            required_evidence=required,
            retrieved_evidence=[EvidenceRef.model_validate(item) for item in trace["retrieved_evidence"]],
            execution_error=execution_error,
            quote_match_mode="substring",
            slices={"category": case.get("category", "unknown")},
        ).model_dump_json()))
    return cases


def _chunk_ranked_records(sess: CleanedSession, window_size: int = 6, step: int = 3) -> tuple[list[ChunkRankedRecord], int, int]:
    """对单个 session 做滑动窗口切块，返回排名记录与原始 token 数。"""
    records = capture_production_chunks(sess, window_size=window_size, step=step)
    original_tokens = max(1, sum(estimate_tokens(t.text) for t in sess.turns))
    emitted = sum(item.token_count for item in records)
    return records, original_tokens, emitted


def build_chunking_cases(golden: list[dict], sessions: dict[int, CleanedSession]) -> list[dict]:
    """
    chunking 语义: 目标 session 的生产滑动窗口是否覆盖黄金引用的 turn。
    ranked_chunks 保持生产/真实检索顺序，不进行目标命中 oracle 排序。
    """
    cases = []
    for idx, case in enumerate(golden, 1):
        quotes = case.get("verbatim_grounding_quotes", [])
        if not quotes:
            continue
        target_turns = [TurnKey(session_id=q["session_id"], turn_id=q["turn_id"]) for q in quotes]
        session_ids = sorted({q["session_id"] for q in quotes})

        ranked: list[ChunkRankedRecord] = []
        original_total = 0
        emitted_total = 0
        for sid in session_ids:
            sess = sessions[sid]
            recs, orig, emitted = _chunk_ranked_records(sess)
            original_total += orig
            emitted_total += emitted
            ranked.extend(order_ranked_chunks(recs))

        started = time.perf_counter()
        _ = capture_production_chunks(sessions[session_ids[0]])
        latency_ms = (time.perf_counter() - started) * 1000.0

        cases.append(json.loads(ChunkingEvalCase(
            case_id=f"golden-{idx:02d}-chunking",
            query=case["question"],
            target_turns=target_turns,
            ranked_chunks=ranked,
            original_token_count=max(original_total, 1),
            emitted_chunk_token_count=emitted_total,
            latency_ms=round(latency_ms, 3),
            slices={"category": case.get("category", "unknown")},
        ).model_dump_json()))
    return cases


def build_chunk_structure_cases(sessions: dict[int, CleanedSession], indexed_session_ids: set[int]) -> list[dict]:
    """Structural checks intentionally have no query or retriever ranking input."""
    return [
        json.loads(audit_production_chunk_structure(sessions[session_id]).model_dump_json())
        for session_id in sorted(indexed_session_ids)
        if session_id in sessions
    ]


def build_chunk_retrieval_cases(golden: list[dict], retriever) -> list[dict]:
    """Use the actual fallback-window ranking from the isolated index."""
    cases = []
    for index, case in enumerate(golden, 1):
        quotes = case.get("verbatim_grounding_quotes", [])
        if not quotes:
            continue
        started = time.perf_counter()
        candidates = retriever.retrieve_fallback_windows(case["question"], top_k=50)
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked = []
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            raw_turns = metadata.get("source_turn_ids", [])
            source_turn_ids = json.loads(raw_turns) if isinstance(raw_turns, str) else raw_turns
            ranked.append(ChunkRankedRecord(
                chunk_id=str(candidate["chunk_id"]),
                session_id=int(metadata["session_id"]),
                source_turn_ids=set(source_turn_ids),
                token_count=estimate_tokens(str(candidate.get("document", ""))),
            ))
        target_turns = [TurnKey(session_id=q["session_id"], turn_id=q["turn_id"]) for q in quotes]
        cases.append(json.loads(ChunkingEvalCase(
            case_id=f"golden-{index:02d}-chunk-retrieval",
            query=case["question"],
            target_turns=target_turns,
            ranked_chunks=ranked,
            original_token_count=max(1, sum(estimate_tokens(str(t.get("text", ""))) for t in retriever.storage.get_dialogue_turns(quotes[0]["session_id"]))),
            emitted_chunk_token_count=sum(item.token_count for item in ranked),
            latency_ms=round(latency_ms, 3),
            inflation_applicable=False,
            slices={"category": case.get("category", "unknown"), "source": "fallback_windows_actual_ranking"},
        ).model_dump_json()))
    return cases


def build_document_qrels(case: dict, *, query_id: str) -> list[dict]:
    """Deduplicate quote evidence into derived document-level qrels.

    Human labels identify evidence turns.  The document relevance is derived
    from those labels, so provenance must not claim a direct human document
    judgment.
    """
    intent = case.get("category", "UNKNOWN")
    suffix = "misconception" if intent == "STUDENT_INSIGHT" else "tutor_strategy"
    grouped: dict[str, set[tuple[int, int]]] = {}
    for quote in case.get("verbatim_grounding_quotes", []):
        doc_id = f"session_{quote['session_id']}_{suffix}"
        grouped.setdefault(doc_id, set()).add((int(quote["session_id"]), int(quote["turn_id"])))
    return [
        {
            "query_id": query_id,
            "document_id": doc_id,
            "slot": "strategy" if suffix == "tutor_strategy" else "misconception",
            "relevance": 3,
            "label_source": "derived_from_human_quote",
            "relevance_source": "human_verbatim_grounding_quote",
            "evidence_turns": [
                {"session_id": sid, "turn_id": tid}
                for sid, tid in sorted(turns)
            ],
        }
        for doc_id, turns in sorted(grouped.items())
    ]


def build_assembler_qrels(case: dict, *, query_id: str) -> list[dict]:
    """Build both fixed assembler slot targets for every cited session."""
    base = build_document_qrels(case, query_id=query_id)
    sessions = sorted({q["session_id"] for q in case.get("verbatim_grounding_quotes", [])})
    evidence = [
        {"session_id": int(q["session_id"]), "turn_id": int(q["turn_id"])}
        for q in case.get("verbatim_grounding_quotes", [])
    ]
    rows = {row["document_id"]: row for row in base}
    for sid in sessions:
        for suffix, slot in (("misconception", "misconception"), ("tutor_strategy", "strategy")):
            doc_id = f"session_{sid}_{suffix}"
            rows.setdefault(doc_id, {
                "query_id": query_id,
                "document_id": doc_id,
                "slot": slot,
                "relevance": 3,
                "label_source": "derived_from_human_quote",
                "relevance_source": "human_verbatim_grounding_quote_session",
                "evidence_turns": evidence,
            })
    return [rows[key] for key in sorted(rows)]


def _ranked_session_ids(retriever, query: str, collection: str, top_k: int = 20) -> list[str]:
    """执行真实检索，返回按排名排列的 chunk_id 列表。"""
    if collection == "student_misconceptions":
        candidates = retriever.retrieve_misconceptions(query, top_k=top_k, fetch_evidence=False)
    else:
        candidates = retriever.retrieve_strategies(query, top_k=top_k, fetch_evidence=False)
    return [str(c["chunk_id"]) for c in candidates]


def _rewrite_query(raw_query: str, rewriter, lane: str = "misconception") -> str:
    """Use the lane selected by intent; never force every query through misconception."""
    result = rewriter.rewrite(raw_query)
    field = {
        "misconception": "misconception_query",
        "strategy": "strategy_query",
        "curriculum": "curriculum_query",
    }.get(lane)
    return getattr(result, field, None) or raw_query


def _extract_protected_tokens(question: str) -> set[str]:
    """题目中的数字与字母选项是必须保留的关键 token。"""
    return set(re.findall(r"\d+(?:\.\d+)?|[A-D](?![a-z])", question))


def build_rewrite_cases(
    golden: list[dict],
    sessions: dict[int, CleanedSession],
    retriever,
    rewriter,
) -> list[dict]:
    """
    rewrite 语义: 改写后的 query 检索效果不应差于原始 query。
    ranked_ids 来自真实检索 (raw vs rewritten 各跑一次)。
    qrels: 黄金 session 的卡片 (理想目标 = 3)。
    """
    # 预建 session -> 卡片 id 映射
    card_ids_by_session: dict[int, set[str]] = {}
    for sid in sessions:
        card_ids_by_session[sid] = {f"session_{sid}_misconception", f"session_{sid}_tutor_strategy"}

    cases = []
    for idx, case in enumerate(golden, 1):
        raw_query = case["question"]
        quotes = case.get("verbatim_grounding_quotes", [])
        if not quotes:
            continue
        target_session = quotes[0]["session_id"]
        collection = (
            "student_misconceptions" if case.get("category") == "STUDENT_INSIGHT"
            else "tutor_strategies"
        )
        target_suffix = "misconception" if collection == "student_misconceptions" else "tutor_strategy"
        target_id = f"session_{target_session}_{target_suffix}"

        started = time.perf_counter()
        raw_ranked = _ranked_session_ids(retriever, raw_query, collection, top_k=20)
        intent = case.get("category", "UNKNOWN")
        lane = query_lane_for_intent(intent)
        rewritten_query = _rewrite_query(raw_query, rewriter, lane)
        rewritten_ranked = _ranked_session_ids(retriever, rewritten_query, collection, top_k=20)
        latency_ms = (time.perf_counter() - started) * 1000.0

        protected = _extract_protected_tokens(raw_query)
        # protected token 必须出现在 raw_query 中 (契约校验)
        protected = {t for t in protected if t.casefold() in raw_query.casefold()}
        entities = {t for t in protected if not t.isdigit() and "." not in t}

        cases.append(json.loads(RewriteEvalCase(
            case_id=f"golden-{idx:02d}-rewrite",
            raw_query=raw_query,
            rewritten_query=rewritten_query,
            protected_tokens=protected,
            critical_entities=entities,
            intent_preserved=None,
            intent_label_source=None,
            domain_injection_expected=False,
            raw_ranked_ids=raw_ranked,
            rewritten_ranked_ids=rewritten_ranked,
            qrels={row["document_id"]: row["relevance"] for row in build_document_qrels(case, query_id=f"eedi-l1-{idx:04d}")},
            latency_ms=round(latency_ms, 3),
            slices={"category": case.get("category", "unknown")},
        ).model_dump_json()))
    return cases


def build_fusion_cases(golden: list[dict], retriever) -> list[dict]:
    """Capture paired raw, lane, rewrite-only and raw-inclusive RRF results."""
    cases = []
    for index, case in enumerate(golden, 1):
        quotes = case.get("verbatim_grounding_quotes", [])
        if not quotes:
            continue
        intent = case.get("category", "UNKNOWN")
        collection = "student_misconceptions" if intent == "STUDENT_INSIGHT" else "tutor_strategies"
        suffix = "misconception" if collection == "student_misconceptions" else "tutor_strategy"
        target_id = f"session_{quotes[0]['session_id']}_{suffix}"
        started = time.perf_counter()
        raw_ranked = _ranked_session_ids(retriever, case["question"], collection, top_k=20)
        result = retriever.retrieve_multi_perspective_rrf(case["question"], top_k_each=20, fetch_evidence=False)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prefix = "misconception" if collection == "student_misconceptions" else "strategy"
        trace = result["trace"]
        lanes = trace["lanes"]
        cases.append(json.loads(FusionEvalCase(
            case_id=f"golden-{index:02d}-fusion",
            raw_query=case["question"],
            intent=intent if intent in {"STUDENT_INSIGHT", "TUTOR_INTERVENTION", "CONTENT_IMPROVEMENT"} else "UNKNOWN",
            lane_ranked_ids={
                "raw": raw_ranked,
                prefix: [item["chunk_id"] for item in lanes[prefix]],
                "curriculum": [item["chunk_id"] for item in lanes[f"{prefix}_curriculum"]],
            },
            rewrite_only_ranked_ids=trace["fused_rankings"][f"{prefix}_rewrite_only"],
            raw_inclusive_ranked_ids=trace["fused_rankings"][f"{prefix}_raw_inclusive"],
            qrels={row["document_id"]: row["relevance"] for row in build_document_qrels(case, query_id=f"eedi-l1-{index:04d}")},
            latency_ms=round(latency_ms, 3),
            slices={"category": intent, "collection": collection},
        ).model_dump_json()))
    return cases


def build_assembler_cases(
    golden: list[dict],
    sessions: dict[int, CleanedSession],
    retriever,
    assembler,
) -> list[dict]:
    """
    assembler 语义: MMR 装配后黄金目标卡是否保留、选择精度、压缩率、预算。
    input = 真实检索候选 (两集合合并)，final = 真实 MMR 输出。
    """
    cases = []
    for idx, case in enumerate(golden, 1):
        query = case["question"]
        quotes = case.get("verbatim_grounding_quotes", [])
        if not quotes:
            continue
        target_session = quotes[0]["session_id"]
        collection = (
            "student_misconceptions" if case.get("category") == "STUDENT_INSIGHT"
            else "tutor_strategies"
        )
        target_suffix = "misconception" if collection == "student_misconceptions" else "tutor_strategy"
        target_id = f"session_{target_session}_{target_suffix}"

        started = time.perf_counter()
        retrieval_res = retriever.retrieve_multi_perspective_rrf(
            query, top_k_each=5, fetch_evidence=True
        )
        candidates = retrieval_res.get("misconceptions", []) + retrieval_res.get("strategies", [])
        input_ranked = [str(c["chunk_id"]) for c in candidates]

        trace = assemble_with_trace(query, retrieval_res, assembler)
        gold_ctx = assembler.assemble(raw_query=query, retrieval_results=retrieval_res)
        final_ids: list[str] = []
        for key in ("selected_misconception", "selected_strategy"):
            obj = getattr(gold_ctx, key, None)
            if obj and obj.get("chunk_id"):
                final_ids.append(str(obj["chunk_id"]))
        latency_ms = (time.perf_counter() - started) * 1000.0

        candidate_chars = sum(len(c.get("document", "")) for c in candidates)
        assembled_chars = sum(
            len(candidates[i].get("document", ""))
            for cid in final_ids
            for i, c in enumerate(candidates)
            if str(c["chunk_id"]) == cid
        )

        slot_by_id = {
            str(c["chunk_id"]): ("misconception" if c.get("metadata", {}).get("chunk_type", "").startswith("mis") or "misconception" in str(c.get("chunk_id", "")) else "strategy")
            for c in candidates
            if c.get("chunk_id")
        }
        qrel_rows = build_assembler_qrels(case, query_id=f"eedi-l1-{idx:04d}")
        qrels = {row["document_id"]: row["relevance"] for row in qrel_rows}
        slot_by_id.update({
            row["document_id"]: row["slot"]
            for row in qrel_rows
        })

        cases.append(json.loads(AssemblerEvalCase(
            case_id=f"golden-{idx:02d}-assembler",
            input_ranked_ids=input_ranked,
            final_context_ids=final_ids,
            qrels=qrels,
            required_evidence_ids=set(qrels),
            slot_by_id=slot_by_id,
            final_evidence_turns=[EvidenceRef(session_id=t["session_id"], turn_id=t["turn_id"], speaker=t["speaker"], quote_text=t["text"]) for t in trace["final_evidence_turns"]],
            actual_prompt_context=trace["actual_prompt_context"],
            trace=trace,
            candidate_char_count=candidate_chars,
            assembled_char_count=assembled_chars,
            estimated_tokens=assembled_chars // 4,
            max_prompt_tokens=1500,
            evidence_count=len(trace["final_evidence_turns"]),
            latency_ms=round(latency_ms, 3),
            slices={"category": case.get("category", "unknown")},
        ).model_dump_json()))
    return cases


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory has no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def build_manifest_provenance(
    *,
    db_path: Path,
    chroma_path: Path,
    source_session_count: int,
    indexed_session_count: int,
    golden_case_count: int,
    system_config: dict,
    source_files: dict[str, str] | None = None,
    component_case_counts: dict[str, int] | None = None,
    message_count: int | None = None,
    input_artifact_hashes: dict[str, str] | None = None,
) -> dict:
    """Build manifest provenance from the exact artifacts used for the run."""
    db_path = db_path.resolve(strict=True)
    chroma_path = chroma_path.resolve(strict=True)
    post_db_hash = _file_sha256(db_path)
    post_chroma_hash = _directory_sha256(chroma_path)
    post_combined_hash = hash_storage_artifacts(db_path, chroma_path)
    return {
        "schema_version": "l1-component-suite/v2",
        "dataset_version": "golden30-v2",
        "qrels_version": "golden-verbatim-v2-derived-document-v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_files": source_files or {},
        "source_session_count": source_session_count,
        "indexed_session_count": indexed_session_count,
        "golden_case_count": golden_case_count,
        "message_count": message_count,
        "component_case_counts": component_case_counts or {},
        "input_artifact_hashes": input_artifact_hashes or {},
        "artifact_hashes": {
            "duckdb_sha256": post_db_hash,
            "chroma_sha256": post_chroma_hash,
            "combined_sha256": post_combined_hash,
        },
        "artifact_mutated_during_build": bool(
            input_artifact_hashes
            and input_artifact_hashes.get("combined_sha256")
            != post_combined_hash
        ),
        "system_config_hash": canonical_system_config_hash(system_config),
        "embedding": {
            "backend": system_config.get("embedding_backend"),
            "model": "FastDeterministicEmbeddingFunction",
            "version": "128d-v1",
        },
        "retrieval_config": {
            "query_rewrite_mode": system_config.get("query_rewrite_mode"),
            "mmr_lambda": system_config.get("mmr_lambda"),
            "retrieval_backend": "dense-only-chroma",
        },
        "grounding_output_policy": "explicit-deterministic-system-trace",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned L1 component evaluation datasets")
    parser.add_argument("--schema-version", default="l1-component-suite/v2")
    parser.add_argument("--output", "--output-dir", dest="output_dir", default="evals/datasets/l1-v2", type=Path)
    parser.add_argument("--db", required=True, type=Path, help="isolated DuckDB artifact copy")
    parser.add_argument("--chroma", required=True, type=Path, help="isolated Chroma artifact copy")
    args = parser.parse_args(argv)
    if args.schema_version not in {"l1-component-suite/v1", "l1-component-suite/v2"}:
        parser.error("--schema-version must be l1-component-suite/v1 or l1-component-suite/v2")
    if not args.db.is_file():
        parser.error(f"--db must name an existing file: {args.db}")
    if not args.chroma.is_dir():
        parser.error(f"--chroma must name an existing directory: {args.chroma}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    golden = load_golden_cases()
    sessions = load_sessions()
    print(f"黄金集: {len(golden)} 题 | 会话库: {len(sessions)} 场")

    input_artifact_hashes = {
        "duckdb_sha256": _file_sha256(args.db.resolve(strict=True)),
        "chroma_sha256": _directory_sha256(args.chroma.resolve(strict=True)),
        "combined_sha256": hash_storage_artifacts(args.db, args.chroma),
    }

    storage = DualEngineStorageManager(
        db_path=str(args.db),
        chroma_dir=str(args.chroma),
        embedding_backend="deterministic",
    )
    from src.retriever import DualMetricRetriever
    from src.reranker import PedagogicalGoldAssembler
    from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
    from src.query_rewriter import MultiPerspectiveQueryRewriter

    retriever = DualMetricRetriever(
        storage_manager=storage,
        query_rewrite_mode="deterministic",
        alpha=0.5,
    )
    assembler = PedagogicalGoldAssembler(lambda_diversity=0.7)
    pipeline = EndToEndPedagogicalRAGPipeline(retriever=retriever, assembler=assembler)
    rewriter = MultiPerspectiveQueryRewriter(mode="deterministic")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    indexed_session_ids = {
        int(row[0])
        for row in storage.duck_conn.execute("SELECT DISTINCT session_id FROM sliding_window_chunks").fetchall()
    }

    datasets = {
        "grounding.json": build_grounding_cases(golden, sessions, pipeline),
        "chunk_structure.json": build_chunk_structure_cases(sessions, indexed_session_ids),
        "chunking.json": build_chunk_retrieval_cases(golden, retriever),
        "rewrite.json": build_rewrite_cases(golden, sessions, retriever, rewriter),
        "fusion.json": build_fusion_cases(golden, retriever),
        "assembler.json": build_assembler_cases(golden, sessions, retriever, assembler),
    }

    for name, cases in datasets.items():
        system_config = {
            "source": "golden_test_set.json",
            "case_count": len(cases),
            "embedding_backend": "deterministic",
            "query_rewrite_mode": "deterministic_rules",
            "mmr_lambda": 0.7,
        }
        payload = {
            "schema_version": args.schema_version,
            "context": {
                "run_id": f"golden30-{name.replace('.json', '')}",
                "dataset_version": "golden30-v2",
                "qrels_version": "golden-verbatim-v2-derived-document-v1",
                "system_config_hash": canonical_system_config_hash(system_config),
                "system_config": system_config,
            },
        }
        key_map = {
            "grounding.json": "grounding_cases",
            "chunk_structure.json": "chunk_structural_cases",
            "chunking.json": "chunking_cases",
            "rewrite.json": "rewrite_cases",
            "fusion.json": "fusion_cases",
            "assembler.json": "assembler_cases",
        }
        payload[key_map[name]] = cases
        (out_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ {name}: {len(cases)} cases")

    # Unified executable suite, plus versioned query/qrels/grounding streams
    # used by downstream controls.  These files are generated from the same
    # run and therefore cannot drift through hand-edited aggregate counts.
    suite_config = {
        "source": "golden_test_set.json",
        "component_case_counts": {name.removesuffix(".json"): len(rows) for name, rows in datasets.items()},
        "embedding_backend": "deterministic",
        "query_rewrite_mode": "deterministic_rules",
        "mmr_lambda": 0.7,
    }
    suite_payload = {
        "schema_version": args.schema_version,
        "context": {
            "run_id": "l1-v2-suite",
            "dataset_version": "golden30-v2",
            "qrels_version": "golden-verbatim-v2-derived-document-v1",
            "system_config_hash": canonical_system_config_hash(suite_config),
            "system_config": suite_config,
        },
        "grounding_cases": datasets["grounding.json"],
        "chunk_structural_cases": datasets["chunk_structure.json"],
        "chunking_cases": datasets["chunking.json"],
        "rewrite_cases": datasets["rewrite.json"],
        "fusion_cases": datasets["fusion.json"],
        "assembler_cases": datasets["assembler.json"],
    }
    (out_dir / "suite.json").write_text(json.dumps(suite_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(golden, 1):
            if item.get("question"):
                handle.write(json.dumps({
                    "query_id": f"eedi-l1-{idx:04d}",
                    "query": item["question"],
                    "intent": item.get("category", "UNKNOWN"),
                    "split": "dev",
                    "source_session_ids": sorted({q["session_id"] for q in item.get("verbatim_grounding_quotes", [])}),
                    "slices": {"category": item.get("category", "unknown")},
                }, ensure_ascii=False) + "\n")
    with (out_dir / "qrels.jsonl").open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(golden, 1):
            for row in build_document_qrels(item, query_id=f"eedi-l1-{idx:04d}"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "grounding.jsonl").open("w", encoding="utf-8") as handle:
        for case in datasets["grounding.json"]:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    # Release DuckDB/Chroma handles before hashing artifacts on Windows.
    # Hashing an open DuckDB file is a construction error, not an evaluation
    # result; close the evaluator-owned connection explicitly first.
    storage.close()
    manifest = build_manifest_provenance(
        db_path=args.db,
        chroma_path=args.chroma,
        source_session_count=len(sessions),
        indexed_session_count=len(indexed_session_ids),
        golden_case_count=len(golden),
        message_count=sum(len(session.turns) for session in sessions.values()),
        component_case_counts={name.removesuffix(".json"): len(rows) for name, rows in datasets.items()},
        source_files={
            str(path): _file_sha256(path)
            for path in (GOLDEN_PATH, SESSIONS_PATH)
            if path.exists()
        },
        system_config=suite_config,
        input_artifact_hashes=input_artifact_hashes,
    )
    manifest["schema_version"] = args.schema_version
    qrel_rows = [
        row
        for idx, item in enumerate(golden, 1)
        for row in build_document_qrels(item, query_id=f"eedi-l1-{idx:04d}")
    ]
    manifest["qrels_provenance"] = {
        "document_labels": "derived_from_human_quote",
        "multi_relevance_annotated": False,
        "quote_evidence_rows": sum(len(item.get("verbatim_grounding_quotes", [])) for item in golden),
        "unique_query_document_pairs": len({(row["query_id"], row["document_id"]) for row in qrel_rows}),
        "limitation": "document relevance is derived from human quote/session evidence; independent multi-document human judgments are not present",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n全部数据集构建完成")


if __name__ == "__main__":
    main()
