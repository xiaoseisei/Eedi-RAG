from __future__ import annotations

"""Run the complete L1 v2 suite against an isolated Qwen embedding index."""

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.contracts import (
    AssemblerEvalCase,
    ChunkStructuralEvalCase,
    ChunkingEvalCase,
    FusionEvalCase,
    GroundingEvalCase,
    L1ComponentSuiteSpec,
    RewriteEvalCase,
    RetrievalEvalCase,
    RunContext,
    canonical_system_config_hash,
)
from evals.live_storage import hash_storage_artifacts
from evals.adapters.storage import capture_storage_snapshot_from_paths
from evals.reporting import write_report
from evals.suite import run_component_suite
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction


def _index_rows(storage, table: str) -> list[dict[str, Any]]:
    """Read immutable chunk metadata used to derive strategy-specific qrels."""
    if table not in {"misconception_chunks", "tutor_strategy_chunks", "sliding_window_chunks"}:
        raise ValueError(f"unsupported chunk table: {table}")
    return [
        {
            "chunk_id": str(row[0]),
            "session_id": int(row[1]),
            "subject_path": row[2] or "",
            "source_turn_ids": list(row[3] or []),
        }
        for row in storage.duck_conn.execute(
            f"SELECT chunk_id, session_id, subject_path, source_turn_ids FROM {table} ORDER BY chunk_id"
        ).fetchall()
    ]


def _strategy_slot(case: dict[str, Any]) -> str:
    """Select the role-bearing card lane without consulting system output."""
    intent = case.get("category", "UNKNOWN")
    preferred = "misconception" if intent == "STUDENT_INSIGHT" else "strategy"
    speakers = {str(item.get("speaker")) for item in case.get("verbatim_grounding_quotes", [])}
    expected = "student" if preferred == "misconception" else "tutor"
    if expected in speakers:
        return preferred
    return "strategy" if preferred == "misconception" else "misconception"


def _ranked_window_records(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert live retriever output to the minimal Chunking contract shape."""
    from src.chunker import estimate_tokens

    records = []
    for candidate in candidates:
        metadata = candidate.get("metadata", {})
        raw_turn_ids = metadata.get("source_turn_ids", [])
        if isinstance(raw_turn_ids, str):
            raw_turn_ids = json.loads(raw_turn_ids)
        records.append({
            "chunk_id": str(candidate["chunk_id"]),
            "session_id": int(metadata["session_id"]),
            "subject_path": metadata.get("subject_path") or "",
            "source_turn_ids": [int(value) for value in (raw_turn_ids or [])],
            "token_count": estimate_tokens(str(candidate.get("document", ""))),
        })
    return records


def _query_lane_for_intent(intent: str) -> str:
    if intent == "STUDENT_INSIGHT":
        return "misconception"
    if intent == "TUTOR_INTERVENTION":
        return "strategy"
    return "curriculum"


def _rewrite_query(raw_query: str, rewriter, lane: str) -> str:
    result = rewriter.rewrite(raw_query)
    field = {
        "misconception": "misconception_query",
        "strategy": "strategy_query",
        "curriculum": "curriculum_query",
    }.get(lane)
    return getattr(result, field, None) or raw_query


def _protected_tokens(question: str) -> set[str]:
    import re

    return set(re.findall(r"\d+(?:\.\d+)?|[A-D](?![a-z])", question))


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _load_golden() -> list[dict]:
    return json.loads((PROJECT_ROOT / "data/golden_test_set.json").read_text(encoding="utf-8"))


def _target_id(case: dict) -> tuple[str, str]:
    quotes = case.get("verbatim_grounding_quotes") or []
    if not quotes:
        raise ValueError("golden case has no verbatim grounding quotes")
    intent = case.get("category", "UNKNOWN")
    slot = "misconception" if intent == "STUDENT_INSIGHT" else "strategy"
    return f"session_{quotes[0]['session_id']}_{slot if slot == 'misconception' else 'tutor_strategy'}", slot


def build_retrieval_cases(
    golden: list[dict],
    retriever,
    *,
    top_k: int,
    chunk_strategy: str = "card",
    index_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> list[RetrievalEvalCase]:
    if chunk_strategy not in {"card", "fallback"}:
        raise ValueError("chunk_strategy must be card or fallback")
    if index_rows is None:
        index_rows = {
            "misconception": _index_rows(retriever.storage, "misconception_chunks"),
            "strategy": _index_rows(retriever.storage, "tutor_strategy_chunks"),
            "fallback": _index_rows(retriever.storage, "sliding_window_chunks"),
        }
    cases: list[RetrievalEvalCase] = []
    for index, case in enumerate(golden, 1):
        slot = _strategy_slot(case)
        collection = "fallback_windows" if chunk_strategy == "fallback" else (
            "student_misconceptions" if slot == "misconception" else "tutor_strategies"
        )
        query_id = f"eedi-l1-{index:04d}"
        started = time.perf_counter()
        if chunk_strategy == "fallback":
            candidates = retriever.retrieve_fallback_windows(case["question"], top_k=top_k, fetch_evidence=False)
            from evals.graded_qrels import build_window_graded_qrels

            qrel_rows = build_window_graded_qrels({**case, "query_id": query_id}, index_rows["fallback"])
        else:
            candidates = (
                retriever.retrieve_misconceptions(case["question"], top_k=top_k, fetch_evidence=False)
                if slot == "misconception"
                else retriever.retrieve_strategies(case["question"], top_k=top_k, fetch_evidence=False)
            )
            from evals.graded_qrels import build_card_graded_qrels

            qrel_rows = build_card_graded_qrels(
                {**case, "query_id": query_id},
                index_rows[slot],
                target_slot=slot,
            )

        cases.append(RetrievalEvalCase(
            case_id=f"golden-{index:02d}-{chunk_strategy}-qwen-retrieval",
            query=case["question"],
            ranked_ids=[str(item["chunk_id"]) for item in candidates],
            qrels={row["document_id"]: int(row["relevance"]) for row in qrel_rows},
            latency_ms=(time.perf_counter() - started) * 1000.0,
            slices={
                "category": case.get("category", "unknown"),
                "collection": collection,
                "embedding": "qwen3-0.6b",
                "chunk_strategy": chunk_strategy,
            },
        ))
    return cases


def build_card_chunking_cases(golden: list[dict], retriever, index_rows: dict[str, list[dict[str, Any]]]) -> list[dict]:
    """Measure card pointer coverage using the actual card ranking, not windows."""
    from evals.contracts import ChunkRankedRecord, ChunkingEvalCase, TurnKey
    from src.chunker import estimate_tokens

    cases: list[dict] = []
    for index, case in enumerate(golden, 1):
        quotes = case.get("verbatim_grounding_quotes", [])
        if not quotes:
            continue
        slot = _strategy_slot(case)
        expected_speaker = "student" if slot == "misconception" else "tutor"
        target_turns = [
            TurnKey(session_id=int(item["session_id"]), turn_id=int(item["turn_id"]))
            for item in quotes
            if str(item.get("speaker")) == expected_speaker
        ]
        if not target_turns:
            target_turns = [TurnKey(session_id=int(item["session_id"]), turn_id=int(item["turn_id"])) for item in quotes]
        started = time.perf_counter()
        candidates = (
            retriever.retrieve_misconceptions(case["question"], top_k=20, fetch_evidence=False)
            if slot == "misconception"
            else retriever.retrieve_strategies(case["question"], top_k=20, fetch_evidence=False)
        )
        ranked = []
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            raw_turn_ids = metadata.get("source_turn_ids", [])
            if isinstance(raw_turn_ids, str):
                raw_turn_ids = json.loads(raw_turn_ids)
            ranked.append(ChunkRankedRecord(
                chunk_id=str(candidate["chunk_id"]),
                session_id=int(metadata["session_id"]),
                source_turn_ids={int(value) for value in (raw_turn_ids or [])},
                token_count=estimate_tokens(str(candidate.get("document", ""))),
            ))
        # Retrieved candidates are not chunk-generation output; inflation is N/A.
        original_tokens = max(1, sum(
            estimate_tokens(str(item.get("document", "")))
            for item in candidates
        ))
        cases.append(json.loads(ChunkingEvalCase(
            case_id=f"golden-{index:02d}-card-chunk-retrieval",
            query=case["question"],
            target_turns=target_turns,
            ranked_chunks=ranked,
            original_token_count=original_tokens,
            emitted_chunk_token_count=sum(item.token_count for item in ranked),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            inflation_applicable=False,
            slices={"category": case.get("category", "unknown"), "chunk_strategy": "card", "source": "card_actual_ranking"},
        ).model_dump_json()))
    return cases


def build_fallback_rewrite_cases(golden: list[dict], retriever, rewriter, index_rows: list[dict[str, Any]]) -> list[dict]:
    """Run rewrite diagnostics against real windows; fallback does not use fusion."""
    from evals.contracts import RewriteEvalCase
    from evals.graded_qrels import build_window_graded_qrels

    cases: list[dict] = []
    for index, case in enumerate(golden, 1):
        if not case.get("verbatim_grounding_quotes"):
            continue
        query_id = f"eedi-l1-{index:04d}"
        started = time.perf_counter()
        raw = retriever.retrieve_fallback_windows(case["question"], top_k=20, fetch_evidence=False)
        rewritten_query = _rewrite_query(case["question"], rewriter, _query_lane_for_intent(case.get("category", "UNKNOWN")))
        rewritten = retriever.retrieve_fallback_windows(rewritten_query, top_k=20, fetch_evidence=False)
        qrels = build_window_graded_qrels({**case, "query_id": query_id}, index_rows)
        protected = _protected_tokens(case["question"])
        cases.append(json.loads(RewriteEvalCase(
            case_id=f"golden-{index:02d}-fallback-rewrite",
            raw_query=case["question"],
            rewritten_query=rewritten_query,
            protected_tokens=protected,
            critical_entities=set(),
            intent_preserved=None,
            intent_label_source=None,
            domain_injection_expected=False,
            raw_ranked_ids=[str(item["chunk_id"]) for item in raw],
            rewritten_ranked_ids=[str(item["chunk_id"]) for item in rewritten],
            qrels={row["document_id"]: int(row["relevance"]) for row in qrels},
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            slices={"category": case.get("category", "unknown"), "chunk_strategy": "fallback", "rewrite_usage": "not_applicable_content_first"},
        ).model_dump_json()))
    return cases


def build_fallback_fusion_cases(golden: list[dict], retriever, rewriter, index_rows: list[dict[str, Any]]) -> list[dict]:
    """Record the content-first route's explicit no-fusion behavior."""
    from evals.contracts import FusionEvalCase
    from evals.graded_qrels import build_window_graded_qrels

    cases: list[dict] = []
    for index, case in enumerate(golden, 1):
        if not case.get("verbatim_grounding_quotes"):
            continue
        query_id = f"eedi-l1-{index:04d}"
        started = time.perf_counter()
        raw = retriever.retrieve_fallback_windows(case["question"], top_k=20, fetch_evidence=False)
        rewritten_query = _rewrite_query(case["question"], rewriter, _query_lane_for_intent(case.get("category", "UNKNOWN")))
        rewritten = retriever.retrieve_fallback_windows(rewritten_query, top_k=20, fetch_evidence=False)
        qrels = build_window_graded_qrels({**case, "query_id": query_id}, index_rows)
        raw_ids = [str(item["chunk_id"]) for item in raw]
        rewritten_ids = [str(item["chunk_id"]) for item in rewritten]
        cases.append(json.loads(FusionEvalCase(
            case_id=f"golden-{index:02d}-fallback-fusion",
            raw_query=case["question"],
            intent=case.get("category") if case.get("category") in {"STUDENT_INSIGHT", "TUTOR_INTERVENTION", "CONTENT_IMPROVEMENT"} else "UNKNOWN",
            lane_ranked_ids={"raw": raw_ids, "fallback_rewrite_diagnostic": rewritten_ids},
            rewrite_only_ranked_ids=rewritten_ids,
            raw_inclusive_ranked_ids=raw_ids,
            selected_ranked_ids=raw_ids,
            selected_strategy="raw_first",
            qrels={row["document_id"]: int(row["relevance"]) for row in qrels},
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            slices={"category": case.get("category", "unknown"), "chunk_strategy": "fallback", "fusion_usage": "not_applicable_content_first"},
        ).model_dump_json()))
    return cases


def build_fallback_assembler_cases(golden: list[dict], retriever, assembler, index_rows: list[dict[str, Any]]) -> list[dict]:
    """Assemble the real fallback windows and retain their evidence trace."""
    from evals.adapters.assembler import assemble_with_trace
    from evals.contracts import AssemblerEvalCase, EvidenceRef
    from evals.graded_qrels import build_window_graded_qrels

    cases: list[dict] = []
    for index, case in enumerate(golden, 1):
        if not case.get("verbatim_grounding_quotes"):
            continue
        query_id = f"eedi-l1-{index:04d}"
        retrieval_res = {
            "chunk_strategy": "fallback",
            "windows": retriever.retrieve_fallback_windows(case["question"], top_k=20, fetch_evidence=True),
            "misconceptions": [],
            "strategies": [],
            "rewritten_queries": {"extracted_keywords": []},
        }
        started = time.perf_counter()
        trace = assemble_with_trace(case["question"], retrieval_res, assembler)
        context = assembler.assemble(raw_query=case["question"], retrieval_results=retrieval_res)
        qrel_rows = build_window_graded_qrels({**case, "query_id": query_id}, index_rows)
        qrels = {row["document_id"]: int(row["relevance"]) for row in qrel_rows}
        final_ids = [str(item["chunk_id"]) for item in (context.selected_windows or [])]
        slot_by_id = {row["document_id"]: "fallback_window" for row in qrel_rows}
        final_evidence = [EvidenceRef(session_id=int(item["session_id"]), turn_id=int(item["turn_id"]), speaker=item["speaker"], quote_text=item["text"]) for item in trace["final_evidence_turns"]]
        cases.append(json.loads(AssemblerEvalCase(
            case_id=f"golden-{index:02d}-fallback-assembler",
            input_ranked_ids=[str(item["chunk_id"]) for item in retrieval_res["windows"]],
            final_context_ids=final_ids,
            qrels=qrels,
            required_evidence_ids={row["document_id"] for row in qrel_rows if int(row["relevance"]) >= 2},
            slot_by_id=slot_by_id,
            final_evidence_turns=final_evidence,
            actual_prompt_context=trace["actual_prompt_context"],
            trace=trace,
            candidate_char_count=sum(len(str(item.get("document", ""))) for item in retrieval_res["windows"]),
            assembled_char_count=len(context.prompt_context_markdown),
            estimated_tokens=context.estimated_token_count,
            max_prompt_tokens=assembler.max_prompt_tokens,
            evidence_count=len(final_evidence),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            slices={"category": case.get("category", "unknown"), "chunk_strategy": "fallback"},
        ).model_dump_json()))
    return cases


def _attach_split(cases: list, split_assignments: dict[str, str]) -> None:
    for index, case in enumerate(cases, 1):
        query_id = f"eedi-l1-{index:04d}"
        if query_id not in split_assignments:
            raise ValueError(f"split manifest is missing {query_id}")
        case.slices = {**case.slices, "split": split_assignments[query_id]}


def _write_closeout(
    *,
    run_dir: Path,
    report,
    dataset_manifest: dict,
    source_hash: str,
    after_hash: str,
    reproduction_command: str,
) -> dict:
    hard_blockers = [
        item.model_dump(mode="json")
        for item in report.observations
        if item.hard_gate and item.status.value != "SUCCESS"
    ]
    decision = "GO_TO_L2_TECHNICAL" if not hard_blockers else "BLOCKED"
    stage_actions = {
        "storage": "repair index/relational parity or metadata and rebuild a new isolated index",
        "chunk_structure": "repair production chunk pointers/boundaries before retrieval tuning",
        "chunking": "inspect failed window ranking and query/document representation",
        "grounding": "inspect the exact citation quad and final-evidence authorization",
        "rewrite": "obtain human/DeepEval labels or repair protected-token preservation",
        "fusion": "retain raw_first or validate another non-degrading strategy on dev/holdout",
        "retrieval": "improve candidate representation/recall on the frozen qrels",
        "assembler": "inspect retriever_miss versus assembler_drop and slot retention",
    }
    blocker_rows = []
    for item in hard_blockers:
        blocker_rows.append({
            "case_id": item["case_id"],
            "stage": item["stage"],
            "metric": item["metric_name"],
            "scope": item["scope"],
            "value": item["value"],
            "threshold": item["threshold"],
            "status": item["status"],
            "evidence_refs": item.get("evidence_refs", []),
            "action": stage_actions.get(item["stage"], "inspect the recorded case trace"),
        })
    aggregate = [
        item.model_dump(mode="json")
        for item in report.observations
        if item.scope == "aggregate"
    ]
    payload = {
        "schema_version": "l1-closeout/v1",
        "decision": decision,
        "business_quality_passed": False,
        "business_quality_reason": dataset_manifest["qrels_provenance"]["limitation"],
        "run_id": report.run_id,
        "dataset_version": report.dataset_version,
        "qrels_version": report.qrels_version,
        "system_config_hash": report.system_config_hash,
        "status_counts": report.status_counts,
        "artifact_hash_before": source_hash,
        "artifact_hash_after": after_hash,
        "artifact_mutated_during_evaluation": source_hash != after_hash,
        "api_usage": report.system_config.get("embedding_telemetry", {}),
        "api_cost": {
            "status": "UNMEASURED",
            "reason": "provider pricing was not configured; token usage is recorded without fabricating a cost",
        },
        "aggregate_metrics": aggregate,
        "hard_blocker_count": len(blocker_rows),
        "hard_blockers": blocker_rows,
        "reproduction_command": reproduction_command,
        "l2": {
            "admitted": decision == "GO_TO_L2_TECHNICAL",
            "started": False,
            "reason": (
                "L1 hard gates passed; L2 may start in a separate run"
                if decision == "GO_TO_L2_TECHNICAL"
                else "L1 hard gates failed; DeepEval entry is prohibited"
            ),
        },
    }
    (run_dir / "closeout.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage_order = ["storage", "chunk_structure", "chunking", "grounding", "rewrite", "fusion", "retrieval", "assembler"]
    lines = [
        "# Eedi-RAG L1 Closeout",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Run: `{report.run_id}`",
        f"- Dataset: `{report.dataset_version}`",
        f"- Qrels: `{report.qrels_version}`",
        f"- Artifact hash unchanged: `{source_hash == after_hash}`",
        f"- Hard blockers: `{len(blocker_rows)}`",
        "- Business quality claim: `NOT_ALLOWED` (qrels are not independent multi-document graded judgments)",
        "",
    ]
    for stage in stage_order:
        lines.extend([f"## {stage}", ""])
        rows = [item for item in aggregate if item["stage"] == stage]
        if not rows:
            lines.append("No aggregate observation.")
        else:
            lines.extend(["| Metric | Status | Value | Threshold | Hard gate |", "| --- | --- | ---: | ---: | --- |"])
            for item in rows:
                lines.append(
                    f"| {item['metric_name']} | {item['status']} | {item['value']} | {item['threshold']} | {item['hard_gate']} |"
                )
        lines.append("")
    lines.extend(["## Reproduce", "", f"`{reproduction_command}`", ""])
    (run_dir / "closeout.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run complete L1 v2 evaluation on Qwen3-Embedding-0.6B")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, required=True, help="isolated Qwen Chroma index")
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-artifact-hash", type=str, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--chunk-strategy", choices=("card", "fallback"), default="card")
    args = parser.parse_args(argv)
    _load_dotenv_if_available()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    db_path = args.db.resolve(strict=True)
    chroma_path = args.chroma.resolve(strict=True)
    dataset_manifest_path = args.dataset_manifest.resolve(strict=True)
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    split_file = args.split_manifest.resolve(strict=True) if args.split_manifest else dataset_manifest_path.parent / dataset_manifest["split"]["file"]
    split_manifest = json.loads(split_file.read_text(encoding="utf-8"))
    split_assignments = split_manifest["query_assignments"]
    provider = SiliconFlowQwen3EmbeddingFunction.from_env()
    if provider.model != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError(f"expected Qwen/Qwen3-Embedding-0.6B, got {provider.model!r}")

    source_hash = hash_storage_artifacts(db_path, chroma_path)
    expected_hash = args.expected_artifact_hash or dataset_manifest["artifact_hashes"]["qwen_combined_sha256"]
    if source_hash != expected_hash:
        raise ValueError(f"Qwen source hash does not match frozen dataset: expected={expected_hash} actual={source_hash}")
    storage_snapshot = capture_storage_snapshot_from_paths(db_path, chroma_path, case_id="storage-qwen-closeout")
    if hash_storage_artifacts(db_path, chroma_path) != source_hash:
        raise RuntimeError("immutable storage audit changed the Qwen source artifact")
    run_id = args.run_id or datetime.now(UTC).strftime("l1-qwen3-0.6b-full-%Y%m%dT%H%M%SZ")
    golden = _load_golden()
    output_root = args.output_root.resolve()
    scratch_root = output_root / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="l1-qwen-full-", dir=str(scratch_root), ignore_cleanup_errors=True) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)

        from scripts.build_l1_component_datasets import (
            build_assembler_cases,
            build_chunk_retrieval_cases,
            build_chunk_structure_cases,
            build_fusion_cases,
            build_grounding_cases,
            build_rewrite_cases,
            load_sessions,
        )
        from src.query_rewriter import MultiPerspectiveQueryRewriter
        from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
        from src.reranker import PedagogicalGoldAssembler
        from src.retriever import DualMetricRetriever
        from src.storage_manager import DualEngineStorageManager

        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_function=provider,
            embedding_index_version=provider.index_version,
        )
        try:
            retriever = DualMetricRetriever(
                storage_manager=storage,
                query_rewrite_mode="deterministic",
                alpha=0.5,
                fusion_strategy="raw_first",
            )
            assembler = PedagogicalGoldAssembler(
                lambda_diversity=0.7,
                max_prompt_tokens=1500,
                chunk_strategy=args.chunk_strategy,
            )
            pipeline = EndToEndPedagogicalRAGPipeline(
                retriever=retriever,
                assembler=assembler,
                chunk_strategy=args.chunk_strategy,
            )
            rewriter = MultiPerspectiveQueryRewriter(mode="deterministic")
            sessions = load_sessions()
            indexed_session_ids = {
                int(row[0])
                for row in storage.duck_conn.execute("SELECT DISTINCT session_id FROM sliding_window_chunks").fetchall()
            }
            index_rows = {
                "misconception": _index_rows(storage, "misconception_chunks"),
                "strategy": _index_rows(storage, "tutor_strategy_chunks"),
                "fallback": _index_rows(storage, "sliding_window_chunks"),
            }
            retrieval_cases = build_retrieval_cases(
                golden,
                retriever,
                top_k=args.top_k,
                chunk_strategy=args.chunk_strategy,
                index_rows=index_rows,
            )
            grounding_cases = [GroundingEvalCase.model_validate(item) for item in build_grounding_cases(golden, sessions, pipeline)]
            chunk_structure_cases = [ChunkStructuralEvalCase.model_validate(item) for item in build_chunk_structure_cases(sessions, indexed_session_ids)]
            if args.chunk_strategy == "fallback":
                chunking_cases = [ChunkingEvalCase.model_validate(item) for item in build_chunk_retrieval_cases(golden, retriever)]
                rewrite_cases = [RewriteEvalCase.model_validate(item) for item in build_fallback_rewrite_cases(golden, retriever, rewriter, index_rows["fallback"])]
                fusion_cases = [FusionEvalCase.model_validate(item) for item in build_fallback_fusion_cases(golden, retriever, rewriter, index_rows["fallback"])]
                assembler_cases = [AssemblerEvalCase.model_validate(item) for item in build_fallback_assembler_cases(golden, retriever, assembler, index_rows["fallback"])]
            else:
                chunking_cases = [ChunkingEvalCase.model_validate(item) for item in build_card_chunking_cases(golden, retriever, index_rows)]
                rewrite_cases = [RewriteEvalCase.model_validate(item) for item in build_rewrite_cases(golden, sessions, retriever, rewriter)]
                fusion_cases = [FusionEvalCase.model_validate(item) for item in build_fusion_cases(golden, retriever)]
                assembler_cases = [AssemblerEvalCase.model_validate(item) for item in build_assembler_cases(golden, sessions, retriever, assembler, top_k_each=20)]
            for cases in (retrieval_cases, grounding_cases, chunking_cases, rewrite_cases, fusion_cases, assembler_cases):
                _attach_split(cases, split_assignments)
            for case in chunk_structure_cases:
                case.slices = {**case.slices, "split": "all_indexed"}
        finally:
            storage.close()
            try:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()

    after_hash = hash_storage_artifacts(db_path, chroma_path)
    if after_hash != source_hash:
        raise RuntimeError("source Qwen artifact changed during complete L1 evaluation")

    system_config = {
        "component": "l1-closeout-all-components",
        "embedding_backend": "siliconflow_openai_compatible",
        "embedding_model": provider.model,
        "embedding_dimension": provider.dimension,
        "embedding_telemetry": provider.telemetry,
        "embedding_provider": provider.name(),
        "embedding_base_url": provider.base_url,
        "embedding_index_version": provider.index_version,
        "embedding_batch_size": provider.batch_size,
        "embedding_timeout_seconds": provider.timeout_seconds,
        "embedding_cache_key_fields": ["provider", "model", "index_version", "query_text"],
        "embedding_cache_success_only": True,
        "query_rewrite_mode": "deterministic_rules",
        "retrieval_backend": "dense-chroma-plus-dialogue-numeric-formula-exact",
        "fusion_strategy": "raw_first",
        "chunk_strategy": args.chunk_strategy,
        "rewrite_fusion_semantics": (
            "multi_perspective_rrf_on_role_card_lanes"
            if args.chunk_strategy == "card"
            else "not_applicable_content_first; raw_window_ranking_retained_as_diagnostic"
        ),
        "rrf_k": 60,
        "mmr_lambda": 0.7,
        "assembler_top_k_each": 20,
        "assembler_max_prompt_tokens": 1500,
        "top_k": args.top_k,
        "artifact_hash": source_hash,
        "source_access": "isolated-artifact-copy",
        "dataset_manifest_sha256": __import__("hashlib").sha256(dataset_manifest_path.read_bytes()).hexdigest(),
    }
    context = RunContext(
        run_id=run_id,
        dataset_version=dataset_manifest["dataset_version"],
        qrels_version=dataset_manifest["qrels_version"],
        system_config_hash=canonical_system_config_hash(system_config),
        system_config=system_config,
    )
    report = run_component_suite(L1ComponentSuiteSpec(
        schema_version="l1-component-suite/v2",
        context=context,
        storage=storage_snapshot,
        grounding_cases=grounding_cases,
        chunking_cases=chunking_cases,
        chunk_structural_cases=chunk_structure_cases,
        rewrite_cases=rewrite_cases,
        fusion_cases=fusion_cases,
        retrieval_cases=retrieval_cases,
        assembler_cases=assembler_cases,
    ))
    run_dir = write_report(report, output_root)
    (run_dir / "embedding_usage.json").write_text(json.dumps({
        "provider": provider.name(),
        "model": provider.model,
        "dimension": provider.dimension,
        "usage": provider.telemetry,
        "artifact_hash": source_hash,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": report.run_id,
        "release_status": report.release_status,
        "status_counts": report.status_counts,
        "report_dir": str(run_dir),
        "artifact_hash": source_hash,
        "embedding_model": provider.model,
        "embedding_dimension": provider.dimension,
        "embedding_usage": provider.telemetry,
    }, ensure_ascii=False))
    reproduction_command = (
        f".venv\\Scripts\\python.exe scripts\\run_l1_qwen_full_eval.py --db {args.db} "
        f"--chroma {args.chroma} --dataset-manifest {args.dataset_manifest} "
        f"--output-root {args.output_root} --run-id <new-unique-run-id> --top-k {args.top_k}"
        f" --chunk-strategy {args.chunk_strategy}"
    )
    closeout = _write_closeout(
        run_dir=run_dir,
        report=report,
        dataset_manifest=dataset_manifest,
        source_hash=source_hash,
        after_hash=after_hash,
        reproduction_command=reproduction_command,
    )
    print(json.dumps({"closeout_decision": closeout["decision"], "hard_blockers": closeout["hard_blocker_count"]}, ensure_ascii=False))
    return 0 if closeout["decision"] == "GO_TO_L2_TECHNICAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
