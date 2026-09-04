from __future__ import annotations

"""Independent A/B evaluation for Anchored Logical Windows.

Both routes use the same Top-20 hybrid card retrieval and the same Qwen card
rerank.  The baseline reranks every W6/S3 logical window belonging to the
retrieved card sessions.  The anchored route first keeps Parent Top-3/5 cards,
then filters windows to those intersecting their ``source_turn_ids`` and
reranks the reduced pool with compact parent metadata injected only into the
reranker document.  Final context is assembled from the window's real Turn
text; parent metadata is never citation evidence and gold qrels are never
passed to selection.

This is a diagnostic evaluation.  It reads production artifacts, copies them
to a temporary directory for Chroma isolation, and never writes the source
DuckDB/Chroma files or changes production defaults.
"""

import argparse
import gc
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.live_storage import hash_storage_artifacts
from scripts.eval_card_vs_evidence_rerank import (
    _all_required,
    _apply_pointer_map,
    _card_slot_with_required_evidence,
    _card_text,
    _estimate_tokens,
    _load_pointer_map,
    _required,
)
from src.evidence_index import build_evidence_index
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction
from src.reranker_provider import SiliconFlowQwen3Reranker
from src.reranker import PedagogicalGoldAssembler
from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever

logger = logging.getLogger(__name__)

WINDOW_SIZE = 6
WINDOW_STEP = 3
PARENT_TOP_KS = (3, 5)
WINDOW_SELECTION_KS = (4, 5)
MAX_CONTEXT_TOKENS = 1500


def _pointer_ids(candidate: dict[str, Any]) -> list[int]:
    raw = (candidate.get("metadata", {}) or {}).get("source_turn_ids", [])
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [int(value) for value in (raw or [])]


def _turn_keys(candidate: dict[str, Any]) -> set[tuple[int, int]]:
    metadata = candidate.get("metadata", {}) or {}
    session_id = metadata.get("session_id")
    if session_id is None:
        raise ValueError("candidate is missing metadata.session_id")
    return {(int(session_id), turn_id) for turn_id in _pointer_ids(candidate)}


def _parent_context(parent: dict[str, Any]) -> dict[str, str]:
    """Return compact, non-citation metadata for reranker input only."""

    metadata = parent.get("metadata", {}) or {}
    return {
        "card_id": str(parent.get("chunk_id", "")),
        "session_id": str(metadata.get("session_id", "")),
        "subject_path": str(metadata.get("subject_path", "")),
        "parent_title": str(
            metadata.get("misconception_name")
            or metadata.get("key_aha_question")
            or parent.get("chunk_id", "")
        )[:300],
        "parent_summary": str(
            metadata.get("deep_mechanism")
            or metadata.get("pedagogical_goal")
            or ""
        )[:500],
    }


def _window_units(
    cards: list[dict[str, Any]],
    sessions: dict[int, Any],
    storage: DualEngineStorageManager,
    *,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> list[dict[str, Any]]:
    """Expand all logical windows for the sessions represented by cards."""

    indexes: dict[int, Any] = {}
    units: dict[str, dict[str, Any]] = {}
    for card in cards:
        metadata = card.get("metadata", {}) or {}
        session_id = int(metadata["session_id"])
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"missing session {session_id} for logical-window expansion")
        index = indexes.setdefault(
            session_id,
            build_evidence_index(session, window_size=window_size, step=step),
        )
        card_id = str(card.get("chunk_id", ""))
        for window in index.windows:
            unit = units.setdefault(
                window.index_id,
                {
                    "chunk_id": window.index_id,
                    "document": window.content,
                    "metadata": {
                        "session_id": session_id,
                        "question_id": int(session.question_id),
                        "subject_path": str(metadata.get("subject_path", "")),
                        "window_start_turn": int(window.window_start_turn),
                        "window_end_turn": int(window.window_end_turn),
                        "source_turn_ids": list(window.source_turn_ids),
                        "source_card_ids": [],
                        "parent_contexts": [],
                    },
                    "evidence_turns": storage.get_dialogue_turns(
                        session_id, list(window.source_turn_ids)
                    ),
                },
            )
            source_cards = unit["metadata"]["source_card_ids"]
            if card_id and card_id not in source_cards:
                source_cards.append(card_id)
    return list(units.values())


def filter_anchored_windows(
    windows: list[dict[str, Any]], parent_cards: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep windows intersecting selected parents' real source Turn pointers."""

    parent_keys = set().union(*(_turn_keys(card) for card in parent_cards)) if parent_cards else set()
    selected: list[dict[str, Any]] = []
    for window in windows:
        overlap = _turn_keys(window) & parent_keys
        if not overlap:
            continue
        item = dict(window)
        metadata = dict(item.get("metadata", {}) or {})
        contexts: list[dict[str, str]] = []
        window_session = int(metadata["session_id"])
        window_turns = _turn_keys(window)
        for parent in parent_cards:
            parent_meta = parent.get("metadata", {}) or {}
            if int(parent_meta.get("session_id", -1)) != window_session:
                continue
            if window_turns & _turn_keys(parent):
                contexts.append(_parent_context(parent))
        metadata["parent_contexts"] = contexts
        metadata["anchor_overlap_turn_ids"] = sorted(
            turn_id for sid, turn_id in overlap if sid == window_session
        )
        item["metadata"] = metadata
        selected.append(item)
    return selected


def _plain_window_rank_text(candidate: dict[str, Any]) -> str:
    """Baseline renderer: logical window document only."""

    return str(candidate.get("document", ""))


def _anchored_window_rank_text(candidate: dict[str, Any]) -> str:
    """Anchored renderer: compact parent metadata plus one real window."""

    metadata = candidate.get("metadata", {}) or {}
    lines = ["[Parent metadata: ranking context only]"]
    for context in metadata.get("parent_contexts", []) or []:
        lines.extend(
            [
                f"[Parent Session {context.get('session_id', '?')}]",
                f"[Topic] {context.get('subject_path', '')}",
                f"[Parent Card] {context.get('parent_title', '')}",
                f"[Parent Summary] {context.get('parent_summary', '')}",
            ]
        )
    lines.extend(["[Logical Window]", str(candidate.get("document", ""))])
    return "\n".join(lines)


def _rerank_units(
    provider: SiliconFlowQwen3Reranker,
    query: str,
    candidates: list[dict[str, Any]],
    text_fn: Callable[[dict[str, Any]], str],
    unit_name: str,
) -> tuple[list[dict[str, Any]], float, int]:
    if not candidates:
        return [], 0.0, 0
    documents = [text_fn(candidate) for candidate in candidates]
    input_tokens = _estimate_tokens(query) + sum(
        _estimate_tokens(document) for document in documents
    )
    started = time.perf_counter()
    scores = provider.rerank(query, documents, top_n=len(documents))
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    ranked: list[dict[str, Any]] = []
    for rank, score in enumerate(scores, start=1):
        if score.index < 0 or score.index >= len(candidates):
            raise RuntimeError(f"reranker returned invalid candidate index {score.index}")
        item = dict(candidates[score.index])
        item["reranker_score"] = float(score.relevance_score)
        item["reranker_rank"] = rank
        item["reranking_applied"] = True
        item["reranker_unit"] = unit_name
        ranked.append(item)
    if len(ranked) != len(candidates):
        raise RuntimeError("reranker returned an incomplete permutation")
    return ranked, elapsed_ms, input_tokens


def _assemble(
    query: str,
    selected_windows: list[dict[str, Any]],
    parent_cards: list[dict[str, Any]],
    slot: str,
) -> Any:
    if not selected_windows:
        raise ValueError("cannot assemble an empty logical-window selection")
    assembler = PedagogicalGoldAssembler(
        model_reranker=None,
        chunk_strategy="card",
        rerank_unit="logical_evidence",
        evidence_selection_count=len(selected_windows),
        max_prompt_tokens=MAX_CONTEXT_TOKENS,
    )
    retrieval_results = {
        "chunk_strategy": "card",
        # Parent cards decide the anchored candidate pool upstream.  Their
        # summaries are intentionally not sent to the final Assembler: this
        # keeps the A/B context renderer identical and makes real Turn text
        # the only citation-bearing material.  The arguments remain in the
        # helper signature for traceability of the selected parent route.
        "misconceptions": [],
        "strategies": [],
        "evidence_units": selected_windows,
        "rewritten_queries": {"extracted_keywords": []},
    }
    return assembler.assemble(query, retrieval_results)


def _selected_turns(units: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return set().union(*(_turn_keys(unit) for unit in units)) if units else set()


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _variant_row(
    *,
    case_index: int,
    split: str,
    case: dict[str, Any],
    slot: str,
    variant: str,
    parent_cards: list[dict[str, Any]],
    all_windows: list[dict[str, Any]],
    candidate_windows: list[dict[str, Any]],
    ranked_windows: list[dict[str, Any]],
    selection_count: int,
    required_role: set[tuple[int, int]],
    required_all: set[tuple[int, int]],
    card_rerank_ms: float,
    card_input_tokens: int,
    window_rerank_ms: float,
    window_input_tokens: int,
    retrieval_ms: float,
    assembler_ms: float,
    context: Any,
) -> dict[str, Any]:
    selected = ranked_windows[:selection_count]
    selected_turns = _selected_turns(selected)
    candidate_turns = _selected_turns(candidate_windows)
    all_window_turns = _selected_turns(all_windows)
    context_turns = {
        (int(turn["session_id"]), int(turn["turn_id"]))
        for turn in context.evidence_turns
    }
    required_sessions = {session_id for session_id, _ in required_all}
    parent_sessions = {
        int((card.get("metadata", {}) or {})["session_id"]) for card in parent_cards
    }
    role_hits = selected_turns & required_role
    all_hits = selected_turns & required_all
    context_hits = context_turns & required_all
    return {
        "case_id": f"{variant}-case-{case_index:04d}-w{selection_count}",
        "case_index": case_index,
        "split": split,
        "category": case.get("category", "UNKNOWN"),
        "slot": slot,
        "variant": variant,
        "parent_card_count": len(parent_cards),
        "parent_session_ids": sorted(parent_sessions),
        "parent_session_recall": _ratio(len(parent_sessions & required_sessions), len(required_sessions)),
        "parent_turn_union_recall": _ratio(len(_selected_turns(parent_cards) & required_all), len(required_all)),
        "all_window_candidate_count": len(all_windows),
        "candidate_window_count": len(candidate_windows),
        "candidate_reduction_ratio": 1.0 - _ratio(len(candidate_windows), len(all_windows)),
        "all_window_union_recall": _ratio(len(all_window_turns & required_all), len(required_all)),
        "candidate_window_union_recall": _ratio(len(candidate_turns & required_all), len(required_all)),
        "ranked_window_recall": _ratio(len(all_hits), len(required_all)),
        "ranked_window_precision": _ratio(len(all_hits), len(selected_turns)),
        "context_evidence_recall": _ratio(len(context_hits), len(required_all)),
        "context_evidence_precision": _ratio(len(context_hits), len(context_turns)),
        "role_recall": _ratio(len(role_hits), len(required_role)),
        "role_precision": _ratio(len(role_hits), len(selected_turns)),
        "selected_window_count": len(selected),
        "assembled_window_count": len(context.selected_evidence_units),
        "selected_window_ids": [str(item["chunk_id"]) for item in selected],
        "selected_window_ranks": [int(item.get("reranker_rank", 0)) for item in selected],
        "selected_evidence_turn_count": len(selected_turns),
        "context_evidence_turn_count": len(context_turns),
        "context_token_count": int(context.estimated_token_count),
        "budget_violation": bool(context.budget_violation),
        "truncation_loss": int(context.truncation_loss),
        "retrieval_ms": retrieval_ms,
        "card_reranker_ms": card_rerank_ms,
        "window_reranker_ms": window_rerank_ms,
        "assembler_ms": assembler_ms,
        "card_reranker_input_tokens": card_input_tokens,
        "window_reranker_input_tokens": window_input_tokens,
    }


def _aggregate(rows: list[dict[str, Any]], variant: str, metric_names: tuple[str, ...]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("variant") == variant and row.get("status") == "MEASURED"]
    result: dict[str, Any] = {"measured_cases": len(selected)}
    for metric in metric_names:
        values = [float(row[metric]) for row in selected]
        latency_stats: dict[str, float] = {}
        if metric.endswith("_ms") and values:
            latency_stats = {
                "p50": median(values),
                "p95": quantiles(values, n=20, method="inclusive")[18]
                if len(values) > 1
                else values[0],
            }
        result[metric] = {
            "mean": mean(values) if values else None,
            "dev_mean": mean([float(row[metric]) for row in selected if row["split"] == "dev"])
            if any(row["split"] == "dev" for row in selected)
            else None,
            "holdout_mean": mean(
                [float(row[metric]) for row in selected if row["split"] == "holdout"]
            )
            if any(row["split"] == "holdout" for row in selected)
            else None,
            **latency_stats,
        }
    return result


def evaluate(
    *,
    db_path: Path,
    chroma_path: Path,
    golden_path: Path,
    split_path: Path,
    output_dir: Path,
    top_k_cards: int = 20,
    bm25_weight: float = 0.35,
) -> dict[str, Any]:
    if top_k_cards <= 0:
        raise ValueError("top_k_cards must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    golden = json.loads(golden_path.resolve(strict=True).read_text(encoding="utf-8"))
    split_assignments = json.loads(split_path.resolve(strict=True).read_text(encoding="utf-8"))["query_assignments"]
    source_hash_before = hash_storage_artifacts(db_path, chroma_path)
    embedding = SiliconFlowQwen3EmbeddingFunction.from_env()
    reranker = SiliconFlowQwen3Reranker.from_env()
    pointer_map = _load_pointer_map(db_path)
    from scripts.incremental_card_ingest import _load_sessions_from_db

    sessions = {session.intervention_id: session for session in _load_sessions_from_db(db_path)}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="anchored-logical-window-eval-",
        dir=str(chroma_path.parent),
        ignore_cleanup_errors=True,
    ) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_function=embedding,
            embedding_index_version=embedding.index_version,
        )
        retriever = DualMetricRetriever(
            storage_manager=storage,
            query_rewrite_mode="deterministic",
            fusion_strategy="raw_first",
            retrieval_mode="bm25_dense",
            bm25_weight=bm25_weight,
        )
        try:
            for case_index, case in enumerate(golden, start=1):
                try:
                    slot, slot_fallback = _card_slot_with_required_evidence(case)
                    required_role = _required(case, slot)
                    required_all = _all_required(case)
                    split = split_assignments[f"eedi-l1-{case_index:04d}"]
                    started = time.perf_counter()
                    raw_cards = (
                        retriever.retrieve_misconceptions(
                            case["question"], top_k=top_k_cards, fetch_evidence=False
                        )
                        if slot == "misconception"
                        else retriever.retrieve_strategies(
                            case["question"], top_k=top_k_cards, fetch_evidence=False
                        )
                    )
                    retrieval_ms = round((time.perf_counter() - started) * 1000.0, 3)
                    cards = _apply_pointer_map(storage, raw_cards, slot, pointer_map)
                    card_ranked, card_ms, card_input_tokens = _rerank_units(
                        reranker, case["question"], cards, _card_text, "card"
                    )
                    all_windows = _window_units(card_ranked, sessions, storage)
                    baseline_ranked, baseline_window_ms, baseline_input_tokens = _rerank_units(
                        reranker,
                        case["question"],
                        all_windows,
                        _plain_window_rank_text,
                        "logical_window",
                    )
                    for count in WINDOW_SELECTION_KS:
                        if baseline_ranked:
                            started = time.perf_counter()
                            context = _assemble(
                                case["question"], baseline_ranked[:count], card_ranked, slot
                            )
                            assembler_ms = round((time.perf_counter() - started) * 1000.0, 3)
                            row = _variant_row(
                                case_index=case_index,
                                split=split,
                                case=case,
                                slot=slot,
                                variant="baseline",
                                parent_cards=card_ranked,
                                all_windows=all_windows,
                                candidate_windows=all_windows,
                                ranked_windows=baseline_ranked,
                                selection_count=count,
                                required_role=required_role,
                                required_all=required_all,
                                card_rerank_ms=card_ms,
                                card_input_tokens=card_input_tokens,
                                window_rerank_ms=baseline_window_ms,
                                window_input_tokens=baseline_input_tokens,
                                retrieval_ms=retrieval_ms,
                                assembler_ms=assembler_ms,
                                context=context,
                            )
                            row["slot_fallback"] = slot_fallback
                            row["status"] = "MEASURED"
                            rows.append(row)
                    for parent_top_k in PARENT_TOP_KS:
                        parents = card_ranked[:parent_top_k]
                        anchored = filter_anchored_windows(all_windows, parents)
                        anchored_ranked, anchored_window_ms, anchored_input_tokens = _rerank_units(
                            reranker,
                            case["question"],
                            anchored,
                            _anchored_window_rank_text,
                            f"anchored_logical_window_parent{parent_top_k}",
                        )
                        variant = f"anchored_parent{parent_top_k}"
                        for count in WINDOW_SELECTION_KS:
                            if not anchored_ranked:
                                raise RuntimeError(
                                    f"{variant} has no anchored windows for case {case_index}"
                                )
                            started = time.perf_counter()
                            context = _assemble(
                                case["question"], anchored_ranked[:count], parents, slot
                            )
                            assembler_ms = round((time.perf_counter() - started) * 1000.0, 3)
                            row = _variant_row(
                                case_index=case_index,
                                split=split,
                                case=case,
                                slot=slot,
                                variant=variant,
                                parent_cards=parents,
                                all_windows=all_windows,
                                candidate_windows=anchored,
                                ranked_windows=anchored_ranked,
                                selection_count=count,
                                required_role=required_role,
                                required_all=required_all,
                                card_rerank_ms=card_ms,
                                card_input_tokens=card_input_tokens,
                                window_rerank_ms=anchored_window_ms,
                                window_input_tokens=anchored_input_tokens,
                                retrieval_ms=retrieval_ms,
                                assembler_ms=assembler_ms,
                                context=context,
                            )
                            row["slot_fallback"] = slot_fallback
                            row["status"] = "MEASURED"
                            rows.append(row)
                except Exception as exc:  # record a transparent per-case failure
                    logger.exception("Anchored logical-window case %s failed", case_index)
                    failures.append(
                        {
                            "case_index": case_index,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        finally:
            storage.close()
            try:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except ImportError:
                pass
            except Exception as exc:  # cleanup warning is observable, not a fake result
                logger.warning("Unable to clear Chroma client cache during cleanup: %s", exc)
            gc.collect()

    source_hash_after = hash_storage_artifacts(db_path, chroma_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError(
            "source DB/Chroma changed during read-only evaluation; refusing to report a moving snapshot"
        )

    metric_names = (
        "parent_session_recall",
        "parent_turn_union_recall",
        "all_window_candidate_count",
        "candidate_window_count",
        "candidate_reduction_ratio",
        "all_window_union_recall",
        "candidate_window_union_recall",
        "ranked_window_recall",
        "ranked_window_precision",
        "context_evidence_recall",
        "context_evidence_precision",
        "role_recall",
        "role_precision",
        "selected_window_count",
        "assembled_window_count",
        "selected_evidence_turn_count",
        "context_evidence_turn_count",
        "context_token_count",
        "budget_violation",
        "truncation_loss",
        "retrieval_ms",
        "card_reranker_ms",
        "window_reranker_ms",
        "assembler_ms",
        "card_reranker_input_tokens",
        "window_reranker_input_tokens",
    )
    variants = {
        "baseline_w4": _aggregate(
            [dict(row, variant="baseline_w4") for row in rows if row["variant"] == "baseline" and row["selected_window_count"] == 4],
            "baseline_w4",
            metric_names,
        ),
        "baseline_w5": _aggregate(
            [dict(row, variant="baseline_w5") for row in rows if row["variant"] == "baseline" and row["selected_window_count"] == 5],
            "baseline_w5",
            metric_names,
        ),
    }
    for parent_top_k in PARENT_TOP_KS:
        for count in WINDOW_SELECTION_KS:
            key = f"anchored_parent{parent_top_k}_w{count}"
            variants[key] = _aggregate(
                [
                    dict(row, variant=key)
                    for row in rows
                    if row["variant"] == f"anchored_parent{parent_top_k}"
                    and row["selected_window_count"] == count
                ],
                key,
                metric_names,
            )
    status = "MEASURED" if not failures and rows else "PARTIAL" if rows else "FAILED"
    payload = {
        "schema_version": "anchored-logical-windows-ab-eval/v1",
        "status": status,
        "source_artifact_hash": source_hash_before,
        "db_path": str(db_path),
        "chroma_path": str(chroma_path),
        "golden_path": str(golden_path),
        "split_path": str(split_path),
        "card_pool_top_k": top_k_cards,
        "retrieval_mode": "bm25_dense",
        "bm25_weight": bm25_weight,
        "window_size": WINDOW_SIZE,
        "window_step": WINDOW_STEP,
        "parent_top_ks": PARENT_TOP_KS,
        "window_selection_ks": WINDOW_SELECTION_KS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "case_count": len(golden),
        "measured_row_count": len(rows),
        "failures": failures,
        "variants": variants,
        "reranker_telemetry": reranker.telemetry,
        "limitations": [
            "qrels are provisional human-quote-derived labels, not business truth",
            "parent metadata is used only in anchored reranker input and never as citation evidence",
            "context/evidence token counts are estimates, not provider billing tokens",
            "this evaluates evidence coverage; final answer quality still needs L2/DeepEval validation",
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Anchored Logical Windows A/B",
        "",
        f"Status: `{status}`; cases: `{len(golden)}`; card pool: Top-`{top_k_cards}`; window: W{WINDOW_SIZE}/S{WINDOW_STEP}",
        "",
        "| Variant | All-required Recall | Context Precision | Candidate Windows | Candidate Union Recall | Context Tokens | Window Input Tokens | Window Rerank ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, aggregate in variants.items():
        def value(metric: str) -> str:
            item = aggregate[metric]["mean"]
            return "n/a" if item is None else f"{item:.4f}" if "recall" in metric or "precision" in metric else f"{item:.1f}"

        lines.append(
            f"| {name} | {value('context_evidence_recall')} | {value('context_evidence_precision')} | "
            f"{value('candidate_window_count')} | {value('candidate_window_union_recall')} | "
            f"{value('context_token_count')} | {value('window_reranker_input_tokens')} | "
            f"{value('window_reranker_ms')} |"
        )
    lines.extend(
        [
            "",
            "Baseline: all logical windows from Top-20 card sessions. Anchored: only windows intersecting Parent Top-3/5 source_turn_ids.",
            "Detailed rows: `cases.jsonl`; failures: `failures.json`; selection never reads qrels.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data/golden_test_set.json")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-cards", type=int, default=20)
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    args = parser.parse_args(argv)
    payload = evaluate(
        db_path=args.db.resolve(strict=True),
        chroma_path=args.chroma.resolve(strict=True),
        golden_path=args.golden.resolve(strict=True),
        split_path=args.split.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        top_k_cards=args.top_k_cards,
        bm25_weight=args.bm25_weight,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "case_count": payload["case_count"],
                "measured_row_count": payload["measured_row_count"],
                "failure_count": len(payload["failures"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
