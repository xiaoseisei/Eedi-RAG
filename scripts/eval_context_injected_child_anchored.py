from __future__ import annotations

"""Evaluate anchored Context-Injected Child rerank.

Controlled route:
Top-20 cards -> card rerank -> Parent Top-3/5 cards -> source_turn_ids
anchored Turn±1 children -> per-session quota -> child rerank ->
coverage-aware selection (3/5/8) -> assembler.

Selection never reads gold qrels. Gold qrels are used only after selection to
measure required evidence recall/precision.
"""

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_card_evidence_child_rerank import _all_required, _required
from scripts.eval_card_vs_evidence_rerank import (
    _apply_pointer_map,
    _card_text,
    _card_slot_with_required_evidence,
    _estimate_tokens,
    _load_pointer_map,
    _rerank as _base_rerank,
)
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction
from src.reranker_provider import SiliconFlowQwen3Reranker
from src.reranker import PedagogicalGoldAssembler
from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever


PARENT_TOP_KS = (3, 5)
CHILD_QUOTAS = (2, 3, 5)
CHILD_SELECTION_KS = (3, 5, 8)
MAX_CONTEXT_TOKENS = 1500


def _child_rank_text(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata", {}) or {}
    lines = [
        f"[Parent Session {metadata.get('session_id', '?')}]",
        f"[Topic] {metadata.get('subject_path', '')}",
        f"[Question] {metadata.get('question_text', '')}",
        f"[Parent Card] {metadata.get('parent_title', '')}",
        f"[Parent Summary] {metadata.get('parent_summary', '')}",
        "[Child Dialogue]",
    ]
    lines.extend(
        f"[Turn {turn.get('turn_id')}] [{turn.get('speaker')}] {turn.get('text', '')}"
        for turn in candidate.get("evidence_turns", []) or []
    )
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
    input_tokens = _estimate_tokens(query) + sum(_estimate_tokens(document) for document in documents)
    started = time.perf_counter()
    scores = provider.rerank(query, documents, top_n=len(documents))
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    ranked: list[dict[str, Any]] = []
    for rank, score in enumerate(scores, start=1):
        item = dict(candidates[score.index])
        item["reranker_score"] = float(score.relevance_score)
        item["reranker_rank"] = rank
        item["reranking_applied"] = True
        item["reranker_unit"] = unit_name
        ranked.append(item)
    if len(ranked) != len(candidates):
        raise RuntimeError("reranker returned an incomplete permutation")
    return ranked, elapsed_ms, input_tokens


def _pointer_ids(candidate: dict[str, Any]) -> list[int]:
    raw = candidate.get("metadata", {}).get("source_turn_ids", [])
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [int(value) for value in (raw or [])]


def _turn_keys(candidate: dict[str, Any]) -> set[tuple[int, int]]:
    session_id = int(candidate.get("metadata", {}).get("session_id"))
    return {(session_id, turn_id) for turn_id in _pointer_ids(candidate)}


def _anchored_child_units(
    parent_cards: list[dict[str, Any]],
    sessions: dict[int, Any],
    storage: DualEngineStorageManager,
    *,
    radius: int = 1,
) -> list[dict[str, Any]]:
    """Expand only source_turn_ids from selected parent cards to Turn±1 units."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    units: dict[str, dict[str, Any]] = {}
    for parent in parent_cards:
        parent_meta = parent.get("metadata", {}) or {}
        session_id = int(parent_meta["session_id"])
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"missing session {session_id} for child expansion")
        turns = list(session.turns)
        by_id = {int(turn.turn_id): index for index, turn in enumerate(turns)}
        anchors = _pointer_ids(parent)
        parent_title = parent_meta.get("misconception_name") or parent_meta.get("key_aha_question") or parent.get("chunk_id", "")
        parent_summary = parent_meta.get("deep_mechanism") or parent_meta.get("pedagogical_goal") or ""
        for anchor_id in anchors:
            if anchor_id not in by_id:
                raise ValueError(f"card pointer Turn {anchor_id} missing from session {session_id}")
            center_index = by_id[anchor_id]
            start = max(0, center_index - radius)
            end = min(len(turns), center_index + radius + 1)
            window_turns = turns[start:end]
            child_id = f"session_{session_id}_child_anchor_{anchor_id}_r{radius}"
            metadata = {
                "session_id": session_id,
                "question_id": session.question_id,
                "subject_path": session.subjects.paths[0] if session.subjects.paths else "",
                "question_text": session.question.question_text,
                "window_start_turn": int(window_turns[0].turn_id),
                "window_end_turn": int(window_turns[-1].turn_id),
                "center_turn_id": int(anchor_id),
                "source_turn_ids": [int(turn.turn_id) for turn in window_turns],
                "anchor_turn_ids": [int(anchor_id)],
                "parent_card_id": str(parent.get("chunk_id", "")),
                "parent_title": str(parent_title),
                "parent_summary": str(parent_summary)[:500],
            }
            if child_id not in units:
                units[child_id] = {
                    "chunk_id": child_id,
                    "document": "\n".join(
                        [
                            f"[Session {session_id}] [Child anchor Turn {anchor_id}]",
                            *[
                                f"[Turn {turn.turn_id}] [{('tutor' if turn.is_tutor else 'student')}] {turn.text}"
                                for turn in window_turns
                            ],
                        ]
                    ),
                    "metadata": metadata,
                    "evidence_turns": [
                        {
                            "session_id": session_id,
                            "turn_id": int(turn.turn_id),
                            "speaker": "tutor" if turn.is_tutor else "student",
                            "text": turn.text,
                        }
                        for turn in window_turns
                    ],
                }
            else:
                existing = units[child_id]["metadata"]
                existing["anchor_turn_ids"] = sorted(set(existing["anchor_turn_ids"]) | {int(anchor_id)})
    return list(units.values())


def _coverage_aware_select(
    ranked_children: list[dict[str, Any]],
    *,
    selection_count: int,
    per_session_quota: int,
    anchor_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Greedily maximize reranker relevance plus new anchor coverage.

    This uses only system-produced card pointers and candidate metadata; gold
    required Turns are intentionally not passed to the selector.
    """
    if selection_count <= 0 or per_session_quota <= 0:
        raise ValueError("selection_count and per_session_quota must be positive")
    remaining = list(ranked_children)
    selected: list[dict[str, Any]] = []
    covered: set[tuple[int, int]] = set()
    session_counts: dict[int, int] = {}
    scores = [float(item.get("reranker_score", 0.0)) for item in remaining]
    minimum = min(scores, default=0.0)
    maximum = max(scores, default=1.0)
    scale = max(1e-9, maximum - minimum)
    while remaining and len(selected) < selection_count:
        eligible = [
            item
            for item in remaining
            if session_counts.get(int(item["metadata"]["session_id"]), 0) < per_session_quota
        ]
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda item: (
                0.70 * ((float(item.get("reranker_score", 0.0)) - minimum) / scale)
                + 0.30 * (len((_turn_keys(item) & anchor_keys) - covered) / max(1, len(anchor_keys))),
                -int(item.get("reranker_rank", 10**9)),
                str(item.get("chunk_id", "")),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        session_id = int(best["metadata"]["session_id"])
        session_counts[session_id] = session_counts.get(session_id, 0) + 1
        covered.update(_turn_keys(best) & anchor_keys)
    return selected


def _assemble_selected_children(
    query: str,
    selected_children: list[dict[str, Any]],
    parent_cards: list[dict[str, Any]],
) -> Any:
    """Pass pre-ranked children to the production assembler without reranking."""
    assembler = PedagogicalGoldAssembler(
        model_reranker=None,
        chunk_strategy="card",
        rerank_unit="logical_evidence",
        evidence_selection_count=len(selected_children),
        max_prompt_tokens=MAX_CONTEXT_TOKENS,
    )
    slot = parent_cards[0].get("metadata", {}).get("chunk_type", "") if parent_cards else ""
    result = assembler.assemble(
        query,
        {
            "chunk_strategy": "card",
            "misconceptions": parent_cards if slot.startswith("mis") else [],
            "strategies": parent_cards if not slot.startswith("mis") else [],
            "evidence_units": selected_children,
            "rewritten_queries": {"extracted_keywords": []},
        },
    )
    return result


def evaluate(*, db_path: Path, chroma_path: Path, golden_path: Path, split_path: Path, output_dir: Path, top_k_cards: int = 20, bm25_weight: float = 0.35) -> dict[str, Any]:
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
    embedding = SiliconFlowQwen3EmbeddingFunction.from_env()
    reranker = SiliconFlowQwen3Reranker.from_env()
    pointer_map = _load_pointer_map(db_path)
    from scripts.incremental_card_ingest import _load_sessions_from_db

    sessions = {session.intervention_id: session for session in _load_sessions_from_db(db_path)}
    with tempfile.TemporaryDirectory(prefix="anchored-child-eval-", dir=str(chroma_path.parent), ignore_cleanup_errors=True) as temp_name:
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
        rows: list[dict[str, Any]] = []
        try:
            for case_index, case in enumerate(golden, start=1):
                slot, slot_fallback = _card_slot_with_required_evidence(case)
                required_role = _required(case, slot)
                required_all = _all_required(case)
                retrieval_started = time.perf_counter()
                raw_cards = (
                    retriever.retrieve_misconceptions(case["question"], top_k=top_k_cards, fetch_evidence=False)
                    if slot == "misconception"
                    else retriever.retrieve_strategies(case["question"], top_k=top_k_cards, fetch_evidence=False)
                )
                retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000.0, 3)
                cards = _apply_pointer_map(storage, raw_cards, slot, pointer_map)
                card_ranked, card_rerank_ms, card_input_tokens = _rerank_units(
                    reranker, case["question"], cards, _card_text, "card"
                )

                for parent_top_k in PARENT_TOP_KS:
                    parent_cards = card_ranked[:parent_top_k]
                    parent_sessions = sorted({int(item["metadata"]["session_id"]) for item in parent_cards})
                    children = _anchored_child_units(parent_cards, sessions, storage, radius=1)
                    child_ranked, child_rerank_ms, child_input_tokens = _rerank_units(
                        reranker, case["question"], children, _child_rank_text, "context_injected_child"
                    )
                    anchor_keys = {
                        (int(item["metadata"]["session_id"]), turn_id)
                        for item in parent_cards
                        for turn_id in _pointer_ids(item)
                    }
                    for child_quota in CHILD_QUOTAS:
                        for child_selection_k in CHILD_SELECTION_KS:
                            selected_children = _coverage_aware_select(
                                child_ranked,
                                selection_count=child_selection_k,
                                per_session_quota=child_quota,
                                anchor_keys=anchor_keys,
                            )
                            context = _assemble_selected_children(case["question"], selected_children, parent_cards)
                            context_keys = {
                                (int(turn["session_id"]), int(turn["turn_id"]))
                                for turn in context.evidence_turns
                            }
                            child_keys = set().union(*(_turn_keys(item) for item in selected_children)) if selected_children else set()
                            rows.append({
                                "case_id": f"anchored-child-{case_index:04d}-p{parent_top_k}-q{child_quota}-k{child_selection_k}",
                                "case_index": case_index,
                                "split": split_assignments[f"eedi-l1-{case_index:04d}"],
                                "category": case.get("category", "UNKNOWN"),
                                "slot": slot,
                                "slot_fallback": slot_fallback,
                                "parent_card_count": len(parent_cards),
                                "parent_session_count": len(parent_sessions),
                                "parent_session_ids": parent_sessions,
                                "child_candidate_count": len(children),
                                "anchor_turn_count": len(anchor_keys),
                                "child_quota_per_session": child_quota,
                                "child_selection_k": child_selection_k,
                                "selected_child_count": len(selected_children),
                                "selected_child_ids": [str(item["chunk_id"]) for item in selected_children],
                                "selected_child_ranks": [int(item.get("reranker_rank", 0)) for item in selected_children],
                                "role_recall": len(child_keys & required_role) / len(required_role) if required_role else 0.0,
                                "role_precision": len(child_keys & required_role) / len(child_keys) if child_keys else 0.0,
                                "all_required_recall": len(child_keys & required_all) / len(required_all) if required_all else 0.0,
                                "all_required_precision": len(child_keys & required_all) / len(child_keys) if child_keys else 0.0,
                                "context_evidence_recall": len(context_keys & required_all) / len(required_all) if required_all else 0.0,
                                "context_evidence_precision": len(context_keys & required_all) / len(context_keys) if context_keys else 0.0,
                                "anchor_coverage": len(child_keys & anchor_keys) / len(anchor_keys) if anchor_keys else 0.0,
                                "selected_evidence_turn_count": len(child_keys),
                                "context_evidence_turn_count": len(context_keys),
                                "context_token_count": int(context.estimated_token_count),
                                "budget_violation": bool(context.budget_violation),
                                "truncation_loss": int(context.truncation_loss),
                                "card_reranker_input_tokens": card_input_tokens,
                                "child_reranker_input_tokens": child_input_tokens,
                                "retrieval_ms": retrieval_ms,
                                "card_reranker_ms": card_rerank_ms,
                                "child_reranker_ms": child_rerank_ms,
                            })
        finally:
            storage.close()
            try:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()

    variant_names = [
        f"parent{parent_top_k}_quota{quota}_child{child_k}"
        for parent_top_k in PARENT_TOP_KS
        for quota in CHILD_QUOTAS
        for child_k in CHILD_SELECTION_KS
    ]
    aggregates: dict[str, Any] = {}
    for name in variant_names:
        parent_top_k, quota, child_k = (int(part.replace("parent", "")) if idx == 0 else int(part.replace("quota", "")) if idx == 1 else int(part.replace("child", "")) for idx, part in enumerate(name.split("_")))
        selected = [
            row
            for row in rows
            if row["parent_card_count"] == parent_top_k
            and row["child_quota_per_session"] == quota
            and row["child_selection_k"] == child_k
        ]
        aggregates[name] = {
            metric: {
                "mean": mean(float(item[metric]) for item in selected),
                "dev_mean": mean(float(item[metric]) for item in selected if item["split"] == "dev"),
                "holdout_mean": mean(float(item[metric]) for item in selected if item["split"] == "holdout"),
            }
            for metric in (
                "role_recall",
                "role_precision",
                "all_required_recall",
                "all_required_precision",
                "context_evidence_recall",
                "context_evidence_precision",
                "anchor_coverage",
                "selected_evidence_turn_count",
                "context_evidence_turn_count",
                "context_token_count",
                "budget_violation",
                "truncation_loss",
                "child_candidate_count",
                "card_reranker_input_tokens",
                "child_reranker_input_tokens",
                "retrieval_ms",
                "card_reranker_ms",
                "child_reranker_ms",
            )
        }
        aggregates[name]["measured_cases"] = len(selected)
    payload = {
        "schema_version": "anchored-context-injected-child-eval/v1",
        "status": "MEASURED",
        "card_pool_top_k": top_k_cards,
        "parent_top_ks": PARENT_TOP_KS,
        "child_quotas": CHILD_QUOTAS,
        "child_selection_ks": CHILD_SELECTION_KS,
        "child_radius": 1,
        "bm25_weight": bm25_weight,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "case_count": len(golden),
        "aggregates": aggregates,
        "reranker_telemetry": reranker.telemetry,
        "limitations": [
            "qrels are provisional human-quote-derived labels",
            "selection uses source_turn_ids/card metadata and never reads gold required Turns",
            "Parent Top-K is card count; duplicate cards may map to fewer unique sessions",
            "token counts are estimates, not provider billing tokens",
            "this run measures evidence completeness, not LLM answer quality",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    lines = [
        "# Anchored Context-Injected Child Rerank",
        "",
        f"Cases: `{len(golden)}`; card pool: Top-`{top_k_cards}`; child radius: `±1`",
        "",
        "| Variant | Role Recall | Role Precision | All-required Recall | All-required Precision | Context Recall | Context Precision | Anchor Coverage | Child Candidates | Context Tokens | Budget Violations | Card Rerank ms | Child Rerank ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in variant_names:
        agg = aggregates[name]
        lines.append(
            f"| {name} | {agg['role_recall']['mean']:.4f} | {agg['role_precision']['mean']:.4f} | "
            f"{agg['all_required_recall']['mean']:.4f} | {agg['all_required_precision']['mean']:.4f} | "
            f"{agg['context_evidence_recall']['mean']:.4f} | {agg['context_evidence_precision']['mean']:.4f} | "
            f"{agg['anchor_coverage']['mean']:.4f} | {agg['child_candidate_count']['mean']:.1f} | "
            f"{agg['context_token_count']['mean']:.1f} | {agg['budget_violation']['mean']:.2f} | "
            f"{agg['card_reranker_ms']['mean']:.1f} | {agg['child_reranker_ms']['mean']:.1f} |"
        )
    lines.extend(["", "Detailed per-case traces are in `cases.jsonl`; selection did not use gold qrels."])
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
    print(json.dumps({"status": payload["status"], "case_count": payload["case_count"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
