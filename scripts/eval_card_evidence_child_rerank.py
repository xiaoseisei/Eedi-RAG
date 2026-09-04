from __future__ import annotations

"""Unified A/B/C rerank evaluation on the same post-rebind card snapshot.

A = card-level rerank (select one card)
B = logical evidence window rerank (select up to five W6/S3 windows)
C = context-injected child rerank: raw Parent Top-5 cards expand to Turn±1
    child windows whose rerank text contains parent topic/question/card summary.

Only real dialogue turns are counted as evidence. Parent metadata is ranking
context and is never authorized as a citation. This script is diagnostic and
does not modify production artifacts.
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

from scripts.eval_card_vs_evidence_rerank import (
    _all_required,
    _apply_pointer_map,
    _card_text,
    _card_slot_with_required_evidence,
    _evidence_text,
    _estimate_tokens,
    _evidence_units,
    _load_pointer_map,
    _required,
    _turns_from_candidates,
)
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction
from src.reranker_provider import SiliconFlowQwen3Reranker
from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from src.evidence_index import build_evidence_index


CHILD_PARENT_COUNT = 5
CHILD_KS = (3, 5, 8)
MAX_CONTEXT_TOKENS = 1500


def _child_text(candidate: dict[str, Any]) -> str:
    """Render ranking text with compact parent context and raw child turns."""
    metadata = candidate.get("metadata", {}) or {}
    parent_summary = str(metadata.get("parent_summary", "")).strip()
    lines = [
        f"[Parent Session {metadata.get('session_id', '?')}]",
        f"[Topic] {metadata.get('subject_path', '')}",
        f"[Question] {metadata.get('question_text', '')}",
        f"[Parent Card] {metadata.get('parent_title', '')}",
    ]
    if parent_summary:
        lines.append(f"[Parent Summary] {parent_summary}")
    lines.append("[Child Dialogue]")
    for turn in candidate.get("evidence_turns", []) or []:
        lines.append(f"[Turn {turn.get('turn_id')}] [{turn.get('speaker')}] {turn.get('text', '')}")
    return "\n".join(lines)


def _rerank(
    provider: SiliconFlowQwen3Reranker,
    query: str,
    candidates: list[dict[str, Any]],
    text_fn: Callable[[dict[str, Any]], str],
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
        item["reranker_unit"] = (
            "card" if text_fn is _card_text
            else "logical_evidence" if text_fn is _evidence_text
            else "context_injected_child"
        )
        ranked.append(item)
    if len(ranked) != len(candidates):
        raise RuntimeError("reranker returned an incomplete permutation")
    return ranked, elapsed_ms, input_tokens


def _child_units(
    parents: list[dict[str, Any]],
    sessions: dict[int, Any],
    storage: DualEngineStorageManager,
    *,
    radius: int = 1,
) -> list[dict[str, Any]]:
    """Create one child unit per center Turn with deterministic ±radius context."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    units: dict[str, dict[str, Any]] = {}
    for parent in parents:
        parent_meta = parent.get("metadata", {}) or {}
        session_id = int(parent_meta["session_id"])
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"missing session {session_id} for child expansion")
        turns = list(session.turns)
        parent_title = (
            parent_meta.get("misconception_name")
            or parent_meta.get("key_aha_question")
            or str(parent.get("chunk_id", ""))
        )
        parent_summary = (
            parent_meta.get("deep_mechanism")
            or parent_meta.get("pedagogical_goal")
            or ""
        )
        for center_index, center in enumerate(turns):
            start = max(0, center_index - radius)
            end = min(len(turns), center_index + radius + 1)
            window_turns = turns[start:end]
            child_id = f"session_{session_id}_child_{center.turn_id}_r{radius}"
            metadata = {
                "session_id": session_id,
                "question_id": session.question_id,
                "subject_path": session.subjects.paths[0] if session.subjects.paths else "",
                "question_text": session.question.question_text,
                "window_start_turn": int(window_turns[0].turn_id),
                "window_end_turn": int(window_turns[-1].turn_id),
                "center_turn_id": int(center.turn_id),
                "source_turn_ids": [int(turn.turn_id) for turn in window_turns],
                "parent_card_id": str(parent.get("chunk_id", "")),
                "parent_title": str(parent_title),
                "parent_summary": str(parent_summary)[:500],
            }
            units.setdefault(
                child_id,
                {
                    "chunk_id": child_id,
                    "document": "\n".join(
                        [
                            f"[Session {session_id}] [Child Turn {center.turn_id}]",
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
                },
            )
    return list(units.values())


def _variant_metrics(
    ranked: list[dict[str, Any]],
    *,
    required_role: set[tuple[int, int]],
    required_all: set[tuple[int, int]],
    count: int,
    text_fn: Callable[[dict[str, Any]], str],
    rerank_ms: float,
    reranker_input_tokens: int,
    retrieval_ms: float,
    unit: str,
) -> dict[str, Any]:
    selected = ranked[:count]
    selected_turns = _turns_from_candidates(selected, count)
    role_hits = selected_turns & required_role
    all_hits = selected_turns & required_all
    context_tokens = sum(_estimate_tokens(text_fn(item)) for item in selected)
    return {
        "unit": unit,
        "selected_count": count,
        "retrieved_candidate_count": len(ranked),
        "role_recall": len(role_hits) / len(required_role) if required_role else 0.0,
        "role_precision": len(role_hits) / len(selected_turns) if selected_turns else 0.0,
        "all_required_recall": len(all_hits) / len(required_all) if required_all else 0.0,
        "all_required_precision": len(all_hits) / len(selected_turns) if selected_turns else 0.0,
        "evidence_turn_count": len(selected_turns),
        "context_token_count": context_tokens,
        "budget_violation": context_tokens > MAX_CONTEXT_TOKENS,
        "reranker_input_tokens": reranker_input_tokens,
        "retrieval_ms": retrieval_ms,
        "reranker_ms": rerank_ms,
    }


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
    with tempfile.TemporaryDirectory(prefix="card-child-rerank-", dir=str(chroma_path.parent), ignore_cleanup_errors=True) as temp_name:
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
                started = time.perf_counter()
                raw_cards = (
                    retriever.retrieve_misconceptions(case["question"], top_k=top_k_cards, fetch_evidence=False)
                    if slot == "misconception"
                    else retriever.retrieve_strategies(case["question"], top_k=top_k_cards, fetch_evidence=False)
                )
                retrieval_ms = round((time.perf_counter() - started) * 1000.0, 3)
                cards = _apply_pointer_map(storage, raw_cards, slot, pointer_map)

                card_ranked, card_ms, card_input_tokens = _rerank(reranker, case["question"], cards, _card_text)
                evidence_candidates = _evidence_units(cards, sessions, storage)
                evidence_ranked, evidence_ms, evidence_input_tokens = _rerank(reranker, case["question"], evidence_candidates, _evidence_text)
                child_candidates = _child_units(cards[:CHILD_PARENT_COUNT], sessions, storage, radius=1)
                child_ranked, child_ms, child_input_tokens = _rerank(reranker, case["question"], child_candidates, _child_text)

                rows.append({
                    "case_id": f"card-child-rerank-{case_index:04d}",
                    "case_index": case_index,
                    "split": split_assignments[f"eedi-l1-{case_index:04d}"],
                    "category": case.get("category", "UNKNOWN"),
                    "slot": slot,
                    "slot_fallback": slot_fallback,
                    "parent_card_count": len(cards),
                    "child_parent_count": CHILD_PARENT_COUNT,
                    "child_candidate_count": len(child_candidates),
                    "evidence_candidate_count": len(evidence_candidates),
                    "retrieval_ms": retrieval_ms,
                    "variants": [
                        _variant_metrics(
                            card_ranked,
                            required_role=required_role,
                            required_all=required_all,
                            count=1,
                            text_fn=_card_text,
                            rerank_ms=card_ms,
                            reranker_input_tokens=card_input_tokens,
                            retrieval_ms=retrieval_ms,
                            unit="card_1",
                        ),
                        _variant_metrics(
                            evidence_ranked,
                            required_role=required_role,
                            required_all=required_all,
                            count=5,
                            text_fn=_evidence_text,
                            rerank_ms=evidence_ms,
                            reranker_input_tokens=evidence_input_tokens,
                            retrieval_ms=retrieval_ms,
                            unit="logical_evidence_5",
                        ),
                        *[
                            _variant_metrics(
                                child_ranked,
                                required_role=required_role,
                                required_all=required_all,
                                count=count,
                                text_fn=_child_text,
                                rerank_ms=child_ms,
                                reranker_input_tokens=child_input_tokens,
                                retrieval_ms=retrieval_ms,
                                unit=f"context_injected_child_{count}",
                            )
                            for count in CHILD_KS
                        ],
                    ],
                })
        finally:
            storage.close()
            try:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()

    variant_names = ["card_1", "logical_evidence_5", *[f"context_injected_child_{count}" for count in CHILD_KS]]
    aggregates: dict[str, Any] = {}
    for name in variant_names:
        selected = [variant for row in rows for variant in row["variants"] if variant["unit"] == name]
        aggregates[name] = {
            metric: {
                "mean": mean(float(item[metric]) for item in selected),
                "dev_mean": mean(float(item[metric]) for item in selected if next(row["split"] for row in rows if item in row["variants"]) == "dev"),
                "holdout_mean": mean(float(item[metric]) for item in selected if next(row["split"] for row in rows if item in row["variants"]) == "holdout"),
            }
            for metric in (
                "role_recall",
                "role_precision",
                "all_required_recall",
                "all_required_precision",
                "evidence_turn_count",
                "context_token_count",
                "budget_violation",
                "reranker_input_tokens",
                "retrieval_ms",
                "reranker_ms",
            )
        }
        aggregates[name]["measured_cases"] = len(selected)
    payload = {
        "schema_version": "card-evidence-child-rerank-eval/v1",
        "status": "MEASURED",
        "card_pool_top_k": top_k_cards,
        "child_parent_count": CHILD_PARENT_COUNT,
        "child_context_radius": 1,
        "bm25_weight": bm25_weight,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "case_count": len(rows),
        "aggregates": aggregates,
        "reranker_telemetry": reranker.telemetry,
        "limitations": [
            "qrels are provisional human-quote-derived labels",
            "parent metadata helps ranking but is never citation evidence",
            "child candidates are expanded only from raw Top-5 parent cards; parent misses cannot be recovered",
            "token counts are estimates, not provider billing tokens",
            "answer completeness still requires L2 human/DeepEval validation",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    lines = [
        "# Card / Logical Evidence / Context-Injected Child Rerank",
        "",
        f"Cases: `{len(rows)}`; Card pool: Top-`{top_k_cards}`; Parent Top-`{CHILD_PARENT_COUNT}`; Child radius: `±1`",
        "",
        "| Variant | Role Recall | Role Precision | All-required Recall | All-required Precision | Evidence Turns | Context Tokens | Budget violations | Reranker input tokens | Reranker ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in variant_names:
        agg = aggregates[name]
        lines.append(
            f"| {name} | {agg['role_recall']['mean']:.4f} | {agg['role_precision']['mean']:.4f} | "
            f"{agg['all_required_recall']['mean']:.4f} | {agg['all_required_precision']['mean']:.4f} | "
            f"{agg['evidence_turn_count']['mean']:.2f} | {agg['context_token_count']['mean']:.1f} | "
            f"{agg['budget_violation']['mean']:.2f} | {agg['reranker_input_tokens']['mean']:.1f} | "
            f"{agg['reranker_ms']['mean']:.1f} |"
        )
    lines.extend(["", "Detailed case rows are in `cases.jsonl`; this is a diagnostic report, not a business-quality pass."])
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
