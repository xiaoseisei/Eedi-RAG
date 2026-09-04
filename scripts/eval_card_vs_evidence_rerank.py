from __future__ import annotations

"""Compare card-level and logical-evidence-level reranking.

The experiment keeps the same Qwen embedding Top-20 card pool and changes only
the unit sent to the cross-encoder reranker.  It is diagnostic: no production
artifact is modified and all qrels are the existing provisional labels.
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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.live_storage import hash_storage_artifacts
from scripts.run_turn_graded_eval import _card_slot_with_required_evidence
from src.evidence_index import build_evidence_index
from src.reranker_provider import SiliconFlowQwen3Reranker
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction


CARD_KS = (1, 2, 3)
EVIDENCE_KS = (3, 5, 8, 10)


def _required(case: dict[str, Any], slot: str) -> set[tuple[int, int]]:
    expected = "student" if slot == "misconception" else "tutor"
    return {
        (int(item["session_id"]), int(item["turn_id"]))
        for item in case.get("verbatim_grounding_quotes", [])
        if str(item.get("speaker")) == expected
    }


def _all_required(case: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(item["session_id"]), int(item["turn_id"]))
        for item in case.get("verbatim_grounding_quotes", [])
    }


def _pointer_ids(candidate: dict[str, Any]) -> list[int]:
    raw = candidate.get("metadata", {}).get("source_turn_ids", [])
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [int(value) for value in (raw or [])]


def _turns_from_candidates(candidates: list[dict[str, Any]], k: int) -> set[tuple[int, int]]:
    return {
        (int(candidate.get("metadata", {}).get("session_id")), turn_id)
        for candidate in candidates[:k]
        for turn_id in _pointer_ids(candidate)
    }


def _recall(candidates: list[dict[str, Any]], required: set[tuple[int, int]], k: int) -> float:
    return len(_turns_from_candidates(candidates, k) & required) / len(required) if required else 0.0


def _precision(candidates: list[dict[str, Any]], required: set[tuple[int, int]], k: int) -> float:
    pointers = _turns_from_candidates(candidates, k)
    return len(pointers & required) / len(pointers) if pointers else 0.0


def _estimate_tokens(text: str) -> int:
    from src.chunker import estimate_tokens

    return estimate_tokens(text)


def _card_text(candidate: dict[str, Any]) -> str:
    evidence = candidate.get("evidence_turns", []) or []
    evidence_text = "\n".join(
        f"[Turn {turn.get('turn_id')}] [{turn.get('speaker')}] {turn.get('text', '')}"
        for turn in evidence
    )
    return f"{candidate.get('document', '')}\n【直接证据】\n{evidence_text}" if evidence_text else str(candidate.get("document", ""))


def _rerank(provider: SiliconFlowQwen3Reranker, query: str, candidates: list[dict[str, Any]], text_fn) -> tuple[list[dict[str, Any]], float, int]:
    if not candidates:
        return [], 0.0, 0
    documents = [text_fn(candidate) for candidate in candidates]
    estimated_input_tokens = _estimate_tokens(query) + sum(_estimate_tokens(document) for document in documents)
    started = time.perf_counter()
    scores = provider.rerank(query, documents, top_n=len(documents))
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    ranked: list[dict[str, Any]] = []
    for rank, score in enumerate(scores, start=1):
        item = dict(candidates[score.index])
        item["reranker_score"] = float(score.relevance_score)
        item["reranker_rank"] = rank
        item["reranking_applied"] = True
        item["reranker_unit"] = "card" if text_fn is _card_text else "logical_evidence"
        ranked.append(item)
    if len(ranked) != len(candidates):
        raise RuntimeError("reranker returned an incomplete permutation")
    return ranked, latency_ms, estimated_input_tokens


def _load_pointer_map(db_path: Path) -> dict[tuple[int, str], list[int]]:
    import duckdb

    connection = duckdb.connect(database=str(db_path.resolve(strict=True)), read_only=True)
    try:
        result: dict[tuple[int, str], list[int]] = {}
        for slot, table in (("misconception", "misconception_chunks"), ("strategy", "tutor_strategy_chunks")):
            for sid, ids in connection.execute(f"SELECT session_id, source_turn_ids FROM {table}").fetchall():
                result[(int(sid), slot)] = [int(value) for value in (ids or [])]
        return result
    finally:
        connection.close()


def _apply_pointer_map(storage, candidates: list[dict[str, Any]], slot: str, pointer_map: dict[tuple[int, str], list[int]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        metadata = dict(item.get("metadata", {}))
        sid = int(metadata["session_id"])
        ids = pointer_map.get((sid, slot))
        if ids is None:
            raise ValueError(f"missing pointer map for session={sid}, slot={slot}")
        metadata["source_turn_ids"] = ids
        item["metadata"] = metadata
        item["evidence_turns"] = storage.get_dialogue_turns(sid, ids)
        updated.append(item)
    return updated


def _evidence_units(candidates: list[dict[str, Any]], sessions: dict[int, Any], storage) -> list[dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        metadata = candidate.get("metadata", {})
        sid = int(metadata["session_id"])
        index = build_evidence_index(sessions[sid], window_size=6, step=3)
        card_id = str(candidate["chunk_id"])
        for window in index.windows:
            item = units.setdefault(window.index_id, {
                "chunk_id": window.index_id,
                "metadata": {
                    "session_id": sid,
                    "source_turn_ids": list(window.source_turn_ids),
                    "window_start_turn": window.window_start_turn,
                    "window_end_turn": window.window_end_turn,
                    "subject_path": metadata.get("subject_path", ""),
                    "source_card_ids": [],
                },
                "document": window.content,
            })
            if card_id not in item["metadata"]["source_card_ids"]:
                item["metadata"]["source_card_ids"].append(card_id)
            item["evidence_turns"] = storage.get_dialogue_turns(sid, list(window.source_turn_ids))
    return list(units.values())


def _evidence_text(candidate: dict[str, Any]) -> str:
    evidence = candidate.get("evidence_turns", []) or []
    evidence_text = "\n".join(
        f"[Turn {turn.get('turn_id')}] [{turn.get('speaker')}] {turn.get('text', '')}"
        for turn in evidence
    )
    return f"{candidate.get('document', '')}\n【逻辑链证据】\n{evidence_text}"


def _evaluate_snapshot(
    *,
    db_path: Path,
    chroma_path: Path,
    golden: list[dict[str, Any]],
    top_k: int,
    bm25_weight: float,
    reranker: SiliconFlowQwen3Reranker,
) -> dict[str, Any]:
    from scripts.incremental_card_ingest import _load_sessions_from_db
    from src.retriever import DualMetricRetriever
    from src.storage_manager import DualEngineStorageManager

    pointer_map = _load_pointer_map(db_path)
    with tempfile.TemporaryDirectory(prefix="card-vs-evidence-", dir=str(chroma_path.parent), ignore_cleanup_errors=True) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        provider = SiliconFlowQwen3EmbeddingFunction.from_env()
        storage = DualEngineStorageManager(
            db_path=copied_db,
            chroma_dir=copied_chroma,
            embedding_function=provider,
            embedding_index_version=provider.index_version,
        )
        retriever = DualMetricRetriever(
            storage_manager=storage,
            query_rewrite_mode="deterministic",
            fusion_strategy="raw_first",
            retrieval_mode="bm25_dense",
            bm25_weight=bm25_weight,
        )
        sessions = {session.intervention_id: session for session in _load_sessions_from_db(db_path)}
        rows: list[dict[str, Any]] = []
        try:
            for index, case in enumerate(golden, start=1):
                slot, slot_fallback = _card_slot_with_required_evidence(case)
                role_required = _required(case, slot)
                all_required = _all_required(case)
                retrieval_started = time.perf_counter()
                raw_cards = (
                    retriever.retrieve_misconceptions(case["question"], top_k=top_k, fetch_evidence=False)
                    if slot == "misconception"
                    else retriever.retrieve_strategies(case["question"], top_k=top_k, fetch_evidence=False)
                )
                retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000.0, 3)
                cards = _apply_pointer_map(storage, raw_cards, slot, pointer_map)

                card_ranked, card_rerank_ms, card_input_tokens = _rerank(reranker, case["question"], cards, _card_text)
                units = _evidence_units(cards, sessions, storage)
                evidence_ranked, evidence_rerank_ms, evidence_input_tokens = _rerank(reranker, case["question"], units, _evidence_text)

                for selection_count in CARD_KS:
                    selected = card_ranked[:selection_count]
                    selected_pointer_turns = _turns_from_candidates(selected, selection_count)
                    selected_turns = selected_pointer_turns & role_required
                    all_turns = selected_pointer_turns & all_required
                    rows.append({
                        "case_id": f"card-{index:04d}-n{selection_count}",
                        "case_index": index,
                        "split": "unknown",
                        "category": case.get("category", "UNKNOWN"),
                        "slot": slot,
                        "slot_fallback": slot_fallback,
                        "unit": "card",
                        "selected_count": selection_count,
                        "retrieved_card_count": len(cards),
                        "evidence_unit_count": len(units),
                        "role_recall": len(selected_turns) / len(role_required) if role_required else 0.0,
                        "role_precision": len(selected_turns) / len(selected_pointer_turns) if selected_pointer_turns else 0.0,
                        "all_required_recall": len(all_turns & all_required) / len(all_required) if all_required else 0.0,
                        "evidence_turn_count": len(selected_pointer_turns),
                        "context_token_count": sum(_estimate_tokens(_card_text(item)) for item in selected),
                        "reranker_input_tokens": card_input_tokens,
                        "retrieval_ms": retrieval_ms,
                        "reranker_ms": card_rerank_ms,
                        "assembler_ms": 0.0,
                    })
                for selection_count in EVIDENCE_KS:
                    selected = evidence_ranked[:selection_count]
                    selected_pointer_turns = _turns_from_candidates(selected, selection_count)
                    role_turns = selected_pointer_turns & role_required
                    all_turns = selected_pointer_turns & all_required
                    rows.append({
                        "case_id": f"evidence-{index:04d}-m{selection_count}",
                        "case_index": index,
                        "split": "unknown",
                        "category": case.get("category", "UNKNOWN"),
                        "slot": slot,
                        "slot_fallback": slot_fallback,
                        "unit": "logical_evidence",
                        "selected_count": selection_count,
                        "retrieved_card_count": len(cards),
                        "evidence_unit_count": len(units),
                        "role_recall": len(role_turns & role_required) / len(role_required) if role_required else 0.0,
                        "role_precision": len(role_turns) / len(selected_pointer_turns) if selected_pointer_turns else 0.0,
                        "all_required_recall": len(all_turns & all_required) / len(all_required) if all_required else 0.0,
                        "evidence_turn_count": len(selected_pointer_turns),
                        "context_token_count": sum(_estimate_tokens(_evidence_text(item)) for item in selected),
                        "reranker_input_tokens": evidence_input_tokens,
                        "retrieval_ms": retrieval_ms,
                        "reranker_ms": evidence_rerank_ms,
                        "assembler_ms": 0.0,
                    })
        finally:
            storage.close()
            try:
                from chromadb.api.client import SharedSystemClient

                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()
    return {"rows": rows, "case_count": len(golden)}


def evaluate(*, before_db: Path, before_chroma: Path, after_db: Path, after_chroma: Path, golden_path: Path, split_path: Path, output_dir: Path, top_k: int, bm25_weight: float) -> dict[str, Any]:
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
    reranker = SiliconFlowQwen3Reranker.from_env()
    before = _evaluate_snapshot(db_path=before_db, chroma_path=before_chroma, golden=golden, top_k=top_k, bm25_weight=bm25_weight, reranker=reranker)
    after = _evaluate_snapshot(db_path=after_db, chroma_path=after_chroma, golden=golden, top_k=top_k, bm25_weight=bm25_weight, reranker=reranker)
    for dataset in (before, after):
        for row in dataset["rows"]:
            row["split"] = split_assignments[f"eedi-l1-{int(row['case_index']):04d}"]

    metrics = ["role_recall", "role_precision", "all_required_recall", "evidence_turn_count", "context_token_count", "reranker_input_tokens", "retrieval_ms", "reranker_ms"]
    variants: dict[str, Any] = {}
    for unit, counts in (("card", CARD_KS), ("logical_evidence", EVIDENCE_KS)):
        for count in counts:
            key = f"{unit}_{count}"
            variants[key] = {}
            for name, dataset in (("before", before), ("after", after)):
                selected = [row for row in dataset["rows"] if row["unit"] == unit and row["selected_count"] == count]
                variants[key][name] = {
                    metric: {
                        "mean": mean(float(row[metric]) for row in selected),
                        "dev_mean": mean(float(row[metric]) for row in selected if row["split"] == "dev"),
                        "holdout_mean": mean(float(row[metric]) for row in selected if row["split"] == "holdout"),
                    }
                    for metric in metrics
                }
    # Rerank-unit A/B is the primary comparison after rebinding.  The before
    # snapshot is retained for audit, while selection variants are compared
    # on the same post-rebind evidence.
    comparison = {}
    for unit, counts in (("card", CARD_KS), ("logical_evidence", EVIDENCE_KS)):
        for count in counts:
            key = f"{unit}_{count}"
            current = variants[key]["after"]
            comparison[key] = {metric: current[metric] for metric in metrics}
    payload = {
        "schema_version": "card-vs-evidence-rerank-eval/v1",
        "status": "MEASURED",
        "top_k_cards": top_k,
        "bm25_weight": bm25_weight,
        "case_count": len(golden),
        "variants_before_after": variants,
        "post_rebind_variants": comparison,
        "reranker_telemetry": reranker.telemetry,
        "limitations": [
            "qrels are provisional human-quote-derived labels",
            "logical evidence units are deterministic W6/S3 windows expanded from the same Top-20 card pool",
            "context_token_count and reranker_input_tokens are estimates, not provider billing tokens",
            "answer completeness requires L2 human/DeepEval evaluation; evidence recall is its deterministic proxy",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "before.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in before["rows"]), encoding="utf-8")
    (output_dir / "after.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in after["rows"]), encoding="utf-8")
    lines = [
        "# Card-level vs Logical-evidence-level Rerank",
        "",
        f"Cases: `{len(golden)}`; card pool: Top-`{top_k}`; BM25 weight: `{bm25_weight}`",
        "",
        "## Post-rebind variants",
        "",
        "| Variant | Role Recall | Role Precision | All-required Recall | Evidence Turns | Context Tokens | Rerank Input Tokens | Rerank ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in comparison.items():
        lines.append(
            f"| {key} | {values['role_recall']['mean']:.4f} | {values['role_precision']['mean']:.4f} | "
            f"{values['all_required_recall']['mean']:.4f} | {values['evidence_turn_count']['mean']:.2f} | "
            f"{values['context_token_count']['mean']:.1f} | {values['reranker_input_tokens']['mean']:.1f} | "
            f"{values['reranker_ms']['mean']:.1f} |"
        )
    lines.extend(["", "Detailed per-case rows and before/after snapshot metrics are in `before.jsonl`, `after.jsonl`, and `results.json`."])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-db", type=Path, required=True)
    parser.add_argument("--before-chroma", type=Path, required=True)
    parser.add_argument("--after-db", type=Path, required=True)
    parser.add_argument("--after-chroma", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data/golden_test_set.json")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    args = parser.parse_args(argv)
    result = evaluate(
        before_db=args.before_db.resolve(strict=True),
        before_chroma=args.before_chroma.resolve(strict=True),
        after_db=args.after_db.resolve(strict=True),
        after_chroma=args.after_chroma.resolve(strict=True),
        golden_path=args.golden.resolve(strict=True),
        split_path=args.split.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        top_k=args.top_k,
        bm25_weight=args.bm25_weight,
    )
    print(json.dumps({"status": result["status"], "case_count": result["case_count"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
