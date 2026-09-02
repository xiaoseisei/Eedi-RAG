from __future__ import annotations

"""Compare embedding candidates on the frozen L1 v2 query/qrels set.

The benchmark deliberately does not open Chroma. Documents are reconstructed
from the read-only DuckDB source using the same builders as ingestion, and
all rankings are computed in memory. This keeps candidate runs isolated and
prevents model loading or Chroma housekeeping writes from changing production
artifacts.
"""

import argparse
import json
import math
import os
import platform
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from evals.live_storage import hash_storage_artifacts
from evals.metrics.retrieval import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from src.storage_manager import (
    FastDeterministicEmbeddingFunction,
    build_misconception_embedding_doc,
    build_tutor_strategy_embedding_doc,
)


MODEL_IDS = {
    "deterministic": None,
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "bge-m3": "BAAI/bge-m3",
}


def _load_queries(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    queries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qrels_path = path.with_name("qrels.jsonl")
    qrels: dict[str, dict[str, int]] = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            qrels.setdefault(row["query_id"], {})[row["document_id"]] = int(row["relevance"])
    return queries, qrels


def _load_corpus(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    connection = duckdb.connect(database=str(db_path.resolve(strict=True)), read_only=True)
    try:
        questions = {
            int(row[0]): str(row[1] or "")
            for row in connection.execute("SELECT intervention_id, question_text FROM tutoring_sessions").fetchall()
        }
        corpus: dict[str, list[dict[str, Any]]] = {"misconception": [], "strategy": []}
        misc_rows = connection.execute("SELECT * FROM misconception_chunks ORDER BY chunk_id").fetchall()
        misc_cols = [row[0] for row in connection.execute("DESCRIBE misconception_chunks").fetchall()]
        for raw in misc_rows:
            row = dict(zip(misc_cols, raw, strict=True))
            card = _object_from_row(row, "misconception")
            corpus["misconception"].append({
                "id": str(row["chunk_id"]),
                "text": build_misconception_embedding_doc(card, question_text=questions.get(int(row["session_id"]), ""), subject_path=row.get("subject_path")),
            })
        strat_rows = connection.execute("SELECT * FROM tutor_strategy_chunks ORDER BY chunk_id").fetchall()
        strat_cols = [row[0] for row in connection.execute("DESCRIBE tutor_strategy_chunks").fetchall()]
        for raw in strat_rows:
            row = dict(zip(strat_cols, raw, strict=True))
            card = _object_from_row(row, "strategy")
            corpus["strategy"].append({
                "id": str(row["chunk_id"]),
                "text": build_tutor_strategy_embedding_doc(card, question_text=questions.get(int(row["session_id"]), ""), subject_path=row.get("subject_path") or ""),
            })
        return corpus
    finally:
        connection.close()


def _object_from_row(row: dict[str, Any], kind: str) -> Any:
    """Create the existing Pydantic card model without changing source data."""
    from src.models import StudentMisconceptionProfile, TutorStrategyProfile

    if kind == "misconception":
        return StudentMisconceptionProfile(
            session_id=int(row["session_id"]), question_id=int(row["question_id"]),
            subject_path=row.get("subject_path") or "",
            misconception_name=row["misconception_name"],
            error_choice=row.get("error_choice") or None, deep_mechanism=row["deep_mechanism"],
            confusion_triggers=list(row.get("confusion_triggers") or []),
            verbatim_student_quotes=list(row.get("verbatim_student_quotes") or []),
            source_turn_ids=list(row.get("source_turn_ids") or []),
        )
    return TutorStrategyProfile(
        session_id=int(row["session_id"]), question_id=int(row["question_id"]),
        pedagogical_goal=row["pedagogical_goal"],
        strategy_category=row["strategy_category"], key_aha_question=row["key_aha_question"],
        scaffolding_steps=list(row.get("scaffolding_steps") or []),
        analogy_or_metaphor=row.get("analogy_or_metaphor") or None,
        talk_moves=list(row.get("talk_moves") or []), resolution_outcome=row["resolution_outcome"],
        source_turn_ids=list(row.get("source_turn_ids") or []),
    )


def _embedder(candidate: str) -> tuple[Callable[[list[str], bool], list[list[float]]], dict[str, Any]]:
    if candidate == "deterministic":
        fn = FastDeterministicEmbeddingFunction()
        return lambda texts, is_query: fn(texts), {"backend": "deterministic", "model": "FastDeterministicEmbeddingFunction", "version": "128d-v1", "dimension": 128, "prefix": None}

    from sentence_transformers import SentenceTransformer

    model_id = MODEL_IDS[candidate]
    model = SentenceTransformer(model_id, device="cpu")
    dimension = int(model.get_sentence_embedding_dimension())
    prefix = "e5" if candidate.startswith("multilingual-e5") else "none"

    def encode(texts: list[str], is_query: bool) -> list[list[float]]:
        if prefix == "e5":
            marker = "query: " if is_query else "passage: "
            texts = [marker + text for text in texts]
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()

    return encode, {"backend": "sentence-transformers", "model": model_id, "version": "unresolved-local-revision", "dimension": dimension, "prefix": prefix}


def _cosine_rank(query: list[float], docs: list[list[float]], ids: list[str]) -> list[str]:
    scored = []
    for doc_id, vector in zip(ids, docs, strict=True):
        score = sum(float(a) * float(b) for a, b in zip(query, vector, strict=True))
        scored.append((score, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored]


def _tokens(query: str) -> tuple[set[str], set[str], set[str]]:
    numbers = set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", query))
    formulas = set(re.findall(r"[-+]?\d+\s*[A-Za-z]?\s*[+*/=<>-]\s*[-+]?\d+[A-Za-z]?", query))
    options = set(re.findall(r"\b[A-D]\b", query))
    return numbers, formulas, options


def _content_metrics(query: str, ranked_ids: list[str], documents: dict[str, str], qrels: dict[str, int]) -> dict[str, float | None]:
    numbers, formulas, options = _tokens(query)
    top_docs = " ".join(documents[item] for item in ranked_ids[:5] if item in documents)
    def hit(tokens: set[str]) -> float | None:
        if not tokens:
            return None
        return float(all(token in top_docs for token in tokens))
    return {"numeric_exact_hit_at_5": hit(numbers), "formula_exact_hit_at_5": hit(formulas), "option_exact_hit_at_5": hit(options)}


def run_candidate(candidate: str, db_path: Path, query_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    queries, qrels = _load_queries(query_path)
    corpus = _load_corpus(db_path)
    load_started = time.perf_counter()
    encoder, model_meta = _embedder(candidate)
    load_seconds = time.perf_counter() - load_started
    encode_started = time.perf_counter()
    encoded: dict[str, tuple[list[str], list[list[float]], dict[str, str]]] = {}
    for slot, rows in corpus.items():
        ids = [row["id"] for row in rows]
        texts = [row["text"] for row in rows]
        encoded[slot] = (ids, encoder(texts, False), dict(zip(ids, texts, strict=True)))
    corpus_encode_seconds = time.perf_counter() - encode_started

    observations = []
    query_seconds = []
    for query in queries:
        query_id = query["query_id"]
        rel = qrels.get(query_id, {})
        positive = [doc_id for doc_id, score in rel.items() if score > 0]
        if not positive:
            continue
        slot = "strategy" if positive[0].endswith("_tutor_strategy") else "misconception"
        ids, vectors, documents = encoded[slot]
        t0 = time.perf_counter()
        query_vector = encoder([query["query"]], True)[0]
        ranked = _cosine_rank(query_vector, vectors, ids)
        query_seconds.append(time.perf_counter() - t0)
        observations.append({
            "query_id": query_id, "slot": slot, "ranked_ids": ranked,
            "qrels": rel, "content": _content_metrics(query["query"], ranked, documents, rel),
        })

    def avg(metric: str) -> float | None:
        vals = [
            float(item["content"][metric])
            for item in observations
            if item["content"].get(metric) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    metrics = {
        "candidate_recall_at_20": sum(candidate_recall(item["ranked_ids"], item["qrels"], 20) for item in observations) / len(observations),
        "candidate_recall_at_50": sum(candidate_recall(item["ranked_ids"], item["qrels"], 50) for item in observations) / len(observations),
        "recall_at_1": sum(recall_at_k(item["ranked_ids"], item["qrels"], 1) for item in observations) / len(observations),
        "recall_at_3": sum(recall_at_k(item["ranked_ids"], item["qrels"], 3) for item in observations) / len(observations),
        "recall_at_5": sum(recall_at_k(item["ranked_ids"], item["qrels"], 5) for item in observations) / len(observations),
        "mrr": sum(mean_reciprocal_rank(item["ranked_ids"], item["qrels"]) for item in observations) / len(observations),
        "ndcg_at_5": sum(ndcg_at_k(item["ranked_ids"], item["qrels"], 5) for item in observations) / len(observations),
        "numeric_exact_hit_at_5": avg("numeric_exact_hit_at_5"),
        "formula_exact_hit_at_5": avg("formula_exact_hit_at_5"),
        "option_exact_hit_at_5": avg("option_exact_hit_at_5"),
        "query_latency_ms_avg": sum(query_seconds) / len(query_seconds) * 1000.0,
        "query_latency_ms_p95": sorted(query_seconds)[max(0, math.ceil(len(query_seconds) * 0.95) - 1)] * 1000.0,
    }
    for item in observations:
        for key, value in item["content"].items():
            item[key] = value
    return {
        "candidate_id": candidate,
        "status": "SUCCESS",
        "model": model_meta,
        "case_count": len(observations),
        "metrics": metrics,
        "timings": {"model_load_seconds": load_seconds, "corpus_encode_seconds": corpus_encode_seconds, "total_seconds": time.perf_counter() - started},
        "slice_metrics": {
            slot: {
                "case_count": sum(item["slot"] == slot for item in observations),
                "recall_at_5": sum(recall_at_k(item["ranked_ids"], item["qrels"], 5) for item in observations if item["slot"] == slot) / max(1, sum(item["slot"] == slot for item in observations)),
            }
            for slot in ("misconception", "strategy")
        },
    }


def candidate_recall(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    return float(bool(set(ranked[:k]) & {doc for doc, score in qrels.items() if score > 0}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated embedding candidate benchmark")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--queries", type=Path, default=Path("evals/datasets/l1-v2/queries.jsonl"))
    parser.add_argument("--candidates", nargs="+", choices=sorted(MODEL_IDS), default=["deterministic", "multilingual-e5-small", "multilingual-e5-base", "bge-m3"])
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval/embedding-benchmark"))
    args = parser.parse_args(argv)
    db = args.db.resolve(strict=True); chroma = args.chroma.resolve(strict=True); queries = args.queries.resolve(strict=True)
    source_hash = hash_storage_artifacts(db, chroma)
    results = []
    for candidate in args.candidates:
        try:
            result = run_candidate(candidate, db, queries)
        except Exception as exc:
            result = {"candidate_id": candidate, "status": "UNMEASURED", "error_type": type(exc).__name__, "error_message": str(exc)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    after_hash = hash_storage_artifacts(db, chroma)
    if after_hash != source_hash:
        raise RuntimeError("source artifact changed during embedding benchmark")
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("embedding-benchmark-%Y%m%dT%H%M%SZ")
    payload = {"schema_version": "embedding-benchmark/v1", "run_id": run_id, "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "source_artifact_hash": source_hash, "runtime": {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(), "cuda": False}, "results": results}
    (output / f"{run_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(output / f"{run_id}.md", payload)
    return 0 if all(item["status"] == "SUCCESS" for item in results) else 2


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Embedding Benchmark", "", f"Run: `{payload['run_id']}`", f"Source artifact hash: `{payload['source_artifact_hash']}`", "", "| Candidate | Status | Candidate Recall@20 | Recall@5 | MRR | nDCG@5 | P95 ms |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in payload["results"]:
        m = item.get("metrics", {})
        lines.append(f"| {item['candidate_id']} | {item['status']} | {m.get('candidate_recall_at_20', '')} | {m.get('recall_at_5', '')} | {m.get('mrr', '')} | {m.get('ndcg_at_5', '')} | {m.get('query_latency_ms_p95', '')} |")
        if item["status"] != "SUCCESS": lines.append(f"\n`{item['candidate_id']}` unavailable: {item.get('error_type')}: {item.get('error_message')}\n")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
