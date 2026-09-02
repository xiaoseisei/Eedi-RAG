from __future__ import annotations

import importlib
import time
from typing import Any, Literal

from evals.contracts import RetrievalEvalCase


def capture_legacy_retrieval_cases(
    retriever: Any,
    *,
    mode: Literal["raw", "rrf"],
    top_k: int = 20,
    limit: int | None = None,
) -> list[RetrievalEvalCase]:
    """Run the existing 50-query diagnostic set without changing its qrels semantics."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero when supplied")
    benchmark_module = importlib.import_module("scripts.eval_50_queries")
    benchmark = benchmark_module.BENCHMARK_50_QUERIES
    selected_items = benchmark[:limit] if limit is not None else benchmark
    cases: list[RetrievalEvalCase] = []
    for item in selected_items:
        started = time.perf_counter()
        if mode == "rrf":
            result = retriever.retrieve_multi_perspective_rrf(
                item["query"],
                top_k_each=top_k,
                fetch_evidence=False,
            )
            candidates = (
                result["misconceptions"]
                if item["target_collection"] == "student_misconceptions"
                else result["strategies"]
            )
        else:
            if item["target_collection"] == "student_misconceptions":
                candidates = retriever.retrieve_misconceptions(
                    item["query"], top_k=top_k, fetch_evidence=False
                )
            else:
                candidates = retriever.retrieve_strategies(
                    item["query"], top_k=top_k, fetch_evidence=False
                )
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranked_ids = [str(candidate["chunk_id"]) for candidate in candidates]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"retriever returned duplicate IDs for benchmark query {item['id']}")
        target_suffix = (
            "misconception"
            if item["target_collection"] == "student_misconceptions"
            else "tutor_strategy"
        )
        target_id = f"session_{item['target_session_id']}_{target_suffix}"
        cases.append(
            RetrievalEvalCase(
                case_id=f"legacy50-{item['id']}-{mode}",
                query=item["query"],
                ranked_ids=ranked_ids,
                qrels={target_id: 3},
                latency_ms=latency_ms,
                slices={
                    "mode": mode,
                    "card_type": item["type"],
                    "collection": item["target_collection"],
                },
            )
        )
    return cases

