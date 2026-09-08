"""Fast comparison script for Top-20 -> Parent-6 with different window counts and sizes.

Evaluates on the 30 Golden Test Set cases:
1. Top-20 -> Parent-6 -> 5 窗口 (W6 / Sel-5)
2. Top-20 -> Parent-6 -> 7 窗口 (W6 / Sel-7)
3. Top-20 -> Parent-6 -> 5 窗口 (W7 / Sel-5)
4. Top-20 -> Parent-6 -> 7 窗口 (W7 / Sel-7)

Outputs side-by-side comparison of:
- Context Evidence Turn Recall (真实 Turn 覆盖率)
- Contextual Recall / Claim Coverage Proxy (语义事实覆盖率)
- Average / Max Prompt Context Tokens (Prompt Token 消耗)
- Candidate Window Coverage (候选展开覆盖率)
- Rerank Latency P95 (重排耗时)
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

PROJECT_ROOT = Path(r"E:\PIAgent\10-projects\AgentLearn\Eedi-RAG")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.storage_manager import DualEngineStorageManager
from src.retriever import DualMetricRetriever
from src.reranker_provider import SiliconFlowQwen3Reranker
from src.reranker import PedagogicalGoldAssembler
from scripts.run_l2_deepeval import _context_coverage
from evals.contracts import EvidenceRef

TRACES_PATH = PROJECT_ROOT / "reports" / "eval" / "l2-qwen-full-final-20260907" / "traces.jsonl"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_test_set.json"
DB_PATH = PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb"
CHROMA_DIR = PROJECT_ROOT / "reports" / "eval" / "embedding-qwen3-full-20260907-rerun" / "chroma"


def select_parent_cards(ranked_lanes: dict, parent_limit: int = 6) -> list:
    """Greedily pick top misconception & strategy, then fill up to parent_limit by rank."""
    parent_cards = []
    for lane in ("misconceptions", "strategies"):
        lane_cards = ranked_lanes.get(lane, [])
        if lane_cards and len(parent_cards) < parent_limit:
            parent_cards.append(lane_cards[0])

    remaining_cards = sorted(
        [
            card
            for cards in ranked_lanes.values()
            for card in cards
            if card not in parent_cards
        ],
        key=lambda card: (
            -float(card.get("reranker_score", 0.0)),
            int(card.get("reranker_rank", 10**9)),
            str(card.get("chunk_id", "")),
        ),
    )
    parent_cards.extend(remaining_cards[: max(0, parent_limit - len(parent_cards))])
    return parent_cards


def evaluate_configuration(
    golden_cases: list,
    traces: list,
    storage: DualEngineStorageManager,
    retriever: DualMetricRetriever,
    model_reranker: SiliconFlowQwen3Reranker,
    window_size: int,
    step: int,
    sel_count: int,
    parent_k: int = 6,
    budget: int = 3500,
    cached_reranks: dict | None = None,
) -> dict:
    assembler = PedagogicalGoldAssembler(
        lambda_diversity=0.7,
        max_prompt_tokens=budget,
        model_reranker=model_reranker,
        rerank_unit="anchored_logical_window",
        reranker_pool_size=20,
        parent_card_count=parent_k,
        evidence_selection_count=sel_count,
    )

    turn_recalls = []
    claim_recalls = []
    cand_recalls = []
    parent_recalls = []
    token_counts = []
    max_tokens = 0
    rerank_times = []

    for idx, (case, trace) in enumerate(zip(golden_cases, traces), start=1):
        query = case["question"]
        rt = trace.get("retrieval_trace", {})
        ranked_lanes = {
            "misconceptions": rt.get("misconceptions", []),
            "strategies": rt.get("strategies", []),
        }
        card_candidates = list(ranked_lanes["misconceptions"]) + list(ranked_lanes["strategies"])
        parent_cards = select_parent_cards(ranked_lanes, parent_limit=parent_k)

        cache_key = (idx, window_size, step, parent_k)
        if cached_reranks is not None and cache_key in cached_reranks:
            ranked_units, elapsed_ms = cached_reranks[cache_key]
        else:
            anchored_units = retriever.expand_anchored_logical_windows(
                card_candidates,
                parent_cards,
                window_size=window_size,
                step=step,
            )
            t1 = time.perf_counter()
            ranked_units = assembler.rerank_candidates(
                query,
                anchored_units,
                unit_name="anchored_logical_window",
            )
            elapsed_ms = (time.perf_counter() - t1) * 1000
            if cached_reranks is not None:
                cached_reranks[cache_key] = (ranked_units, elapsed_ms)

        rerank_times.append(elapsed_ms)

        retrieval_res = {
            "chunk_strategy": "card",
            "misconceptions": ranked_lanes["misconceptions"],
            "strategies": ranked_lanes["strategies"],
            "anchored_parent_cards": parent_cards,
            "evidence_units": ranked_units,
            "rewritten_queries": rt.get("rewritten_queries", {}),
        }
        gold_ctx = assembler.assemble(query, retrieval_res)

        quotes = case.get("verbatim_grounding_quotes", [])
        required = [
            EvidenceRef(
                session_id=int(q["session_id"]),
                turn_id=int(q["turn_id"]),
                speaker=q["speaker"],
                quote_text=q["quote_text"],
            )
            for q in quotes
        ]

        cov = _context_coverage(
            case,
            required,
            gold_ctx.evidence_turns,
            gold_ctx.deepeval_context_nodes,
            parent_cards=parent_cards,
            candidate_units=ranked_units,
        )

        turn_recalls.append(cov["turn_recall"])
        claim_recalls.append(cov.get("claim_recall") or 0.0)
        cand_recalls.append(cov["candidate_turn_recall"])
        parent_recalls.append(cov["parent_pointer_recall"])
        tok = gold_ctx.estimated_token_count
        token_counts.append(tok)
        if tok > max_tokens:
            max_tokens = tok

    p95_time = float(np.percentile(rerank_times, 95))
    mean_turn = float(np.mean([r for r in turn_recalls if r is not None]))
    mean_claim = float(np.mean(claim_recalls))
    mean_cand = float(np.mean([c for c in cand_recalls if c is not None]))
    mean_tok = float(np.mean(token_counts))

    return {
        "turn_recall": mean_turn,
        "claim_recall": mean_claim,
        "candidate_recall": mean_cand,
        "avg_tokens": mean_tok,
        "max_tokens": max_tokens,
        "p95_rerank_ms": p95_time,
    }


def main():
    print("=" * 70)
    print("  Eedi-RAG Parent-6 Window Configuration Comparison (30 Golden Cases)")
    print("=" * 70)

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    with open(TRACES_PATH, "r", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f if line.strip() and json.loads(line).get("track") == "real"]

    storage = DualEngineStorageManager(
        db_path=str(DB_PATH),
        chroma_dir=str(CHROMA_DIR),
        embedding_backend="deterministic",
    )
    retriever = DualMetricRetriever(
        storage_manager=storage,
        query_rewrite_mode="deterministic",
        fusion_strategy="raw_first",
    )
    model_reranker = SiliconFlowQwen3Reranker.from_env()

    configs = [
        # (Label, window_size, step, sel_count)
        ("Top-20 -> Parent-6 -> 5 个窗口 (W6 / Sel-5)", 6, 3, 5),
        ("Top-20 -> Parent-6 -> 7 个窗口 (W6 / Sel-7)", 6, 3, 7),
        ("Top-20 -> Parent-6 -> 5 个窗口 (W7 / Sel-5)", 7, 3, 5),
        ("Top-20 -> Parent-6 -> 7 个窗口 (W7 / Sel-7)", 7, 3, 7),
    ]

    cached_reranks = {}
    results = []

    try:
        for name, w_size, step, sel_count in configs:
            print(f"\n[Running] {name}...")
            res = evaluate_configuration(
                golden_cases=golden,
                traces=traces,
                storage=storage,
                retriever=retriever,
                model_reranker=model_reranker,
                window_size=w_size,
                step=step,
                sel_count=sel_count,
                parent_k=6,
                budget=3500,
                cached_reranks=cached_reranks,
            )
            res["name"] = name
            results.append(res)
            print(f"  -> 真实 Turn Recall:        {res['turn_recall']*100:.2f}%")
            print(f"  -> Contextual Recall (代理): {res['claim_recall']*100:.2f}%")
            print(f"  -> 平均 Prompt Tokens:      {res['avg_tokens']:.1f} (Max: {res['max_tokens']})")
            print(f"  -> 候选池窗口覆盖率:        {res['candidate_recall']*100:.2f}%")
            print(f"  -> Rerank P95 耗时:         {res['p95_rerank_ms']:.1f} ms")
    finally:
        storage.close()

    print("\n" + "=" * 70)
    print("  横向对比汇总表 (Markdown)")
    print("=" * 70)
    print("| 配置方案 | 真实 Turn Recall | Contextual Recall (语义代理) | 平均 Prompt Token | 最大 Prompt Token | 候选池覆盖率 | Rerank P95 |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        print(f"| {r['name']} | **{r['turn_recall']*100:.2f}%** | {r['claim_recall']*100:.2f}% | {r['avg_tokens']:.0f} | {r['max_tokens']} | {r['candidate_recall']*100:.2f}% | {r['p95_rerank_ms']:.1f} ms |")
    print("=" * 70)


if __name__ == "__main__":
    main()
