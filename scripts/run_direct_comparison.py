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
from src.reranker import PedagogicalGoldAssembler, estimate_text_tokens, GoldAssembledContext
from scripts.run_l2_deepeval import _context_coverage
from evals.contracts import EvidenceRef

TRACES_PATH = PROJECT_ROOT / "reports" / "eval" / "l2-qwen-full-final-20260907" / "traces.jsonl"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_test_set.json"
DB_PATH = PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb"
CHROMA_DIR = PROJECT_ROOT / "reports" / "eval" / "embedding-qwen3-full-20260907-rerun" / "chroma"


def select_parent_cards(ranked_lanes: dict, parent_limit: int = 6) -> list:
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


def assemble_greedy_budget(assembler, raw_query, retrieval_results, max_windows):
    evidence_units = retrieval_results.get("evidence_units", []) or []
    ranked_units = sorted(
        (dict(unit) for unit in evidence_units),
        key=lambda unit: (
            int(unit.get("reranker_rank", 10**9)),
            -float(unit.get("reranker_score", 0.0)),
            str(unit.get("chunk_id", "")),
        ),
    )
    parent_cards = retrieval_results.get("anchored_parent_cards") or []
    selected_misc = assembler._select_mmr_best(retrieval_results.get("misconceptions", []), raw_query, [], [])
    selected_strat = assembler._select_mmr_best(retrieval_results.get("strategies", []), raw_query, [], [])
    parent_fact_nodes = assembler._render_parent_fact_nodes(parent_cards)

    def render(units):
        ordered_units = sorted(
            units,
            key=lambda unit: (
                int(unit.get("reranker_rank", 10**9)),
                -float(unit.get("reranker_score", 0.0)),
                str(unit.get("chunk_id", "")),
            ),
        )
        ordered = assembler._ordered_unit_evidence(ordered_units)
        catalog_text, evidence_ids = assembler._render_compact_evidence_catalog(ordered) if ordered else ("", {})
        window_nodes = assembler._render_unique_window_nodes(ordered_units, anchored=True, evidence_ids=evidence_ids)
        context_nodes = [*parent_fact_nodes, *window_nodes]
        lines = [
            "# 【权威教研参考知识基座 (Anchored Logical-Window Context)】",
            "> Parent Card 提供有来源的派生事实；真实 Turn 仍是唯一可引用的原始证据。",
        ]
        if parent_fact_nodes:
            lines.extend(["", "## 一、 Parent Card 派生事实", *parent_fact_nodes])
        lines.extend(["", "## 二、 Anchored Reranker 排序后的真实逻辑链证据", *window_nodes])
        lines.extend(["", "## 三、 可引用的真实 Turn 证据", "- 仅允许引用上方逻辑链窗口中逐字出现的 `[Turn N]` 原文。"])
        prompt = "\n\n".join(lines)
        final_context = "\n\n".join(part for part in (prompt, catalog_text) if part)
        estimated = estimate_text_tokens(final_context)
        return prompt, ordered, estimated, context_nodes, estimate_text_tokens(catalog_text) if catalog_text else 0

    selected_units = []
    seen_turn_keys = set()
    prompt_markdown, ordered_evidence, estimated_tokens, context_nodes, catalog_tokens = render(selected_units)

    for unit in ranked_units:
        if len(selected_units) >= max_windows:
            break
        unit_turn_keys = {(int(t.get("session_id", -1)), int(t["turn_id"])) for t in unit.get("evidence_turns", [])}
        if selected_units and unit_turn_keys and unit_turn_keys.issubset(seen_turn_keys):
            continue
        trial_units = selected_units + [unit]
        trial_prompt, trial_evidence, trial_tokens, trial_nodes, trial_catalog = render(trial_units)
        if selected_units and trial_tokens > assembler.max_prompt_tokens:
            continue
        selected_units.append(unit)
        seen_turn_keys.update(unit_turn_keys)
        prompt_markdown, ordered_evidence, estimated_tokens, context_nodes, catalog_tokens = (
            trial_prompt, trial_evidence, trial_tokens, trial_nodes, trial_catalog
        )

    return GoldAssembledContext(
        raw_query=raw_query,
        prompt_context_markdown=prompt_markdown,
        selected_misconception=selected_misc,
        selected_strategy=selected_strat,
        selected_windows=[],
        selected_evidence_units=selected_units,
        chunk_strategy="card",
        rerank_unit="anchored_logical_window",
        evidence_selection_count=len(selected_units),
        evidence_turns=ordered_evidence,
        estimated_token_count=estimated_tokens,
        compression_ratio=0.0,
        budget_violation=False,
        truncation_loss=0,
        deepeval_context_nodes=context_nodes,
        catalog_token_count=catalog_tokens,
    )


def main():
    print("Loading datasets and test cases...", flush=True)
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
    assembler = PedagogicalGoldAssembler(
        lambda_diversity=0.7,
        max_prompt_tokens=4000,
        model_reranker=model_reranker,
        rerank_unit="anchored_logical_window",
        reranker_pool_size=20,
        parent_card_count=6,
        evidence_selection_count=5,
    )

    # Pre-rank windows for all 30 cases for W6 and W7
    print("Pre-ranking candidate windows for 30 cases (W6 and W7)...", flush=True)
    w6_ranked = []
    w7_ranked = []
    
    for idx, (case, trace) in enumerate(zip(golden, traces), start=1):
        query = case["question"]
        rt = trace.get("retrieval_trace", {})
        ranked_lanes = {"misconceptions": rt.get("misconceptions", []), "strategies": rt.get("strategies", [])}
        card_candidates = list(ranked_lanes["misconceptions"]) + list(ranked_lanes["strategies"])
        parent_cards = select_parent_cards(ranked_lanes, parent_limit=5)

        # W6
        u6 = retriever.expand_anchored_logical_windows(card_candidates, parent_cards, window_size=6, step=3)
        r6 = assembler.rerank_candidates(query, u6, unit_name="anchored_logical_window")
        w6_ranked.append((parent_cards, r6, ranked_lanes))

        # W7
        u7 = retriever.expand_anchored_logical_windows(card_candidates, parent_cards, window_size=7, step=3)
        r7 = assembler.rerank_candidates(query, u7, unit_name="anchored_logical_window")
        w7_ranked.append((parent_cards, r7, ranked_lanes))

    storage.close()
    print("Ranking complete! Evaluating 4 configurations...", flush=True)

    configs = [
        ("Top-20 → Parent-6 → 5 个窗口 (W6/S3 窗口选 5 个)", w6_ranked, 5),
        ("Top-20 → Parent-6 → 7 个窗口 (W6/S3 窗口选 7 个)", w6_ranked, 7),
        ("Top-20 → Parent-6 → 5 个窗口 (W7/S3 窗口选 5 个)", w7_ranked, 5),
        ("Top-20 → Parent-6 → 7 个窗口 (W7/S3 窗口选 7 个)", w7_ranked, 7),
    ]

    summary_rows = []

    for name, dataset, max_w in configs:
        recalls = []
        claim_recalls = []
        tokens = []
        max_tok = 0
        actual_window_counts = []

        for idx, (case, (parent_cards, ranked_units, ranked_lanes)) in enumerate(zip(golden, dataset), start=1):
            query = case["question"]
            retrieval_res = {
                "chunk_strategy": "card",
                "misconceptions": ranked_lanes["misconceptions"],
                "strategies": ranked_lanes["strategies"],
                "anchored_parent_cards": parent_cards,
                "evidence_units": ranked_units,
                "rewritten_queries": {},
            }
            gold_ctx = assemble_greedy_budget(assembler, query, retrieval_res, max_windows=max_w)

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

            recalls.append(cov["turn_recall"])
            claim_recalls.append(cov.get("claim_recall") or 0.0)
            tok = gold_ctx.estimated_token_count
            tokens.append(tok)
            if tok > max_tok:
                max_tok = tok
            actual_window_counts.append(len(gold_ctx.selected_evidence_units))

        row = {
            "name": name,
            "turn_recall": float(np.mean([r for r in recalls if r is not None])),
            "claim_recall": float(np.mean(claim_recalls)),
            "avg_tokens": float(np.mean(tokens)),
            "max_tokens": max_tok,
            "avg_windows": float(np.mean(actual_window_counts)),
        }
        summary_rows.append(row)
        print(f"\n[{name}]")
        print(f"  真实 Turn Recall:        {row['turn_recall']*100:.2f}%")
        print(f"  Contextual Recall (代理): {row['claim_recall']*100:.2f}%")
        print(f"  平均 Prompt Tokens:      {row['avg_tokens']:.0f} (Max: {row['max_tokens']})")
        print(f"  实际装载窗口数:          {row['avg_windows']:.1f}")

    print("\n" + "=" * 80)
    print("RESULTS_JSON:" + json.dumps(summary_rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
