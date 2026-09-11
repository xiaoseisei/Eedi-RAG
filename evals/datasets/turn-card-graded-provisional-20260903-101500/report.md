# Turn-level graded retrieval evaluation

Status: **PROVISIONAL — HUMAN REVIEW REQUIRED**

Cases: `30`; window graded qrels: `21570`; card graded qrels: `3000`; source artifact unchanged: `True`

## Aggregates

| Metric | Mean | Dev | Holdout |
| --- | ---: | ---: | ---: |
| graded_recall_at_3 | 0.5811 | 0.5493 | 0.7083 |
| graded_recall_at_5 | 0.7983 | 0.7861 | 0.8472 |
| graded_recall_at_10 | 0.8700 | 0.8757 | 0.8472 |
| graded_recall_at_20 | 0.9433 | 0.9292 | 1.0000 |
| graded_precision_at_5 | 0.8600 | 0.8667 | 0.8333 |
| graded_ndcg_at_5 | 0.8110 | 0.7889 | 0.8993 |
| graded_mrr | 0.8461 | 0.8076 | 1.0000 |
| turn_coverage_at_1 | 0.4722 | 0.4236 | 0.6667 |
| turn_coverage_at_3 | 0.7944 | 0.7569 | 0.9444 |
| turn_coverage_at_5 | 0.9000 | 0.8750 | 1.0000 |
| latency_ms | 593.2249 | 633.6772 | 431.4157 |
| card_graded_recall_at_3 | 0.8333 | 0.8333 | 0.8333 |
| card_graded_recall_at_5 | 0.8333 | 0.8333 | 0.8333 |
| card_graded_recall_at_20 | 0.8333 | 0.8333 | 0.8333 |
| card_graded_precision_at_5 | 0.3600 | 0.3583 | 0.3667 |
| card_graded_ndcg_at_5 | 0.8982 | 0.9070 | 0.8628 |
| card_graded_mrr | 0.8333 | 0.8333 | 0.8333 |

## Attribution

- Card pointer loss cases: `23` / `30`
- Retrieval rank loss cases (Turn@3 < 1): `12` / `30`
- Role-correct card pointer recall mean: `0.6111`
- Turn coverage@5 mean: `0.9000`
- Window generation loss cases: `0` in this dataset; production window union covers all required turns.

## Interpretation

If card pointer recall is low while full window coverage is complete, the first loss is extraction/pointer selection. If pointer coverage is complete but top-k Turn coverage is low, the loss is window retrieval/ranking. Assembler and Generator must be evaluated separately against final context and citation traces.

Detailed per-case data is in `raw_cases.jsonl`; all qrels carry `needs_human_review=true`.
