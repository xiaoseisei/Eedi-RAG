# Eedi-RAG L1 Closeout

Decision: **BLOCKED**

- Run: `l1-closeout-qwen-20260903-025500`
- Dataset: `eedi-golden30-l1-closeout-v1`
- Qrels: `golden-verbatim-v2-derived-document-v1`
- Artifact hash unchanged: `True`
- Hard blockers: `67`
- Business quality claim: `NOT_ALLOWED` (qrels are not independent multi-document graded judgments)

## storage

No aggregate observation.

## chunk_structure

| Metric | Status | Value | Threshold | Hard gate |
| --- | --- | ---: | ---: | --- |
| pointer_validity | SUCCESS | 1.0 | 1.0 | True |
| verbatim_integrity | SUCCESS | 1.0 | 1.0 | True |
| turn_coverage | SUCCESS | 1.0 | 1.0 | True |
| boundary_integrity | SUCCESS | 1.0 | 1.0 | True |
| orphan_count | SUCCESS | 0.0 | 0.0 | True |
| out_of_bounds_turn_count | SUCCESS | 0.0 | 0.0 | True |
| inflation_ratio | SUCCESS | 3.801842150130753 | 4.0 | False |

## chunking

| Metric | Status | Value | Threshold | Hard gate |
| --- | --- | ---: | ---: | --- |
| session_recall_at_1 | SUCCESS | 0.9 | 0.0 | False |
| session_recall_at_3 | SUCCESS | 0.9333333333333333 | 0.9 | True |
| turn_recall_at_1 | SUCCESS | 0.4722222222222222 | 0.0 | False |
| turn_recall_at_3 | FAILED | 0.7944444444444444 | 0.9 | True |
| turn_mrr | FAILED | 0.8460858585858586 | 0.85 | True |
| inflation_ratio | UNMEASURED | None | 4.0 | False |
| latency_p95_ms | FAILED | 153.55034999999998 | 150.0 | False |

## grounding

| Metric | Status | Value | Threshold | Hard gate |
| --- | --- | ---: | ---: | --- |
| citation_precision | SUCCESS | 0.9888888888888889 | 0.95 | True |
| citation_recall | FAILED | 0.5888888888888888 | 0.95 | True |
| role_accuracy | SUCCESS | 0.9888888888888889 | 0.95 | True |
| quote_grounding_precision | SUCCESS | 0.9888888888888889 | 0.95 | True |
| candidate_authorization_rate | SUCCESS | 1.0 | 0.95 | True |

## rewrite

| Metric | Status | Value | Threshold | Hard gate |
| --- | --- | ---: | ---: | --- |
| protected_token_preservation | SUCCESS | 1.0 | 1.0 | True |
| entity_preservation | SUCCESS | 1.0 | 0.99 | True |
| intent_preservation | UNMEASURED | None | 0.95 | False |
| domain_injection_coverage | UNMEASURED | None | 1.0 | False |
| raw_mrr | SUCCESS | 1.0 | 0.0 | False |
| rewritten_mrr | SUCCESS | 0.8722222222222222 | 0.0 | False |
| retrieval_mrr_lift | FAILED | -0.1277777777777778 | 0.0 | False |
| raw_recall_at_5 | SUCCESS | 1.0 | 0.0 | False |
| rewritten_recall_at_5 | SUCCESS | 1.0 | 0.0 | False |
| retrieval_recall_at_5_lift | SUCCESS | 0.0 | 0.0 | False |
| latency_p95_ms | FAILED | 116.9964 | 100.0 | False |

## fusion

| Metric | Status | Value | Threshold | Hard gate |
| --- | --- | ---: | ---: | --- |
| raw_rrf_mrr | SUCCESS | 1.0 | 0.0 | False |
| rewrite_only_rrf_mrr | SUCCESS | 0.8777777777777778 | 0.0 | False |
| raw_inclusive_rrf_mrr | SUCCESS | 0.961111111111111 | 0.0 | False |
| rewrite_only_rrf_mrr_lift | FAILED | -0.12222222222222223 | 0.0 | False |
| raw_inclusive_rrf_mrr_lift | FAILED | -0.03888888888888889 | 0.0 | False |
| raw_inclusive_rrf_recall_at_5_lift | SUCCESS | 0.0 | 0.0 | False |
| selected_fusion_mrr | SUCCESS | 1.0 | 0.0 | False |
| selected_fusion_mrr_lift | SUCCESS | 0.0 | 0.0 | True |
| selected_fusion_recall_at_5_lift | SUCCESS | 0.0 | 0.0 | True |

## retrieval

| Metric | Status | Value | Threshold | Hard gate |
| --- | --- | ---: | ---: | --- |
| candidate_recall_at_20 | SUCCESS | 1.0 | 0.98 | True |
| recall_at_1 | SUCCESS | 1.0 | 0.0 | False |
| recall_at_3 | SUCCESS | 1.0 | 0.9 | True |
| recall_at_5 | SUCCESS | 1.0 | 0.95 | True |
| recall_at_20 | SUCCESS | 1.0 | 0.98 | True |
| precision_at_5 | SUCCESS | 0.2 | 0.0 | False |
| noise_ratio_at_5 | FAILED | 0.8 | 0.05 | False |
| mrr | SUCCESS | 1.0 | 0.85 | True |
| ndcg_at_5 | SUCCESS | 1.0 | 0.9 | True |
| latency_p95_ms | FAILED | 443.80876999930456 | 300.0 | True |

## assembler

| Metric | Status | Value | Threshold | Hard gate |
| --- | --- | ---: | ---: | --- |
| candidate_recall_retention | SUCCESS | 1.0 | 0.9 | True |
| retriever_miss_rate | SUCCESS | 0.0 | 1.0 | False |
| assembler_drop_rate | SUCCESS | 0.0 | 1.0 | False |
| final_evidence_recall | SUCCESS | 1.0 | 0.9 | True |
| selection_precision | FAILED | 0.5 | 0.9 | False |
| noise_ratio | FAILED | 0.5 | 0.1 | False |
| token_budget_violation_rate | SUCCESS | 0.0 | 0.0 | True |
| evidence_empty_rate | SUCCESS | 0.0 | 0.0 | True |
| compression_ratio | SUCCESS | 0.943696409988701 | 0.0 | False |
| rerank_ndcg_gain | SUCCESS | 0.252371901428583 | 0.0 | False |
| latency_p95_ms | SUCCESS | 3.635599999999998 | 500.0 | True |

## Reproduce

`.venv\Scripts\python.exe scripts\run_l1_qwen_full_eval.py --db data\db\tutoring_knowledge.duckdb --chroma reports\eval\embedding-qwen3-0.6b-closeout-20260903-014700\chroma --dataset-manifest evals\datasets\l1-closeout-20260903-015400\manifest.json --output-root reports\eval --run-id <new-unique-run-id> --top-k 20`
