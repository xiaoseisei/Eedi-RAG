# Eedi-RAG L1/L2/L3 多维分析报告

最终交付状态：**L1_BLOCKED_L2_VALIDATED_L3_BUSINESS_UNMEASURED**

## 1. 结论摘要

L1 已完成可审计 closeout，但 Grounding citation recall、fallback Chunk Turn Recall 和 Retrieval latency 未过门禁；因此 L1 不能发布。L2 Gold/Real + DeepEval 闭环已在真实 API/Judge 上跑通，但只完成 2-case smoke，且受 L1 前置阻塞。L3 已完成 30-case Eedi 技术 proxy 召回测评，真实业务效果尚无数据。

## 2. 原始数据与 provenance

- L1 dataset: `eedi-golden30-l1-closeout-v1`；L2 dataset: `eedi-l2-golden30-v1`；L3 dataset: `eedi-l3-business-proxy-v1`
- Artifact hash 一致：`True`（6bb387b4b9698009fffca6193a35d02b98b0424d017fc4add166f89428e9d67a）
- L1 raw observations: `E:\PIAgent\10-projects\AgentLearn\Eedi-RAG\reports\eval\l1-closeout-qwen-20260903-025500\observations.jsonl`
- L2 traces/metrics: `E:\PIAgent\10-projects\AgentLearn\Eedi-RAG\reports\eval\l2-deepeval-smoke-20260903-060000\traces.jsonl` / `E:\PIAgent\10-projects\AgentLearn\Eedi-RAG\reports\eval\l2-deepeval-smoke-20260903-060000\metrics.jsonl`
- L3 per-case raw data: `E:\PIAgent\10-projects\AgentLearn\Eedi-RAG\reports\eval\l3-business-proxy-20260903-051500\raw_cases.jsonl`

## 3. L1 指标

| Stage | Metric | Status | Value | Threshold | Hard gate |
| --- | --- | --- | ---: | ---: | --- |
| grounding | citation_precision | SUCCESS | 0.9888888888888889 | 0.95 | True |
| grounding | citation_recall | FAILED | 0.5888888888888888 | 0.95 | True |
| grounding | role_accuracy | SUCCESS | 0.9888888888888889 | 0.95 | True |
| grounding | quote_grounding_precision | SUCCESS | 0.9888888888888889 | 0.95 | True |
| grounding | candidate_authorization_rate | SUCCESS | 1.0 | 0.95 | True |
| chunking | session_recall_at_1 | SUCCESS | 0.9 | 0.0 | False |
| chunking | session_recall_at_3 | SUCCESS | 0.9333333333333333 | 0.9 | True |
| chunking | turn_recall_at_1 | SUCCESS | 0.4722222222222222 | 0.0 | False |
| chunking | turn_recall_at_3 | FAILED | 0.7944444444444444 | 0.9 | True |
| chunking | turn_mrr | FAILED | 0.8460858585858586 | 0.85 | True |
| chunking | inflation_ratio | UNMEASURED | None | 4.0 | False |
| chunking | latency_p95_ms | FAILED | 153.55034999999998 | 150.0 | False |
| chunk_structure | pointer_validity | SUCCESS | 1.0 | 1.0 | True |
| chunk_structure | verbatim_integrity | SUCCESS | 1.0 | 1.0 | True |
| chunk_structure | turn_coverage | SUCCESS | 1.0 | 1.0 | True |
| chunk_structure | boundary_integrity | SUCCESS | 1.0 | 1.0 | True |
| chunk_structure | orphan_count | SUCCESS | 0.0 | 0.0 | True |
| chunk_structure | out_of_bounds_turn_count | SUCCESS | 0.0 | 0.0 | True |
| chunk_structure | inflation_ratio | SUCCESS | 3.801842150130753 | 4.0 | False |
| rewrite | protected_token_preservation | SUCCESS | 1.0 | 1.0 | True |
| rewrite | entity_preservation | SUCCESS | 1.0 | 0.99 | True |
| rewrite | intent_preservation | UNMEASURED | None | 0.95 | False |
| rewrite | domain_injection_coverage | UNMEASURED | None | 1.0 | False |
| rewrite | raw_mrr | SUCCESS | 1.0 | 0.0 | False |
| rewrite | rewritten_mrr | SUCCESS | 0.8722222222222222 | 0.0 | False |
| rewrite | retrieval_mrr_lift | FAILED | -0.1277777777777778 | 0.0 | False |
| rewrite | raw_recall_at_5 | SUCCESS | 1.0 | 0.0 | False |
| rewrite | rewritten_recall_at_5 | SUCCESS | 1.0 | 0.0 | False |
| rewrite | retrieval_recall_at_5_lift | SUCCESS | 0.0 | 0.0 | False |
| rewrite | latency_p95_ms | FAILED | 116.9964 | 100.0 | False |
| fusion | raw_rrf_mrr | SUCCESS | 1.0 | 0.0 | False |
| fusion | rewrite_only_rrf_mrr | SUCCESS | 0.8777777777777778 | 0.0 | False |
| fusion | raw_inclusive_rrf_mrr | SUCCESS | 0.961111111111111 | 0.0 | False |
| fusion | rewrite_only_rrf_mrr_lift | FAILED | -0.12222222222222223 | 0.0 | False |
| fusion | raw_inclusive_rrf_mrr_lift | FAILED | -0.03888888888888889 | 0.0 | False |
| fusion | raw_inclusive_rrf_recall_at_5_lift | SUCCESS | 0.0 | 0.0 | False |
| fusion | selected_fusion_mrr | SUCCESS | 1.0 | 0.0 | False |
| fusion | selected_fusion_mrr_lift | SUCCESS | 0.0 | 0.0 | True |
| fusion | selected_fusion_recall_at_5_lift | SUCCESS | 0.0 | 0.0 | True |
| retrieval | candidate_recall_at_20 | SUCCESS | 1.0 | 0.98 | True |
| retrieval | recall_at_1 | SUCCESS | 1.0 | 0.0 | False |
| retrieval | recall_at_3 | SUCCESS | 1.0 | 0.9 | True |
| retrieval | recall_at_5 | SUCCESS | 1.0 | 0.95 | True |
| retrieval | recall_at_20 | SUCCESS | 1.0 | 0.98 | True |
| retrieval | precision_at_5 | SUCCESS | 0.2 | 0.0 | False |
| retrieval | noise_ratio_at_5 | FAILED | 0.8 | 0.05 | False |
| retrieval | mrr | SUCCESS | 1.0 | 0.85 | True |
| retrieval | ndcg_at_5 | SUCCESS | 1.0 | 0.9 | True |
| retrieval | latency_p95_ms | FAILED | 443.80876999930456 | 300.0 | True |
| assembler | candidate_recall_retention | SUCCESS | 1.0 | 0.9 | True |
| assembler | retriever_miss_rate | SUCCESS | 0.0 | 1.0 | False |
| assembler | assembler_drop_rate | SUCCESS | 0.0 | 1.0 | False |
| assembler | final_evidence_recall | SUCCESS | 1.0 | 0.9 | True |
| assembler | selection_precision | FAILED | 0.5 | 0.9 | False |
| assembler | noise_ratio | FAILED | 0.5 | 0.1 | False |
| assembler | token_budget_violation_rate | SUCCESS | 0.0 | 0.0 | True |
| assembler | evidence_empty_rate | SUCCESS | 0.0 | 0.0 | True |
| assembler | compression_ratio | SUCCESS | 0.943696409988701 | 0.0 | False |
| assembler | rerank_ndcg_gain | SUCCESS | 0.252371901428583 | 0.0 | False |
| assembler | latency_p95_ms | SUCCESS | 3.635599999999998 | 500.0 | True |

## 4. L2 Gold/Real DeepEval 指标

| Track | Metric | Measured | Mean | Threshold |
| --- | --- | ---: | ---: | ---: |
| gold | answer_relevancy | 2 | 1.0 | 0.85 |
| gold | citation_audit | 2 | 0.8333333333333333 | 0.95 |
| gold | contextual_precision | 2 | 1.0 | 0.85 |
| gold | contextual_recall | 2 | 0.8333333333333333 | 0.9 |
| gold | faithfulness | 2 | 0.875 | 0.9 |
| gold | pedagogical_geval | 2 | 0.6000000000000001 | 0.8 |
| real | answer_relevancy | 2 | 1.0 | 0.85 |
| real | citation_audit | 2 | 0.25 | 0.95 |
| real | contextual_precision | 2 | 1.0 | 0.85 |
| real | contextual_recall | 2 | 1.0 | 0.9 |
| real | faithfulness | 2 | 0.75 | 0.9 |
| real | pedagogical_geval | 2 | 0.6 | 0.8 |

## 5. L3 业务 proxy 指标

| Metric | Mean | Dev | Holdout | Status |
| --- | ---: | ---: | ---: | --- |
| actionability | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| candidate_recall_at_20 | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| final_document_recall | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| final_evidence_recall | 0.6111 | 0.6319 | 0.5278 | DIAGNOSTIC_PROXY |
| learning_gain | — | — | — | UNMEASURED |
| mrr | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| online_feedback | — | — | — | UNMEASURED |
| recall_at_5 | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| researcher_adoption | — | — | — | UNMEASURED |
| task_completion | — | — | — | UNMEASURED |

## 6. 组件优化结论

| Layer | Component | Evidence | Conclusion |
| --- | --- | --- | --- |
| L1 | Storage / Chunk Structure | ID parity, metadata, orphan, pointer, verbatim and boundary gates passed in the latest closeout. | Keep current artifact contract and immutable read path; no structural rollback needed. |
| L1 | Retriever / Fusion | Qwen single-slot Recall@5/MRR/nDCG=1.0; selected raw_first fusion lift=0.0; retrieval P95 exceeded 300ms. | Preserve raw_first as non-degrading diagnostic policy; optimize API/Chroma latency and replace single-positive qrels before claiming quality. |
| L1 | Fallback Chunk | Turn Recall@3=0.7944 and MRR=0.8461; dev lower than holdout. | Improve dialogue-window query/representation and validate on larger independent graded qrels. |
| L1 | Grounding | Citation precision/role/quote≈0.9889, authorization=1.0, citation recall=0.5889. | Keep quadruple/candidate authorization hard gates; improve generator citation coverage without copying gold evidence. |
| L1 | Assembler | Candidate retention/evidence recall=1.0, budget/evidence-empty=0, latency P95≈3.6ms after slot/qrels correction. | Independent slots and raw rank boost are effective; global two-slot precision/noise remains diagnostic under one-slot qrels. |
| L2 | Gold/Real Generator + DeepEval | 2 Gold + 2 Real traces, 24 DeepEval metric observations, Judge readiness SUCCESS; Gold pedagogical GEval=0.6000000000000001, Real faithfulness=0.75. | The semantic loop is runnable, but expand calibration/holdout sample and resolve L1 precondition before release. |
| L3 | Business proxy / feedback | 30 proxy cases; retrieval/document recall and actionability=1.0, final evidence recall=0.6111; adoption/task/learning/online feedback UNMEASURED. | Use proxy only for technical prioritization; collect real researcher/user events and pre/post learning outcomes before business claims. |

## 7. 限制与下一步

- Eedi is a technical proxy, not NBCOT customer data.
- L1/L3 qrels are derived from human quote evidence and are not independent multi-document graded judgments.
- L2 smoke sample has 2 cases per track; calibration and full holdout semantic evaluation remain required.
- Real business adoption, task completion, learning gain, and online feedback are unmeasured.
