# Eedi-RAG L1 Component Evaluation

Release status: **BLOCKED**

- Run: `l1-closeout-qwen-20260903-025500`
- Dataset: `eedi-golden30-l1-closeout-v1`
- Qrels: `golden-verbatim-v2-derived-document-v1`
- System config hash: `819ff4dc4b0fee0709ad6f9f6131da7e7def17625c5c291ee8b27b175590d4c4`
- Modules: storage, grounding, chunking, chunk_structure, rewrite, fusion, retrieval, assembler

## Status counts

- SUCCESS: 2110
- FAILED: 344
- UNMEASURED: 262
- DEGRADED: 0
- ERROR: 0

## Non-success observations

| Stage | Metric | Scope | Status | Value | Threshold | Reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| grounding | citation_recall | case | FAILED | 0.3333333333333333 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.3333333333333333 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.0 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.0 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.5 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.3333333333333333 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.5 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.0 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_precision | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.0 | 0.95 |  |
| grounding | role_accuracy | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | quote_grounding_precision | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | case | FAILED | 0.6666666666666666 | 0.95 |  |
| grounding | citation_recall | aggregate | FAILED | 0.5888888888888888 | 0.95 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | latency_ms | case | FAILED | 263.64 | 150.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | latency_ms | case | FAILED | 152.358 | 150.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | latency_ms | case | FAILED | 152.815 | 150.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.5 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.5 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | latency_ms | case | FAILED | 152.657 | 150.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_mrr | case | FAILED | 0.5 | 0.85 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | turn_mrr | case | FAILED | 0.3333333333333333 | 0.85 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | session_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | turn_mrr | case | FAILED | 0.5 | 0.85 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | latency_ms | case | FAILED | 154.152 | 150.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | latency_ms | case | FAILED | 150.932 | 150.0 |  |
| chunking | session_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | session_recall_at_3 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_mrr | case | FAILED | 0.09090909090909091 | 0.85 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | session_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | session_recall_at_3 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_mrr | case | FAILED | 0.125 | 0.85 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | turn_mrr | case | FAILED | 0.3333333333333333 | 0.85 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.0 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | turn_mrr | case | FAILED | 0.5 | 0.85 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | turn_recall_at_3 | case | FAILED | 0.6666666666666666 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_1 | case | FAILED | 0.3333333333333333 | 1.0 |  |
| chunking | inflation_ratio | case | UNMEASURED |  | 4.0 | retrieval candidate expansion is not chunk-generation inflation |
| chunking | turn_recall_at_3 | aggregate | FAILED | 0.7944444444444444 | 0.9 |  |
| chunking | turn_mrr | aggregate | FAILED | 0.8460858585858586 | 0.85 |  |
| chunking | inflation_ratio | aggregate | UNMEASURED |  | 4.0 | all 30 case observations are UNMEASURED or ERROR |
| chunking | latency_p95_ms | aggregate | FAILED | 153.55034999999998 | 150.0 |  |
| chunking | turn_recall_at_3 | slice | FAILED | 0.8333333333333334 | 0.9 |  |
| chunking | inflation_ratio | slice | UNMEASURED |  | 4.0 | all 18 case observations are UNMEASURED or ERROR |
| chunking | latency_p95_ms | slice | FAILED | 170.57519999999982 | 150.0 |  |
| chunking | turn_recall_at_3 | slice | FAILED | 0.7361111111111112 | 0.9 |  |
| chunking | turn_mrr | slice | FAILED | 0.827020202020202 | 0.85 |  |
| chunking | inflation_ratio | slice | UNMEASURED |  | 4.0 | all 12 case observations are UNMEASURED or ERROR |
| chunking | turn_recall_at_3 | slice | FAILED | 0.7944444444444444 | 0.9 |  |
| chunking | turn_mrr | slice | FAILED | 0.8460858585858586 | 0.85 |  |
| chunking | inflation_ratio | slice | UNMEASURED |  | 4.0 | all 30 case observations are UNMEASURED or ERROR |
| chunking | latency_p95_ms | slice | FAILED | 153.55034999999998 | 150.0 |  |
| chunking | turn_recall_at_3 | slice | FAILED | 0.7569444444444445 | 0.9 |  |
| chunking | turn_mrr | slice | FAILED | 0.8076073232323232 | 0.85 |  |
| chunking | inflation_ratio | slice | UNMEASURED |  | 4.0 | all 24 case observations are UNMEASURED or ERROR |
| chunking | latency_p95_ms | slice | FAILED | 153.95145 | 150.0 |  |
| chunking | inflation_ratio | slice | UNMEASURED |  | 4.0 | all 6 case observations are UNMEASURED or ERROR |
| chunk_structure | inflation_ratio | case | FAILED | 4.725663716814159 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.286516853932584 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.596273291925466 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.346303501945525 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.844827586206897 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.796992481203008 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 5.0120481927710845 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.184027777777778 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.390243902439025 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 6.351111111111111 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 5.415584415584416 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.163346613545817 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.353070175438597 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.556603773584905 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.16256157635468 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.595628415300546 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.35978835978836 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.364485981308412 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.173728813559322 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.903114186851211 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.251552795031056 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.258064516129032 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.878787878787879 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.519774011299435 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.7218543046357615 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.152492668621701 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.072368421052632 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.16358024691358 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.878172588832487 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.131868131868132 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.159509202453988 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 5.6923076923076925 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 5.038759689922481 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.185022026431718 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 5.055276381909548 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.35754189944134 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 5.183098591549296 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.364864864864865 | 4.0 |  |
| chunk_structure | inflation_ratio | case | FAILED | 4.456 | 4.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 119.639 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 113.137 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 113.627 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | retrieval_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| rewrite | latency_ms | case | FAILED | 116.244 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 115.246 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 104.355 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 107.681 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 108.69 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 115.111 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 117.612 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 111.696 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 113.106 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | retrieval_mrr_lift | case | FAILED | -0.6666666666666667 | 0.0 |  |
| rewrite | latency_ms | case | FAILED | 110.267 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | retrieval_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| rewrite | latency_ms | case | FAILED | 111.579 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | retrieval_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| rewrite | latency_ms | case | FAILED | 107.028 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 108.476 | 100.0 |  |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | retrieval_mrr_lift | case | FAILED | -0.6666666666666667 | 0.0 |  |
| rewrite | latency_ms | case | FAILED | 105.192 | 100.0 |  |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 111.649 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 112.166 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 113.214 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 109.194 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | retrieval_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| rewrite | latency_ms | case | FAILED | 105.652 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 110.665 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 109.551 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 108.252 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 104.156 | 100.0 |  |
| rewrite | protected_token_preservation | case | UNMEASURED |  | 1.0 | query has no explicitly protected numeric/formula/option tokens |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 110.104 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 110.169 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | latency_ms | case | FAILED | 111.918 | 100.0 |  |
| rewrite | entity_preservation | case | UNMEASURED |  | 1.0 | query has no labeled critical entities |
| rewrite | intent_preservation | case | UNMEASURED |  | 1.0 | intent preservation has no human or DeepEval label |
| rewrite | domain_injection_coverage | case | UNMEASURED |  | 1.0 | domain injection is not applicable to this case |
| rewrite | retrieval_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| rewrite | latency_ms | case | FAILED | 107.88 | 100.0 |  |
| rewrite | intent_preservation | aggregate | UNMEASURED |  | 0.95 | all 30 case observations are UNMEASURED or ERROR |
| rewrite | domain_injection_coverage | aggregate | UNMEASURED |  | 1.0 | all 30 case observations are UNMEASURED or ERROR |
| rewrite | retrieval_mrr_lift | aggregate | FAILED | -0.1277777777777778 | 0.0 |  |
| rewrite | latency_p95_ms | aggregate | FAILED | 116.9964 | 100.0 |  |
| rewrite | intent_preservation | slice | UNMEASURED |  | 0.95 | all 18 case observations are UNMEASURED or ERROR |
| rewrite | domain_injection_coverage | slice | UNMEASURED |  | 1.0 | all 18 case observations are UNMEASURED or ERROR |
| rewrite | retrieval_mrr_lift | slice | FAILED | -0.1851851851851852 | 0.0 |  |
| rewrite | latency_p95_ms | slice | FAILED | 116.75325000000001 | 100.0 |  |
| rewrite | intent_preservation | slice | UNMEASURED |  | 0.95 | all 12 case observations are UNMEASURED or ERROR |
| rewrite | domain_injection_coverage | slice | UNMEASURED |  | 1.0 | all 12 case observations are UNMEASURED or ERROR |
| rewrite | retrieval_mrr_lift | slice | FAILED | -0.041666666666666664 | 0.0 |  |
| rewrite | latency_p95_ms | slice | FAILED | 116.23644999999999 | 100.0 |  |
| rewrite | intent_preservation | slice | UNMEASURED |  | 0.95 | all 24 case observations are UNMEASURED or ERROR |
| rewrite | domain_injection_coverage | slice | UNMEASURED |  | 1.0 | all 24 case observations are UNMEASURED or ERROR |
| rewrite | retrieval_mrr_lift | slice | FAILED | -0.1388888888888889 | 0.0 |  |
| rewrite | latency_p95_ms | slice | FAILED | 117.4068 | 100.0 |  |
| rewrite | intent_preservation | slice | UNMEASURED |  | 0.95 | all 6 case observations are UNMEASURED or ERROR |
| rewrite | domain_injection_coverage | slice | UNMEASURED |  | 1.0 | all 6 case observations are UNMEASURED or ERROR |
| rewrite | retrieval_mrr_lift | slice | FAILED | -0.08333333333333333 | 0.0 |  |
| rewrite | latency_p95_ms | slice | FAILED | 112.82275 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 308.844 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 318.495 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 386.855 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| fusion | latency_ms | case | FAILED | 325.603 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 331.141 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 340.896 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 326.433 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 327.207 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 326.128 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 326.093 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 324.416 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 309.918 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| fusion | latency_ms | case | FAILED | 313.047 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| fusion | latency_ms | case | FAILED | 303.206 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| fusion | raw_inclusive_rrf_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| fusion | latency_ms | case | FAILED | 307.436 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 305.769 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | case | FAILED | -0.6666666666666667 | 0.0 |  |
| fusion | raw_inclusive_rrf_mrr_lift | case | FAILED | -0.6666666666666667 | 0.0 |  |
| fusion | latency_ms | case | FAILED | 313.389 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 307.507 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 315.609 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 318.952 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 303.912 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| fusion | latency_ms | case | FAILED | 308.365 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 317.504 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 330.158 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 329.985 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 326.801 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 331.664 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 330.413 | 100.0 |  |
| fusion | latency_ms | case | FAILED | 329.956 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | case | FAILED | -0.5 | 0.0 |  |
| fusion | latency_ms | case | FAILED | 329.589 | 100.0 |  |
| fusion | rewrite_only_rrf_mrr_lift | aggregate | FAILED | -0.12222222222222223 | 0.0 |  |
| fusion | raw_inclusive_rrf_mrr_lift | aggregate | FAILED | -0.03888888888888889 | 0.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 2855.0046000018483 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 483.75050000322517 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 331.04930000263266 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 302.1250999881886 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 329.8903000104474 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 381.3064999994822 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 353.17319999739993 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 360.5778999917675 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 315.4465999978129 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 394.9910999945132 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 321.24580000527203 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 360.006600007182 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_ms | case | FAILED | 320.8463000046322 | 300.0 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | case | FAILED | 0.8 | 0.05 |  |
| retrieval | noise_ratio_at_5 | aggregate | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_p95_ms | aggregate | FAILED | 443.80876999930456 | 300.0 |  |
| retrieval | noise_ratio_at_5 | slice | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_p95_ms | slice | FAILED | 734.7419049932761 | 300.0 |  |
| retrieval | noise_ratio_at_5 | slice | FAILED | 0.8000000000000002 | 0.05 |  |
| retrieval | latency_p95_ms | slice | FAILED | 434.93282999843353 | 300.0 |  |
| retrieval | noise_ratio_at_5 | slice | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_p95_ms | slice | FAILED | 734.7419049932761 | 300.0 |  |
| retrieval | noise_ratio_at_5 | slice | FAILED | 0.8000000000000002 | 0.05 |  |
| retrieval | latency_p95_ms | slice | FAILED | 434.93282999843353 | 300.0 |  |
| retrieval | noise_ratio_at_5 | slice | FAILED | 0.8 | 0.05 |  |
| retrieval | latency_p95_ms | slice | FAILED | 443.80876999930456 | 300.0 |  |
| retrieval | noise_ratio_at_5 | slice | FAILED | 0.8000000000000002 | 0.05 |  |
| retrieval | latency_p95_ms | slice | FAILED | 470.4365900019182 | 300.0 |  |
| retrieval | noise_ratio_at_5 | slice | FAILED | 0.8000000000000002 | 0.05 |  |
| retrieval | latency_p95_ms | slice | FAILED | 349.29507499327883 | 300.0 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | misconception_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | misconception_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for misconception slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | strategy_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | strategy_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for strategy slot |
| assembler | fallback_window_slot_precision | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | fallback_window_slot_recall | case | UNMEASURED |  | 1.0 | no positive qrels for fallback_window slot |
| assembler | selection_precision | case | FAILED | 0.5 | 1.0 |  |
| assembler | noise_ratio | case | FAILED | 0.5 | 0.1 |  |
| assembler | selection_precision | aggregate | FAILED | 0.5 | 0.9 |  |
| assembler | noise_ratio | aggregate | FAILED | 0.5 | 0.1 |  |
