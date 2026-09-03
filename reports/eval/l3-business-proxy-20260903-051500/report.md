# L3 Business Proxy Report

Release decision: **L3_BUSINESS_UNMEASURED**

- Cases: `30` (dev=24, holdout=6)
- Source artifact unchanged: `True`
- Eedi is a technical proxy; adoption, task completion, learning gain, and online feedback are not measured.

## Proxy metrics

| Metric | Mean | Dev | Holdout | Status |
| --- | ---: | ---: | ---: | --- |
| candidate_recall_at_20 | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| recall_at_5 | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| mrr | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| final_document_recall | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| final_evidence_recall | 0.6111 | 0.6319 | 0.5278 | DIAGNOSTIC_PROXY |
| actionability | 1.0000 | 1.0000 | 1.0000 | DIAGNOSTIC_PROXY |
| researcher_adoption | — | — | — | UNMEASURED |
| task_completion | — | — | — | UNMEASURED |
| learning_gain | — | — | — | UNMEASURED |
| online_feedback | — | — | — | UNMEASURED |

Raw per-case evidence: `raw_cases.jsonl`
