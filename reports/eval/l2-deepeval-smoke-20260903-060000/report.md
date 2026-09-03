# L2 DeepEval Report

Release decision: **BLOCKED_L1_PRECONDITION**
L1 precondition: `BLOCKED`
Judge readiness: `MetricStatus.SUCCESS`
Source artifact unchanged: `True`

## Metric aggregates

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

## Reproduce

`.venv\Scripts\python.exe scripts\run_l2_deepeval.py --db data\db\tutoring_knowledge.duckdb --chroma reports\eval\embedding-qwen3-0.6b-closeout-20260903-014700\chroma --l1-closeout reports\eval\l1-closeout-qwen-20260903-025500\closeout.json --l1-manifest evals\datasets\l1-closeout-20260903-015400\manifest.json --dataset-output evals\datasets\l2-golden30-20260903-060000 --output-root reports\eval --run-id <new-run-id> --max-cases 2 --judge-model mimo-v2.5-pro --judge-api-key-env LLM_API_KEY --judge-base-url https://api.xiaomimimo.com/v1 --allow-l1-blocked`
