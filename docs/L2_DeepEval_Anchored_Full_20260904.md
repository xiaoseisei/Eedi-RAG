# L2 DeepEval Anchored Parent-3/W5 全量回归（2026-09-04）

## 结论

本轮在生产默认的 Anchored Parent-3/W5 上完成了真实 L2 Gold/Real 全量回归。结果没有出现“全线飘绿”：上下文召回提升后，Real 的相关性和上下文排序较好，但 Generator 的引用完整性、忠实度和教学表达仍未达首轮门槛。

这说明当前问题已经从“Assembler 只能塞进约 3 个重叠窗口”转移为两个下游问题：

1. Generator 没有稳定复制最终 Context 中的完整引用四元组，导致 citation recall 低；
2. Generator/DeepEval 对“上下文是否足以支持完整答案”的判断仍不稳定，Faithfulness、Contextual Recall 和 Pedagogical GEval 未达标。

## 运行口径

```text
真实 L2：Qwen Embedding + BM25/Dense → Top-20 Card → Card Rerank
→ Anchored 总 Parent-3 → W6/S3 Window Rerank → overlap-aware Assembler
→ Generator LLM → Citation Audit → DeepEval Judge
```

- Gold：评测控制面直接注入 required evidence，仅作为上限轨，不进入生产检索。
- Real：通过 `EndToEndPedagogicalRAGPipeline.prepare_context()` 使用生产最终上下文，再调用同一 Generator。
- DeepEval Judge：`mimo-v2.5-pro`，OpenAI-compatible endpoint，temperature `0`。
- Generator：`.env` 中配置的生产模型，timeout `60s`。
- 数据：30-case Golden，dev 24 / holdout 6；qrels 为 provisional quote-derived labels。
- artifact：`b45860c80d02b999c9c9bf6d960c50779b84aad1b76fa74bef02da6ac9a8aeb9`，运行前后 hash 一致。
- 报告：`reports/eval/l2-anchored-parent3-w5-full-20260904-160000/`。

本轮采用 4 个可恢复 shard 并行执行：1–8、9–16、17–23、24–30；合并器校验 shard 不重叠、路由一致、artifact hash 一致和 Gold/Real 成对完整。

## 指标结果

首轮门槛：Faithfulness `≥0.90`、Answer Relevancy `≥0.85`、Contextual Recall `≥0.90`、Contextual Precision `≥0.85`、Pedagogical GEval `≥0.80`、Citation Audit `≥0.95`。

| Track | Metric | Mean | Passed cases | 结论 |
| --- | --- | ---: | ---: | --- |
| Gold | Answer Relevancy | **0.9583** | 26/30 | 达标 |
| Gold | Contextual Precision | **0.9667** | 29/30 | 达标 |
| Gold | Contextual Recall | 0.6867 | 15/30 | 未达标 |
| Gold | Faithfulness | 0.6972 | 18/30 | 未达标 |
| Gold | Pedagogical GEval | 0.5033 | 6/30 | 未达标 |
| Gold | Citation Audit | 0.7111 | 14/30 | 未达标 |
| Real | Answer Relevancy | **0.9400** | 25/30 | 达标 |
| Real | Contextual Precision | **0.9550** | 28/30 | 达标 |
| Real | Contextual Recall | 0.8022 | 18/30 | 未达标 |
| Real | Faithfulness | 0.7528 | 17/30 | 未达标 |
| Real | Pedagogical GEval | 0.5600 | 8/30 | 未达标 |
| Real | Citation Audit | **0.2222** | 0/30 | 严重未达标 |

全部指标均为 DeepEval 实测值；没有用关键词规则、旧硬编码分数或 qrels 反向填补。

## 关键归因

### 1. Anchored 上下文已能支持相关回答，但不是完整回答

Real Answer Relevancy `0.9400` 和 Contextual Precision `0.9550` 表明模型大多数时候回答了用户问题，且输入上下文整体相关、排序合理。Real Contextual Recall `0.8022` 仍低于门槛，说明最终上下文虽然平均约 4.87 个窗口、原始 Turn 覆盖较好，但对 expected answer 所需的语义事实覆盖仍不完整。

### 2. Citation Audit 的主要损失在 Generator 输出，不是授权链路

Real Citation Audit 分布为：13/30 为 `0`，13/30 为 `0.3333`，2/30 为 `0.5`，2/30 为 `0.6667`。常见错误是模型把 `[Turn N] [speaker]` 标记一起复制进 `quote_text`，或只引用一个 Turn，未覆盖 required quotes。多数非错误 case 的 `quote_grounding_precision`、`role_accuracy` 和 `candidate_authorization_rate` 仍为 `1.0`，说明系统能校验真实 Turn，问题是 Generator 没有稳定、完整地选择和输出引用。

全量共有 6 条 `generation/audit CitationAuditError`：Gold 1 条，Real 5 条（Real：cases 12、17、26、27、28）。错误均已保留在 `report.json` 的 `errors` 和对应 trace 中。

### 3. Faithfulness 未因 Context Recall 达到 0.90 而自动达标

Real Faithfulness `0.7528`，Gold 也只有 `0.6972`。这说明模型会在真实对白之外加入概括、推断或不受上下文直接支持的陈述；增加窗口只提高可见证据，不会自动约束回答必须逐句由 Context 支持。Gold 轨同样偏低，排除了“只有检索丢失”这一单一归因，但 Gold Context 的 quote-only 形式也可能不足以承载完整教学答案，因此 Gold 结果不是业务上限真值。

### 4. Pedagogical GEval 是当前最弱的语义指标

Real `0.5600`、Gold `0.5033`，显著低于 `0.80`。当前 Generator 输出虽常与问题相关，但未稳定做到：区分学生错因与导师策略、给出完整教学干预、只使用证据支持的事实。需要单独优化 Generator system prompt、结构化答案契约和多相关人工 rubric，不能只继续调检索。

## 发布判断

报告的 `release_decision=BLOCKED_L1_PRECONDITION`，因为历史 L1 closeout 仍是 `BLOCKED`；即使忽略该前置，本轮 L2 语义指标也会因 Faithfulness、Contextual Recall、Pedagogical GEval、Citation Audit 未达门槛而阻断。当前不能宣称 L2 通过或“Generator 全线飘绿”。

## 下一步

1. 保持 Anchored Parent-3/W5 和 overlap-aware Assembler，不回退已验证的上下文优化。
2. 优先修复 Generator citation contract：要求引用只能从最终 Context 逐字复制，拆分 Turn 标记与 `quote_text`，并要求覆盖回答中使用的每个证据主张。
3. 对 Real 失败 cases 做 Generator prompt/输出 schema 专项 A/B；固定 retrieval/context，单独测 citation recall、Faithfulness、Answer Relevancy。
4. 用多相关 graded qrels 和教研员 rubric 校准 DeepEval 的 Contextual Recall、Faithfulness、Pedagogical GEval，避免把 quote-only Gold 误当业务真值。
5. 轮换此前误打印到工具输出中的 API 凭据，并更新本地 `.env`；报告不记录任何密钥。

## 复现与合并

单 shard：

```powershell
.venv\Scripts\python.exe scripts\run_l2_deepeval.py `
  --db reports\eval\card-evidence-rebind-staging-20260904-001500\tutoring_knowledge.duckdb `
  --chroma reports\eval\card-reextraction-qwen-index-20260903-203000\chroma `
  --l1-closeout reports\eval\l1-card-reextraction-qwen-full-20260903-223000\closeout.json `
  --l1-manifest evals\datasets\l1-closeout-20260903-015400\manifest.json `
  --expected-artifact-hash b45860c80d02b999c9c9bf6d960c50779b84aad1b76fa74bef02da6ac9a8aeb9 `
  --dataset-output evals\datasets\l2-anchored-shard-<id> `
  --output-root reports\eval --run-id <shard-run-id> `
  --case-start <start> --max-cases <count> `
  --generator-timeout-seconds 60 `
  --judge-model mimo-v2.5-pro --judge-api-key-env LLM_API_KEY `
  --judge-base-url https://api.xiaomimimo.com/v1 `
  --reranker-backend siliconflow --rerank-unit anchored_logical_window `
  --reranker-pool-size 20 --parent-card-count 3 --evidence-selection-count 5 `
  --retrieval-mode bm25_dense --bm25-weight 0.35 --allow-l1-blocked
```

合并：

```powershell
.venv\Scripts\python.exe scripts\merge_l2_deepeval_reports.py `
  --shard reports\eval\l2-anchored-p3w5-shard1-<id> `
  --shard reports\eval\l2-anchored-p3w5-shard2-<id> `
  --shard reports\eval\l2-anchored-p3w5-shard3-<id> `
  --shard reports\eval\l2-anchored-p3w5-shard4-<id> `
  --output-dir reports\eval\l2-anchored-parent3-w5-full-<id>
```

## Generator-first v2 回归补充（2026-09-05）

本节记录在同一 Anchored Parent-3/W5 路线上切换 Generator v2 后的真实结果。完整报告：`reports/eval/l2-generator-v2-full-20260905-000000/`；A/B 参数与边界：`docs/L2_Generator_First_AB_20260904.md`。

| Track | Answer Relevancy | Contextual Precision | Contextual Recall | Faithfulness | Pedagogical GEval | Citation Audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gold | 0.7932 (29) | 1.0000 (28) | 0.9557 (29) | 0.9241 (29) | 0.9286 (28) | 0.5690 (29) |
| Real | 0.8625 (26) | 0.8979 (26) | 0.7404 (26) | 0.7704 (27) | 0.9231 (26) | 0.4877 (27) |

v2 使 Faithfulness/Pedagogical GEval 显著高于旧基线，但 Citation Audit 仍低于 0.95，Real Contextual Recall 也未达 0.90。4 条 trace 的远端 connection error 已保留，未以默认值替代；release 仍为 `BLOCKED_L1_PRECONDITION`。

API 余额不足时产生的 v1 partial 报告 `reports/eval/l2-generator-v1-full-20260905-0100-partial/` 仅作故障记录，不用于 A/B 选型。API 恢复后已启动新的 v1 分片（`20260905-0200`），待全部完成后再补充同口径对照。

## Gold 对齐修复补充（2026-09-05）

Gold Context v2 现按 required Turn 构造去重的 `[CARD_FACT]`、`[AUTHORITATIVE_TURN]` 和 `[ALLOWED_INFERENCE]` 节点；DeepEval 使用节点列表，并对 Answer Relevancy 单独使用 Generator 核心答案。30-case 本地结构审计为 required Turn 缺失 `0`、ground-truth 泄漏 `0`、平均 `4.13` 节点和 `3006` 字符；case 0025 的卡片 `error_choice=A` 与 Golden `C` 冲突，已作为数据 warning 保留。

Gold alignment smoke 报告：`reports/eval/l2-gold-alignment-smoke-20260905-0901/`。Gold 的 Answer Relevancy、Contextual Recall、Faithfulness、Pedagogical GEval 分别为 `1.00/1.00/0.90/0.90`；Citation Audit `0.50` 仍显示 required evidence 覆盖契约需要继续加强。该 smoke 不是 30-case release gate。
