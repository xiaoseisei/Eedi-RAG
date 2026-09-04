# Card / Fallback Chunk Strategy：生产开关与 L1 全量对照

日期：2026-09-04（Asia/Shanghai）  
数据：Eedi QATD-2k 技术代理数据，30 条 golden cases，固定 dev/holdout split  
Embedding：`Qwen/Qwen3-Embedding-0.6B`（1024 维）  
检索：`BM25 + Dense`，`bm25_weight=0.35`，L1 `top_k=20`；专项 A/B 为 Top-15 pool  
来源 artifact：`reports/eval/card-reextraction-staging-20260903-200000/` + `reports/eval/card-reextraction-qwen-index-20260903-203000/`  
hash：`c47318847151b90023f2dceddc7ab60b265725f546bc633b8d8f6e211c1408bb`

## 边界

- `card` 只检索 misconception/strategy 卡片，再通过 `source_turn_ids` 回溯 DuckDB 原文。
- `fallback` 只检索 `fallback_windows` 真实滑动窗口，不依赖卡片指针。
- 没有引入第三种混合 chunk 结果；生产 DB/Chroma 未被覆盖，评测在隔离副本上执行。
- 两条路径使用相同 30-case、split 和 Qwen artifact。qrels 是由人工 verbatim quote 派生的 provisional labels，不是独立多文档业务真值。
- API 或评测异常只记录为失败/未测量，不填默认分数。

## 生产开关

```powershell
# 默认：卡片优先
python -m src.cli --chunk-strategy card

# 原文窗口优先
python -m src.cli --chunk-strategy fallback
```

两者均可继续配置 `--retrieval-mode dense|bm25_dense`、`--bm25-weight` 和可选的 `--reranker-backend siliconflow`。无外部 Reranker 时默认仍为 `card`；启用 SiliconFlow Reranker 且未指定单位时，生产默认改为 Anchored Parent-3/W5，仍可显式选择 `card` 或 `logical_evidence`。

## 当前 L1 全量门禁

命令：

```text
python scripts/run_l1_qwen_full_eval.py ... --chunk-strategy card    --run-id l1-card-strategy-20260904-090000
python scripts/run_l1_qwen_full_eval.py ... --chunk-strategy fallback --run-id l1-fallback-strategy-20260904-110000
```

| Stage / aggregate metric | Card | Fallback | 解释 |
| --- | ---: | ---: | --- |
| Storage ID/metadata/反向关系 | 通过 | 通过 | 两方案读取同一隔离 artifact |
| Chunk Structure pointer/verbatim/boundary/coverage | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 1.0 | 结构正确性不是差异来源 |
| Chunking session recall@3 | 1.0000 | 0.9000 | Card 目标 session 更容易在前列出现 |
| Chunking turn recall@3 | 0.8722 | 0.6889 | 当前无 reranker 的 L1 实际排序；fallback 仍有排序损失 |
| Chunking turn MRR | 0.9333 | 0.7745 | 同上 |
| Grounding citation precision | 0.8984 | 0.8085 | fallback 送入的连续窗口包含更多非目标对白 |
| Grounding citation recall | 0.7333 | 0.6667 | 两者均未达到 0.95 门禁 |
| Retrieval candidate recall@20 | 1.0000 | 0.9000 | card 文档 qrels 与窗口 qrels 不同，不能单独作为业务优劣结论 |
| Retrieval nDCG@5 | 0.8819 | 0.6994 | 同一原因，且 fallback 候选数量更大 |
| Retrieval MRR | 0.9500 | 0.9000 | card 目标卡定位更集中 |
| Assembler final evidence recall | 0.8667 | 0.6550 | 当前无 reranker 时 card 装配更稳定 |
| Assembler selection precision | 0.4333 | 0.7267 | fallback 选出的窗口中 provisional qrels 命中比例更高 |
| Token budget violation / empty evidence | 0 / 0 | 0 / 0 | 两者均满足 |
| Retrieval P95 latency (ms) | 374.7 | 3478.9 | fallback 的 exact channel + 大窗口候选造成明显延迟 |
| L1 release status | `BLOCKED` | `BLOCKED` | card：461 failed/263 unmeasured；fallback：543 failed/878 unmeasured（Rewrite/Fusion 对 content-first 显式 N/A） |

L1 报告：

- `reports/eval/l1-card-strategy-20260904-090000/`
- `reports/eval/l1-fallback-strategy-20260904-110000/`

## 同一 Top-15 + Qwen Reranker 专项 A/B

该专项使用相同 Top-15 pool、同一 Qwen reranker 和策略专用 graded qrels；它回答“最终真实证据能覆盖多少”，不替代 L1 门禁。

| 指标 | Card | Fallback | 结论 |
| --- | ---: | ---: | --- |
| Pool Recall@5 | 1.0000 | 0.6550 | Card 更容易找回目标文档/窗口 |
| Reranked Recall@5 | 1.0000 | 0.7111 | fallback rerank 可改善窗口目标覆盖，但仍低于 card 文档召回 |
| Reranked nDCG@5 | 0.9218 | 0.7507 | Card 目标卡排序更集中 |
| Reranked Precision@5 | 0.3533 | 0.8000 | fallback Top-5 中直接相关窗口比例更高 |
| Reranked MRR | 0.9833 | 0.7500 | Card 首个目标卡更靠前 |
| Final evidence recall（全部 required Turns） | 0.4889 | 0.8222 | fallback 对完整 Turn 链覆盖明显更强 |
| Target role/card-or-window evidence recall | 0.8556 | 0.8222 | card 在目标角色证据上略高；fallback 统一覆盖学生+导师证据 |
| Context token mean | 756.9 | 1500.0 | card 更省上下文；fallback 达到预算上限 |
| Assembler mean latency (ms) | 0.265 | 0.620 | card 更轻 |
| Reranker P50/P95 (ms) | 413.8 / 533.9 | 339.1 / 406.7 | fallback 文本窗口 rerank 延迟反而较低，但端到端 retrieval 仍需优化 |

专项报告：

- `reports/eval/chunk-strategy-card-full-20260904-061000/`
- `reports/eval/chunk-strategy-fallback-full-20260904-070000/`

## 多角度结论

### Card 的优势

- 结构化语义摘要集中，目标槽位定位和首个目标命中更强。
- Context 平均约 757 tokens，适合摘要、诊断、快速教研问答。
- 在当前 L1 无 reranker 条件下，assembler retention、final evidence recall 和端到端延迟更稳。
- 缺点是卡片生成阶段可能漏掉未被 `source_turn_ids` 指针覆盖的关键对白；因此“命中卡片”不等于“覆盖完整 Turn 因果链”。

### Fallback 的优势

- 直接使用真实窗口，绕过卡片抽取遗漏，完整 required-Turn evidence recall 从 0.4889 提升到 0.8222（专项 A/B）。
- Top-5 直接相关窗口 precision 0.8000，高于 card 的 0.3533；更适合证据型回答、引用和连续师生互动复盘。
- 不依赖卡片指针，卡片抽取失败时仍能提供真实原文（但不能伪造结构化诊断字段）。
- 缺点是窗口更长、上下文达到 1500-token 上限，L1 原始排序 Recall/MRR 和 retrieval P95 仍未达门禁。

### 不能从本次结果推出的结论

- 不能说 fallback 在“所有召回指标”上优于 card；文档级 qrels 和窗口级 qrels 的分母不同，必须看 Turn evidence 指标。
- 不能说任一方案已通过 L1 或已证明 NBCOT 业务价值；两次 L1 都是 `BLOCKED`，qrels 仍需人工多相关 graded review。
- 不能把 reranker 的改善归因给 chunk 生成；它只是在候选已进入 pool 后做重排。

## 建议

1. 保留生产双开关，默认 `card`，避免未经完整门禁验证改变现有行为。
2. 证据型问答、需要完整引用或卡片指针缺失时，显式选择 `fallback`；配合 Top-15 + reranker 时优先观察 Turn evidence recall 和 citation quality。
3. 摘要/诊断型问答优先 `card`，控制 Context 长度和首个目标命中延迟。
4. 下一步不是继续调 prompt，而是：优化 fallback 候选召回与 P95、修复 grounding recall、升级多相关人工 graded qrels，再决定是否将 fallback 设为某类意图的默认路由。

## 卡片 Turn 证据回填实验（2026-09-04）

本次目标只针对 `card hit → Turn evidence`，不重新生成卡片语义。使用 `scripts/rebind_card_evidence_staging.py` 对 100 场/200 张历史卡片做确定性 W6/S3 逻辑链索引和角色绑定，`llm_calls=0`；使用 `scripts/eval_card_evidence_rebind.py` 在 30 个 golden cases 上对比回填前后的卡片命中证据。

| 指标 | 回填前 | 回填后 | 变化 |
| --- | ---: | ---: | ---: |
| Card evidence Recall@1 | 0.6556 | 0.7667 | +0.1111 |
| Card evidence Recall@3 | 0.8056 | 0.9333 | +0.1278 |
| Card evidence Recall@5 | 0.8056 | 0.9333 | +0.1278 |
| Card evidence Recall@10 | 0.8389 | 0.9667 | +0.1278 |
| Card evidence Recall@20 | 0.8722 | 1.0000 | +0.1278 |
| Assembler final Context evidence recall | 0.6556 | 0.7667 | +0.1111 |
| Card pointer count mean | 128.2 | 242.1 | +113.9 |
| Assembler Context Turn count mean | 5.43 | 10.40 | +4.97 |

Top-K full coverage cases changed from `21/30` to `28/30` at Top-3 and from `23/30` to `30/30` at Top-20. The final Context still selects one target card in the current assembler, so its recall equals the Top-1 evidence result; the remaining 7/30 misses are selection/context issues rather than missing raw Turns.

The trade-off is explicit: recall-first binding increased evidence volume and lowered conservative pointer precision at Top-1 from `0.2238` to `0.1394`. These are provisional-qrels diagnostics, not business accuracy. The experiment confirms that a full LLM re-extraction is unnecessary for this objective; the next optimization is deterministic minimal-sufficient evidence selection inside the logical chain.

## Evidence-level Assembler implementation

已将实验中选出的 `logical_evidence_5` 接入生产可选路径：

```powershell
python -m src.cli --embedding-backend siliconflow `
  --retrieval-mode bm25_dense --bm25-weight 0.35 `
  --reranker-backend siliconflow --chunk-strategy card `
  --reranker-pool-size 20 `
  --rerank-unit logical_evidence --evidence-selection-count 5
```

该路径将卡片 Top-15（可配置）展开为真实 W6/S3 逻辑链窗口，由 Qwen Reranker 排序；Assembler 最多选择 5 个单元，按 rank 逐个加入并在 1500-token 预算前停止，去重且保留 rank 顺序，Generator 看到的是排序后的 evidence Context。`rerank_unit=card` 仍保留原有卡片重排逻辑。

真实 staging smoke 已通过 `AUDITED_100_VERIFIED`，按预算实际选择 4 个 logical units、20 条真实引用，估算 `1390 tokens`、无截断，总耗时约 `7.7s`。这证明链路已连通，但不等于 L2 回答完整性已通过。
