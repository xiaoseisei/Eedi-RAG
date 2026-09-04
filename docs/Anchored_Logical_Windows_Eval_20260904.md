# Anchored Logical Windows 独立 A/B 评测（2026-09-04）

## 结论（最终状态）

在 Assembler 增加重叠感知去重后，同一份 30-case provisional qrels、同一 Qwen Embedding、同一 BM25+Dense（`bm25_weight=0.35`）、同一 Top-20 卡片池和同一 Qwen3-Reranker 下，Baseline W5 的最终 Context Recall 为 `0.9222`，Anchored Parent-3 W5 为 `0.9000`。Anchored 仍低 `2.22pp`，但窗口 Reranker 输入和延迟降低约 `72%/82%`，且两边平均可装入约 `4.8` 个窗口。

锚定过滤本身没有删掉 required Turn，Baseline 与 Anchored 的候选窗口 union Recall 都是 `1.0000`。优化前的主要损失来自重叠窗口重复渲染；优化后剩余差距主要来自 Anchored 候选池分布和 Window Reranker 排序。按用户确认的延迟/召回权衡，Anchored Parent-3/W5 固化为真实生产 Reranker 默认，旧全量 logical-evidence W5 保留为显式高召回回退。

## 固定口径

```text
Top-20 Cards
→ Card Rerank
→ Logical W6/S3 windows
→ Window Rerank
→ Top-4 / Top-5 request
→ Assembler（1500-token budget）
```

Baseline 使用 Top-20 卡片所覆盖会话的全部 W6/S3 窗口；Anchored 使用 Card Rerank 后的 Parent Top-3/5，并只保留：

```text
window.source_turn_ids ∩ parent_card.source_turn_ids != ∅
```

Parent 元数据只拼入 Anchored Window Reranker 输入；最终 Assembler 只接收窗口 `document` 和真实 `evidence_turns`，不把父卡摘要当引用证据。选择器不读取 gold qrels，qrels 只用于评测。

数据与报告：

- DB：`reports/eval/card-evidence-rebind-staging-20260904-001500/tutoring_knowledge.duckdb`
- Chroma：`reports/eval/card-reextraction-qwen-index-20260903-203000/chroma`
- Golden：`data/golden_test_set.json`
- Split：`evals/datasets/l1-closeout-20260903-015400/split_manifest.json`
- 结果：`reports/eval/anchored-logical-windows-ab-20260904-152000/`
- 源 artifact hash：`b45860c80d02b999c9c9bf6d960c50779b84aad1b76fa74bef02da6ac9a8aeb9`

运行结果为 `MEASURED`：30/30 cases、180 条变体记录、0 case 失败，源 DB/Chroma 前后 hash 一致。

## 结果矩阵

| Variant | Final Context Recall | Dev | Holdout | Final Precision | Ranked Window Recall | Candidate windows | Candidate reduction | Candidate union Recall | Context tokens | Window input tokens | Window Rerank mean / P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline W4 | 0.7889 | 0.7500 | 0.9444 | 0.1699 | 0.9111 | 154.4 | 0.0% | 1.0000 | 1324.7 | 22862.3 | 2182 / 2036 / 2419 ms |
| Baseline W5 | 0.7889 | 0.7500 | 0.9444 | 0.1699 | 0.9333 | 154.4 | 0.0% | 1.0000 | 1324.7 | 22862.3 | 2182 / 2036 / 2419 ms |
| Anchored Parent-3 W4 | 0.7500 | 0.7431 | 0.7778 | 0.1556 | 0.8444 | 21.2 | 86.2% | 1.0000 | 1282.6 | 6333.6 | 366 / 346 / 516 ms |
| Anchored Parent-3 W5 | 0.7500 | 0.7431 | 0.7778 | 0.1556 | 0.9111 | 21.2 | 86.2% | 1.0000 | 1282.6 | 6333.6 | 366 / 346 / 516 ms |
| Anchored Parent-5 W4 | 0.7278 | 0.7292 | 0.7222 | 0.1539 | 0.8333 | 34.8 | 77.3% | 1.0000 | 1275.9 | 10152.2 | 543 / 521 / 672 ms |
| Anchored Parent-5 W5 | 0.7278 | 0.7292 | 0.7222 | 0.1539 | 0.9000 | 34.8 | 77.3% | 1.0000 | 1275.9 | 10152.2 | 543 / 521 / 672 ms |

`Ranked Window Recall` 是窗口 Top-K 在进入 Assembler 前的 union Recall；`Final Context Recall` 是 Assembler 输出 `context.evidence_turns` 的 Recall。两者必须同时看，不能用前者替代最终上下文指标。

## 分析

1. **Parent 级不是主要瓶颈。** 所有变体的 Parent Session Recall 为 `1.0000`；Parent 卡片的 Turn pointer union Recall 只有约 `0.6056`，但全量逻辑窗口 union 可以回到 `1.0000`。这说明完整窗口确实绕开了卡片 pointer 不完整问题。
2. **Anchored 过滤没有丢证据，但改变了候选分布。** Parent-3 平均只剩 `21.2` 个窗口，Parent-5 为 `34.8`，而 Baseline 为 `154.4`。窗口 union Recall 仍为 `1.0000`，所以下降来自排序/选择，不是过滤条件误删 required Turn。
3. **Parent-3 是成本最优但质量略降。** 相对 Baseline，窗口输入 token 降 `72.3%`，窗口 Rerank 均值降 `83.2%`（约 `6.0×` 更快）；最终 Recall 下降 `3.89` 个百分点，Precision 下降 `1.43` 个百分点。
4. **Parent-5 没有带来质量收益。** 它比 Parent-3 多约 `64%` 候选、Rerank 约 `1.48×`，但最终 Recall 反而再低 `2.22` 个百分点。
5. **W5 请求未转化成 5 个最终窗口。** 在 1500-token 预算下，Assembler 平均只装入约 `3.0` 个窗口；因此 Baseline W4/W5 和 Anchored W4/W5 的最终 Context Recall 相同。W5 只提高了进入 Assembler 前的 Ranked Window Recall，未提高最终输出。
6. **存在远端 Reranker 运行波动。** 同一命令的前一轮复测最终 Recall 为 Baseline `0.8000`、Parent-3 `0.7500`、Parent-5 `0.7278`；最新轮为 `0.7889/0.7500/0.7278`。A/B 方向一致，但单次百分点差异不应解释为统计显著性。

## 决策

- 真实 SiliconFlow Reranker 默认使用 `anchored_logical_window`：总 Parent-3、W6/S3、Window Top-5、overlap-aware Assembler。
- 需要最大证据召回时显式使用 `--rerank-unit logical_evidence --evidence-selection-count 5`。
- 不继续增加 Parent Top-K；当前候选 union 已完整，剩余差距主要是锚定候选分布与窗口排序。
- 这些 qrels 是 provisional 人工 quote-derived labels；上线前仍需多相关 graded qrels、L2/DeepEval 的答案完整性和引用正确性评测。

## 复现命令

```powershell
.venv\Scripts\python.exe scripts/eval_anchored_logical_windows.py `
  --db reports/eval/card-evidence-rebind-staging-20260904-001500/tutoring_knowledge.duckdb `
  --chroma reports/eval/card-reextraction-qwen-index-20260903-203000/chroma `
  --golden data/golden_test_set.json `
  --split evals/datasets/l1-closeout-20260903-015400/split_manifest.json `
  --output-dir reports/eval/anchored-logical-windows-ab-<unique-run-id> `
  --top-k-cards 20 `
  --bm25-weight 0.35
```

## Assembler 重叠感知优化复测（最终基准）

前面的结果是优化前基准：W6/S3 窗口的重叠 Turn 会被重复渲染，1500-token 预算下平均只能装入约 3 个窗口。随后仅修改 Assembler，增加 overlap-aware packing：

- 按 Window Reranker rank 遍历；
- 每个 `(session_id, turn_id)` 只渲染第一次出现的真实 Turn；
- 保留窗口 rank、窗口边界和 `selected_evidence_units` provenance；
- 预算判断使用新增 Turn 的增量文本；
- Parent metadata 不进入最终 Anchored Prompt，只用于 Window Reranker。

最终报告：`reports/eval/anchored-logical-windows-assembler-final-20260904-134055/`。状态为 `MEASURED`：30/30 cases、180 条变体记录、0 case 失败，源 artifact hash 仍为 `b45860c80d02b999c9c9bf6d960c50779b84aad1b76fa74bef02da6ac9a8aeb9`。

| Variant | Target-role Recall | Final All-required Recall | Dev / Holdout | Evidence Precision | Context Tokens | 实际装入窗口 | Window Reranker Input Tokens | 平均 Rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline W5（全量逻辑窗口） | 0.9500 | **0.9222** | 0.9028 / 1.0000 | 0.1342 | 1277.1 | 4.83 | 22862.3 | 2018 ms |
| Anchored Parent-3 W5 | 0.9056 | **0.9000** | 0.9028 / 0.8889 | 0.1340 | 1257.3 | 4.87 | **6354.9** | **372 ms** |
| Anchored Parent-5 W5 | 0.9222 | 0.9000 | 0.9028 / 0.8889 | 0.1344 | 1252.7 | 4.87 | 10218.2 | 548 ms |

与优化前同口径结果相比：

- Baseline W5：Final Recall `0.7889 → 0.9222`，提升 `13.33pp`；实际窗口 `2.97 → 4.83`。
- Anchored Parent-3 W5：Final Recall `0.7500 → 0.9000`，提升 `15.00pp`；实际窗口 `3.03 → 4.87`。
- Anchored Parent-3 相对 Baseline W5 仍低 `2.22pp` Recall，但 Window Reranker 输入减少 `72.2%`，平均 Rerank 延迟减少 `81.6%`。
- Precision 基本持平（约 `0.134`），说明本轮收益主要来自去除重复上下文，而非改变相关性排序。
- 两条 W5 路线均无 budget violation 或 truncation loss。

因此最终生产取舍为：真实 SiliconFlow Reranker 默认 Anchored Parent-3/W5；需要最大证据召回时可显式回退 `logical_evidence` W5。两者都必须继续通过 L2/DeepEval 的答案完整性与引用正确性评测。
