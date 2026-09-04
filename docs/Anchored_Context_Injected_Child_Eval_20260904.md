# Anchored Context-Injected Child 评测结果

日期：2026-09-04  
报告：`reports/eval/anchored-context-injected-child-20260904-130000/`

## 固定实验口径

```text
Top-20 Cards
→ Card Rerank
→ Parent Top-3 / Top-5 Cards
→ 只围绕 source_turn_ids 展开 Turn±1 Child
→ 每个 Parent Session quota = 2 / 3 / 5
→ Child Rerank
→ coverage-aware Child-3 / Child-5 / Child-8
→ Assembler
```

固定 Qwen Embedding、BM25+Dense（weight=0.35）、30 cases、同一 split/qrels、同一 Qwen3-Reranker、1500-token budget。选择器没有读取 gold qrels；gold 仅用于最后计算 Recall/Precision。

## 结果矩阵

| Variant | All-required Recall | All-required Precision | Context Recall | Context tokens | Child candidates | Card/Child Rerank ms |
|---|---:|---:|---:|---:|---:|---:|
| Parent3 / quota2 / Child3 | 0.3667 | 0.1181 | 0.3667 | 773.7 | 32.6 | 514.5 / 493.8 |
| Parent3 / quota3 / Child3 | 0.4778 | 0.1751 | 0.4778 | 782.1 | 32.6 | 514.5 / 493.8 |
| Parent3 / quota5 / Child5 | **0.7056** | **0.1659** | **0.7056** | **1151.9** | 32.6 | 514.5 / 493.8 |
| Parent3 / quota5 / Child8 | 0.7056 | 0.0995 | 0.7056 | 1407.6 | 32.6 | 514.5 / 493.8 |
| Parent5 / quota5 / Child5 | 0.6944 | 0.1629 | 0.6944 | 1145.8 | 53.8 | 514.5 / 740.2 |
| Parent5 / quota5 / Child8 | 0.6944 | 0.0976 | 0.6944 | 1412.0 | 53.8 | 514.5 / 740.2 |
| B Logical-evidence-5（对照） | **0.9556** | 0.1353 | **0.9556** | 1298.9 | 154.4 | — / 2388.1 |

最佳 C 是 `Parent3 / quota5 / Child5`：Recall=`0.7056`、Precision=`0.1659`、Context=`1151.9 tokens`。它仍比 B 低 `25.0` 个百分点 Recall；Child-8 没有增加 Recall，只增加上下文和噪声。

## Dev / Holdout

最佳配置 `Parent3 / quota5 / Child5`：

- Dev all-required Recall：`0.7153`
- Holdout all-required Recall：`0.6667`
- Context 平均：`1151.9 tokens`
- Budget violation：`0/30`

这不是稳定达到 0.95 的方案，且 holdout 低于 dev，不能上线为证据型默认。

## 归因

额外检查显示目标 Session 在 Parent Top-5 中为 `30/30`，因此 C 不是父级召回问题。C 的损失来自：

1. 每个 Parent Session 的 source pointers 往往覆盖多个 Turn；围绕所有 pointer 展开后，平均仍有 `32.6`（Parent3）或 `53.8`（Parent5）个 Child。
2. `coverage-aware` 当前只知道“是否覆盖 Parent anchor”，不知道哪个 anchor 是 required Turn；它会偏好高分或新的 anchor，但可能漏掉关键 required Turn。
3. Child±1 窗口会带入邻接 Turn；不同中心窗口高度重叠，导致同一 Session 的普通上下文挤占选择名额。
4. Parent 摘要注入改善了代词理解，但不会自动提供 Turn 级相关性标注；语义可理解性提升没有转化为 required Recall 提升。

## 决策

当前不将锚定 Child 路线接入生产默认。继续采用：

- 证据型问答：Logical-evidence-5，上限 5 个窗口，Assembler 按预算实际选 4～5 个；
- 低延迟摘要：Card-1；
- Anchored Child：保留实验候选。

下一版 C 若继续，应加入不依赖 gold 的结构化 coverage 目标，例如：每个 Parent Session 至少覆盖一个学生 anchor、一个导师 anchor、一个修正/确认阶段；并对 Child 按 session/role/阶段做配额。仅增加 Child 数量不能解决排序目标不明确的问题。
