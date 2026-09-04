# Card / Logical Evidence / Context-Injected Child 三路评测

日期：2026-09-04  
评测报告：`reports/eval/card-evidence-child-rerank-20260904-110000/`  
数据：30 cases、同一 dev/holdout split、同一 provisional required-Turn qrels  
索引：Qwen Embedding 卡片索引 + 确定性 evidence rebind staging  
检索：BM25 + Dense，`bm25_weight=0.35`，Top-20 card pool  
重排：同一 Qwen3-Reranker-0.6B

## 固定边界

- A/B/C 只改变 Reranker 的输入单位和选择数量，不改 Query、Embedding、BM25、qrels 或生成器。
- Parent metadata 只作为排序辅助，不作为引用证据；引用只来自真实 Turn。
- 所有测试在隔离副本执行，不修改生产 DB/Chroma。
- Token 为估算值，qrels 是人工 quote 派生的 provisional labels，不能宣称业务准确率。

## 方案定义

```text
A Card-1:
  Top-20 cards → card rerank → select 1 card → expand card evidence

B Logical-evidence-5:
  Top-20 cards → expand all related W6/S3 logical windows → rerank windows → select up to 5

C Context-injected Child-3/5/8:
  raw Top-5 parent cards → expand each parent session to Turn±1 child windows
  → rerank [topic + question + parent card summary + child dialogue]
  → select 3/5/8 children
```

## 结果

| Variant | Target-role Recall | All-required Recall | All-required Precision | Evidence Turns | Context Tokens | Budget violations | Reranker Input Tokens | Mean Rerank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A Card-1 | 0.9667 | 0.5833 | 0.1739 | 9.90 | 476.0 | 0/30 | 10,338 | 683 ms |
| B Logical-evidence-5 | **0.9833** | **0.9556** | 0.1353 | 20.27 | 1,298.9 | 5/30 | 38,974 | 2,388 ms |
| C Child-3 | 0.4111 | 0.4611 | **0.1907** | 7.00 | 636.8 | 0/30 | 27,212 | 1,726 ms |
| C Child-5 | 0.6278 | 0.6278 | 0.1810 | 10.27 | 1,056.3 | 3/30 | 27,212 | 1,726 ms |
| C Child-8 | 0.7722 | 0.7667 | 0.1597 | 14.23 | 1,680.4 | 16/30 | 27,212 | 1,726 ms |

Dev/holdout all-required Recall：

| Variant | Dev | Holdout |
|---|---:|---:|
| A Card-1 | 0.5764 | 0.6111 |
| B Logical-evidence-5 | 0.9444 | 1.0000 |
| C Child-3 | 0.5069 | 0.2778 |
| C Child-5 | 0.6319 | 0.6111 |
| C Child-8 | 0.7778 | 0.7222 |

## 归因

额外 Parent Top-5 检查显示，30/30 cases 的目标 Session 都在原始卡片 Top-5 内，且目标卡在 Top-20 内。因此 C 的损失不是 Parent 初检漏召回，而是 Child Reranker 排序阶段：

1. Parent Top-5 展开后平均产生 `123.2` 个 child candidates（范围 `84～182`），远高于 B 的逻辑窗口候选。
2. 只有 `3/5/8` 个 child 被选入 Context；在大候选池中，短 required Turn 仍容易被同 Session 的普通邻近 Turn 淹没。
3. Context 注入提高了 Child 的可理解性，但没有自动提高“必要 Turn 优先级”；Parent 摘要还会使同一 Parent Session 的多个 child 获得相似分数。
4. C-8 的平均 Context 已达 `1680 tokens`，53% cases 超过 1500-token 预算；C-3/C-5 虽便宜，但 Recall 明显不足。

## 决策

当前不采用 C 作为生产默认，也不将其结果解释为 Context Injection 机制本身无效。当前数据选择：

- 证据型问答：继续使用 B 的 logical-evidence 路线，Evidence-5 作为上限，并由 Assembler 按预算实际选择 4～5 个窗口。
- 低延迟摘要：继续使用 A Card-1。
- C 保留为后续实验候选，但下一版必须减少 Child 候选池并增加 Turn-aware coverage 约束，例如只从 Parent evidence anchors ±1 展开、按 Session/role 做候选配额，或采用 coverage-aware rerank selection。

## 重要限制

- 本次 C 只测了 `Parent Top-5 + 全部中心 Turn±1`，不是所有可能的 Child 生成策略。
- 当前 Assembler 的最终回答完整性仍需 L2 Generator/DeepEval 验证。
- 报告中的 precision 是 provisional qrels 下的诊断指标，不是人工确认的业务准确率。
