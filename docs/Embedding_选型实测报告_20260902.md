# Eedi-RAG Embedding 选型实测报告

> 报告编号：`EEDI-RETRIEVAL-EMBEDDING-SELECT-REPORT-001`
> 测量日期：2026-09-02（Asia/Shanghai）
> 实验分支：`embedding-benchmark`（已合并回 `master`）
> 评测脚本：`scripts/run_embedding_benchmark.py`
> 数据集：`evals/datasets/l1-v2/queries.jsonl` + 同目录 `qrels.jsonl`

## 1. 决策摘要

本轮在同一份 DuckDB 卡片、同一批 30 条 Query、同一份 qrels、同一 CPU 环境下完成了三个可比较状态中的两个真实 Dense 模型：

```text
deterministic hash：已完成
multilingual-e5-small：已完成
multilingual-e5-base：已完成
BGE-M3：未完成（2.27GB 权重，当前 CPU/网络条件下中止下载）
```

当前建议不是立即把某个模型切换到生产，而是：

1. 将 `multilingual-e5-small` 作为性价比候选，继续做量化/批量化和 Hybrid Retrieval 实验；
2. 将 `multilingual-e5-base` 作为质量候选，适合离线教研或有 GPU 的部署；
3. 在 CPU-only、P95 150ms 预算下，两个 E5 模型都暂不直接替换生产；
4. 后续在更合适的环境补测 BGE-M3，再决定长期 Hybrid 方案。

## 2. 实验真实性和边界

### 2.1 输入和隔离

- 只读 DuckDB 构造 100 张 misconception/strategy 文档；
- 不打开或写入生产 Chroma，仅在内存中编码和排序；
- Query 与 qrels 固定，不把 qrels 注入被测模型；
- 每次运行前后检查组合 artifact hash；
- 当前生产 artifact hash：`711f1ed14e8f7014a2a57421665c08037fd8189c80bf922217209b7789718fa4`。

### 2.2 qrels 限制

当前 qrels 已完成 quote evidence 去重和来源标注，但仍是：

```text
30 个 Query
30 个唯一 query-document pair
每个 Query 当前只有一个主要正例
document relevance 来源：derived_from_human_quote
multi-relevance human annotation：false
```

因此本报告适合比较模型相对效果，不应把 Recall/Precision 直接解释成最终业务准确率。正式发布门禁仍需多相关、0/1/2/3 graded qrels。

## 3. 实测结果

### 3.1 总体指标

| 候选 | 维度 | Candidate Recall@20 | Recall@5 | MRR | nDCG@5 | 数字命中@5 | 公式命中@5 | 查询 P95 | Corpus 编码 | 模型加载 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic hash | 128 | 0.7000 | 0.4333 | 0.3368 | 0.3337 | 0.5455 | 0.3333 | 1.94 ms | 0.15 s | 0.00 s |
| multilingual-e5-small | 384 | 0.9667 | 0.8667 | 0.7976 | 0.8064 | 0.7727 | 0.5000 | 153.96 ms | 60.98 s | 43.54 s |
| multilingual-e5-base | 768 | 0.9667 | 0.9333 | 0.8660 | 0.8798 | 0.8182 | 0.5000 | 185.04 ms | 176.46 s | 261.56 s |
| BGE-M3 | — | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — | — |

硬件：Windows 11、16 CPU、CPU-only、无 CUDA。E5 模型使用 `query:`/`passage:` 前缀；模型版本已记录为本地包实际加载的 model ID，但尚未锁定 Hub revision hash。

### 3.2 相对 deterministic 的提升

| 候选 | Candidate Recall@20 | Recall@5 | MRR | nDCG@5 | 查询 P95 |
|---|---:|---:|---:|---:|---:|
| e5-small | +26.67 个百分点 | +43.33 个百分点 | +0.4608 | +0.4727 | +152.02 ms |
| e5-base | +26.67 个百分点 | +50.00 个百分点 | +0.5293 | +0.5461 | +183.10 ms |

### 3.3 按卡片类型

| 候选 | Misconception Recall@5 | Strategy Recall@5 |
|---|---:|---:|
| deterministic hash | 0.4444 | 0.4167 |
| e5-small | 0.8889 | 0.8333 |
| e5-base | 0.9444 | 0.9167 |

e5-base 对策略卡也有提升，但代价是更高的 CPU 延迟和约 3 倍 Corpus 编码时间。

## 4. 结果解释

### 4.1 质量结论

两个 E5 模型都显著优于当前 hash baseline：

- Candidate Recall@20 从 70% 提升到 96.67%；
- e5-small Recall@5 达到 86.67%；
- e5-base Recall@5 达到 93.33%；
- MRR 和 nDCG 均显著提升；
- 策略卡 Recall@5 从 41.67% 提升到 83.33%/91.67%。

这证明当前第一瓶颈确实是 Dense 表示能力，而不是 DuckDB、Chroma ID 或 Chunk 原文完整性。

### 4.2 延迟结论

当前 CPU 延迟预算按 P95 150ms 观察：

- deterministic 远低于预算；
- e5-small 为 153.96ms，略超预算；
- e5-base 为 185.04ms，明显超预算。

这两个延迟是“每条 Query 重新编码”的 CPU 结果，不代表 GPU 或量化部署结果。它们可以用于当前本机性价比比较，不能直接外推线上 QPS。

### 4.3 数字和公式结论

e5-small/e5-base 的数字精确命中均高于 deterministic，但公式命中都只有 0.5。Dense 模型不能替代精确词法通道，后续必须增加：

```text
BM25 / Sparse
数字与公式 exact channel
subject_path / card_type metadata filter
```

不能因为 Dense Recall 上升就删除精确匹配路径。

## 5. 选择建议

### 推荐排序

| 方案 | 当前结论 | 适用场景 | 阻断项 |
|---|---|---|---|
| e5-small + BM25/RRF | 首选性价比候选 | CPU/成本敏感、在线候选召回 | 当前未实现 BM25；CPU P95 略超 150ms |
| e5-base + BM25/RRF | 首选质量候选 | GPU/离线教研、质量优先 | 当前 CPU P95 超预算；资源成本更高 |
| deterministic hash | 保留为回归基线 | 单元测试、无模型诊断 | 质量不足，不作最终语义模型 |
| BGE-M3 | 待补测 | 长期多语言 Hybrid 候选 | 本轮未取得真实分数 |

### 为什么不直接选择 e5-base

e5-base 的质量最好，但：

```text
查询 P95 比 e5-small 高约 20%（185ms vs 154ms）
Corpus 编码时间约高 2.9 倍（176s vs 61s）
维度是 e5-small 的 2 倍（768 vs 384）
```

在当前数据规模下，它的质量收益是真实的，但是否值得线上成本，需要等 BM25 + Dense 和量化实验后再定。

### 为什么不直接选择 e5-small

e5-small 是最合理的下一步工程候选，但也不应未经优化直接切生产：

```text
P95 153.96ms，略高于当前 150ms 观察预算
公式 exact-hit 仍不足
当前 qrels 尚未完成多相关人工标注
```

因此建议先用它做 BM25 + Dense RRF 的基础 Dense 模型，并在同一 qrels 上验证是否能用 Sparse 通道补足精确内容、同时降低 Dense 调用次数。

## 6. 硬性决策规则

后续候选必须满足：

1. Artifact 构建前后 hash 不变；
2. 模型可重复加载，model ID/revision 可追溯；
3. Candidate Recall@20 不低于 deterministic baseline；
4. 数字/公式/选项命中不发生预设幅度退化；
5. Strategy slice 不得严重退化；
6. 在目标部署硬件上满足 P95 预算，或有明确量化/GPU 方案；
7. 通过 dev 选型后在 holdout 上复测。

综合分只能在硬性条件通过后计算，暂不把未完成的 BGE-M3 纳入分数排序。

## 7. 下一步实施

```text
1. 固定 e5-small/e5-base 的 model revision hash
2. 用 e5-small 建立独立 Dense index（不覆盖 data/chroma）
3. 增加 BM25/数字公式精确通道
4. 比较 Dense-only、BM25-only、BM25+Dense RRF
5. 对最优候选测 Cross-Encoder rerank
6. 在完整 graded qrels 和 holdout 上复测
7. 再补测 BGE-M3（建议 GPU/高速网络环境）
8. 通过完整 L1 后才做生产蓝绿切换
```

## 8. 原始证据

本报告数字来自以下不可覆盖运行文件：

```text
reports/eval/embedding-benchmark/embedding-benchmark-20260902T121200Z.json
reports/eval/embedding-benchmark/embedding-benchmark-20260902T120928Z.json
reports/eval/embedding-benchmark/embedding-benchmark-20260902T121942Z.json
```

其中：

- `121200Z`：deterministic，含修正后的数字/公式指标；
- `120928Z`：e5-small，首次完整成功运行；
- `121942Z`：e5-base，首次完整成功运行。

BGE-M3 的中止下载不产生成功结果，不能作为模型质量结论。
