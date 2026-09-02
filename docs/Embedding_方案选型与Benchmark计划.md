# Eedi-RAG Embedding 方案选型与 Benchmark 实施文档

> 文档编号：`EEDI-RETRIEVAL-EMBEDDING-SELECT-001`
> 版本：`v1.0`
> 日期：2026-09-02（Asia/Shanghai）
> 当前状态：方案冻结，候选模型实测待执行

## 1. 一句话结论

当前可以开始 Embedding 选型，但不能凭模型名称或公开榜单直接宣布最终模型。应将每个候选模型放进同一套 L1 评测，在同一份数据、同一份 qrels、同一套 Query、同一硬件和同一 Top-K 下比较，再按质量、速度、成本和维护性决策。

本项目的暂定候选优先级为：

```text
第一轮：deterministic hash（现有基线）→ multilingual-e5-base → BGE-M3
第二轮：multilingual-e5-small、multilingual-e5-large、MiniLM（仅在第一轮结果不足以决策时加入）
```

“第一候选”不等于“最终选定”：最终选择必须由本项目真实 benchmark 结果决定。

## 2. 当前基线和可比条件

### 2.1 当前 artifact

当前生产输入是已通过 Storage L1 的数据资产：

```text
DuckDB tutoring_knowledge.duckdb：100 个 indexed Session
misconception cards：100
tutor strategy cards：100
fallback windows：719
session dialogue turns：2335
```

当前生产 Chroma 的组合 hash：

```text
711f1ed14e8f7014a2a57421665c08037fd8189c80bf922217209b7789718fa4
```

只读 artifact 审计结果：

```text
ID parity：100%
metadata contract：100%
orphan rate：0
immutable audit 前后 hash：一致
```

因此 Embedding 实验的首要问题不是重做清洗或 Chunk，而是提升 Query 到正确卡片/窗口的候选召回。

### 2.2 语料特征

当前索引文本主要是英文教学对话和英文结构化知识卡，黄金 Query 主要是中文，部分包含英文术语、数字、公式和选项。例如：

```text
5.4598
0.02
-3(1-2p)
A / B / C / D
```

因此模型必须同时处理：

- 中文 Query → 英文知识卡/对话窗口的跨语言匹配；
- 学生口语 → 数学专业术语的语义映射；
- 数字、公式、选项和题目原文的精确保留；
- misconception 与 tutor strategy 两种不同文本分布；
- 短 Query 与较长知识卡之间的匹配。

### 2.3 必须保持一致的实验条件

每个方案都必须满足：

```text
同一 DuckDB 原卡片事实
同一 Chunk/Document 文本
同一 30 条黄金 Query
同一 qrels 版本
同一 Top-K 和 RRF 参数
同一 CPU/GPU 环境
同一 tokenizer/计时口径
不启用 Reranker
不修改 Assembler
不修改 Query 文本和 qrels
```

每个候选必须使用独立 Chroma 目录。禁止在 `data/chroma` 上直接重建或覆盖：

```text
reports/eval/embedding-benchmark/<candidate>/chroma
```

实验构建前后都要计算 DuckDB、Chroma 和组合 hash；hash 变化必须标记 `artifact_mutated` 并阻断正式对比。

## 3. 候选方案及选择依据

### 3.1 Deterministic hash：必须保留的零成本基线

实现：`FastDeterministicEmbeddingFunction`，128 维 token + 字符 2-gram 哈希。

用途：

- 测量当前系统真实起点；
- 验证 benchmark 没有把模型变化误当成评测变化；
- 在离线或依赖不可用时提供可复现诊断。

限制：

- 不具备真正语义理解；
- 中文 Query 到英文文档能力弱；
- 数字相近但语义不同的区分能力有限；
- 不能作为最终生产语义方案。

### 3.2 multilingual-e5-base：第一轮平衡候选

选择依据：

- 面向多语言语义检索；
- 规模和推理成本处于中间档；
- 适合先验证“跨语言 Dense 是否能显著超过哈希基线”；
- 生态和集成路径相对清晰。

必须验证的使用规则：

```text
Query 使用 query 前缀
Document 使用 passage 前缀
```

如果不按模型要求编码，不能把得到的分数与正确配置的模型比较。

风险：

- 可能对数字/公式的精确区分不足；
- CPU 延迟可能高于当前基线；
- 需要实际确认 Python 3.14、CPU 推理和模型下载是否稳定。

### 3.3 BGE-M3：长期 Hybrid 方向候选

选择依据：

- 多语言能力覆盖当前中文 Query + 英文文档场景；
- 对较长知识卡和混合语言文本具有较强适配潜力；
- 未来可以扩展 Dense、Sparse/lexical 和多向量能力；
- 与后续 BM25 + Dense Hybrid 的路线一致。

风险：

- 模型和索引资源开销高于 small/base；
- CPU 构建和查询耗时可能成为工程约束；
- 仅使用 Dense 部分时，不能声称已经启用了 BGE 的 Sparse 能力；
- license、模型文件和部署方式必须在落地前复核。

### 3.4 multilingual-e5-small：低资源候选

选择依据：

- 适合 CPU 或低延迟部署对照；
- 可用于判断小模型是否已经足够满足当前 100-card 规模。

风险：

- 复杂教学意图和专业术语区分可能弱于 base；
- 如果 Recall 明显下降，不能因速度快而选用。

### 3.5 multilingual-e5-large：离线质量上限候选

选择依据：

- 用于估计 Dense 模型质量上限；
- 不作为第一默认生产候选。

风险：

- 模型加载、内存和 CPU 延迟较高；
- 当前 100 张卡规模不足以证明其额外成本值得；
- 如果只提升少数 Query，不得据此替换平衡方案。

### 3.6 multilingual-MiniLM：轻量 fallback

选择依据：

- 小模型、依赖简单、便于快速跑通；
- 作为低资源/高并发的工程对照。

风险：

- 数字、公式和数学专业语义可能弱于 E5/BGE；
- 只能在质量不退化的情况下作为性价比候选。

## 4. 评价维度

### 4.1 召回质量：第一优先级

必须输出 aggregate 和 slice：

```text
Candidate Recall@20
Candidate Recall@50
Recall@1/@3/@5/@20
MRR
nDCG@5
```

至少拆分：

```text
student_misconceptions
tutor_strategies
fallback_windows
STUDENT_INSIGHT
TUTOR_INTERVENTION
```

候选召回优先于 Reranker 指标：正确文档没有进 Candidate Top-N 时，Reranker 不可能补救。

### 4.2 精确内容能力

对 Query 中出现数字、公式或选项的 Case，额外计算：

```text
numeric exact-hit rate
formula exact-hit rate
option exact-hit rate
subject_path exact-hit rate
```

这些指标用于防止“平均语义 Recall 上升，但关键数学符号被召回错”的情况。

### 4.3 性能

每个候选必须记录：

```text
model load time
embedding batch throughput
index build time
index disk size
average query latency
P95/P99 query latency
peak RSS/memory
embedding dimension
```

当前机器是 CPU-only；本机没有 CUDA。模型必须同时报告 CPU 结果，不能用 GPU 结果冒充本机部署成本。

### 4.4 成本和维护

记录：

```text
模型下载大小
运行时依赖
是否需要特殊 Query/Passage 前缀
Python/torch/transformers 兼容性
离线部署可行性
license 与模型来源
升级/回滚复杂度
```

当前项目 `.venv` 原本没有 `sentence-transformers`；依赖安装必须固定版本并记录到实验 manifest。安装失败的候选必须为 `UNMEASURED`，不能填估计分数。

## 5. 硬性淘汰条件和综合评分

### 5.1 硬性淘汰

满足任一条件即淘汰该候选：

```text
构建期间 artifact 未授权变化
模型无法稳定加载或重复运行
Candidate Recall@20 低于 deterministic baseline
关键数字/公式 exact-hit 明显下降
strategy Recall@5 严重退化
P95 超过已批准延迟预算
无法满足项目 license/离线部署要求
```

“严重退化”必须在实验前写入配置，例如相对基线下降超过 5 个百分点；不得看完结果后临时改规则。

### 5.2 综合评分

硬性淘汰后才计算综合分。建议初始权重：

| 维度 | 权重 |
|---|---:|
| Candidate Recall@20/@50 | 25% |
| Recall@5、MRR、nDCG@5 | 25% |
| 数字/公式/选项精确命中 | 10% |
| P95 latency | 15% |
| 内存、索引大小、构建时间 | 10% |
| 部署/维护/依赖成本 | 15% |

质量指标取归一化后的 dev 分数，性能和成本指标取反向归一化分数。所有归一化上下界必须记录在报告中。

推荐公式：

```text
score = 0.25 * recall_score
      + 0.25 * ranking_score
      + 0.10 * exact_content_score
      + 0.15 * latency_score
      + 0.10 * resource_score
      + 0.15 * maintenance_score
```

综合分只用于排序，不能抵消硬性失败。例如 Recall 低于基线的模型，即使成本很低，也不能因为综合分高而入选。

## 6. 实验分阶段执行

### Batch 0：环境和基线

1. 锁定当前生产 DB/Chroma hash。
2. 复制到只读/隔离实验目录。
3. 跑 deterministic baseline 两次，确认指标和 hash 稳定。
4. 记录 CPU、内存、Python、Torch、Chroma 版本。

### Batch 1：Dense 模型横向比较

固定：Raw Query、Dense-only、同一 30 Query、同一 qrels、无 Reranker。

运行：

```text
deterministic
multilingual-e5-base
BGE-M3 Dense
```

如果 base 与 BGE-M3 已足以决策，再不必增加 large；如果质量接近，则优先选择成本更低者。

### Batch 2：轻量/上限对照

仅在 Batch 1 无法决策时加入：

```text
multilingual-e5-small
multilingual-e5-large
multilingual-MiniLM
```

### Batch 3：同一 Dense 模型下的检索结构

Embedding 选出候选后，再固定模型比较：

```text
Dense-only
BM25-only
BM25 + Dense RRF
Multi-query Dense RRF
BM25 + Multi-query Dense RRF
```

本批次不与模型选择同时调参，否则无法归因收益来自模型还是检索结构。

### Batch 4：Reranker 与 Assembler

在候选召回稳定后比较：

```text
无 Reranker
Cross-Encoder Reranker
不同 Assembler 参数
```

Assembler 评测必须继续区分 `retriever_miss` 与 `assembler_drop`。

## 7. 数据和报告产物

每个候选至少生成：

```text
reports/eval/embedding-benchmark/<candidate>/manifest.json
reports/eval/embedding-benchmark/<candidate>/run.json
reports/eval/embedding-benchmark/<candidate>/metrics.json
reports/eval/embedding-benchmark/<candidate>/summary.md
```

统一比较报告：

```text
reports/eval/embedding-benchmark/model-comparison.md
reports/eval/embedding-benchmark/model-comparison.json
```

候选 manifest 至少包含：

```text
candidate_id
model_id + revision/hash
embedding_dimension
query_prefix/document_prefix
runtime versions
hardware
source artifact hashes
index artifact hash
dataset/qrels/system config hashes
build/load/query timings
memory/disk measurements
status: SUCCESS/FAILED/UNMEASURED
failure reason
```

## 8. 当前推荐和决策规则

在没有实测结果前，只能给出“实验优先级”，不能给出最终生产模型：

```text
首轮平衡候选：multilingual-e5-base
长期 Hybrid 候选：BGE-M3
低资源候选：multilingual-e5-small / MiniLM
质量上限对照：multilingual-e5-large
```

选择规则：

1. 先淘汰 artifact、兼容性、Recall 和关键符号命中失败的模型。
2. 在剩余模型中优先选择 Candidate Recall@20/@50 更高者。
3. 若质量差距小于预设容忍度，选择 P95、内存、索引大小和维护成本更优者。
4. 若策略卡 slice 明显更好，即使 aggregate 接近，也要单独记录业务价值。
5. 通过 dev 选择后，用 holdout 重跑；holdout 退化则不能合并生产。

## 9. 当前阻断与下一步

当前可以确认：

```text
artifact 健康
Dense hash 是主要质量瓶颈候选
当前没有 BM25/Sparse
当前机器 CPU-only
```

当前不能声称：

```text
某个候选模型已经优于 deterministic
某个模型一定达到 Recall@5 95%
BGE-M3 已经启用 Sparse
Embedding 选择已经完成
```

下一步施工顺序：

```text
1. 固定当前 artifact hash 和运行环境
2. 在 benchmark 分支完成统一候选 Adapter/Index Builder
3. 跑 deterministic 重复基线
4. 安装并锁定 sentence-transformers 兼容依赖
5. 跑 multilingual-e5-base
6. 跑 BGE-M3 Dense
7. 自动生成模型比较报告
8. 选择模型后再施工 BM25 + Dense
```

本轮依赖安装若在 Python 3.14/CPU 下持续不稳定，应建立独立 Python 3.11/3.12 benchmark 环境；不能为了方便修改生产运行时或将未运行结果写入选择表。
