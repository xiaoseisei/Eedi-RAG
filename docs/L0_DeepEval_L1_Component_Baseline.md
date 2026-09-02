# Eedi-RAG L0 DeepEval 评测基座与 L1 组件层基线

> 文档编号：`EEDI-EVAL-L0-L1-BASELINE-001`  
> 基线版本：`l1-v0.1-diagnostic`  
> 测量日期：2026-09-01（Asia/Shanghai）  
> 代码提交：`b7077518e53aa23c739d6835434b3c42f7263108`  
> 状态：**冻结候选，当前不可作为发布基线**

## 1. 结论先行

本项目把 DeepEval 确定为唯一的语义评测框架，并把一个显式配置的 LLM 作为 DeepEval 的 evaluation model。原有自研 `RagasEvaluatorEngine`/Judge 只保留为历史实现，不再用于新基线、回归门禁或发布决策。

本文件中的 “L0” 指评测控制平面（工具、模型配置、数据集、Trace、指标适配器、门禁和报告契约），不是 RAG 业务链路中的一个功能组件；“L1” 指可以隔离运行并独立归因的 RAG 组件层。

当前真实状态如下：

| 层级/对象 | 当前状态 | 解释 |
|---|---|---|
| L0 DeepEval 工具 | `BLOCKED` | 当前 Python 环境未安装 `deepeval`。 |
| L0 evaluation model | `BLOCKED` | 当前环境未配置评测模型名称和凭据。 |
| L0 统一报告 | `NOT_READY` | 现有脚本各自输出报告，还没有统一的 `Trace/Observation` 结果仓库。 |
| L1 确定性组件指标 | `DIAGNOSTIC_AVAILABLE` | Chunk、检索、存储一致性、引用结构和性能已有可复现测量。 |
| L1 DeepEval 语义指标 | `UNMEASURED` | 在 DeepEval 与 evaluation model 就绪前，不填充任何语义分数。 |
| L1 发布可用性 | `BLOCKED` | 持久化索引元数据契约漂移导致确定性端到端 30/30 失败。 |

因此，本文件不是宣称系统已经达到 95% 的成绩单，而是冻结当前系统可重复测量的事实、暴露阻断项，并规定下一次正式 DeepEval 基线必须遵守的口径。

## 2. 评测边界和真实性规则

### 2.1 L0/L1/L2/L3 的边界

```text
L0 评测控制平面
 ├─ DeepEval runner + evaluation model 配置
 ├─ 数据集注册、版本、切分和 qrels
 ├─ Trace / MetricObservation / Report schema
 ├─ 确定性指标适配器
 └─ 阈值、回归比较和发布门禁

L1 组件层
 ├─ 数据抽取与证据绑定
 ├─ Chunking / Indexing
 ├─ Query Rewrite / Router
 ├─ Embedding + Retriever
 ├─ RRF
 ├─ Reranker / MMR / Context Assembler
 ├─ Generator
 └─ Citation / Output Contract

L2 端到端链式层（本文件只规定接口，不给当前成绩）
 └─ 真实检索、Gold Context、Oracle Rerank 等反事实链路

L3 业务层（本文件不把代理分数冒充业务真值）
 └─ 教研员采纳、编辑、任务完成、学习增益和线上反馈
```

### 2.2 真实性硬规则

1. **DeepEval 是语义评测框架，不是被测 RAG。** Evaluation model 必须显式配置、记录名称/提供方/版本；不得依赖隐式默认模型。
2. **没有评测模型就记 `UNMEASURED`，不切关键词 heuristic。** DeepEval 调用失败时只能是 `FAILED`，不能改写成成功或降级分数。
3. **确定性事实和语义判断分开。** ID、Session、Turn、Speaker、引用原文、Schema、延迟、Token、成本由代码或数据库核验；Faithfulness、Relevancy、教学质量由 DeepEval 指标核验。
4. **每个指标独立记录。** 不用一个 LLM 同时返回多个自定义分数，也不把一个综合分作为唯一发布依据。
5. **Ground Truth 不进入被测 Pipeline。** Gold 只在评测侧用于构造 qrels、`expected_output`、required claims 和业务 rubric。
6. **所有降级都必须显式声明。** `DEGRADED`、`FAILED`、`UNMEASURED` 与 `SUCCESS` 不可互换。
7. **引用是信任边界。** 任何引用四元组不匹配真实原文，都应阻断发布；不能用平均分掩盖单条错误。

## 3. 当前冻结的系统配置

正式比较时，以下配置必须作为 run provenance 的一部分。修改任一项都应产生新的 `system_config_hash`，不能与本基线直接横比。

| 配置项 | 当前值 | 代码/证据位置 |
|---|---|---|
| Embedding | `FastDeterministicEmbeddingFunction`, 128 维，`fast_deterministic` | `src/storage_manager.py:55` |
| Query Rewrite | `deterministic_rules` | `src/query_rewriter.py:191` |
| Hybrid alpha | `0.5`（Cosine/L2） | `src/retriever.py:79` |
| 单路候选 | `top_k=3`；RRF 各透视最多 15 个候选 | `src/retriever.py:140,297` |
| RRF | `k=60`，最终每集合 `top_k_each=5` | `src/retriever.py:274` |
| MMR | `lambda=0.7` | `src/reranker.py:72` |
| Context budget | 估算上限 1500 token | `src/reranker.py:75` |
| 离线生成 | `mode=deterministic`，只用于契约和链路诊断 | `src/rag_pipeline.py:447` |
| 语义评测 | DeepEval（目标标准；本快照尚未安装） | L0 待建设 |
| Evaluation model | 未配置 | `UNMEASURED` |

### 3.1 数据和索引 artifact

当前持久化数据库和向量索引不是 10 条 sample fixture，而是 100 个会话对应的扩展快照；10 条和 50 条测试集分别用于组件诊断。由于这些文件目前有工作树外的生成改动，正式 CI 必须把它们纳入版本化 artifact registry。

| Artifact | 规模/用途 | SHA-256（当前快照） |
|---|---:|---|
| `data/db/tutoring_knowledge.duckdb` | 100 sessions、2335 turns、100 misconception、100 strategy、719 windows | `ee3dd817f123819e41bbd7256feb0821509d93f3d2257eb4c595adaa1ee0f050` |
| `data/chroma/chroma.sqlite3` | Chroma 持久化索引 | `80c8339e53dbda1fe3ac83888ac1a5b9f6819c5a17cdb0675f38e52ae7b56959` |
| `data/golden_test_set.json` | 30 条端到端黄金用例（18 `STUDENT_INSIGHT`、12 `TUTOR_INTERVENTION`） | `035e4fbb35ad00b75661a792688f8b16821b3d4014c13a9a1839d1ce4555f6b9` |
| `scripts/eval_50_queries.py` | 50 条检索 Query（25 misconception、25 strategy） | 脚本内常量，待注册为独立 JSON artifact |
| `tests/chunk/ground_truth_queries.json` | 10 条 Turn 级 Chunking qrels | `24393befe46da9249d87401525362f075c1e04185ffdf7740095b9037b837777` |
| `data/authoring_catalog.json` | 100 条作者/业务目录卡片 | `c0b8451ab8812897556e09652f2a5b47e2dc8f171272965c2b6092a30d5895e9` |

当前索引内部数量和外键健康度实测为：

```text
DuckDB:
  tutoring_sessions       100
  session_dialogue_turns  2335
  misconception_chunks    100
  tutor_strategy_chunks   100
  sliding_window_chunks   719

Chroma:
  student_misconceptions  100
  tutor_strategies        100
  fallback_windows        719

外键孤儿记录：0
DuckDB/Chroma 数量同步：100%
```

数量同步只证明写入数量一致，不证明 metadata 字段满足下游 Generator 契约；后者目前不成立（见第 6 节）。

## 4. L0 DeepEval 基线契约

### 4.1 Evaluation model 配置

正式运行前必须显式设置并写入报告的配置至少包括：

```text
evaluation_provider
evaluation_model
evaluation_model_version（若提供方支持）
deepeval_version
metric_versions
temperature / seed（若适用）
```

凭据只能来自环境变量或 secret manager，不写入数据集、日志和报告。生成模型和评测模型应尽量分离；如果必须使用同一提供方，也要记录这一事实并重新建立基线。

当前环境检查结果：

```text
DEEPEVAL_INSTALLED      = False
EVAL_MODEL_CONFIGURED   = False
LLM_API_KEY_CONFIGURED  = False
```

因此本文不报告任何 DeepEval `score`。

### 4.2 DeepEval TestCase 映射

Generator 的 DeepEval 用例应使用**生成模型实际看到的最终 Context**，而不是调试候选集合：

```python
LLMTestCase(
    input=case.question,
    actual_output=trace.final_answer,
    expected_output=case.ground_truth,
    retrieval_context=trace.final_assembled_context,
)
```

建议的第一批指标：

```text
FaithfulnessMetric
AnswerRelevancyMetric
ContextualPrecisionMetric
ContextualRecallMetric
GEval: misconception diagnosis quality
GEval: pedagogical actionability
GEval: Socratic / non-answer-leakage quality
```

上面的类名和参数以锁定的 DeepEval 版本文档为准；安装前不在代码中假定版本。每个 Metric 都要独立落库：

```json
{
  "run_id": "run-...",
  "case_id": "golden-001",
  "layer": "L1",
  "stage": "generator",
  "metric_name": "faithfulness",
  "value": null,
  "threshold": 0.95,
  "status": "UNMEASURED",
  "evaluator_type": "deepeval",
  "evaluator_model": null,
  "deepeval_version": null,
  "dataset_version": "golden-v...",
  "system_config_hash": "...",
  "evidence_refs": [],
  "error": "evaluation model is not configured"
}
```

### 4.3 统一 Trace / Observation 最小字段

每个 Case 必须关联同一个 `run_id` 和 `case_id`。L1 评测至少记录：

```text
run_id, case_id, layer, stage
input_hash, output_hash
dataset_version, split, qrels_version
prompt_version, index_version, embedding_version
candidate_ids, ranked_ids, final_context_ids
latency_ms, input_tokens, output_tokens, estimated_cost
metric_name, value, threshold, status
evidence_refs, error_type, error_message, created_at
```

状态枚举建议固定为：

```text
SUCCESS       已真实测量且通过
FAILED        已真实测量但未通过，或被测组件明确失败
UNMEASURED    条件未满足，未执行或不适用
DEGRADED      显式授权的真实替代路径
ERROR         评测器或被测组件异常
```

`UNMEASURED` 不得参与平均值；`DEGRADED` 必须单独计数并默认阻断发布。

## 5. L1 组件清单、指标和实现

### 5.1 数据抽取与证据绑定

**隔离输入**：固定 `CleanedSession`，不经过 Chroma、不调用生成器。  
**实现位置**：`src/extract_knowledge.py`、`src/models.py`、`scripts/expand_knowledge_base.py`。

指标：

| 指标 | 定义/公式 | 实现方式 | 初始门槛 |
|---|---|---|---:|
| Schema pass rate | 通过严格 Pydantic 契约的记录数 / 总记录数 | `model_validate`；禁止 extra、空必填字段 | 100% |
| Quote grounding precision | 能在对应真实 Turn 原文找到的引用数 / 输出引用数 | 逐字子串 + `(session, turn, speaker)` 联合校验 | 100% |
| Source role accuracy | 学生卡只绑定 student Turn，策略卡只绑定 tutor Turn | DuckDB 反查 `session_dialogue_turns` | 100% |
| Extraction coverage | 成功生成完整卡片的会话数 / 输入会话数 | 记录 `success/partial/failed/skipped`，不伪造 | ≥95% |
| PII leakage | 通过脱敏检查的记录数 / 总记录数 | 正则和人工抽样复核 | 100% |

### 5.2 存储与索引一致性

**隔离输入**：固定已验证的会话、卡片和 Chunk，执行入库后审计。  
**实现位置**：`src/storage_manager.py`。

指标：

```text
id_parity = 1 - |DuckDB IDs △ Chroma IDs| / expected_ids
orphan_rate = orphan_rows / total_rows
metadata_contract_rate = complete_metadata_rows / indexed_rows
```

门槛：`id_parity=100%`、`orphan_rate=0`、关键 metadata contract `=100%`。数量同步不能替代字段契约检查。

### 5.3 Chunking

**隔离输入**：会话 + Turn 级 qrels。  
**实现位置**：`src/chunker.py`、`tests/chunk/benchmark_eval.py`。

指标：

```text
Session Recall@K = 命中目标 session 的 query 数 / query 总数
Turn Recall@K    = 命中至少一个 target_turn 的 query 数 / query 总数
MRR              = mean(1 / 第一个相关 Chunk 的 rank；未命中为 0)
inflation_ratio  = chunk token 估算总量 / 原始会话 token 估算总量
```

Chunking 的“最佳”不是只看 Recall；应在 Turn MRR、Token 膨胀和构建成本的 Pareto 前沿上选择。当前 10 条样本仅为诊断，尚未形成正式 holdout。

### 5.4 Query Rewrite / Router

**隔离输入**：原始 Query；比较 identity、deterministic rules 和未来 LLM rewrite。  
**实现位置**：`src/query_rewriter.py`。

指标：

| 指标 | 定义/实现 | 门槛 |
|---|---|---:|
| Numeric/formula preservation | 改写前后数字、公式、选项名集合的精确保留率 | 100% |
| Entity preservation | 关键学科实体/Session/题型实体未丢失的比例 | ≥99% |
| Intent preservation | 人工标注或 DeepEval rubric 判定意图未漂移的比例 | ≥95% |
| Domain injection coverage | 应匹配领域词典的 Query 中实际命中的比例 | 诊断指标，不作为唯一质量门槛 |
| Retrieval lift | 配对 Query 在 Rewrite 后的 Recall/MRR 增量 | 必须显著优于 identity 或说明不值得保留 |

字符串仍包含原始 Query 不是语义保持证明；必须检查结构化实体和下游 Recall。

### 5.5 Embedding 与 Retriever

**隔离输入**：固定 Query、固定索引、固定 graded qrels。  
**实现位置**：`src/storage_manager.py`、`src/retriever.py`、`scripts/eval_50_queries.py`。

推荐指标：

```text
Candidate Recall@20 = qrels 中至少一个相关文档进入候选池的比例
Recall@K            = top-K 覆盖的相关 qrels / 相关 qrels 总数
Precision@K         = top-K 相关 qrels 数 / K
MRR                 = mean reciprocal rank
nDCG@K              = 按 graded relevance 计算的归一化折损累计增益
noise_ratio         = top-K 不相关结果数 / K
P50/P95/P99 latency = 检索阶段耗时分位数
```

正式 qrels 必须允许多个真实相关 Session/Chunk，并至少区分：

```text
0 = 无关
1 = 同考点但不能直接支持结论
2 = 可支持结论
3 = 直接证据
```

当前脚本的“唯一目标 Session 命中”适合作为旧诊断，不足以作为最终 nDCG 真值。

### 5.6 RRF

RRF 不应被当作一个脱离结果的“业务质量组件”。它应以消融实验测量：

```text
ΔRecall = Recall(Rewrite + RRF) - Recall(Rewrite + no RRF)
ΔMRR    = MRR(Rewrite + RRF) - MRR(Rewrite + no RRF)
Δlatency = latency(RRF) - latency(no RRF)
```

只有在质量增益超过延迟和复杂度成本时才保留。RRF `k`、各透视候选数和融合前后排名必须进入 Trace。

### 5.7 Reranker / MMR / Context Assembler

**隔离输入**：固定 Retriever 候选集；不重新调用 Embedding 或 Generator。  
**实现位置**：`src/reranker.py`。

指标：

| 指标 | 定义 |
|---|---|
| Rerank nDCG gain | 重排后 nDCG@K - 重排前 nDCG@K |
| Candidate recall retention | 重排/截断后仍保留至少一个相关证据的比例 |
| Final evidence recall | 最终注入 Context 覆盖 required evidence 的比例 |
| Selection precision | 最终选中的卡片中真正相关卡片比例 |
| Noise ratio | 最终 Context 中无关文本占比 |
| Token budget violation | `estimated_tokens > max_prompt_tokens` 的 Case 比例 |
| Evidence empty rate | 最终 Context 没有可审计证据的 Case 比例 |
| Compression ratio | `1 - assembled_chars / candidate_chars`，只作成本/容量指标 |

MMR 的 `lambda` 必须用固定候选集做网格消融；只选一张错因卡时没有同集合多样性惩罚，不能把当前 `lambda=0.7` 视为已验证最优。

### 5.8 Generator

**隔离输入**：优先使用 Gold Context，判断生成器自身上限；再使用真实最终 Context 判断链路损失。  
**实现位置**：`src/rag_pipeline.py`；语义评测由 DeepEval Adapter 执行。

指标和初始门槛：

| DeepEval/业务指标 | 单例阈值 | 数据集通过率 |
|---|---:|---:|
| Faithfulness | 0.95 | ≥95% |
| Answer Relevancy | 0.90 | ≥95% |
| Contextual Precision | 0.90 | ≥95% |
| Contextual Recall | 0.90 | ≥95% |
| Misconception diagnosis G-Eval | 0.90 | ≥95%（须人工校准） |
| Pedagogical actionability G-Eval | 0.90 | ≥95%（须人工校准） |
| Socratic/non-answer-leakage G-Eval | 0.95 | ≥95%（关键违规需 0） |

“分数 0.95”与“95% 用户满意”不是同一个概念。必须同时报告单例 score、阈值、pass/fail、数据集 pass rate、slice 和置信区间。

### 5.9 Citation 与输出契约

Citation 四元组定义为：

```text
(session_id, turn_id, speaker, quote_text)
```

分别计算：

```text
Citation Precision = 有效输出引用数 / 输出引用数
Citation Recall    = 被覆盖的 required evidence 数 / required evidence 数
```

实现位置：`src/rag_pipeline.py:348`、`src/evaluation.py:408` 和 DuckDB `session_dialogue_turns`。当前代码主要实现了 Precision 风格的命中率，后续必须补充 Recall；任何一项不是 100% 都不得发布。

## 6. 当前 L1 实测结果

### 6.1 测试契约

运行全量测试：

```text
87 passed, 1 failed
```

失败项：`tests/test_expand_knowledge_base.py::test_llm_expansion_extractor_uses_strict_shared_gate`。

真实错误是异常路径返回了 `extraction_status="llm_failed"`，而 `src/models.py` 的 Literal 契约只允许：

```text
success / partial / failed / skipped
```

这不是普通低分，而是失败状态契约错误；发布门槛为 100% 通过，必须先修复。

评测模块定向测试单独运行结果：

```text
tests/test_evaluation.py: 8 passed
```

它证明了严格 Judge schema、缺少 LLM 配置时失败、显式 heuristic 降级标记、引用四元组审计和不泄漏 Gold 等测试契约；不代表 DeepEval 已经运行。

### 6.2 存储与索引健康度

| 指标 | 结果 | 结论 |
|---|---:|---|
| DuckDB/Chroma 数量同步 | 100% | 写入数量一致 |
| 外键孤儿记录 | 0 | 会话/卡片/Turn 外键关系健康 |
| Misconception 卡完整 metadata contract | 0% | Chroma 快照缺少 `deep_mechanism` 等关键字段 |
| Tutor strategy 卡完整 metadata contract | 0% | Chroma 快照缺少 `pedagogical_goal`、`scaffolding_steps`、`resolution_outcome` |

这解释了为什么“索引数量正常”仍然不能支撑生成器：索引 schema 与当前代码契约不一致，必须重建或迁移索引。

### 6.3 抽取卡片证据绑定

在 100 张卡片上逐条回查 DuckDB：

| 指标 | 结果 | 目标 |
|---|---:|---:|
| Misconception source Turn 角色正确 | 99% | 100% |
| Misconception verbatim quote grounding | 98% | 100% |
| Strategy source Turn 角色正确 | 99% | 100% |
| Strategy Aha 原句 grounding | 96% | 100% |

已发现的异常包括 Session 14、29、42、69、80。引用真实性不能用总体均值掩盖单条错误，应修复卡片并重建索引。

### 6.4 Chunking 基线（10 sessions / 10 Turn-qrels）

| 策略 | Session Hit@1 | Turn Hit@1 | Turn MRR | Token 膨胀 |
|---|---:|---:|---:|---:|
| SlidingWindow W4/S2 | 100% | 80% | 0.900 | 4.20x |
| SlidingWindow W6/S3 | 100% | 90% | 0.925 | 4.02x |
| ParentChild C2/S2 | 90% | 70% | 0.825 | 4.71x |
| KnowledgeDistilled Cards | 100% | 100% | 1.000 | 3.20x |

当前诊断候选是 `KnowledgeDistilled Cards`，但 `n=10` 太小，不能形成 95% 置信度的工业结论；必须扩展并按 `intervention_id` 做数据隔离。

### 6.5 Query Rewrite 与 Retriever 消融（50 queries）

| 模式 | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Raw Retriever | 18% | 28% | 36% | 0.2447 |
| Identity + RRF | 18% | 28% | 36% | 0.2447 |
| Deterministic Rewrite + RRF | 22% | 40% | 48% | 0.3200 |

配对增量：

```text
Hit@1  +4 个百分点
Hit@3  +12 个百分点
Hit@5  +12 个百分点
MRR    +0.0753
```

按目标类型拆分，Rewrite + RRF 的 Hit@5 为：

```text
misconception: 60%，MRR 0.4233
strategy:      36%，MRR 0.2167
```

这表明当前优先问题是策略库召回/表示/qrels，而不是先调生成 Prompt。

确定性 Rewrite 自身的结构指标：

```text
原始 Query 字符串保留率：100%
领域词典匹配覆盖率：    86%
平均改写耗时：           0.028 ms
```

三组 Hybrid alpha 消融 `0.0/0.5/1.0` 的排名和 Recall 完全相同，说明当前 Cosine 与 L2 排序近似单调等价；继续调 alpha 暂无收益证据。

### 6.6 Assembler / MMR（50 queries）

```text
最终选中目标 Session 比例：24%
被提升到目标的案例：        4
被错误降权的案例：          3
Token 超预算率：            0%
空证据率：                  0%
平均估算 Token：            690
平均压缩率：                80.22%
```

当前 MMR 结果必须谨慎解释：错因集合只选一张卡，第一张卡没有同集合多样性惩罚；因此 `lambda=0.7` 尚未被有效网格实验验证。

### 6.7 Generator / 端到端契约诊断

使用当前持久化索引和确定性生成模式运行 `data/golden_test_set.json` 的 30 条 Query：

```text
成功：0/30
失败：30/30
共同原因：确定性生成缺少真实 misconception_name 或 deep_mechanism
```

这不是生成质量分数，而是被测链路在进入语义生成前的契约阻断。系统正确地拒绝伪造回答，但当前不能进入 DeepEval Generator 基线。

DeepEval 语义指标当前全部为：

```text
Faithfulness             UNMEASURED
Answer Relevancy         UNMEASURED
Contextual Precision     UNMEASURED
Contextual Recall        UNMEASURED
教学 G-Eval              UNMEASURED
```

### 6.8 性能

50 条 Rewrite + RRF 检索：

```text
P50：106.17 ms
P95：117.45 ms
P99：119.89 ms
最大：603.78 ms
```

最大值需要在 Trace 中拆分 DuckDB、Chroma、Rewrite、RRF 和证据回查耗时；不能只保留总耗时。

### 6.9 被排除的旧报告

`data/ragas_evaluation_report.json` 仅作为历史诊断材料，不纳入新基线，原因是：

1. 使用自研 Judge/关键词逻辑，不是 DeepEval；
2. 当前报告 schema 与当前代码不一致（旧报告顶层为 `total_test_cases/average_metrics/case_details`）；
3. `answer_relevance` 30/30 为 1.0，存在明显饱和；
4. 当前评测脚本传给 Judge 的是 `retrieved_sources_debug`，不保证等于生成模型实际看到的最终 Context；
5. 旧报告不能与新的 DeepEval 分数直接横向比较。

## 7. 量化标准和门禁

### 7.1 分数、通过率和置信区间必须分开

对每个 Metric 同时报告：

```text
case_score       单条用例的 DeepEval 或确定性得分
threshold        该指标的单例阈值
case_pass        score >= threshold
pass_rate        通过用例数 / 可测用例数
unmeasured_rate  未测用例数 / 总用例数
slice_pass_rate  按业务类型/学科/难度的通过率
confidence_interval 通过率置信区间
```

`UNMEASURED` 不得作为 0，也不得从分母中悄悄删除而不报告。

### 7.2 初始发布门槛

| 类别 | 指标 | 初始目标 |
|---|---|---:|
| 数据/接口真实性 | Schema、失败状态、输出契约 | 100% |
| 索引健康 | ID parity、关键 metadata contract | 100% |
| 外键健康 | orphan rate | 0 |
| 证据真实性 | Citation Precision、Citation Recall、角色绑定 | 100% |
| 抽取质量 | 关键字段人工/DeepEval 通过率 | ≥95% |
| Chunking | Turn Recall@3 | ≥95% |
| Chunking | Turn MRR | ≥0.90 |
| Rewrite | 数字/公式/关键实体保留率 | 100% / ≥99% |
| Rewrite | 意图保持率 | ≥95% |
| Retriever | Candidate Recall@20 | ≥98% |
| Retriever | Evidence Recall@5 | ≥95% |
| Retriever | Recall@3 | ≥90%（成熟目标 95%） |
| Retriever | nDCG@5 | ≥0.90（成熟目标 0.95） |
| Retriever | MRR | ≥0.85（成熟目标 0.90） |
| Assembler | Candidate recall retention | ≥98% |
| Assembler | Final evidence relevance | ≥95% |
| Assembler | Token budget violation | 0% |
| Generator | DeepEval Faithfulness | 单例 ≥0.95，pass rate ≥95% |
| Generator | DeepEval Answer Relevancy | 单例 ≥0.90，pass rate ≥95% |
| Generator | 教学 G-Eval | 单例 ≥0.90，pass rate ≥95%（人工校准） |
| 稳定性 | 未声明降级、异常、非法 JSON | 0% |
| 性能 | 本地检索 P95 | ≤150 ms |
| 性能 | 生成前链路 P95 | ≤300 ms |

这些是当前项目的初始工程门槛，不是未经校准的行业定律。上线前必须用教研员标注集校准 DeepEval rubric 和阈值。

### 7.3 为什么单个组件 95% 还不够

如果五个关键阶段各自只有 95% 的成功率，独立近似下链路成功率为：

```text
0.95^5 = 77.38%
```

若希望五阶段串联后达到约 95%，每一阶段需要约：

```text
0.95^(1/5) = 98.98%
```

因此：

- 数据真实性、引用、Schema、失败状态属于硬门禁，目标 100%；
- Candidate Recall、索引契约等上游指标应接近 98%~99%；
- 语义质量可以先用 95% 的案例通过率作为阶段目标；
- L3 业务效果不能用 L1 分数替代。

当希望在 95% 置信度下支持“失败率低于 5%”时，零失败样本至少需要约 59 个独立用例；当前 10/10 或 30/30 不足以作此统计声明。

## 8. 指标到优化方向的决策矩阵

| 观察组合 | 首要归因 | 先做的实验 | 当前项目优先级 |
|---|---|---|---:|
| Schema/metadata contract <100% | 抽取、存储或索引迁移 | 重建索引并逐字段校验 | P0 |
| E2E 成功率为 0，但检索有结果 | Retriever 输出与 Generator 契约断层 | 对照 DuckDB 原卡片和 Chroma metadata | P0 |
| Quote/role grounding <100% | 抽取卡片或证据回溯 | 修复错误卡片，重跑 grounding gate | P0 |
| Candidate Recall@20 <98% | 数据、Chunk、Rewrite、Embedding | qrels 对照 + Oracle candidate | P1 |
| Candidate Recall 高，nDCG/MRR 低 | RRF、排序或 Reranker | Oracle Rerank、Raw/RRF 配对消融 | P1 |
| Raw 低，Rewrite 明显提升 | Query Rewrite/领域词典 | identity vs rules vs LLM rewrite | P1 |
| Raw 与 Identity+RRF 完全相同 | RRF 只增加复杂度/延迟 | 检查融合是否真正改变排名 | P1 |
| alpha 0/0.5/1 排名完全相同 | Cosine/L2 冗余 | 删除一个度量并比较延迟 | P2 |
| Strategy slice 显著低于 Misconception | 策略卡表示、样本或 qrels | 重建策略文档、补术语和多相关 qrels | P1 |
| Assembler 后 Evidence Recall 下降 | MMR、boost、截断或槽位 | Oracle Assembler 对照 | P1 |
| Gold Context 下 DeepEval 仍失败 | Generator、Prompt 或输出契约 | 固定 Context 做模型/Prompt 实验 | P2 |
| Gold Context 成功，真实 Context 失败 | Retriever/Assembler | 比较最终 Context 的 claim 覆盖 | P1 |
| Faithfulness 高、Relevancy 低 | 路由或回答结构 | Query intent 和输出 schema 实验 | P2 |
| Citation Precision/Recall <100% | 证据链 | 阻断发布，定位四元组差异 | P0 |
| P95 正常、Max 长尾异常 | I/O、重试或证据回查 | 阶段级 latency trace | P2 |

当前实测对应的直接结论是：先处理 P0 的索引 metadata、端到端契约、失败状态和证据绑定；其次处理 strategy 检索；暂不把 alpha 调参或生成 Prompt 作为第一优化方向。

## 9. 正式 L0/L1 基线运行流程

### 9.1 建基线前置条件

- [ ] 修复 `llm_failed` 与 `ExtractionStatus` 枚举不一致。
- [ ] 重建 Chroma，确保所有 Generator 所需 metadata 字段存在且非空。
- [ ] 修复 Session 14、29、42、69、80 的证据角色/逐字绑定问题。
- [ ] 给 DuckDB、Chroma、卡片和 Chunk 增加明确的 `index_version` / `schema_version`。
- [ ] 将 50 条检索 Query 从脚本常量导出成版本化 JSON，并补充 graded qrels。
- [ ] 按 `intervention_id` 做 train/dev/test 隔离，避免同会话泄漏。
- [ ] 安装并锁定 DeepEval 版本；显式配置 evaluation model。
- [ ] 让 Pipeline 输出 rewrite、候选、RRF、assembler 和**最终实际 Context**的 Trace。

### 9.2 建议运行命令

以下命令描述目标流程；当前环境在满足前置条件前，不应把 DeepEval 命令当作已成功的证据。

```powershell
# 确定性契约与组件测试
.\.venv\Scripts\python.exe -m pytest -q

# Chunking L1
.\.venv\Scripts\python.exe tests\chunk\benchmark_eval.py

# Retriever L1（Raw/RRF 配对）
.\.venv\Scripts\python.exe scripts\eval_50_queries.py

# DeepEval L1（安装并配置 evaluation model 后）
deepeval test run tests\eval_deepeval_l1.py -n 1
```

正式 runner 应生成一个带 schema version 的 JSON/Parquet 报告，而不是依赖控制台文本：

```text
reports/eval/{run_id}/run.json
reports/eval/{run_id}/observations.jsonl
reports/eval/{run_id}/slices.parquet
reports/eval/{run_id}/failures.jsonl
```

### 9.3 回归比较规则

1. 只比较同一 `dataset_version`、`qrels_version`、`system_config_hash` 和 metric/evaluation model 版本。
2. 同一 Case 做 paired comparison；同时报告绝对值、增量和置信区间。
3. 更换 evaluation model 后建立新基线，不把新旧 DeepEval 分数混合平均。
4. 任一硬门禁失败、未声明降级或关键指标 `UNMEASURED`，发布状态为 `BLOCKED`。
5. 只允许在候选版本相对 baseline 没有硬回归且目标 slice 改善时合入优化。

## 10. 当前待办顺序

```text
P0  修复状态枚举和索引 metadata schema，重建可运行的 E2E 链路
P0  修复四元组证据绑定，补 Citation Recall
P0  安装/锁定 DeepEval，配置独立 evaluation model
P1  统一 EvalCase、Trace、MetricObservation 和报告 schema
P1  将 50-query 数据集升级为版本化 graded qrels + 独立 holdout
P1  跑 Retriever/Assembler 的 Candidate Recall、nDCG、MRR 和 Oracle 对照
P2  跑 Gold Context Generator DeepEval 基线
P2  比较 strategy/misconception slice，决定策略库重建方向
P2  做 Rewrite、RRF、alpha、MMR 的正交消融
P3  接入 L2 反事实链路和 L3 教研员/线上业务数据
```

## 11. 证据日志

本文件引用的关键结果来自以下只读运行：

| 检查 | 结果 |
|---|---|
| `python -m pytest tests/test_evaluation.py -q` | `8 passed` |
| `python -m pytest -q` | `87 passed, 1 failed`（状态枚举异常） |
| 10-session Chunking benchmark | KnowledgeDistilled Cards：Turn Hit@1 100%、Turn MRR 1.000、膨胀 3.20x |
| 50-query Raw/RRF benchmark | Raw Hit@5 36%、RRF Hit@5 48%、RRF MRR 0.3200 |
| Rewrite + RRF ablation | Hit@3 +12pp、Hit@5 +12pp、MRR +0.0753 |
| alpha ablation | 0.0/0.5/1.0 排名和 Recall 相同 |
| 100-card storage audit | ID parity 100%、orphan 0 |
| 100-card grounding audit | student role 99%、student quote 98%、tutor role 99%、Aha 96% |
| 30-case deterministic E2E | `0/30`，metadata contract 断层 |
| DeepEval environment check | 未安装、evaluation model 未配置 |

## 12. 基线签署标准

当且仅当以下条件全部满足，`l1-v0.1-diagnostic` 才能升级为正式 `l1-v1.0`：

- L0 DeepEval 和 evaluation model 配置可复现；
- 所有报告均含完整 provenance 和 schema version；
- 关键数据/索引/引用硬门禁通过；
- L1 确定性指标有独立 holdout 和 graded qrels；
- L1 Generator 指标由 DeepEval 实际测量，不混入旧自研 Judge；
- 每个指标有阈值、pass rate、slice 和置信区间；
- 所有 `UNMEASURED`、`DEGRADED`、`FAILED` 都显式出现在报告中；
- 通过一次完整回归，并且没有 P0/P1 未解释的失败。

在此之前，本项目应对外表述为：**“已建立可审计的 L1 诊断框架，DeepEval 语义基线和正式发布门禁尚未完成。”**
