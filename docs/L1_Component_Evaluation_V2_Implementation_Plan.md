# Eedi-RAG L1 组件评测 v2 具体实施说明书

> 文档编号：`EEDI-EVAL-L1-V2-IMPLEMENTATION-001`
> 版本：`v2.0-plan`
> 日期：2026-09-02（Asia/Shanghai）
> 状态：待施工
> 适用范围：Eedi-RAG 当前 100 Session 开发快照

## 1. 目标与交付边界

本轮目标不是让报告“变绿”，而是让每个 L1 指标真实对应一个生产组件，并能准确回答：问题出在数据、切块、候选召回、查询扩展、融合、重排、上下文装配，还是引用链。

交付后应形成六个可以独立复现的评测模块：

```text
Storage / Index Integrity
Grounding / Citation
Chunk Structural Quality
Query Expansion / Fusion
Hybrid Retrieval
Reranker / Context Assembler
```

DeepEval 不参与上述确定性指标。DeepEval 只在后续 Generator L1/L2 中使用独立配置的 evaluation LLM，评估 Faithfulness、Answer Relevancy 和教学质量。

本轮不做以下事情：

- 不降低门槛以换取绿色状态；
- 不用关键词重叠冒充语义 Judge；
- 不直接覆盖源 Chroma；
- 不凭当前 30/50 条单目标 qrels 承诺工业 Recall；
- 不把 Eedi 数学代理数据结果表述成 NBCOT 业务结论。

## 2. 已冻结的事实结论

以下结论已经由源码或当前真实 artifact 直接证明，可作为后续施工的固定基线。

### 2.1 Storage 已通过

当前生产 `data/chroma` 已完成 metadata 受控迁移，生产复测：

```text
run_id: l1-storage-production-20260902-105043
artifact_hash: af6798c5c4aa815b1515bfa1da697fc0849c4fa966a8d3ada87d0a9d427ae839
SUCCESS: 7
FAILED: 0
UNMEASURED: 0
release_status: PASSED
```

这只证明 ID parity、metadata contract 和 orphan 门禁通过，不证明检索质量。

### 2.2 Grounding 现有失败属于黄金数据完整性，不是生产 Grounding 基线

`grounding-v1.json` 的 `output_citations` 直接复制自 `required_evidence`，没有采集真实 RAG 输出。当前 3 个失败引用分别为：

| Case | Session/Turn | 黄金文本 | DuckDB 权威文本 | 差异 |
|---|---|---|---|---|
| golden-19 | 275/4 | `I'd` | `I’d` | ASCII/Unicode apostrophe |
| golden-22 | 278/9 | `That's` | `That’s` | ASCII/Unicode apostrophe |
| golden-25 | 42/12 | `don't` | `don’t` | ASCII/Unicode apostrophe |

正确统计是 27/30 case 全部通过（90%），87/90 citations 有效（96.67%），role accuracy 100%。旧汇总中的“4 个失败 case、27/31、87%”不作为可信结论。

### 2.3 生产滑窗内容完整，但现有 Chunking 排名不是实际检索排名

对 DuckDB 当前 719 个 `sliding_window_chunks` 的只读核验结果：

| 结构指标 | 实测 |
|---|---:|
| source pointer validity | 4192/4192 = 100% |
| Turn 正文逐字一致 | 4192/4192 = 100% |
| Chunk Turn 列表一致 | 719/719 = 100% |
| 窗口边界一致 | 719/719 = 100% |
| 100 个已索引 Session 的 Turn 覆盖 | 2335/2335 = 100% |
| 越界 Turn | 0 |

现有评测构建器没有调用生产 `SlidingWindowChunker`，30 个黄金 case 中有 23 个窗口序列与生产实现不同；同时构建器人为执行 `hit + miss` 排序。因此现有 `turn_recall@1/@3` 是理想窗口装载能力，不是实际检索召回率。

### 2.4 完整多方向 RRF 有正增益，但绝对召回仍低

legacy-50 的同 Query、同单目标 qrels 对照：

| 指标 | Raw | 多方向 RRF | 差值 |
|---|---:|---:|---:|
| Candidate Recall@20 | 0.48 | 0.64 | +0.16 |
| Recall@5 | 0.26 | 0.48 | +0.22 |
| MRR | 0.1522 | 0.3081 | +0.1559 |
| nDCG@5 | 0.1639 | 0.3406 | +0.1767 |

因此不能用现有 Rewrite 单路 `MRR lift=-0.091` 推断完整融合有负收益。完整 RRF 明显优于 Raw，但仍未达到生产门槛。

### 2.5 当前 Retriever 不是 Sparse + Dense Hybrid

当前 `DualMetricRetriever` 对同一个归一化哈希向量组合 Cosine 和 Euclidean L2，再对多个 Dense Query 做 RRF。归一化向量满足 `L2² = 2 - 2 × cosine`，两种度量基本为单调等价排序，不能提供独立的词法召回。

当前没有 BM25/Sparse，也没有 Cross-Encoder。`FastDeterministicEmbeddingFunction` 是 token/字符 2-gram 哈希基线，不具备生产语义表示能力。

### 2.6 现有 Assembler precision/noise 指标无法通过

30 个 Assembler case 全部只有一个正例 qrel，但生产 Assembler 固定选择一张 misconception 和一张 strategy。即使目标卡正确，单 case precision 理论上限也只有 `1/2=0.5`，不可能通过 `0.95/1.0` 门槛。

此外数据集使用 `fetch_evidence=False`，没有测试生产路径中的 DuckDB evidence assembly。因此旧报告中的 `selection_precision=0.10` 和 `noise_ratio=0.90` 不能直接用来判定 Assembler 算法质量。

## 3. 必须废止或降级为诊断的旧结论

| 旧指标/结论 | 处置 | 原因 |
|---|---|---|
| Grounding 87% | 废止 | 统计错误且没有真实系统输出 |
| Chunk turn_recall@1=55% 代表召回差 | 废止 | 使用 oracle `hit + miss` 排序 |
| Rewrite MRR lift=-0.091 代表融合变差 | 废止 | 只测 misconception 单路，策略 Query 也用错方向 |
| Intent preservation=100% | 废止 | 30/30 均由脚本硬编码 `True` |
| Assembler precision=10%、noise=90% | 废止 | 单正例 qrels 与固定双槽位输出冲突 |
| legacy-50 noise=90%+ 为真实业务噪声率 | 仅诊断 | 单目标 Session qrels 会把其他相关卡误记为噪声 |

旧 JSON/Markdown 报告保留作为审计历史，不覆盖、不删除；新报告必须使用新的 dataset/qrels/schema 版本，禁止与 v1 混算。

## 4. v2 组件边界和数据流

```text
真实 Query + graded qrels
       │
       ├─ Chunk Structural Auditor（不排名）
       │
       ├─ Raw BM25 lane ──────────────┐
       ├─ Raw Dense lane ─────────────┤
       ├─ Misconception Dense lane ───┤
       ├─ Strategy Dense lane ────────┤→ Weighted RRF → Candidate Top-N
       └─ Curriculum Dense lane ──────┘                       │
                                                              ▼
                                                    Cross-Encoder Rerank
                                                              │
                                                              ▼
                                                    Context Assembler
                                                              │
                                                              ▼
                                                  Final Context + Citations
```

组件归因原则：

- Chunk 结构审计不使用检索排名；
- Query Expansion 记录每个 lane 的独立结果；
- Fusion 只比较同候选源下的 Raw 与 fused；
- Retrieval 关注相关对象是否进入 Candidate Top-N；
- Reranker 只在固定候选集上比较 before/after；
- Assembler 只评价最终上下文保留、槽位和预算；
- Grounding 只评价真实最终引用与 DuckDB 权威四元组。

## 5. 数据契约 v2

### 5.1 统一 qrels

新增版本化数据文件，建议路径：

```text
evals/datasets/l1-v2/queries.jsonl
evals/datasets/l1-v2/qrels.jsonl
evals/datasets/l1-v2/grounding.jsonl
evals/datasets/l1-v2/manifest.json
```

Query 最小字段：

```json
{
  "query_id": "eedi-l1-0001",
  "query": "...",
  "intent": "STUDENT_INSIGHT",
  "split": "dev",
  "source_session_ids": [10],
  "slices": {
    "language": "zh-en",
    "subject": "Number",
    "contains_formula": "true"
  }
}
```

qrel 最小字段：

```json
{
  "query_id": "eedi-l1-0001",
  "document_id": "session_10_misconception",
  "slot": "misconception",
  "relevance": 3,
  "label_source": "human",
  "evidence_turns": [{"session_id": 10, "turn_id": 6}]
}
```

约束：

- `relevance` 只能为 0/1/2/3；
- 同一 `intervention_id` 只能属于一个 split；
- 每个 Query 可有多个相关文档；
- misconception、strategy、fallback_window 分槽标注；
- 自动派生标签只能标 `derived`，不能冒充 `human`；
- 缺少标签的指标必须为 `UNMEASURED`。

### 5.2 Trace 契约

每次真实执行至少记录：

```text
query_id
raw_query
expanded_queries by lane
lane_name + retriever_type
candidate_ids + raw ranks + raw scores
fusion score + fused rank
reranker score + reranked rank
final_context_ids by slot
final_evidence_turns
actual prompt context
stage latency
artifact/config/model hashes
failure/degradation reason
```

Gold/qrels 不允许进入被测生产调用，只能在执行结束后由评测控制面 join。

## 6. 施工包

### WP1：Grounding v2

修改范围：

- `scripts/build_l1_component_datasets.py`
- `evals/contracts.py`
- `evals/metrics/citation.py`
- `evals/runners/grounding.py`
- 新增 `evals/adapters/grounding.py`
- `src/rag_pipeline.py` 仅在需要暴露真实 Trace 时做最小侵入修改

施工：

1. 删除 `output = list(required)` 作为正式组件基线的路径。
2. 将黄金数据自身检查命名为 `grounding_dataset_integrity`。
3. 修正 3 条 apostrophe 引用，或在 manifest 中明确唯一允许的 Unicode canonicalization。
4. 从真实 `PedagogicalGuidanceResponse.dialogue_citations` 采集输出。
5. 对每条引用核验 `(session_id, turn_id, speaker, exact_quote)`。
6. 引用必须来自本次检索候选及其 `source_turn_ids`；仅存在于数据库但未被召回的引用也判失败。

指标与门禁：

| 指标 | 门禁 |
|---|---:|
| dataset_quote_integrity | 100% |
| citation_precision | 100% |
| citation_recall | 100% |
| role_accuracy | 100% |
| quote_grounding_precision | 100% |
| candidate_authorization_rate | 100% |

红灯测试：错误 Session、错误 Turn、错误 speaker、弯/直引号策略外变更、未召回证据、空引用必须失败。

### WP2：Chunking v2

修改范围：

- `scripts/build_l1_component_datasets.py`
- `evals/contracts.py`
- `evals/runners/chunking.py`
- 新增 `evals/adapters/chunking.py`

施工：

1. 删除评测脚本中的 `_chunk_ranked_records()` 重实现，直接调用生产 `SlidingWindowChunker`。
2. 删除 `hit + miss` oracle 排名。
3. 将结构质量与真实 fallback retrieval 分为两个 suite。
4. 结构 suite 读取生产 DuckDB/Chunk 输出；检索 suite 使用真实 ranked window IDs。

结构指标：

```text
pointer_validity = valid source pointers / all source pointers
verbatim_integrity = exact source Turn lines / all declared Turn lines
turn_coverage = covered source Turns / all expected source Turns
boundary_integrity = valid window boundaries / all windows
inflation_ratio = emitted tokens / source tokens
```

检索指标：

```text
Candidate Turn Recall@20/@50
Turn Recall@3/@5
Context Precision@5
Relevant-turn Density@5
MRR / nDCG@5
Duplicate-turn Ratio
P95 latency
```

结构硬门禁全部为 100%，orphan/越界为 0。检索门禁在完成 graded qrels 后校准，校准前只报告不签署发布。

### WP3：Query Expansion 与 Fusion v2

修改范围：

- `scripts/build_l1_component_datasets.py`
- `evals/contracts.py`
- `evals/runners/rewrite.py`，建议重命名/新增 `fusion.py`
- `src/retriever.py` 增加 lane trace，不改变默认结果

施工：

1. STUDENT_INSIGHT 使用 misconception lane；TUTOR_INTERVENTION 使用 strategy lane。
2. 单独记录 raw、curriculum、misconception、strategy 的排名。
3. 记录 rewrite-only RRF 与 raw-inclusive RRF。
4. 删除硬编码 `intent_preserved=True`；没有 human/DeepEval label 时写 `UNMEASURED`。
5. `domain_injection_expected` 必须来自注入器真实输出。

对照矩阵：

```text
A Raw Dense
B Intent Dense
C Curriculum Dense
D Rewrite-only Dense RRF
E Raw + Rewrite Dense RRF
```

主要门禁：E 相对 A 的 Candidate Recall@20、Recall@5、MRR 不得下降；同时报告各 slice 和 P95 latency。单 lane 下降不直接阻断，只用于归因。

### WP4：Assembler v2

修改范围：

- `evals/contracts.py`
- `evals/runners/assembler.py`
- `scripts/build_l1_component_datasets.py`
- `src/reranker.py` 仅增加 trace/预算执行，不修改相关性公式

施工：

1. qrels 按 misconception/strategy 槽位拆分。
2. `fetch_evidence=True`，使用生产 evidence assembly。
3. 区分 `retriever_miss` 与 `assembler_drop`。
4. token 数使用实际目标模型 tokenizer；没有 tokenizer 时明确标 `estimated`。
5. 检查 `max_prompt_tokens` 是否真正执行，不只检查字段。

指标：

```text
misconception_slot_recall/precision
strategy_slot_recall/precision
candidate_recall_retention_by_slot
final_evidence_turn_recall
context_relevant_token_ratio
subject/session consistency
token_budget_violation_rate
truncation_loss
evidence_empty_rate
rerank_ndcg_gain
P95 latency
```

### WP5：BM25 + Dense Hybrid Retrieval

此施工必须在 WP1-WP4 修正后的测量系统上进行，并写入隔离索引。

建议候选架构：

```text
BM25 Top-50（题干、专业词、公式、原声短语）
Dense Top-50（同义、口语、跨语言语义）
        ↓
Weighted RRF Top-50
        ↓
Cross-Encoder Top-10
        ↓
Assembler
```

实现要求：

- Sparse 索引与 Dense 索引均有独立版本/hash；
- misconception、strategy、fallback window 可分索引或带强类型字段；
- 精确数字、公式、选项和 subject path 不得在分词时丢失；
- Dense 模型优先评测多语言模型，因为存在中文 Query 与英文对话/题干；
- 新索引写新目录，通过门禁后再蓝绿切换。

消融顺序：

```text
deterministic hash baseline
semantic Dense
BM25
BM25 + Dense RRF
BM25 + Multi-Query Dense RRF
+ Cross-Encoder
```

先看 Candidate Recall@20/@50，再看 reranked Recall@5、MRR、nDCG 和延迟。Candidate 没召回时禁止归因给 Reranker/Assembler。

### WP6：自动报告

新增统一报告生成器，禁止手工抄写计数：

- 所有汇总从 `run.json` 自动生成；
- 同时展示 case pass rate、citation-level rate 和 observation count；
- 展示 hard gate 与 soft diagnostic；
- 报告必须声明数据集/qrels/schema/artifact/config hash；
- 不同 qrels 版本禁止横向合并；
- 失真或过期报告标记 `SUPERSEDED`，但保留审计历史。

## 7. TDD 施工顺序

每个 WP 遵循红—绿—重构：

1. 先新增会在旧实现下失败的契约测试。
2. 只实现使该测试通过的最小代码。
3. 运行对应组件测试。
4. 运行全部 `tests/evals`。
5. 用真实 artifact 生成新的不可覆盖报告。
6. 检查源 artifact hash 前后不变。

首批必须增加的测试：

```text
test_grounding_dataset_integrity_is_not_system_output
test_grounding_rejects_evidence_outside_retrieved_candidates
test_chunk_adapter_uses_production_tail_window
test_chunk_retrieval_does_not_oracle_sort_hits
test_strategy_query_uses_strategy_lane
test_unlabeled_intent_is_unmeasured
test_fusion_compares_raw_with_full_rrf
test_assembler_precision_is_slot_scoped
test_assembler_trace_contains_real_evidence
```

## 8. 执行批次与验收

### Batch A：修正评测真实性

完成 WP1-WP4 与 WP6，不改变生产检索排名。

验收：

- 旧失真结论被标记 superseded；
- Grounding 使用真实系统输出或明确 `UNMEASURED`；
- Chunk 结构/检索完全分离；
- Rewrite/Fusion 使用完整多 lane 对照；
- Assembler 使用槽位 qrels；
- `tests/evals` 全通过；
- 新报告无手工统计矛盾。

### Batch B：构建正式 qrels

至少完成 dev/holdout 隔离、多相关 graded relevance、主要 slice 和人工复核。没有该批次，不签署正式 Retrieval/Assembler 门禁。

### Batch C：Hybrid Retrieval 消融

在隔离 artifact 上依次运行 Dense、BM25、RRF、Cross-Encoder。只提升通过同一 dev qrels 且 holdout 不退化的方案。

### Batch D：受控切换

切换前：停止占用进程、记录源 hash、备份源索引、验证 staging hash。切换后：重新运行 Storage、Retrieval、Assembler 评测并保留回滚目录。

## 9. 推荐命令

```powershell
$env:PYTHONPATH='.'

# 控制面测试
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp reports\runtime\pytest-l1-v2 `
  tests\evals

# 构建 v2 数据（施工后命令）
.venv\Scripts\python.exe scripts\build_l1_component_datasets.py `
  --schema-version l1-component-suite/v2 `
  --output evals\datasets\l1-v2

# 统一运行（施工后命令）
.venv\Scripts\python.exe -m evals.cli `
  --input evals\datasets\l1-v2\suite.json `
  --output-root reports\eval
```

上述数据集构建命令已实现；统一 Suite 命令可直接读取 canonical `evals\\datasets\\l1-v2\\suite.json`。构建必须使用隔离 DB/Chroma 副本，不应直接让生产 artifact 参与可写初始化。

## 10. Definition of Done

L1 v2 只有在以下条件全部满足后才完成：

- [ ] Grounding 使用真实最终 citation，四元组及候选授权率可核验；
- [ ] Chunk 结构完整性与真实检索召回分开出值；
- [ ] Query Expansion 每个 lane 和完整 Fusion 都有 paired 指标；
- [ ] Retrieval 使用多相关 graded qrels 和 session-isolated holdout；
- [ ] Assembler 使用槽位级 qrels 和真实 final Context/evidence；
- [ ] 所有结果绑定 dataset/qrels/artifact/config/model hash；
- [ ] `FAILED/UNMEASURED/DEGRADED/ERROR` 均准确反映状态；
- [ ] 报告完全自动生成，不存在手工汇总与 JSON 不一致；
- [ ] 测试通过且保留可复现日志；
- [ ] 源 artifact 在只读评测前后 hash 不变；
- [ ] 优化方案通过同集 paired comparison 和独立 holdout；
- [ ] 未使用 DeepEval 的组件不得产生 LLM 语义分数。

## 11. 当前立即执行顺序

```text
P0-1 修正 Grounding 数据/真实输出边界（已完成）
P0-2 Chunk 复用生产实现并移除 oracle 排名（已完成）
P0-3 Rewrite 改为完整 multi-lane Fusion 对照（已完成）
P0-4 Assembler 改为槽位级 qrels 和真实 evidence（已完成）
P0-5 自动重建六组件报告（已完成）
P1-1 建立多相关 graded qrels
P1-2 在隔离索引施工 BM25 + semantic Dense
P1-3 Cross-Encoder 与 Assembler 重新基线
```

在 P0 完成前，不根据旧 Rewrite/Assembler 数值修改生产排序参数。

## 12. 2026-09-02 四项修复实施记录

以下修复已落地并通过 `tests/evals`（`39 passed`）：

- Chunk retrieval 的 `inflation_ratio` 不再以 `original_token_count=1` 计算；该 suite 明确标记 `inflation_applicable=false`，结果为 `UNMEASURED`。真实 Chunk 生成 inflation 只在 `chunk_structure` 中测量，当前 aggregate 为 `3.8018 <= 4.0`。
- qrels 按 `(query_id, document_id)` 去重，quote evidence 聚合到 `evidence_turns`；`label_source=derived_from_human_quote`，并在 manifest 中声明目前没有独立人工多文档相关性标注。
- Assembler 新增 `retriever_miss_rate` 与 `assembler_drop_rate`：前者表示正例未进入候选，后者只在正例进入候选后计算未被最终上下文保留的比例；两者不再混为 Assembler 失败。
- manifest 新增 `source_session_count`、`indexed_session_count`、`golden_case_count`、输入/产后 DuckDB/Chroma/combined hash、system config hash、embedding/retrieval provenance、生成时间和 qrels 局限说明。

最终数据集目录：`evals/datasets/l1-v2`（修复版已提升为 canonical；旧版归档于 `reports/runtime/l1-v2-pre-fix-20260902-174150`）。
最终报告目录：`reports/eval/l1-v2-final-fixed-20260902-174150/l1-v2-suite`。
该报告仍为 `BLOCKED`，这是检索/引用质量未达门槛的真实结果，不是四项评测修复失败。
