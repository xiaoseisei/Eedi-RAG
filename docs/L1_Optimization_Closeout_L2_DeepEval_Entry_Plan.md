# L1 测试结果优化收尾与 L2 DeepEval 入场计划

> 文档编号：`EEDI-EVAL-L1-CLOSEOUT-L2-ENTRY-001`
> 版本：`v1.0`
> 日期：2026-09-03（Asia/Shanghai）
> 目标：完成一次可验收的 L1 优化闭环，关闭 L1 悬项后进入 L2 端到端评价。

## 1. 交付目标

本轮只允许三种最终状态：

```text
BUSINESS_QUALITY_PASSED
GO_TO_L2_TECHNICAL
BLOCKED（附失败 Case、责任模块、动作和复现命令）
```

不能留下“已发现但没有归属、没有验收、没有复现方式”的尾项。

L1 收尾必须产出：

```text
L1 最终数据集与 qrels manifest
Qwen 0.6B 候选索引及 hash
L1 closeout report
每个失败指标的最终分流
旧索引回滚目录
```

L2 入场必须产出：

```text
真实端到端 Trace
L2 数据集与 split manifest
DeepEval evaluation-model 配置
Gold Context 上限报告
Real Context 链路报告
L2 release decision
```

## 2. 当前冻结基线

生产 deterministic artifact 保持不动：

```text
data/chroma
combined hash = 711f1ed14e8f7014a2a57421665c08037fd8189c80bf922217209b7789718fa4
```

Qwen 0.6B 隔离索引：

```text
reports/eval/embedding-qwen3-0.6b-contract-20260903-002138/chroma
combined hash = 442ef0cf314182eb521801930bfc8d871172553e0d1cb6012c1876488b906e51
provider = openai_compatible:Qwen/Qwen3-Embedding-0.6B
dimension = 1024
index_version = qwen3-embedding-0.6b-v1
```

最新完整 Qwen L1 报告：

```text
reports/eval/l1-qwen3-0.6b-full-contract-20260903-002138
```

```text
SUCCESS = 1720
FAILED = 644
UNMEASURED = 196
ERROR = 0
release_status = BLOCKED
```

`BLOCKED` 表示真实门禁未全部通过，不表示评测程序执行崩溃。

## 3. 当前问题的最终分流

### 3.1 必须修复

| 问题 | 当前证据 | 收尾动作 | 关闭条件 |
|---|---|---|---|
| Assembler 丢弃正确候选 | `retriever_miss=0`、`assembler_drop=0.60`、`final_evidence_recall=0.40` | misconception/strategy 槽位独立选择；取消跨槽位冗余惩罚；固定候选做 paired 实验 | 有正候选 Case 的 drop ≤0.10，槽位 retention ≥0.90 |
| Grounding 引用质量低 | precision=`0.4321`、recall=`0.2722`、authorization=`1.0` | 恢复完整四元组；引用只能来自最终 evidence；删除“查到 Session 即可信”的跳过路径 | precision/recall/role/quote/authorization ≥0.95 |
| API 逐 lane 调用延迟高 | Retrieval P95 约 497ms，Assembler P95 约 1389ms | 同一 Query 的多 lane 向量批量请求；增加带模型/版本的成功结果 cache | Retrieval P95 ≤300ms，Assembler P95 ≤500ms |
| fallback Chunk 检索不足 | `turn_recall@3≈0.739`、`turn_mrr≈0.799` | 检查窗口 Query、候选数、数字/公式 exact channel；不改已通过的 Chunk 结构 | Turn Recall@3 ≥0.90，MRR ≥0.85 |
| Fusion 扰动 Raw 排名 | raw-inclusive MRR lift≈-0.039 | dev qrels 比较 weighted RRF、raw-first fallback、lane gating | 选定策略相对 Raw 不退化；无稳定增益则标 `diagnostic_only` |

### 3.2 必须重分类，不能伪造修复

| 指标 | 处理 |
|---|---|
| Chunk retrieval `inflation_ratio` | 保持 `UNMEASURED`；它只属于 Chunk generation/structure |
| Rewrite `intent_preservation` | 无 human/DeepEval label 时保持 `UNMEASURED` |
| Rewrite `domain_injection_coverage` | 只有真实注入预期为 true 才测，不适用不阻断 |
| Assembler fallback-window slot | 没有该槽位正例时记不适用，不算算法失败 |
| derived qrels 的 absolute precision/noise | 只作相对诊断，不宣称业务准确率 |

### 3.3 必须保留的限制

- 当前 qrels 是 30 Query、每题一个主要文档正例；文档相关性由人工引用/Session 派生，不是独立多文档人工 graded judgment。
- Qwen Retrieval 的 Recall/MRR/nDCG 触顶只说明目标卡被找回，不能证明 Top-5 其余结果都不相关。
- Eedi 数学数据只验证技术链路，不能推出 NBCOT 业务效果。
- API 成本、限流、可用性和跨供应商稳定性必须在 L2 前单独记录。

## 4. 一次性 L1 收尾施工包

施工顺序固定为 O1→O6；每包完成后必须有测试和报告。

### O1：冻结输入与实验边界

涉及：`evals/datasets/l1-v2/`、数据集 builder、manifest。

要求：

1. 固定 DB/Chroma hash、Provider/model/dimension/index_version。
2. 固定 Query、qrels、Top-K、RRF k、Assembler 配置、API batch/timeout。
3. 固定 dev/holdout；同一 `intervention_id` 不跨 split。
4. 新 run 使用新 ID，不覆盖历史报告。

验收：manifest 能回答使用哪份数据、索引、模型、配置和生成时间。

### O2：修复引用闭环

涉及：`src/rag_pipeline.py`、`evals/adapters/grounding.py`、Grounding runner/metrics/tests。

要求：

1. 每条 citation 都是 `(session_id, turn_id, speaker, quote_text)`。
2. citation 必须来自最终 Assembler evidence，不得仅凭 Session 查询自动生成前六轮对白。
3. 删除 `verifiable_in_duckdb=True` 导致的逐字段审计跳过。
4. 错 Session、错 Turn、错 speaker、改写 quote、未进入候选的 citation 必须失败或 `GROUNDING_DEGRADED`。

### O3：修复 Assembler 选择与证据保留

涉及：`src/reranker.py`、Assembler adapter/runner/tests。

要求：

1. misconception、strategy 独立选择和独立评分，不用跨槽位 Jaccard 惩罚压低互补卡。
2. 目标不在候选池只记 `retriever_miss`；候选存在但最终丢失才记 `assembler_drop`。
3. 保留真实 `final_evidence_turns`、`actual_prompt_context`、token budget。
4. 固定候选比较 `before=RRF` 与 `after=Assembler/Rerank`。
5. 生产选择逻辑不得读取 qrels、Gold Context 或目标卡 ID。

验收：

```text
assembler_drop_rate ≤0.10（有正候选 Case）
final_evidence_recall ≥0.90
slot precision/recall ≥0.90
budget violation = 0
evidence empty = 0
```

### O4：降低 API Embedding 延迟

涉及：`src/embedding_provider.py`、`src/retriever.py`、tests。

要求：

1. 同一 Query 的 raw/misconception/strategy/curriculum 向量合并为批请求。
2. cache key 必须包含 `provider/model/index_version/query_text`，只缓存成功向量。
3. 记录 request、cache hit、retry、tokens、P50/P95；不记录 key。
4. API 失败、超时、维度错误显式失败，不切回 deterministic。

验收：批返回顺序正确；失败不缓存；不同模型/版本不共享 cache；达到 Retrieval ≤300ms、Assembler ≤500ms。

### O5：修正 Fusion 与 fallback Chunk 检索

涉及：`src/retriever.py`、Rewrite/Fusion/Chunking runner。

要求：

1. 在 dev qrels 上配对比较 Raw、各 lane、rewrite-only RRF、raw-inclusive RRF。
2. 若 Raw 已强，使用 dev 验证过的 weighted/fallback 策略；不能凭一次 30 Query 结果删除 RRF。
3. fallback window 保持 Query/document 同一 Qwen 模型；增加 numeric/formula exact channel 诊断。
4. 输出按 category、slot、语言、数字/公式 slice 的 Recall/MRR/nDCG/P95。

验收：Fusion 相对 Raw 不退化；Turn Recall@3 ≥0.90；MRR ≥0.85；Chunk pointer/verbatim/boundary/coverage 仍 100%。

### O6：生成 L1 closeout report

报告顺序：Storage → Chunk Structure → Chunk retrieval → Grounding → Rewrite/Fusion → Retrieval → Assembler。

报告必须包含 aggregate/case/slice、hard gate/diagnostic、状态计数、所有 hash、API usage/cost/P95、失败 Case ID、`retriever_miss`/`assembler_drop` 和 paired delta。

L1 技术收尾门禁：

```text
Storage：ID parity=1.0，metadata=1.0，orphan=0
Chunk structure：pointer/verbatim/boundary/coverage=1.0，out_of_bounds=0
Retrieval：Candidate Recall@20≥0.98，Recall@5≥0.95，MRR≥0.85，nDCG@5≥0.90
Fusion：选定策略相对 Raw 不退化
Assembler：retention≥0.90，evidence recall≥0.90，budget/evidence empty=0
Grounding：各精度/召回指标≥0.95
Latency：Retrieval≤300ms，Assembler≤500ms
Provenance：构建和评测前后 artifact hash 不变
```

qrels 尚未升级为多相关人工 graded 时，最终标签只能是：

```text
GO_TO_L2_TECHNICAL
```

不能写成 `BUSINESS_QUALITY_PASSED`。

## 5. L2 DeepEval 入场标准

L2 只在 L1 达到 `GO_TO_L2_TECHNICAL` 或更高状态后开始；L2 不重新定义 L1 确定性指标。

### 5.1 测试对象

每个 Case 必须经过真实链路：

```text
raw query → expansion → retrieval → fusion → reranker/assembler
→ final context → generator → final answer + citations
```

Gold Context 只能做上限实验，不能进入被测生产调用。

### 5.2 L2 数据集

建议目录：

```text
evals/datasets/l2/
├── queries.jsonl
├── expected_answers.jsonl
├── expected_evidence.jsonl
├── split_manifest.json
└── README.md
```

每个 Case 至少包含 `case_id`、`query`、`intent`、expected answer/rubric、required evidence quads、expected facts、split、source sessions 和 slice metadata。

规则：dev/holdout 按 Session 隔离；Gold 只在评测控制面 join；每条 Case 绑定 dataset/qrels/artifact/model/config hash。

### 5.3 DeepEval 配置

DeepEval 是唯一语义评测框架，使用独立 evaluation LLM：

```text
DEEPEVAL_MODEL
DEEPEVAL_BASE_URL
DEEPEVAL_API_KEY_ENV
DEEPEVAL_TEMPERATURE
DEEPEVAL_SEED（若 provider 支持）
```

依赖缺失、Judge 不可用或请求失败只能记 `UNMEASURED/ERROR`，不得以关键词规则填补 DeepEval 分数；报告不写 API key。

### 5.4 L2 首轮指标

| 指标 | 评估对象 | 首轮门槛 |
|---|---|---:|
| Faithfulness | 答案是否被 final context 支持 | ≥0.90 |
| Answer Relevancy | 答案是否回答 Query | ≥0.85 |
| Contextual Recall | final context 是否覆盖必要事实 | ≥0.90 |
| Contextual Precision | final context 排序/噪声 | ≥0.85 |
| Deterministic citation audit | 四元组和候选授权 | ≥0.95 |
| Pedagogical GEval | 教研/教学适配性 | ≥0.80，需人工校准 |

首轮必须用人工小样本校准 Judge 偏差；校准前标 `calibration_pending`。

### 5.5 Gold/Real 双轨

每个 Case 跑两次：

```text
Gold Context → Generator → DeepEval
Real final Context → Generator → DeepEval
```

解释固定为：Gold 高/Real 低表示上游链路损失；两者都低表示 Generator 或数据问题；不能只凭一个分数归因。

### 5.6 L2 发布条件

```text
DeepEval 依赖/Judge 可用
Gold/Real 两轨均有真实结果
Faithfulness、Answer Relevancy、Context Recall/Precision 达门槛
Citation audit ≥0.95
无未解释 ERROR
holdout 不低于 dev 的预设容差
```

## 6. 测试与回滚

```powershell
$env:PYTHONPATH='.'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp reports\runtime\pytest-l1-closeout `
  tests\evals tests\test_embedding_provider.py tests\test_storage_manager.py
.venv\Scripts\python.exe -m compileall -q src evals scripts tests
git diff --check
```

所有新索引必须新目录写入；切换前停止 CLI、记录新旧 hash、跑 Storage/L1 smoke；失败恢复旧目录，不删除新目录。

## 7. 最终完成定义

施工结束时 closeout report 必须明确输出：

```text
GO_TO_L2_TECHNICAL
BUSINESS_QUALITY_PASSED
或 BLOCKED（失败 Case、责任模块、下一动作、复现命令）
```

不得使用“基本完成”“暂时可用”“后续再看”等状态。进入 `GO_TO_L2_TECHNICAL` 后，由 L2 DeepEval 接管语义质量评价，L1 不再反复调整同一门槛。
