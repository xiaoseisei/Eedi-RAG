# Eedi-RAG L1 组件评测施工计划与操作说明书

> 文档版本：1.0  
> 日期：2026-09-02  
> 基线：L0 使用 DeepEval 作为语义评测框架，配置 LLM 作为评测模型  
> 本阶段：先交付确定性 L1 组件评测；DeepEval 只建立外部适配边界，不进入 `src/`

## 1. 交付结论

本阶段将 L1 从零散脚本改造成了独立、严格、可审计的评测控制面：

```text
真实 artifact / 固定数据集
          │
          ▼
严格评测契约 ──> 确定性 Metrics ──> 组件 Runner ──> 版本化报告
          │                                      │
          └──────── DeepEval Adapter（语义阶段） ┘
```

当前已经完成：

| 能力 | 状态 | 说明 |
|---|---|---|
| 统一 `RunContext`、`MetricObservation`、状态机 | 已实现 | 严禁伪造 SUCCESS，禁止空分母默认为 100% |
| Storage / Index 一致性 Runner | 已实现并连接真实 artifact | DuckDB 只读；Chroma SQLite 使用 `mode=ro&immutable=1` |
| Grounding / Citation Runner | 已实现 | 等待正式四元组数据集接入 |
| Chunking Runner | 已实现 | 支持 Session/Turn Recall、Turn MRR、token inflation、P95 |
| Query Rewrite Runner | 已实现 | 支持数字/公式/实体/意图/领域注入和下游检索 lift |
| Retrieval Runner | 已实现 | 支持 graded qrels、case/aggregate/slice、P95 |
| Assembler Runner | 已实现 | 支持 retention、evidence recall、precision、noise、预算、压缩、nDCG gain |
| Suite CLI 和报告写入 | 已实现 | `PASSED` 返回 0，`BLOCKED` 返回 2，执行错误返回 1 |
| DeepEval 可用性与 TestCase 映射边界 | 已实现 | 当前未安装，语义指标仍为 `UNMEASURED` |
| Generator 的 DeepEval 正式基线 | 未执行 | 当前 E2E Context 契约仍阻断，且 evaluation model 未配置 |
| 正式 Retrieval / Assembler 业务基线 | 未执行 | 需要版本化 graded qrels；旧 50-query 单目标 Session 只能作为诊断集 |

“Runner 已实现”和“当前业务达到门槛”必须分开。未具备真实数据或外部模型条件的指标一律保持 `UNMEASURED`，不填 0，也不填模拟分数。

## 2. 施工目标与边界

### 2.1 本阶段目标

1. 每个组件可以脱离整条 RAG 链路独立测量。
2. 每个数值都能回溯到 Case、qrels、artifact 和系统配置。
3. 确定性故障先于语义评测暴露，避免用 LLM Judge 掩盖 Schema、索引和引用错误。
4. 组件变更后可以通过配对数据判断优化方向。
5. 评测代码不改变被测系统的排序、输出或降级行为。

### 2.2 明确不做

- 不把 DeepEval import 到 `src/`。
- 不增加 `evaluation_mode` 业务分支。
- 不将 Gold、expected answer 或 qrels 传入被测组件。
- 不用 Mock 结果冒充真实基线。
- 不把旧 RAGAS/自研 Judge 报告转换成 DeepEval 分数。
- 不因报告被阻断而降低硬门槛。

## 3. 代码结构

```text
evals/
├── contracts.py                 # 严格输入、Observation、Report 契约
├── metrics/
│   ├── retrieval.py             # Recall/MRR/nDCG/P95/Wilson CI
│   └── citation.py              # 四元组、角色、逐字证据审计
├── runners/
│   ├── storage.py
│   ├── grounding.py
│   ├── chunking.py
│   ├── rewrite.py
│   ├── retrieval.py
│   ├── assembler.py
│   └── common.py                # case -> aggregate/slice 汇总
├── adapters/
│   └── storage.py               # 当前 DuckDB/Chroma 的只读适配器
├── deepeval_adapter.py          # 延迟 import、可用性探测、TestCase 映射
├── suite.py                     # 多组件统一执行
├── reporting.py                 # 原子写入报告，不覆盖已有 run_id
├── cli.py                       # 固定 JSON Suite 入口
├── live_storage.py              # 当前真实 Storage artifact 入口
└── config_hash.py               # 配置规范化哈希

tests/evals/                      # L1 控制面的独立测试
reports/eval/{run_id}/            # 每次真实运行的不可覆盖报告
```

生产目录 `src/` 没有引入 DeepEval，也没有为了评测增加测试专用行为。

## 4. 统一真实性契约

### 4.1 RunContext

每次运行必须冻结：

```text
run_id
dataset_version
qrels_version
system_config
system_config_hash = SHA256(canonical JSON(system_config))
```

`system_config` 禁止出现 token、password、cookie、secret、authorization 和真实 API key。允许记录 `api_key_env` 这一类环境变量名称，但不能记录变量值。

### 4.2 Observation

每个指标写成独立 Observation：

```text
run_id / case_id / layer / stage / scope
metric_name / value / threshold / operator / status
hard_gate / evaluator_type
dataset_version / qrels_version / system_config_hash
slices / sample_size / pass counts / confidence interval
evidence_refs / error_type / error_message / created_at
```

状态只有：

| 状态 | 含义 | 是否可进入均值 |
|---|---|---:|
| `SUCCESS` | 已真实测量且满足阈值 | 是 |
| `FAILED` | 已真实测量但未满足阈值 | 是 |
| `UNMEASURED` | 分母为零、依赖未配置或条件不足 | 否 |
| `DEGRADED` | 显式授权的真实替代路径 | 单列，不混合 |
| `ERROR` | 被测组件或评测器异常 | 否 |

契约会反向校验状态：例如 `value=0.2, threshold=0.85, operator=gte, status=SUCCESS` 会直接产生 ValidationError。

### 4.3 发布状态

```text
任一 hard_gate != SUCCESS  -> BLOCKED
所有 hard_gate == SUCCESS  -> PASSED
```

非硬门禁诊断指标失败不会自动放行硬门禁，也不会改变真实状态。

## 5. 各模块施工说明

### 5.1 Storage / Index

输入：

- DuckDB 中的 Session、Turn 和三类 Chunk ID。
- Chroma 中的三类 embedding ID 与 metadata。
- 每类索引的 required metadata fields。

实现：

```text
id_parity
  = 1 - mismatched_distinct_ids / expected_ids

orphan_rate
  = parent_id 不存在的记录数 / 所有受审计子记录数

metadata_contract_rate
  = required fields 均存在且有意义的索引记录数 / 索引记录数
```

空 expected set、空 index 或空关系记录不会返回 100%，而是 `UNMEASURED` 并阻断。

门槛：

| 指标 | 门槛 | 类型 |
|---|---:|---|
| ID parity | 1.00 | 硬门禁 |
| Metadata contract | 1.00 | 硬门禁 |
| Orphan rate | 0.00 | 硬门禁 |

当前路径级适配器不会初始化 `DualEngineStorageManager`，原因是初始化过程包含建表和 `get_or_create_collection`。DuckDB 以 `read_only=True` 打开；Chroma 直接读取 `chroma.sqlite3`，不调用 `PersistentClient`。

### 5.2 Grounding / Citation

输入四元组：

```text
(session_id, turn_id, speaker, quote_text)
```

每个 Case 包含：

- 被测组件输出的 citations。
- DuckDB 权威 Turn。
- required evidence。
- `exact` 或 `substring` quote 匹配模式。
- 可选的预期角色与业务 slice。

指标：

| 指标 | 计算 | 门槛 |
|---|---|---:|
| Citation Precision | 完整四元组有效引用 / 输出引用 | 100% |
| Citation Recall | 被有效引用覆盖的 required evidence / required evidence | 100% |
| Role Accuracy | Session/Turn 对应且角色正确 / 输出引用 | 100% |
| Quote Grounding Precision | 引文可在权威 Turn 找到 / 输出引用 | 100% |

输出 citations 为空时，Precision 和 Role Accuracy 是 `UNMEASURED`，Recall 是真实的 0；不能用“没有错误引用”解释成 Precision=100%。

当前卡片 Schema 将 `source_turn_ids` 和 quote list 分开保存，缺少稳定的逐引用四元组映射。因此正式接入前需要生成显式 Citation artifact，不能在 adapter 中根据 Gold 反向挑选一个“看起来正确”的 Turn。

### 5.3 Chunking

输入是 Query、真实 Chunk 输出顺序、目标 `(session_id, turn_id)` 集合和 token 统计。Runner 计算：

```text
Session Recall@K = 命中目标 Session 数 / 目标 Session 数
Turn Recall@K    = 被 top-K 覆盖的目标 Turn 数 / 目标 Turn 数
Turn MRR        = 第一个覆盖目标 Turn 的 Chunk 的倒数排名
inflation_ratio = emitted chunk tokens / original tokens
```

默认门槛为 Turn Recall@3 ≥95%、Turn MRR ≥0.90；Session/Turn Hit@1 是诊断值，inflation 与 P95 是容量/性能诊断。Chunk Runner 不把 token 膨胀率当作相关性质量，也不会用空 Chunk 集合产生“完美”分数。

### 5.4 Query Rewrite

输入必须显式标注受保护 token（数字、公式、选项名）、关键实体、意图标签来源以及改写前后的同一 qrels 检索结果：

| 指标 | 默认门槛 | 说明 |
|---|---:|---|
| Protected token preservation | 100% | 大小写不敏感的精确子串检查 |
| Entity preservation | ≥99% | 关键学科实体不可丢失 |
| Intent preservation | ≥95% | 需人工或 DeepEval 标签；无标签为 `UNMEASURED` |
| Domain injection coverage | 诊断 | 仅在该 Case 声明需要注入时计分 |
| Retrieval MRR/Recall@5 lift | ≥0 | 与 identity/raw 做配对比较 |

“改写字符串包含原始 Query”不等于意图保持。改写 runner 的 lift 指标只能说明方向，正式保留策略还要在固定 holdout 上做显著性和成本比较。

### 5.5 Retrieval

输入：

```text
case_id
query
ranked_ids（真实组件输出顺序）
qrels: document_id -> relevance(0..3)
latency_ms
slices（card_type / subject / difficulty / mode 等）
```

实现指标：

| 指标 | 默认门槛 | 硬门禁 |
|---|---:|---:|
| Candidate Recall@20 | ≥0.98 | 是 |
| Recall@3 | ≥0.90 | 是 |
| Recall@5 | ≥0.95 | 是 |
| Recall@20 | ≥0.98 | 是 |
| Precision@5 | 诊断 | 否 |
| Noise ratio@5 | ≤0.05 | 是 |
| MRR | ≥0.85 | 是 |
| nDCG@5 | ≥0.90 | 是 |
| P95 latency | ≤150 ms | 是 |
| Recall@1 | 诊断 | 否 |

Runner 同时输出：

- 单 Case 分数与 pass/fail。
- 全集 aggregate。
- 每个一维 slice 的 aggregate。
- pass count、unmeasured count 和 95% Wilson 区间。

正式 qrels 必须允许多个相关文档并使用 0/1/2/3 等级。当前 `scripts/eval_50_queries.py` 的单一目标 Session 可用于回归诊断，但不能签署正式 nDCG 与 Final Evidence Recall 基线。

### 5.6 Reranker / Context Assembler

输入固定候选集，不重新调用 Retriever 或 Generator：

```text
input_ranked_ids
final_context_ids
graded qrels
required_evidence_ids
candidate / assembled chars
estimated / max tokens
evidence_count
latency_ms
```

指标：

| 指标 | 默认门槛 | 说明 |
|---|---:|---|
| Candidate recall retention | ≥0.98 | Retriever 有相关候选时，Assembler 是否保留 |
| Final evidence recall | ≥0.95 | required evidence 是否进入最终 Context |
| Selection precision | ≥0.95 | 最终选择中相关文档占比 |
| Noise ratio | ≤0.05 | 最终选择中无关文档占比 |
| Token budget violation | 0 | `estimated > max` 的比例 |
| Evidence empty rate | 0 | 最终证据为空的比例 |
| Compression ratio | ≥0 | 容量诊断，不是质量代理 |
| Rerank nDCG gain | ≥0 | 初期诊断；需要固定候选消融校准 |
| P95 latency | ≤300 ms | 生成前链路目标 |

如果 Retriever 根本没有提供相关候选，则 retention 标为 `UNMEASURED` 并显式归因上游；不会把它错误计作 Assembler 的 0 分或 1 分。

### 5.7 Generator / DeepEval

本阶段只完成两个边界：

1. `probe_deepeval()`：验证 package、配置模型和凭据环境变量是否存在，不返回语义分数。
2. `DeepEvalCasePayload`：强制 `retrieval_context` 使用 Generator 实际看到的最终 Context。

映射为：

```python
LLMTestCase(
    input=payload.input,
    actual_output=payload.actual_output,
    expected_output=payload.expected_output,
    retrieval_context=payload.final_retrieval_context,
)
```

当前环境 `deepeval=False`，evaluation model 和 API credential 也未配置，所以 Generator 的 Faithfulness、Answer Relevancy、Contextual Precision/Recall 与教学 G-Eval 均不得出分。

## 6. 数据集施工规范

### 6.1 必备 artifact

| Artifact | 最小字段 | 版本要求 |
|---|---|---|
| Grounding set | 四元组输出、权威 Turn、required evidence | `grounding-vN` |
| Retrieval set | Query、graded qrels、slice、split | `retrieval-vN` + `qrels-vN` |
| Assembler set | 固定候选排序、qrels、required evidence、预算 | `assembler-vN` |
| Generator set | input、Gold output、Gold Context、真实 final Context | `generator-vN` |

### 6.2 防泄漏

- 按 `intervention_id` 做 train/dev/test 隔离。
- 同一 Session 的 Chunk 不能跨 split。
- 调参只能看 train/dev；签署基线只看固定 holdout。
- qrels 不得进入生产检索、重排或生成调用。
- 数据集更新后提升版本，不覆盖旧数据集后继续沿用旧基线。

### 6.3 统计签署

每个指标同时报告：

```text
case score
case threshold/pass
aggregate score
dataset pass rate
unmeasured rate
slice score/pass rate
confidence interval
```

10/10 或 30/30 不能证明工业失败率低于 5%。95% 置信度、零失败条件下，至少需要约 59 个独立 Case，且仍需检查 slice 覆盖与样本独立性。

## 7. 执行手册

### 7.1 运行评测控制面测试

当前 Windows 环境的用户级 pytest 临时目录权限异常，因此显式指定工作区临时根：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp reports\runtime\pytest-l1 `
  tests\evals
```

预期：所有 L1 控制面测试通过。临时目录统一放在 `reports\runtime`，目录名可以更换以避免并行任务冲突。

### 7.2 对当前 Storage artifact 做真实只读评测

```powershell
.\.venv\Scripts\python.exe -m evals.live_storage `
  --db data\db\tutoring_knowledge.duckdb `
  --chroma data\chroma `
  --output-root reports\eval `
  --run-id l1-storage-YYYYMMDD-NNN
```

该命令会在读取前后分别计算完整 artifact SHA256。如果读取过程中源 artifact 发生任何变化，命令抛错且不生成基线报告。

### 7.3 执行版本化多模块 Suite

输入文件必须满足 `l1-component-suite/v1`；仓库提供的 `evals/datasets/l1-component-suite.example.json` 只用于契约演示，不代表业务基线：

```powershell
.\.venv\Scripts\python.exe -m evals.cli `
  --input evals\datasets\l1-component-suite.example.json `
  --output-root reports\eval
```

计算 `system_config_hash`：

```powershell
.\.venv\Scripts\python.exe -m evals.config_hash `
  --input evals\datasets\system-config-v1.json
```

将输出的 SHA256 写入 Suite 的 `context.system_config_hash`，同时原样保留不含凭据的 `context.system_config`。两者不一致时评测拒绝执行。

### 7.4 退出码

| 退出码 | 含义 | CI 行为 |
|---:|---|---|
| 0 | 所有硬门禁通过 | 可继续下游步骤 |
| 1 | 输入、评测器或 I/O 执行错误 | 失败并排错 |
| 2 | 评测真实完成，但至少一个硬门禁失败或未测 | 阻断发布 |

退出码 2 不是工具故障，也不能在 CI 中转换为成功。

## 8. 报告与审计

每个 `run_id` 只允许写一次：

```text
reports/eval/{run_id}/run.json
reports/eval/{run_id}/observations.jsonl
reports/eval/{run_id}/failures.jsonl
reports/eval/{run_id}/slices.jsonl
```

- `run.json` 保存配置、版本、状态计数和完整 Observation。
- `observations.jsonl` 便于导入 DuckDB/数据仓库做交叉分析。
- `failures.jsonl` 包含 FAILED、UNMEASURED、DEGRADED、ERROR。
- `slices.jsonl` 只保存 slice 级指标。
- 已存在的 `run_id` 不覆盖，防止历史证据被静默替换。

## 9. 当前真实执行证据

运行：`l1-storage-baseline-20260902-003`

Artifact SHA256：

```text
88750616571f5d7375af188967ea1604d0910262cae42d78dc2f0c05116a546e
```

结果：`BLOCKED`，命令退出码 `2`。

| Case | 指标 | 当前值 | 门槛 | 状态 |
|---|---|---:|---:|---|
| student_misconceptions | ID parity | 100% | 100% | SUCCESS |
| student_misconceptions | metadata contract | 5% | 100% | FAILED |
| tutor_strategies | ID parity | 100% | 100% | SUCCESS |
| tutor_strategies | metadata contract | 5% | 100% | FAILED |
| fallback_windows | ID parity | 100% | 100% | SUCCESS |
| fallback_windows | metadata contract | 100% | 100% | SUCCESS |
| all audited child records | orphan rate | 0% | 0% | SUCCESS |

两类卡片各有 95/100 条索引记录不满足当前 required metadata contract。因此下一步仍然是迁移或重建卡片索引，而不是优先调整生成 Prompt。

同一当前 artifact hash 下的 Retrieval 诊断运行：`l1-retrieval-legacy50-20260902-002`，100 Case（Raw/RRF 各 50），退出码 `2/BLOCKED`。该数据集仍是单目标 Session qrels，不能签署正式 graded 基线，但可用于优化方向：

| 指标 | 当前值 | 门槛 | 结论 |
|---|---:|---:|---|
| Candidate Recall@20 | 56% | ≥98% | FAILED |
| Recall@3 | 24% | ≥90% | FAILED |
| Recall@5 | 37% | ≥95% | FAILED |
| MRR | 0.24 | ≥0.85 | FAILED |
| nDCG@5 | 0.26 | ≥0.90 | FAILED（单目标 qrels 仅诊断） |
| Noise ratio@5 | 93% | ≤5% | FAILED |
| P95 latency | 78.30 ms | ≤150 ms | SUCCESS |

Slice 方向：RRF 的 Recall@5 为 48%，Raw 为 26%；错因卡片为 52%，策略卡片仅 22%。因此当前优先级是补齐 graded qrels、修复策略索引/表示与候选召回，不是生成 Prompt 或延迟优化。

### 9.1 非侵入读取事件记录

初版路径适配器曾用 `chromadb.PersistentClient` 打开真实 Chroma 目录，并只调用读取接口。前后 artifact hash 不一致，评测按保护逻辑立即中止，没有生成报告。

这说明 `PersistentClient` 的“读取调用”不能被视为文件系统只读；该次调用已经改动过源 Chroma artifact，无法事后伪称完全无侵入。当前能确认的是：改动后业务集合数量仍为 100 / 100 / 719，最终 ID parity 为 100%，但没有改动前后的数据库级 diff，不能断言内部只变化了哪张 Chroma 系统表。

整改：

- 路径适配器不再 import 或初始化 Chroma 客户端。
- 直接读取 `chroma.sqlite3` 的 collections / segments / embeddings / metadata 表。
- SQLite URI 固定为 `mode=ro&immutable=1`。
- 每次评测前后计算全 artifact hash；变化即 ERROR，不写报告。

上面的正式 `003` 报告由整改后的 immutable 读取路径生成，读取前后 hash 相同。`001`、`002` 是索引 hash 变化前的过渡报告，仅保留作施工过程证据，不与 `003` 或之后的结果横向比较。

### 9.2 Storage metadata 迁移（隔离副本）

为修复卡片索引中缺失的必需字段，新增 `evals/storage_migration.py` 与
`scripts/migrate_storage_metadata.py`。迁移的唯一事实源是 DuckDB 原卡片表；
迁移计划只比较 `COLLECTION_SPECS` 中的 required metadata 字段，不会因为可选
字段缺失而重写整库。

先做 dry-run（不会写入任何 Chroma 文件）：

```powershell
$env:PYTHONPATH='.'
.venv\Scripts\python.exe scripts/migrate_storage_metadata.py `
  --db data\db\tutoring_knowledge.duckdb `
  --chroma data\chroma `
  --diff reports\eval\storage-metadata-migration-dryrun-YYYYMMDD.json
```

若 diff 经人工审核，必须指定一个不存在的新目录执行 apply；工具会先复制源
目录，再仅更新副本，并拒绝覆盖已存在的目标：

```powershell
.venv\Scripts\python.exe scripts/migrate_storage_metadata.py `
  --db data\db\tutoring_knowledge.duckdb `
  --chroma data\chroma `
  --apply --target-chroma reports\eval\chroma-migrated-YYYYMMDD-vN
```

本次真实 dry-run 报告：
`reports/eval/storage-metadata-migration-dryrun-20260902-v2.json`，共 6 条记录
需要更新（3 条 misconception 各补 3 个字段；3 条 strategy 各补 4 个字段），
没有缺失或多余 ID。隔离副本 `reports/eval/chroma-migrated-20260902-v1` 已执行 apply；源
`chroma.sqlite3` SHA-256 前后均为
`CBB8A9F4D955304D0C06A5AFE5A978412D0B815CAF82F1053CB347D2C947D2FC`。

副本 Storage 复测报告：`reports/eval/l1-storage-migrated-20260902-v1`，
`7 SUCCESS / 0 FAILED / 0 UNMEASURED`，`release_status=PASSED`，退出码 `0`。
该报告证明 metadata contract、ID parity 与 orphan 门禁在迁移副本上通过；源目录
仍保持只读基线，是否切换生产索引需另行审批。

随后在释放 `src/cli.py` 文件锁后完成受控切换。当前生产 Chroma 为
`data/chroma`，完整 artifact hash 为
`af6798c5c4aa815b1515bfa1da697fc0849c4fa966a8d3ada87d0a9d427ae839`；切换前的原始
目录已保存在
`data/chroma-backups/chroma-pre-migration-20260902-105043`，备份 hash 为
`4a035072ab418fd72a928aa6f9cb9f8e766472ba03cf030a7fdfb6d534a0aed3`。生产路径复测
报告为 `reports/eval/l1-storage-production-20260902-105043`，结果为
`7 SUCCESS / 0 FAILED / 0 UNMEASURED`、`release_status=PASSED`、退出码 `0`。
备份未删除，仍可通过反向重命名恢复。

## 10. 指标到施工方向

| 观察 | 归因 | 下一项施工 |
|---|---|---|
| ID parity <100% | 双写、迁移或索引遗漏 | 对比 mismatched IDs，重建对应集合 |
| Metadata <100% | 索引 Schema 落后于当前 Generator 契约 | 从 DuckDB 原卡片重建 Chroma metadata |
| Orphan >0 | 外键/摄取顺序问题 | 阻断摄取，修复 parent 关系 |
| Grounding role/quote <100% | 抽取或证据映射错误 | 修复四元组，不调 Judge 阈值 |
| Candidate Recall@20 低 | 数据、Chunk、Rewrite、Embedding | Oracle candidate 与消融实验 |
| Candidate 高而 MRR/nDCG 低 | 排序/RRF/Reranker | 固定候选做 paired rerank |
| Assembler retention 低 | MMR、槽位、截断 | Oracle assembler 对照 |
| Gold Context 生成仍低 | Generator/Prompt/模型 | 再引入 DeepEval 做语义实验 |
| Gold Context 高、真实 Context 低 | 上游 Context 链路 | 优先 Retrieval/Assembler |

## 11. 后续施工顺序

### P0：先清除真实硬阻断

- [x] 从 DuckDB 原卡片生成 metadata dry-run diff，并在隔离副本完成迁移。
- [x] 迁移副本 Storage L1 达到 metadata contract 100%、ID parity 100%、orphan 0。
- [x] 审批并将迁移副本作为新的生产 artifact；切换前重新计算 artifact hash。
- [ ] 修复卡片引用，使每条证据拥有显式四元组。

验收：Storage 三类 ID parity/metadata 均 100%，orphan=0；Grounding 四项均 100%。

### P1：建立正式 Retrieval 数据基线

- [ ] 将 50-query 常量迁移为版本化 JSON。
- [ ] 每个 Query 标注多个文档与 0/1/2/3 relevance。
- [ ] 按 Session 隔离 split。
- [ ] 运行 Raw、Rewrite、RRF 的同 Case 配对实验。
- [ ] 输出 card_type / subject / difficulty / mode slices。

验收：数据集版本、qrels 版本、系统配置完整，Candidate Recall@20、Recall、MRR、nDCG、P95 均真实出值。

### P1：接入真实 Assembler Trace

- [ ] 暴露候选 IDs、最终 Context IDs、实际 token 估算和 stage latency。
- [ ] 不暴露 Gold 给 Assembler。
- [ ] 使用同一 qrels 计算 before/after。
- [ ] 进行 `lambda` 网格与 Oracle assembler 配对实验。

验收：能区分 Retriever miss 与 Assembler drop，所有预算/空证据指标可审计。

### P2：启用 DeepEval Generator 基线

- [ ] 锁定与 Python 版本兼容的 DeepEval 版本。
- [ ] 配置独立 evaluation model 与凭据环境变量。
- [ ] 固定 Judge 模型版本、temperature、seed（若支持）和 rubric。
- [ ] 先跑 Gold Context，再跑真实 final Context。
- [ ] 使用人工标注集校准 G-Eval 阈值和偏差。

验收：真实 DeepEval 版本和 Judge 信息写入报告；缺少依赖时仍是 `UNMEASURED`，不得启用关键词兜底冒充语义分数。

## 12. Definition of Done

一个 L1 模块只有同时满足以下条件才算完成：

- [ ] 输入数据来自真实 artifact 或明确标记的测试 fixture。
- [ ] 数据集、qrels、系统配置均有版本与哈希。
- [ ] Case、aggregate、slice 和未测率可查询。
- [ ] 空分母、依赖缺失和组件异常均被显式记录。
- [ ] 硬门禁阈值有业务含义，不能为了绿灯临时降低。
- [ ] Runner 专项测试通过。
- [ ] 对当前真实数据至少生成一次不可覆盖报告。
- [ ] 报告退出码与 release status 一致。
- [ ] 生产代码不依赖 DeepEval，Gold 不进入 SUT。
- [ ] 优化前后使用同一 Case 做 paired comparison。

当前 Storage 模块满足“实现与真实运行”条件，但真实数据硬门禁失败；Grounding、Chunking、Rewrite、Retrieval、Assembler 满足“Runner 实现”条件，尚待正式数据/Trace 接入后才能签署业务基线。

最终验证证据：

- L1 专项测试：`21 passed`。
- `compileall evals`：通过。
- `git diff --check`：通过。
- 示例 Suite：`PASSED`，`51 SUCCESS / 3 UNMEASURED`，退出码 `0`；其中 `UNMEASURED` 是无标签的 Rewrite 非适用指标，不参与均值。
- 当前 Storage artifact：`l1-storage-baseline-20260902-003`，`5 SUCCESS / 2 FAILED`，退出码 `2`。
- Storage metadata dry-run：`storage-metadata-migration-dryrun-20260902-v2.json`，6 条记录待补字段。
- 迁移副本 Storage：`l1-storage-migrated-20260902-v1`，`7 SUCCESS / 0 FAILED`，退出码 `0`。
- 当前 Retrieval 诊断：`l1-retrieval-legacy50-20260902-002`，100 Case，`442 SUCCESS / 628 FAILED`，退出码 `2`。
- 全量仓库测试：`109 passed, 1 failed`。唯一失败是预存的 `tests/test_expand_knowledge_base.py::test_llm_expansion_extractor_uses_strict_shared_gate`，与本次 `evals/` 改动无交集；本阶段没有修改该生产代码或测试以掩盖失败。
