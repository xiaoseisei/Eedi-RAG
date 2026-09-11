# Eedi-RAG Graph Gold 交接文档

> 写给没有任何上下文的新对话。先读本文件，再开始操作。本文记录的是 2026-09-10 的真实状态；不要把历史 L1/L2 文档中的旧结论当作当前 Graph Gold 结论。

## 1. 当前任务

工作区：`E:\PIAgent\10-projects\AgentLearn\Eedi-RAG`  
Git 根目录：`E:\PIAgent`（包含多个项目，工作树很脏；只能处理 Eedi-RAG 相关文件，绝不可 `git reset --hard` 或清理整个根仓库。）

正在建设并评测一个 Business Graph RAG 路线：

```text
自然语言问题
→ Router / Planner
→ SQL 或 Concept Graph 定范围
→ Card（错因 / 导师策略）候选
→ Parent Card
→ W7/S3 Window
→ Turn 级原文审计 / 引用
→ 结构化业务响应
```

数据源是 Eedi 辅导对话的本地 DuckDB，不是 NBCOT 私有数据。不可把任何结果表述为 NBCOT 业务结论或学习效果证明。

当前重点不是继续堆 Graph，而是修正一个 Gold 与可观测用户输入不一致的问题，并把原有 embedding 正确接入题目级多 Session 的 RAG 回答。

## 2. 核心结论：`mixed-misconception-intervention-003` 的 Gold 有身份歧义

当前 query：

```text
围绕真实题目 “This pie chart shows how the employees of a company travel to work. ...”
进行 mixed misconception intervention 分析。
```

该题对应 `question_id=76888`，源库中有至少以下 6 个同题真实会话：

```text
282, 522, 3767, 5669, 9242, 10023
```

旧 Gold 却强制要求：

```text
session_522_misconception
Session 522 / Turn 12: "ob wait i bet it now get"
Window: session_522_win_7_13
```

但 query 中没有 Session ID、没有 Session 522 独有学生原话、没有能区分 522 的错因关键词。因此：

- embedding / rerank 可以找到语义相关的真实同题会话；
- embedding / rerank **不能证明用户指的是 522**；
- 无身份锚点时强行回答“Session 522 的学生……”属于无依据身份归因（幻觉）；
- 当前严格 Gold 的这一条不能作为“系统必须猜中 522”的正确门槛。

之前的完整评测因此为 `82/83`；唯一失败正是该 case。它不是普通召回失败，而是 Gold 可辨识性契约不成立。

## 3. 用户已确认的修订方向

用户明确认可：这题应放宽为同题多 Session 场景，并用来测试原本 RAG 的真实能力。

目标契约：

```text
无 Session 身份锚点 + 题干对应多个真实 Session
→ QUESTION（题目级）而非 CASE（某一次唯一会话）
→ embedding / lexical / rerank 召回同题 Card 候选
→ 选择一个或多个真实代表性 Session
→ Card、Window、Turn 引用必须全部来自同一真实 Session 链
→ 明确暴露：session_identity = NOT_UNIQUELY_IDENTIFIED
→ 不得声称该代表 Session 就是用户指定的 Session 522
```

身份锚点存在时仍必须严格：

```text
“Session 522 ...” 或含 Session 522 独有学生原话
→ 严格绑定 Session 522
→ 精确 Card / Window / Turn citation
```

建议的响应/trace 字段：

```json
{
  "scope": "QUESTION",
  "evidence_mode": "SEMANTIC_REPRESENTATIVE",
  "session_identity": "NOT_UNIQUELY_IDENTIFIED",
  "candidate_sessions": [282, 522, 3767, 5669, 9242, 10023],
  "selected_session_ids": [282]
}
```

代表性回答的显示限制应明确类似：

```text
该题在库中对应多个课堂；以下为语义相关的真实代表性课堂证据，
不表示已唯一识别用户指向的某一场会话。
```

## 4. 已完成的工作

### 4.1 Graph 拓扑

已生成候选 Concept Graph：

- `MISCONCEPTION_CONCEPT`
- `STRATEGY_CONCEPT`
- `TYPE_OF`
- `CO_OCCURS_WITH`

候选图：

```text
reports/staging/session-card-graph-v1/graph-concept-topology-20260909.duckdb
Nodes: 42,961
Edges: 87,993
TYPE_OF: 3,331
CO_OCCURS_WITH: 1,576
```

状态是 `READY` 但仍 `PENDING_GOLD_REAPPROVAL`。不得把候选图表述为已获生产或 Gold 批准。

### 4.2 Business Graph 路线

已新增/修改的核心文件：

- `src/business_graph_pipeline.py`
- `src/business_graph_metrics.py`
- `src/business_graph_trace.py`
- `src/business_graph_embedding.py`
- `src/business_models.py`
- `scripts/evaluate_business_graph.py`
- `scripts/export_business_graph_dataset.py`
- `src/cli.py`

支持 `ask_business()`、Router → Planner → SQL/Graph → Card → Parent Card → W7/S3 → Turn audit 的 trace。默认 CLI 路线仍保持 legacy，Graph route 不能静默替换默认生产路径。

### 4.3 Embedding/RAG 接入与安全边界

`src/business_graph_embedding.py` 使用已有 `DualMetricRetriever` 的 dense/BM25/RRF 能力，但只在 isolated Chroma copy 上运行。

- 禁止用 `PersistentClient` 打开或改写 `data/chroma`；历史上已发生 Chroma mutation incident。
- dense 分数只能扩大 Card 候选和辅助排序；不能作为 SQL 统计事实或 citation 授权依据。
- Card metadata 必须重新绑定回源 DuckDB；无源绑定的 dense hit 必须报错。
- 当前排序原则是精确题干/数字/公式 → 题干前缀 → lexical overlap → 稳定 source order → dense score。对于完全同题的 6 个 Session，仍可能按 source order 选代表；这不等于身份识别。

### 4.4 Chroma 稳定性和测试

已修复过 in-memory Chroma 污染、embedding dimension 不一致、HNSW 可见性竞态、stale collection handle、storage/retriever 入口不一致等问题。

最近一次完整测试曾为：

```text
427 passed
```

注意：后续改动后必须重新运行并如实记录结果，不能复用该数字当作新改动的测试证据。

### 4.5 Gold 派生评测集

已从 approved Gold 无损拆分：

```text
evals/datasets/business-graph-v1/
  queries.jsonl                 # 唯一允许传给 SUT 的输入
  expected_facts.jsonl          # evaluator-only
  expected_graph_paths.jsonl    # evaluator-only
  expected_evidence.jsonl       # evaluator-only
  split_manifest.json
```

旧源 Gold：

```text
data/business/business-graph-gold-v1.jsonl
SHA256 d01624167376c0fb93ab4883d762531118e833e6b6533d8ef7e3cece4d121875
```

拆分 manifest 的 source hash 与该源 Gold 完全一致。此前没有修改源 Gold，只创建了派生视图。

### 4.6 最近严格评测证据

报告：

```text
reports/eval/business-graph-v1-split-dataset-full-20260910/report.json
```

条件：83 cases、candidate Concept Graph、canonical Chroma replacement、isolated-copy embedding、确定性 route、无 Qwen reranker、无 generator。

| 指标 | 结果 |
| --- | ---: |
| Router exact match | 100% |
| Planner exact match | 100% |
| SQL/Graph scope confinement | 100% |
| Card evidence validity | 98.7952% |
| Window coverage | 98.7952% |
| Prompt Window coverage | 98.7952% |
| Citation precision | 100% |
| Citation recall | 98.7952% |
| Role accuracy | 100% |
| Exact quote match | 98.7952% |
| Status accuracy | 100% |
| Source artifact unchanged | 100% |

唯一失败：`mixed-misconception-intervention-003`。它实际检索出了真实的同题 `session_282_misconception` 与 Window，严格 evaluator 因期待唯一的 `session_522_*` 而判失败。

## 5. 当前未完成 / 卡点

尚未实际完成以下改动。前一轮只完成了设计、源数据核对与代码定位；用户多次中断，**不要误以为已经实现**。

1. 尚未修改 source Gold、schema、manifest 或派生 dataset。
2. 尚未把该 case 改成 question-level 多 Session 的 expectation。
3. 尚未实现 runtime trace 的 `session_identity` / `evidence_mode` / candidate session fields。
4. Router 对这类包装式 Gold query 仍返回 `CASE`，应调整为没有 Session hint 时的 `QUESTION`；带明确 Session hint 时保留 `CASE`。
5. Evaluator 目前只支持单一 exact Card/Window/Turn expectation，尚不支持“同一 question_id 的任一 source-grounded Session”这一替代集合契约。
6. 因 Gold 内容变更，旧 manifest `review_status=HUMAN_APPROVED_FOR_THIS_EVALUATION` 和旧 hash 不再能覆盖新版本；必须显式标为待人工复核，绝不能自动写回 approved。
7. Chroma replacement 与 candidate Concept Graph 均待人工批准，因此即使题目级评测全绿，release 仍应是 `BLOCKED`。

## 6. 建议的下一步（按顺序执行）

### Step 1：先写测试，再实现契约

修改/新增 `tests/test_business_graph_eval.py` 与 `tests/test_business_graph_pipeline.py`，至少覆盖：

1. 无 Session hint 的 pie-chart query：允许同题任一 Session 的真实 Card/Window/Turn 链通过，且 trace 标记 `NOT_UNIQUELY_IDENTIFIED`。
2. `Session 522: ...` query：必须严格选 `session_522_*`，trace 标记唯一会话。
3. 同题但 Card/Window/Turn 跨不同 Session 拼接：必须失败。
4. 一张不属于 `question_id=76888` 的 Card 即使文本相似也必须失败。
5. 引用必须在 selected Window 且逐字等于源 Turn。

### Step 2：定义最小、可审计的 Gold 字段扩展

推荐给该 case 增加一个明确的 evaluator-only 合约，例如：

```json
"expected_session_resolution": {
  "mode": "QUESTION_LEVEL_REPRESENTATIVE",
  "session_identity": "NOT_UNIQUELY_IDENTIFIED",
  "question_id": 76888,
  "eligible_session_ids": [282, 522, 3767, 5669, 9242, 10023],
  "required_card_type": "MISCONCEPTION",
  "minimum_selected_sessions": 1
}
```

设计原则：

- `eligible_session_ids` 必须由独立 source DB 查询/authoring catalog 生成，不得来自 SUT runtime 输出；
- 不要求指定某一张固定 Card；
- 要求实际 parent Card、selected Window、citation 的 `session_id` 一致，且 Card 的 `question_id=76888`；
- `expected_window_evidence` / required exact Turn 不再写死 Session 522；改成至少一个同题 Session 的真实、已授权引用；
- 保留 `MISCONCEPTION` Card 类型和题目级 SQL filter，不能放宽为“任何相似文本都算过”。

对 schema、loader、exporter 做同步扩展：

- `data/business/business-graph-gold-v1.schema.json`
- `src/business_graph_gold.py`
- `scripts/export_business_graph_dataset.py`
- `scripts/evaluate_business_graph.py`
- `src/business_graph_metrics.py`

可以允许新字段 optional，以避免破坏其余 82 个 case；但若该 case 声明该字段，必须严格校验字段完整性，不能默默忽略。

### Step 3：运行时增加身份解析信息

在 `BusinessGraphRuntimeTrace` 中添加严格字段（建议默认值可明确表示而非猜测）：

```text
evidence_scope: CASE | QUESTION
evidence_mode: EXACT_SESSION | SEMANTIC_REPRESENTATIVE
session_identity: UNIQUELY_IDENTIFIED | NOT_UNIQUELY_IDENTIFIED
candidate_session_ids: list[int]
selected_session_ids: list[int]
```

在 `BusinessGraphPipeline`：

- `_session_id_hint(query)` 非空时，只查该 Session，`EXACT_SESSION`。
- 无 hint 且同题候选 Session 数量大于 1 时，返回 `QUESTION_LEVEL_REPRESENTATIVE`，候选集必须真实来自 source query scope。
- 不要硬编码 `522`，也不要把 source sort 的第一个 `282` 说成“用户指定”。
- `ask_business()` 返回 payload 和 `limitations` 应暴露这个状态；生成器若将来接入，也必须看到该限制语。

Router 是否改为 `QUESTION`：推荐修正。当前 `src/query_intent.py` 的 `_gold_scope_override()` 对任何 “围绕真实题目 / mixed misconception intervention” 强制 `CASE`，这是历史 Gold 专用行为。应让显式 Session hint → `CASE`，无 hint 的题干查询 → `QUESTION`。需要同时更新该 case 的 expected router scope 和所有受影响测试；不要粗暴替换其他独立的 CASE 类 query。

### Step 4：更新 Gold，但必须作为“修订候选”而非伪装原批准

用户已授权“放宽该题”，因此可以改 `mixed-misconception-intervention-003`。修改后：

1. 重新计算 `data/business/business-graph-gold-v1.jsonl` hash。
2. 重新导出到一个新目录，例如 `evals/datasets/business-graph-v1-question-level-r1/`；不要覆盖旧的 `business-graph-v1`，旧目录保留为旧批准版本证据。
3. 新 manifest 明确：`PENDING_HUMAN_REAPPROVAL` / `CANDIDATE_CONTRACT_REVISION`，绝不标 `HUMAN_APPROVED_FOR_THIS_EVALUATION`。
4. 更新 source Gold manifest 的 review scope、hash、revision reason；如果审查状态字段受外部流程控制，则创建一个 revision manifest，不得伪造人工批准。
5. 保留旧 `82/83 strict exact-session` 报告，新增 question-level report；两者不可混写成同一指标。

### Step 5：评测并报告两个维度

新 evaluator/report 应分开显示：

```text
exact_session_identity          # 仅有身份锚点的 case 才是 hard gate
question_level_answerability    # 同题多 Session case 的 hard gate
representative_evidence_validity
cross_session_leakage
```

对该题执行后至少检查：

- 召回候选包含真实同题的多个 session；
- 选中的 Card、Window、Turn 绑定同一 session；
- Card type 是 MISCONCEPTION；
- SQL scope 是 `question_id=76888`；
- `session_identity=NOT_UNIQUELY_IDENTIFIED`；
- 不输出“Session 522”除非 query 明确含 522；
- 所有 citation 都能回查源 DuckDB。

然后运行 focused pytest，再运行相关 full 83-case evaluation（用新的 dataset 目录）。若 Graph/Chroma approval 尚未通过，最终 release 仍然 `BLOCKED`，即便 case 指标全通过。

## 7. 重要文件与路径

### Gold / dataset / reports

- `data/business/business-graph-gold-v1.jsonl`：旧 approved source Gold；当前尚未变更。
- `data/business/business-graph-gold-v1.manifest.json`：旧 approved manifest，case_count 83。
- `data/business/business-graph-gold-v1.schema.json`：Gold JSON schema。
- `data/business/business-graph-gold-v1.authoring-catalog.json`：独立 source-grounded catalog（很大）。
- `scripts/build_business_graph_gold.py`：Gold authoring / source validation；当前通常会覆盖输出，使用前必须先明确是否写入候选目录或创建备份。
- `scripts/export_business_graph_dataset.py`：无损派生评测集 exporter。
- `evals/datasets/business-graph-v1/`：旧 approved Gold 的无损派生视图。
- `reports/eval/business-graph-v1-split-dataset-full-20260910/`：82/83 严格报告。

### Runtime / evaluator

- `src/business_graph_pipeline.py`：Card 召回、Session hint、W7/S3、prompt/citations。
- `src/business_graph_embedding.py`：isolated-copy embedding candidate lane。
- `src/business_graph_trace.py`：runtime trace schema，当前没有 session resolution 字段。
- `src/business_graph_metrics.py`：actual-vs-Gold evaluator，当前 exact single evidence only。
- `src/query_intent.py`：有历史 `_gold_scope_override()` 强制 CASE 的问题。
- `scripts/evaluate_business_graph.py`：CLI evaluator / report / provenance checks。
- `tests/test_business_graph_eval.py`：metric contract test。
- `tests/test_business_graph_pipeline.py`：pipeline integration test，已有 duplicate-question 与 Session 522 hint 测试。

### Artifact hashes / approval

```text
Source DuckDB SHA256:
f089c56f162666ab71b3beecef33a871f3e140608721b3f61f96f9ed384b6157

Candidate Concept Graph SHA256:
fa1b40432b25459c52b87a5a84b00859bed7d2bd0c53e9b7e84d01c52cb81753

Canonical replacement Chroma SQLite SHA256:
4f26d4e59a39e9f770dd0547b1e8cc174f7a4d09ad5240fc550a419aa6d88994

Current production Chroma SHA256:
b24bce32a8a7ae8295ae8a1e5d5353fca00d420ace8450da1438dab809eb16ed
```

Candidate graph and replacement Chroma are both pending human approval. The original frozen production Chroma hash cannot be restored from available backup; do not attempt to paper over that fact.

## 8. 绝对不要踩的坑

1. **不要让系统稳定猜中 Session 522。** 无身份锚点时这是幻觉，不是优化。
2. **不要为了全绿删除失败或把 runtime 输出回写为 Gold。** Gold 必须独立于 SUT 生成。
3. **不要把 embedding 分数当作事实、统计结果或 citation 授权。** 它只是候选召回/排序信号。
4. **不要跨 Session 拼证据。** Parent Card、Window、Turn citation 必须有可验证的 Session 绑定；同题多会话不允许把 A 的 Card 和 B 的 Turn 拼成一条案例。
5. **不要把 source order 等同于用户身份。** 目前 duplicate tie 下 `282` 排在前，只能称代表性会话。
6. **不要通过 `PersistentClient` 打开 production `data/chroma`。** 必须复制到 isolated scratch/candidate artifact；评测前后 hash 检查。
7. **不要覆盖原 Gold / 旧 dataset / 旧报告。** 新契约输出用新版本目录，旧 `82/83` 是重要诊断证据。
8. **不要把候选 Graph 或 replacement Chroma 当 approved artifact。** 两者都 pending human approval。
9. **不要在外部 reranker/generator 依赖不可用时伪造成功。** 明确标 `UNMEASURED`、`FAILED` 或相应受检状态。
10. **不要修改 legacy 默认 route。** Graph business route 仍是显式 opt-in。
11. **不要修改/删除用户已有 dirty changes。** 当前工作树含大量无关文件和上层项目变化；先精确限定路径。
12. **不要声称“完整 pytest 仍 427 passed”。** 那只是修改前历史证据；任何新改动必须重跑。

## 9. 推荐的验证命令

PowerShell 下 `uv` 可能因默认 cache 路径异常而失败。使用项目内临时 cache：

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache-business-graph'
uv run pytest tests/test_business_graph_eval.py tests/test_business_graph_pipeline.py tests/test_query_intent.py -q
```

运行新的完整 Graph eval 时，永远输出新目录：

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache-business-graph'
uv run python scripts/evaluate_business_graph.py `
  --dataset-dir evals/datasets/<new-question-level-dataset> `
  --source-db data/db/tutoring_knowledge.duckdb `
  --graph-db reports/staging/session-card-graph-v1/graph-concept-topology-20260909.duckdb `
  --chroma-dir reports/staging/session-card-graph-v1/chroma-replacement-20260910 `
  --embedding-mode isolated-copy `
  --allow-candidate-graph `
  --output reports/eval/<new-run-id>
```

命令的非零退出码可能只表示 release 被 approval gate 阻断；必须读取 `report.json` 的 per-case metrics、provenance 和 artifact immutability，不能把退出码误报成 runtime 失败。

## 10. 交接时的真实状态一句话

Graph/RAG 基础链路和 83-case 严格评测已完成；唯一失败暴露了同题多 Session 下旧 Gold 强行指定 Session 522 的身份幻觉。用户已要求将该题改为 question-level 的代表性 RAG 评测。设计和定位已完成，代码、Gold revision、派生 dataset、测试与新评测尚未实际落地；且任何新 Gold 修订必须等待人工重新批准。
