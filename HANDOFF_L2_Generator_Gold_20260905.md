# Eedi-RAG L2 Generator / Gold 对齐交接文档

> 给完全没有上下文的新对话使用。本文件是本轮最新交接；不要用旧 `HANDOFF.md` 覆盖本文状态。旧文档只能作为历史背景。

更新时间：2026-09-05  
项目：`E:\PIAgent\10-projects\AgentLearn\Eedi-RAG`  
Git 根：`E:\PIAgent`  
分支：`codex/智能跟单agent`  
当前 HEAD：`bebcb45 fix(eedi-rag): canonicalize declared evidence id padding`

## 1. 我们在做什么

这是一个面向师生数学辅导对话的 RAG 系统。Eedi Tutoring Dialogues 是公开代理数据，不代表 NBCOT 业务数据。

当前 L2 目标是先把 Gold 做成可信对齐基准，再判断 Real 检索和 Generator 的问题：

```text
清洗会话
→ 学生错因卡 / 导师策略卡
→ DuckDB 结构化事实 + 原始 Turn
→ Chroma 向量索引
→ BM25 + Dense Top-20 Cards
→ Card Rerank
→ Anchored Parent-3
→ W6/S3 Window Rerank
→ overlap-aware Assembler
→ Generator
→ 后端证据物化
→ Citation Audit + DeepEval
```

业务最关心的指标：

| 指标 | 含义 | 门槛 |
|---|---|---:|
| Answer Relevancy | 用户问什么，是否直接回答什么 | `>=0.85` |
| Faithfulness | 事实是否有上下文依据，是否抗幻觉 | `>=0.90` |
| Context Recall | 最终 Context 是否覆盖所需知识 | `>=0.90` |
| TTFT/完整延迟 | 用户等待体验 | TTFT `<=2s`，完整回答约 `4–6s` |

Citation Audit 是后端逐字段真实 Turn 审计，门槛 `>=0.95`，不是 DeepEval 语义分。

## 2. 已完成

### 2.1 L1 检索路线已固定

生产默认：

```text
BM25 + Dense → Top-20 Cards → Card Rerank → Parent-3
→ source_turn_ids 锚定 W6/S3 → Window Rerank → Evidence-5 Assembler
```

固定参数：

```text
retrieval_mode=bm25_dense
bm25_weight=0.35
rerank_unit=anchored_logical_window
reranker_pool_size=20
parent_card_count=3
evidence_selection_count=5
window_size=6, window_step=3
max_prompt_tokens=1500
reranker=SiliconFlow Qwen/Qwen3-Reranker-0.6B
```

只读评测 artifact hash：

```text
b45860c80d02b999c9c9bf6d960c50779b84aad1b76fa74bef02da6ac9a8aeb9
```

当前 Generator-first 阶段不要盲目重调 Embedding、Rerank 或窗口；先用 trace 定位损失。

### 2.2 卡片和原文索引已分离

- `src/evidence_index.py`：从真实 `CleanedSession.turns` 确定性构造 W6/S3 窗口，不让 LLM 选择原文指针。
- `src/extraction_cache.py`：session hash + contract hash 缓存成功抽取，失败结果不会成为 cache hit。
- `scripts/incremental_card_ingest.py`：缓存、增量更新、audit-only rebind、失败恢复。
- `src/extract_knowledge.py`：semantic-only 模式不信任 LLM 的 `source_turn_ids`。

### 2.3 Generator v2 已实现

核心文件：`src/generator_contract.py`、`src/rag_pipeline.py`、`src/cli.py`。

- Generator 只输出 `evidence_id`；后端根据最终 Context 物化真实 `session_id/turn_id/speaker/quote_text`。
- `fact` 和 `inference` claim 都必须绑定 evidence；inference 还必须有 qualification。
- claim 声明的全部 evidence ID 必须出现在 `dialogue_citations`。
- 多角色问题必须分别填写 `student_evidence_ids`、`tutor_evidence_ids`，且两组都出现在 citations。
- 首次契约失败触发一次 citation-only repair；repair 状态写入 trace。
- `E1/E04/E11` 等短 ID 只在 catalog 存在对应 `E001/E004/E011` 时补零；未知 ID 仍失败。
- 窄问题允许 `scaffolding_steps=[]`、`pedagogical_intervention=[]`，但字段必须存在。
- Faithfulness 使用 answer + diagnosis + evidence explanation + claim ledger，不把 recommendation 当历史事实。

### 2.4 Prompt v2 已加入红线和示例

`SYSTEM_PEDAGOGICAL_PROMPT_V2` 包含：先回答实际问题、窄问题 1–2 句、最小充分引用、禁止伪造/改写 quote、inference 必须限定、多角色成对引用正例，以及过度展开和错配 evidence 反例。

### 2.5 Gold Context v2 已节点化

Gold 仅用于评测，不进入生产检索：只保留与 required Turn 相交的 W6/S3 窗口，合并重叠区间，每个 Turn 只出现一次；`[CARD_FACT]`、`[AUTHORITATIVE_TURN]`、`[ALLOWED_INFERENCE]` 分离；可显式提供 `[DERIVED_FACT]`，但不能复制 `ground_truth`。

### 2.6 评测基础设施

`scripts/run_l2_deepeval.py` 支持 `--case-start`、`--max-cases`、`--resume`、per-case checkpoint、artifact hash 和 Generator/Judge timeout。`scripts/merge_l2_deepeval_reports.py` 拒绝重叠 case、不同 artifact/route/Judge、缺失 Gold/Real pair，并记录 diagnostics 与 Gold data conflict。

## 3. 当前可用证据

### 3.1 最新完整 10-case

报告：`reports/eval/l2-generator-v2-10case-full-20260905-1700/`。

| Track | Answer Relevancy | Faithfulness | Context Recall | Citation Audit | 状态 |
|---|---:|---:|---:|---:|---|
| Gold | `0.9909` | `0.9200` | `0.9750` | `0.6167` | 语义三项过线，引用未过线 |
| Real | `0.9800` | `0.8600` | `0.6400` | `0.6167` | 相关性过线，其余关键项未过线 |

Gold/Real 各 10 case 全部 SUCCESS，0 ERROR，0 UNMEASURED。该报告在后续短 ID/可空列表修复前生成，是最新完整基准，不是最新代码的最终全量证明。

### 3.2 当前代码后的 partial

`reports/eval/.l2-checkpoints/l2-generator-v2-10case-s1/s2/s3-20260905-1900` 是短 ID 和可空教学列表修复后的运行，但只完成 6/10 case，未合并，不能替代 1700 完整报告。

### 3.3 Citation 诊断

成功样本通常满足：`quote_grounding_precision=1.0`、`role_accuracy=1.0`、`candidate_authorization_rate=1.0`，但 `citation_recall<1.0`。主因是模型只引用部分 required Turn；后端自动闭包只补模型已声明的 ID，绝不根据 qrels 猜证据。

### 3.4 Real Recall 诊断

完整 10-case 的 required raw Turn 约 `77/85=0.906` 能进入最终 Assembler evidence，但 DeepEval Real Context Recall 只有 `0.64`。应分成：

```text
R = required evidence
C = card/source pointer 候选
W = ranked window 候选
A = final Assembler context

R-C：卡片指针/候选展开损失
C-W：Window Rerank 损失
W-A：Assembler budget/selection 损失
A 已有但 Judge 仍低：语义充分性或 qrels/Golden 对齐问题
```

### 3.5 延迟状态

L2 trace 的 `latency_ms` 包含 Generator 和 DeepEval Judge，不是生产 TTFT。10-case 平均约 Gold `67.8s`、Real `79.8s`，只能说明评测成本高。生产 retrieval、Generator、TTFT、完整回答延迟尚未独立埋点，必须标记 `UNMEASURED`。

## 4. 当前卡点

1. Gold Citation Audit `0.6167`，引用真实但覆盖不完整。
2. Real Context Recall `0.64`，需要 R/C/W/A 分段定位。
3. Real Faithfulness `0.86`，部分诊断/推导没有明确标为 inference 或 derived fact。
4. 已知数据冲突：case 0025 卡片 `error_choice=A`，Golden 明确选择 `C`。
5. L1 release gate 仍为 `BLOCKED`，L2 不能伪装成发布通过。
6. DeepEval 有分钟级长尾，不要因长尾重复启动相同 case。

## 5. 下一步计划

### Step 1：最新代码下重新跑 10-case Gold

从新 run-id 开始，使用三路不重叠分片：1–4、5–7、8–10。固定 v2 contract、Gold Context v2、rubric v2、BM25+Dense、Top-20、Anchored Parent-3/W5。Gold 验收条件：10/10 SUCCESS、contract error=0、Answer Relevancy/Faithfulness/Context Recall/Citation Audit 全部达门槛。

### Step 2：解决 Gold citation coverage

逐 case 分类 required evidence：主张必需、角色必需、辅助推理、冗余或冲突。只有人工/规则审查后才可版本化 `qrels_v2`；绝不能自动删除低分 required quote 来抬分。

### Step 3：定位 Real Recall

为每个 case 输出 R/C/W/A 集合：

```text
R = required evidence
C = card/source pointer 候选
W = ranked window 候选
A = final Assembler context
```

只有找到主要损失集合后，才允许改对应组件。

### Step 4：增加真正的生产延迟埋点

独立记录 `retrieval_ms`、`assembler_ms`、`generator_request_ms`、`TTFT`、`generation_complete_ms`、`total_request_ms`；DeepEval Judge 时间单独记录。

### Step 5：通过 Gold 后再扩大规模

```text
10-case Gold smoke
→ Gold citation/faithfulness 修复
→ 10-case Gold gate
→ 10-case Real Recall 分段
→ 30-case Gold
→ 30-case Real
→ 才讨论生产 promotion
```

## 6. 复现命令模板

不包含任何密钥：

```powershell
.venv\Scripts\python.exe scripts\run_l2_deepeval.py `
  --db reports\eval\card-evidence-rebind-staging-20260904-001500\tutoring_knowledge.duckdb `
  --chroma reports\eval\card-reextraction-qwen-index-20260903-203000\chroma `
  --l1-closeout reports\eval\l1-card-reextraction-qwen-full-20260903-223000\closeout.json `
  --l1-manifest evals\datasets\l1-closeout-20260903-015400\manifest.json `
  --expected-artifact-hash b45860c80d02b999c9c9bf6d960c50779b84aad1b76fa74bef02da6ac9a8aeb9 `
  --dataset-output evals\datasets\<new-dataset> --output-root reports\eval `
  --run-id <new-run-id> --case-start <start> --max-cases <count> `
  --generator-contract v2 --gold-context-version v2 --rubric-version v2 `
  --judge-model mimo-v2.5-pro --judge-api-key-env LLM_API_KEY `
  --judge-base-url https://api.xiaomimimo.com/v1 `
  --reranker-backend siliconflow --rerank-unit anchored_logical_window `
  --reranker-pool-size 20 --parent-card-count 3 --evidence-selection-count 5 `
  --retrieval-mode bm25_dense --bm25-weight 0.35 --allow-l1-blocked
```

本地验证：

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src scripts evals tests
git diff --check
```

## 7. 绝对不要踩的坑

### 数据、DB、Chroma

- 评测期间绝不写生产 DB/Chroma；开始/结束 hash 必须一致。
- Gold Context、ground_truth、qrels、expected answer 不能进入生产索引或 Generator Gold 输入。
- 不要让 LLM 重新猜 source_turn_ids；原文指针走确定性 evidence index。
- 不要一次失败就全量重抽取；使用 extraction cache、增量 ingest、checkpoint、`--resume`。

### Generator、引用和推断

- 不信任模型自由填写 quote_text；只接受 evidence ID。
- 不要把 `[Turn N] [speaker]` 复制进 quote_text。
- 不要从 qrels 自动替模型选择未声明证据。
- `precision=1` 不等于 `recall=1`；引用真实不等于引用完整。
- inference、数学推导、recommendation 不能都当历史 fact。
- 不要因窄问题要求模型重复完整诊断；字段可为空，answer 必须切题。
- 一次 repair 成功不代表全量 citation 达标。

### DeepEval、并发和指标

- 不混合不同 Prompt、contract、rubric、artifact、route 或 case 分片。
- 不用 partial report 的均值代表全量；始终报告 attempted、measured、ERROR、UNMEASURED 和分母。
- 不把一个超长字符串节点的 Precision `1.0` 当作内部无噪声。
- 不把 DeepEval Judge 时间当生产 TTFT。
- 不要中途启动重叠 shard；合并器会拒绝且浪费 API。
- 不要因 Judge 长尾盲目增加并发；先查进程、连接和 checkpoint。
- `--allow-l1-blocked` 只是测试开关，不得伪造 release success。
- API 402、connection error、schema error 必须真实记录，禁止默认答案或估计分数补齐。

### Git 与安全

- 根仓库包含多个项目；不要执行 `git reset --hard`、`git clean -fd` 或跨项目覆盖。
- 不要提交 `.env`、API key、token、DB、Chroma、reports 或大批临时数据。
- 之前曾有凭据误打印到工具输出，相关密钥必须轮换；文档只写环境变量名。
- 提交只暂存本任务文件，保留其他项目和未跟踪内容。

## 8. 新对话启动动作

1. 先读本文件，再检查 `git status`、HEAD、评测进程和 checkpoint。
2. 阅读完整 10-case 报告 `l2-generator-v2-10case-full-20260905-1700/report.json`。
3. 认识到该报告早于 1900 最新代码 partial；不要混算。
4. 用最新代码重新跑三路 10-case Gold，先解决 Citation Audit，再分析 Real Recall。
5. Gold 未满足所有硬门槛前，不做 30-case 扩大、不做生产 promotion。
