# Eedi-RAG 项目交接文档

> 面向完全没有上下文的新对话。先读本文件，再读“必读文件”即可恢复项目背景、当前状态和施工顺序。

## 0. 交接摘要

- **工作区**：`E:\PIAgent\10-projects\AgentLearn\Eedi-RAG`
- **根 Git 仓库**：`E:\PIAgent`（注意：根仓库包含多个项目，不能对整个仓库执行清理或重置）
- **当前分支**：`master`
- **当前任务**：在真实教育辅导需求场景下，使用 Hugging Face 的 Eedi Tutoring Dialogues 作为公开代理数据，完成一个从数据治理、知识抽取、RAG 构建、检索与证据优化到业务落地的 0→1→100 完整路线。
- **本次交付**：完成 L1 v2 测量真实性修复、Storage 受控切换记录、Embedding API 小基准，并更新本文件固定最终交接状态；后续仍需合并并行子分支的 PR。
- **当前判断**：架构方向有竞争力，但当前质量基线仍属于诊断基线，不能宣称已经达到生产可用或已证明 NBCOT 业务价值。
- **当前首要方向**：按 `docs/L1_Optimization_Closeout_L2_DeepEval_Entry_Plan.md` 完成 Qwen 0.6B 的 L1 最后一次优化收尾；只有获得明确 `GO_TO_L2_TECHNICAL` 后才进入使用 DeepEval 的 L2。不要先继续堆 Prompt 或调整 MMR 参数。

## 1. 必读文件和阅读顺序

建议新对话按以下顺序阅读：

1. `AGENTS.md`：工程真实性、Fail-Fast、禁止伪造执行和状态的项目守则。
2. `docs/需求.md`：外部业务需求原文及其针对当前项目的总结。
3. `docs/Eedi_RAG_Full_Lifecycle_Roadmap_0_to_100.md`：当前 0→1→100 路线图和架构目标。
4. `docs/L0_DeepEval_L1_Component_Baseline.md`：评测设计、质量门禁及历史证据。
5. `docs/L1_Component_Evaluation_Construction_Guide.md`：当前 L1 评测施工状态、artifact 保护和后续顺序。
6. `docs/L1_Component_Evaluation_V2_Implementation_Plan.md`：L1 v2 指标口径修正、Hybrid Retrieval 和验收施工说明书。
7. `docs/Embedding_方案选型与Benchmark计划.md`：Embedding 候选、统一对照实验、成本/延迟/质量评分和落地规则。
8. `docs/Embedding_选型实测报告_20260902.md`：deterministic、e5-small、e5-base 的真实 CPU 结果和选择依据。
9. `docs/L1_Optimization_Closeout_L2_DeepEval_Entry_Plan.md`：L1 最后一次优化收尾、验收标签和 L2 DeepEval 入场标准。
10. 本文件的“当前真实状态”和“下一步计划”。

源码阅读顺序：

```text
src/models.py
  ↓
src/data_fusion.py
  ↓
src/extract_knowledge.py
  ↓
src/chunker.py
  ↓
src/storage_manager.py
  ↓
src/query_rewriter.py + src/retriever.py
  ↓
src/reranker.py
  ↓
src/rag_pipeline.py
  ↓
src/cli.py / scripts/interactive_cli.py
  ↓
evals/contracts.py + evals/metrics + evals/runners
```

## 2. 项目究竟在做什么

### 2.1 外部业务需求

原始需求是帮助一家教育公司分析约 53 份 NBCOT 辅导记录，判断其中是否包含可用于 AI 的价值，并用小成本 PoC 验证：

- 学生最常问什么；
- 哪些概念最难；
- 哪些认知误区反复出现；
- 学生有哪些考试错误模式；
- 经验导师反复采用哪些教学策略；
- 哪些内容可以改进教材、微课和题库解析；
- 未来可以发展哪些 AI 产品。

核心用户包括教研主管、新导师、课程研发人员和未来的学生端用户。业务原则是先小规模验证价值，不未经批准扩展成复杂 SaaS。

原始业务数据目前不可用，因此不能把当前 Eedi 结果表述成 NBCOT 领域结论。

### 2.2 当前公开代理数据

项目使用 Hugging Face 的 `Eedi/Question-Anchored-Tutoring-Dialogues-2k`，因为它具备：

- 真实师生多轮辅导对话；
- 题目与选项元数据；
- 学科、Topic、Subtopic 层级；
- 学生困惑和导师启发式提问；
- 可用于验证教学对话 RAG 的角色隔离、上下文锚定、证据溯源和检索质量。

数据源 README 的完整数据统计约为 1,971 场 intervention、68,717 条消息；项目目标工程基线按需求侧定义为约 50,000 条消息、约 1,000 场有效课堂。两者不能混为一谈，后续必须用 manifest 记录精确的选取、清洗和入库数量。

### 2.3 项目差异化

本项目的竞争力不在于“调用了 LLM”或“使用了向量库”，而在于面向教学对话做了专门建模：

```text
学生错误表达  →  学生认知误区卡（诊断证据，不是标准答案）
导师教学表达  →  导师策略卡（破局问题、脚手架、教学动作）
题目/考点事实 →  关系型事实表（确定性统计与过滤）
原始对话轮次  →  DuckDB 权威证据（最终引用来源）
```

最终形成：

```text
清洗与脱敏
  →  教学知识结构化抽取
  →  错因卡 / 策略卡 / 原文证据
  →  DuckDB + ChromaDB 双引擎
  →  SQL 统计 + 语义检索
  →  证据装配与生成
  →  四元组引用审计
  →  可复现评测与业务反馈闭环
```

## 3. 当前基线：必须区分三种规模

### 3.1 业务基线

原始需求：约 53 份私有 NBCOT 辅导记录。当前没有该数据，所以不能验证 NBCOT 的领域迁移、业务 ROI 或学生效果。

### 3.2 目标实验基线

建议正式冻结为：

- 代理数据：Eedi QATD-2k；
- 目标规模：约 50,000 条消息、约 1,000 场有效课堂；
- 数据划分：按 `intervention_id` 隔离 train/dev/holdout，不能按 chunk 随机切分；
- 评测对象：学生洞察、导师策略、内容改进建议和证据引用；
- 所有结论明确标注为“Eedi 数学辅导代理数据结果”。

### 3.3 当前可复现工程快照

当前本地 artifact 实际是较小的施工快照，不是目标规模：

| Artifact | 当前数量 |
|---|---:|
| `data/cleaned_sessions.jsonl` | 1,576 行会话记录 |
| DuckDB `tutoring_sessions` | 100 |
| DuckDB `session_dialogue_turns` | 2,335 |
| DuckDB `misconception_chunks` | 100 |
| DuckDB `tutor_strategy_chunks` | 100 |
| DuckDB `sliding_window_chunks` | 719 |
| Chroma `student_misconceptions` | 100 |
| Chroma `tutor_strategies` | 100 |
| Chroma `fallback_windows` | 719 |
| `data/golden_test_set.json` | 30 条端到端用例 |
| `scripts/eval_50_queries.py` | 50 条检索诊断 Query |

因此要准确表述为：**项目设计和路线面向约 1,000 场课堂；当前数据库是 100 场可复现开发快照。**

## 4. 代码架构和真实行为

### 4.1 数据契约：`src/models.py`

主要模型：

- `SubjectHierarchy`：学科树路径、Subject、Topic、Subtopic。
- `Question`：题干、选项、图片。
- `DialogueTurn`：角色、清洗文本、原始碎片、Talk Moves、噪音标记。
- `CleanedSession`：题目事实 + 学科路径 + 时序对话的完整教学交互单元。
- `StudentMisconceptionProfile`：错误选项、深层机理、学生原声、学生 Turn IDs。
- `TutorStrategyProfile`：教学目标、策略类别、Aha 问题、脚手架、Tutor Turn IDs。
- `ExtractedPIU`：两张知识卡和 `success/partial/failed/skipped` 状态。
- `Chunk`：统一的知识卡或滑窗索引对象。
- `MultiPerspectiveQueries`：三个专业检索视角。
- `DialogueCitation`：`session_id + turn_id + speaker + quote_text`。
- `PedagogicalGuidanceResponse`：最终教研回答、证据、审计状态和 Markdown。

模型多数使用 Pydantic 严格校验；知识卡禁止未知字段，引用字段有正数和非空约束，`ExtractedPIU` 会检查状态与卡片载荷是否一致。

### 4.2 ETL：`src/data_fusion.py`

`fuse_all_sessions()` 读取三张 CSV：

1. `dialogue-subjects.csv`：构造知识树路径；
2. `dq-question-metadata.csv`：按 `Sequence` 拼接题干并透视选项；
3. `anchored-dialogues/train.csv` 或 `test.csv`：按 `InterventionId` 聚合对话。

清洗行为：

- LaTeX 先掩码，普通清洗后再恢复；
- 使用规则替换部分姓名、电话、邮箱；
- 连续同角色消息压缩成一个 `DialogueTurn`；
- 合并 Talk Move 标签；
- 标记问候和噪音；
- 双方至少一轮且总轮数至少两轮时标记 `has_valid_tutoring=True`。

注意：当前 PII 脱敏主要是规则模式，不应被描述为通用 NER 级别的隐私保证。接入客户私有数据前需要重新做 PII 扫描、人工抽样和脱敏验收。

### 4.3 知识抽取：`src/extract_knowledge.py`

`build_extraction_prompt()` 将题目、考点和完整会话传给真实 LLM，要求同时输出错因卡和策略卡。

关键真实性门禁：

- 学生 `source_turn_ids` 必须存在且必须指向学生轮次；
- 学生引用必须在对应学生文本中出现；
- Tutor `source_turn_ids` 必须存在且必须指向导师轮次；
- `key_aha_question` 必须出现在对应导师轮次；
- Schema 采用 strict + `extra=forbid`；
- 解析/校验失败最多自纠重试；
- 缺少 API key、模型或依赖时直接报错；
- 批量输出使用临时文件，全部成功后才替换正式 JSONL。

确定性模板知识卡抽取已经禁用，不能为了补数量而伪造知识卡。

### 4.4 切块：`src/chunker.py`

三种策略：

- `primary`：必须有真实 `ExtractedPIU`，产出错因卡和策略卡；没有则抛错。
- `fallback`：纯规则滑动窗口，默认 6 轮窗口、步长 3。
- `hybrid`：有抽取资产时同时产出知识卡和滑窗；没有时透明降级为滑窗并写 WARNING。

滑窗是合法的真实原文替代路径，但不是伪造的结构化知识卡。

### 4.5 双引擎存储：`src/storage_manager.py`

DuckDB 表：

- `tutoring_sessions`：会话和题目事实；
- `session_dialogue_turns`：Turn 级权威原文；
- `misconception_chunks`：错因结构化字段；
- `tutor_strategy_chunks`：策略结构化字段；
- `sliding_window_chunks`：滑窗原文。

Chroma 集合：

- `student_misconceptions`；
- `tutor_strategies`；
- `fallback_windows`。

当前默认/评测 embedding 是显式选择的 `FastDeterministicEmbeddingFunction`：128 维、token + 字符 2-gram 哈希。它适合可复现、零成本诊断，不是最终生产语义 embedding。

存储模块会做 ID、角色、Turn 和原声校验，并提供 SQL 聚合与向量查询接口。注意 DuckDB 和 Chroma 是两个存储系统，当前入库流程没有天然的跨引擎事务；后续要靠 staging、版本和审计快照保证一致性。

### 4.6 Query Rewrite：`src/query_rewriter.py`

当前有两个显式后端：

- `deterministic`：对内置数学 glossary 做关键词子串匹配，注入最多两个知识条目，再生成三个规则查询；
- `llm`：调用 OpenAI-compatible API，严格校验 `MultiPerspectiveQueries`，失败不自动降级。

当前 glossary 是数学领域硬编码，不能直接当作 NBCOT 迁移方案。未来应将 glossary、taxonomy 和 query intent 配置化。

### 4.7 Retriever：`src/retriever.py`

`DualMetricRetriever` 的行为：

- Chroma 初筛候选数为 `max(top_k * 5, 20)` 的上限；
- 用余弦 + L2 组合重新评分；
- 对错因和策略分别检索；
- RRF 入口对每个集合使用 misconception/strategy、curriculum、raw 三个视角，各取 15 条，再用 `k=60` 融合；
- 命中后按 `session_id + source_turn_ids` 从 DuckDB 回查证据。

重要事实：代码中的 RRF 检索调用目前是顺序执行，注释中的“并发”不是实际行为；当前也没有 BM25/Sparse 召回或 Cross-Encoder 重排。

### 4.8 Assembler：`src/reranker.py`

`PedagogicalGoldAssembler`：

- 用 RRF/Hybrid 得分、证据存在、Aha 问题存在、关键词命中做业务加权；
- 以 MMR 从错因候选中选一个、从策略候选中选一个；
- 聚合 DuckDB 证据；
- 输出四槽位 Markdown：错因、策略、真实对白、题目/考点。

注意：类构造器有 `max_prompt_tokens=1500`，但当前 `assemble()` 主要做估算和返回，没有真正对超预算内容执行硬截断/拒绝；后续评测必须检查预算，而不能只相信字段名称。

### 4.9 端到端生成：`src/rag_pipeline.py`

`EndToEndPedagogicalRAGPipeline.ask()` 流程：

```text
Retriever.retrieve_multi_perspective_rrf
  → GoldAssembler.assemble
  → LLM 或显式 deterministic 生成
  → _audit_citations
  → Markdown 渲染
```

支持 `auto`、`llm`、`deterministic` 三种模式：

- `auto` 没有 `LLM_API_KEY` 时直接失败；
- `llm` 需要真实 API key 和显式模型；
- `deterministic` 必须由调用方显式指定，不能伪装成 LLM 结果。

当前工作树里，LLM 输出契约已被改为只返回 `referenced_session_ids`，后处理会对每个 session 取前 6 个 Turn。代码同时把这些 DuckDB 查询结果标为 `verifiable_in_duckdb=True`，引用审计对该标记会跳过逐字段检查。这是当前最重要的证据链风险：Session 存在并不等于模型引用了正确的 Turn。

正确目标应是模型返回或后端确定完整四元组：

```text
(session_id, turn_id, speaker, exact_quote)
```

且引用必须来自本次检索候选和对应 `source_turn_ids`；失败应返回 `GROUNDING_DEGRADED` 或拒答。

### 4.10 CLI 和当前界面

当前实际可运行界面是交互式 CLI：

- `scripts/interactive_cli.py` → `src.cli.main()`；
- 支持 `:help`、`:sample`、`:mode`、`:clear`、`:exit`；
- `scripts/run_demo_query.py` 可执行单次查询。

PRD 和路线图提到 Streamlit `app.py`，但当前项目文件中没有根目录 `app.py` 或 Streamlit 实现。因此 Streamlit 是尚未完成的交付项，不能写成已实现。

## 5. 已完成的工作

### 5.1 业务和架构设计

- 已把外部 53 份 NBCOT 需求映射到 Eedi 公开代理数据；
- 已形成 0→1→100 路线图；
- 已明确学生错因、导师策略、题目事实和原始 Turn 的职责分离；
- 已明确宏观统计应走 SQL，微观案例应走语义检索；
- 已明确隐私、证据溯源和不伪造数据的工程边界。

### 5.2 核心 RAG 链路

- 三表 ETL、题目透视、考点路径构建和 Turn 压缩；
- PII 规则脱敏和 LaTeX 保护；
- strict Pydantic 数据契约；
- LLM 知识抽取、自动校验和自纠重试；
- 双卡片知识结构；
- DuckDB + Chroma 双引擎存储；
- deterministic embedding 诊断基线；
- 多视角 Query Rewrite 和 RRF；
- MMR 上下文装配；
- 生成错误显式抛出；
- 多处 Fail-Fast 和降级 WARNING。

### 5.3 评测和审计框架

`evals/` 已建立 L1 组件评测控制面：

- 严格的 `RunContext`、`MetricObservation` 和版本/哈希契约；
- `UNMEASURED`、`DEGRADED`、`FAILED`、`ERROR` 状态；
- Retrieval 的 Recall、MRR、nDCG、Noise、Latency；
- Grounding 的 Citation Precision/Recall、Role Accuracy、Quote Grounding；
- Storage 的 ID parity、Metadata contract、Orphan rate；
- Chunking 的 Turn Recall、Turn MRR、Token inflation；
- Rewrite 的 token/entity 保留、意图保持和 lift；
- Assembler 的候选保留、证据召回、选择精度、预算、噪声和延迟；
- 只读 artifact 读取和 hash 保护；
- Chroma metadata 隔离副本迁移工具；
- 不把缺少 DeepEval 或 Judge 的情况伪装成语义分数。

## 6. 当前质量证据

### 6.1 30 条端到端 RAGAS 报告

`data/ragas_evaluation_report.json` 当前汇总：

| 指标 | 当前值 |
|---|---:|
| Context Recall | 0.525 |
| Context Precision | 0.6002 |
| Faithfulness | 0.8282 |
| Answer Relevance | 1.0000 |
| Citation Accuracy | 0.2222 |
| Intent Accuracy | 0.8667 |
| Global RAGAS Score | 0.5334 |

这说明主要问题在召回覆盖、噪声和引用闭环，不能只通过继续调生成 Prompt 解决。评测模式和 Judge 配置必须随报告一起确认；没有真实 Judge 时，启发式结果只能标记为 degraded。

### 6.2 历史 50 条检索诊断

`l1-retrieval-legacy50-20260902-002` 是较早的 100 Case（Raw/RRF 各 50）诊断运行，发布状态 `BLOCKED`。它仍使用单目标 Session qrels，不是正式 graded 基线；更新的 Raw/RRF 分模式结论见 6.5，禁止混用两个 run 的聚合值。

文档记录的方向：

- Candidate Recall@20：56%；
- Recall@3：24%；
- Recall@5：37%；
- MRR：0.24；
- nDCG@5：0.26；
- Noise ratio@5：93%；
- RRF Recall@5 高于 Raw，但整体仍远未达到发布门槛；
- 策略卡片 slice 明显弱于错因卡片；
- P95 latency：约 78 ms，已满足当前 ≤150 ms 诊断门槛。

结论：先解决 qrels、候选召回、策略卡表示和检索噪声，不要优先优化延迟。

### 6.3 Storage artifact

当前生产路径曾经完成过 metadata 迁移，但必须区分：

- 迁移副本曾达到 `7 SUCCESS / 0 FAILED / 0 UNMEASURED`；
- 生产路径复测也有通过报告；
- 原始 Chroma 备份保留在 `data/chroma-backups/`；
- 评测使用 immutable 读取路径，避免 `PersistentClient` 的只读调用改写 Chroma housekeeping 文件。

如果重新评测或迁移，必须先对源 DB 和 Chroma 做 hash，并优先写入新目录，不覆盖源 artifact。

### 6.4 当前测试和静态验证

2026-09-02 当前实际运行结果：

```text
pytest -q: 111 passed, 1 failed
compileall src evals scripts tests: PASS
```

唯一失败：

```text
tests/test_expand_knowledge_base.py::test_llm_expansion_extractor_uses_strict_shared_gate
```

原因：`scripts/expand_knowledge_base.py` 当前工作树改动让 `extract_piu_with_llm()` 捕获 `LLMExtractionError`，记录 WARNING 并返回 `extraction_status="failed"`；测试仍期待该异常向上抛出。因此测试契约和当前批处理语义不一致。

正确处理方式：先决定产品契约是“单场失败即整个批次失败”还是“允许逐场失败并显式统计”，再同步代码、测试、报告和文档。禁止仅为了绿灯放宽 strict grounding 或返回伪成功。

### 6.5 2026-09-02 L1 六组件结论复核（固定结论）

详细施工依据见 `docs/L1_Component_Evaluation_V2_Implementation_Plan.md`。以下结论优先于手工汇总报告 `reports/eval/L1_Component_Baseline_All6_20260902.md` 中与之冲突的描述。

#### 当前可以信任的结论

- Storage 生产复测为 `7 SUCCESS / 0 FAILED / 0 UNMEASURED`，`release_status=PASSED`；它只证明存储和索引契约健康。
- 当前 719 个生产滑窗的 4192 个 source pointers 全部有效，Turn 正文逐字一致率、窗口边界一致率均为 100%；100 个已索引 Session 的 2335 个 Turn 覆盖率为 100%。
- legacy-50 上完整 Raw-inclusive 多方向 RRF 相比 Raw-only 有正增益：Candidate Recall@20 `0.48→0.64`、Recall@5 `0.26→0.48`、MRR `0.1522→0.3081`、nDCG@5 `0.1639→0.3406`。
- 虽然 RRF 有正增益，绝对 Retrieval Recall 仍低，不能发布。
- 当前 Retriever 没有 BM25/Sparse 或 Cross-Encoder；所谓 Dual Metric 只是同一归一化哈希向量上的 Cosine+L2。
- 当前 deterministic hash embedding 是可复现诊断基线，不是生产语义 embedding。

#### 当前不能作为生产结论的指标

- Grounding v1 没有使用真实系统 citation；`output_citations` 直接复制自黄金 required evidence。
- Grounding 实际是 3 个 case 的 apostrophe 字符差异；正确统计为 27/30 case 全通过、87/90 citations 有效、aggregate 96.67%，旧报告的“4 case、27/31、87%”不准确。
- Chunking v1 使用评测脚本重实现的窗口，30 个 case 中 23 个与生产窗口序列不同，并使用 `hit + miss` oracle 排序；其 Turn Recall 不能解释为真实检索召回。
- Rewrite v1 对所有题型只评 `misconception_query`，没有评完整多方向 RRF；30/30 intent labels 被硬编码为 `True`，不能作为真实 Intent 结论。
- Assembler v1 每题只有一个正例 qrel，但固定输出两张卡，precision 理论上限为 0.5；同时 `fetch_evidence=False`，因此 precision/noise/evidence 结论不具备正式验收意义。
- legacy-50 使用单目标 Session qrels，Noise/Precision 只能诊断，不能代表真实业务相关性。

#### 当前瓶颈的准确表述

```text
Storage：已通过
Chunk：生产结构完整，真实窗口检索质量待测
Grounding：黄金集完整性发现 3 处字符差异；生产 Citation 基线未测
Rewrite/Fusion：完整 RRF 有正增益；单 lane 与意图质量评测口径需重建
Retrieval：Dense-only 候选召回不足，缺少 BM25/Sparse 和语义 Dense
Assembler：旧 precision/noise 指标失真，需槽位 qrels 后重新测量
```

在修正 L1 v2 测量系统前，不应依据旧 Rewrite/Assembler 指标调整生产排序参数。

### 6.6 2026-09-02 四项 L1 v2 修复结果

已完成以下评测控制面修复：

- Chunk retrieval 不再输出伪造的 inflation ratio；检索扩张不适用时为 `UNMEASURED`，Chunk 生成 inflation 单独由结构 suite 测量。
- qrels 已按 Query-Document 去重，证据轮次聚合，标签来源改为 `derived_from_human_quote`。当前仍只有 30 个唯一 Query-Document 对，且没有独立人工多文档相关性标注；正式 graded qrels 仍待 P1。
- Assembler 已输出 `retriever_miss_rate` 与 `assembler_drop_rate`，区分“正确文档没被召回”和“已召回但被装配器丢弃”。
- manifest 已记录 source/indexed/golden 三种数量、输入与产后 artifact hash、system config、embedding/retrieval provenance、生成时间和 qrels 局限。

最新复现数据集：`evals/datasets/l1-v2`（修复版 canonical；旧版归档于 `reports/runtime/l1-v2-pre-fix-20260902-174150`）。
最新 Suite 报告：`reports/eval/l1-v2-final-fixed-20260902-174150/l1-v2-suite`。
最新报告状态：`BLOCKED`；聚合诊断为 `retriever_miss_rate=0.30`、`assembler_drop_rate=0.50`、Chunk retrieval inflation `UNMEASURED`、Chunk structure inflation `3.8018`。

## 7. 当前卡点和风险

### P0：端到端引用审计存在绕过风险

当前 LLM 只返回 Session ID，后端用前 6 个 Turn 拼引用，并对已从 DuckDB 查询的数据跳过逐字段审计。必须恢复明确的四元组引用和候选约束。现有 Citation Accuracy 0.2222 是必须正视的结果。

### P0：正式索引契约和 artifact 版本要固定

历史评测发现错因卡和策略卡的 Chroma metadata contract 曾只有 5%，虽然迁移副本和后续生产复测有通过证据，但以后所有重建都必须：

- 从 DuckDB 事实表生成 required metadata；
- 写入新版本目录；
- 评测前后做 hash；
- 通过 ID parity、metadata contract、orphan 门禁后才切换。

### P0：当前 Retrieval qrels 不能签署正式质量基线

50-query 数据只有一个目标 Session，适合诊断，不适合正式 nDCG、Evidence Recall 或生产门禁。必须施工 graded、多相关文档和独立 holdout。

### P1：当前语义表示能力不足

确定性哈希 embedding 适合基线，但对于数学数字、公式、英文教学短语和长尾表达，不能作为最终语义方案。当前没有 BM25/Sparse，也没有 Cross-Encoder。

### P1：策略卡检索明显弱于错因卡

需要检查策略卡的 embedding 文本、Aha question、Talk Moves、考点字段和 qrels，而不是直接把问题归因给生成模型。

### P1：路线图中的宏观路由和 Streamlit 尚未完全落地

当前有 DuckDB SQL 查询能力和 CLI，但没有完整的宏观意图路由器、统计看板或 Streamlit `app.py`。这属于待完成的应用层工作。

### P1：Eedi 与 NBCOT 存在领域差异

Eedi 只能证明数学教学对话管线的技术属性。接入真实 NBCOT 数据时，需要重新设计 taxonomy、术语表、抽取 schema、隐私规则、业务 qrels 和人工验收集。

### 工作树风险

根 Git 仓库是 `E:\PIAgent`，当前有其他项目文件和大量工作树外生成物。项目目录内已有用户改动和未跟踪 artifact：

- 已修改：`scripts/expand_knowledge_base.py`、`src/cli.py`、`src/extract_knowledge.py`、`src/rag_pipeline.py`、`src/storage_manager.py`、`tests/test_extract_knowledge.py`；
- 未跟踪：`data/` 中导出、清洗、黄金集、RAGAS 报告，`reports/`，`scripts/data/`；
- 本文件 `HANDOFF.md` 是本次新增。

不要执行全仓库 `git clean`、`git reset --hard`、`git checkout --`，也不要把其他项目改动当成 Eedi-RAG 改动。

## 8. 下一步施工计划

### P0-A：修正 L1 v2 测量真实性

按 `docs/L1_Component_Evaluation_V2_Implementation_Plan.md` 执行：

1. Grounding 分离黄金数据完整性和真实系统 citation。
2. Chunk 直接复用生产 `SlidingWindowChunker`，结构评测与真实检索评测分离，删除 oracle 排名。
3. Rewrite 改为 Raw、各 lane、rewrite-only RRF、raw-inclusive RRF 的同集配对。
4. 删除硬编码 intent 成功；没有 human/DeepEval 标签时写 `UNMEASURED`。
5. Assembler 使用 misconception/strategy 槽位 qrels，并采集真实 evidence。
6. 所有汇总报告从 run JSON 自动生成。

验收：旧失真指标标记 superseded，`tests/evals` 通过，新报告不存在手工计数矛盾。

### P0-B：修复引用和生成契约

1. 让最终 response 的每条 citation 都有完整四元组。
2. citation 的 Session/Turn 必须来自检索候选和对应 `source_turn_ids`。
3. DuckDB 只负责权威核验，不能因为“查到了 Session”就跳过 Turn、角色和原文检查。
4. 空证据、越权证据和无法精确匹配时返回 `GROUNDING_DEGRADED` 或明确失败。
5. 补充端到端测试：错误 Session、错误 Turn、错误 speaker、改写 quote、未召回 Session 都必须失败。

验收：Citation Precision、Citation Recall、Role Accuracy、Quote Grounding Precision 都按正式 Grounding 数据达到 100%。

### P0-C：冻结数据和索引版本

新增一个 manifest，至少包含：

```text
dataset_version
source_files + source hashes
selected_session_count
message_count
cleaned_session_count
extracted_success_count
failed_count
card_count by type
chunk_count by type
db artifact hash
chroma artifact hash
schema_version
```

所有评测报告必须绑定 dataset version、qrels version、system config hash 和 artifact hash。

### P1-A：施工正式 Retrieval qrels

1. 将 `scripts/eval_50_queries.py` 中的常量迁移为版本化 JSON。
2. 每个 Query 支持多个相关文档，使用 0/1/2/3 graded relevance。
3. 按 `intervention_id` 做 dev/holdout 隔离。
4. 增加 no-answer、长尾、跨语言/表达、数字公式、策略和错因 slice。
5. 统一 Raw、Rewrite、RRF 的 paired comparison。

建议第一阶段门槛：Candidate Recall@20 ≥95%、Recall@5 ≥80%、Noise@5 ≤15%；达到稳定后再把发布门槛提升到更严格水平。不要直接把历史理想门槛当成当前成绩。

### P1-B：检索升级

Embedding 方案选择按 `docs/Embedding_方案选型与Benchmark计划.md` 执行：先在 `embedding-benchmark` 分支固定 artifact 和运行环境，比较 deterministic、multilingual-e5-base、BGE-M3，再决定是否加入 small/large/MiniLM；所有候选使用隔离索引和同一 qrels，不能把未实测模型写成已选方案。

2026-09-02 已完成第一轮真实 CPU benchmark，详见 `docs/Embedding_选型实测报告_20260902.md`：deterministic、`intfloat/multilingual-e5-small`、`intfloat/multilingual-e5-base` 均有真实结果；BGE-M3 因 2.27GB 权重和当前 CPU/网络成本中止下载，保持 `UNMEASURED`。当前不直接切生产，下一步用 e5-small/e5-base 做 BM25+Dense RRF 对照，并在 GPU/高速网络环境补测 BGE-M3。

只有 L1 v2 口径和 graded qrels 就绪后，才按以下顺序做消融：

```text
deterministic hash baseline
  → Dense embedding baseline
  → Dense + exact numeric/formula channel
  → Dense + BM25/Sparse + RRF
  → Top-30 Cross-Encoder rerank
  → metadata / subject / card-type filters
```

每一步都必须使用同一份 dev qrels，并保留 holdout 不参与调参。不要因为一个 Query 变好就宣布整体提升。

### P1-C：真实 Assembler Trace

记录：

- 输入候选 IDs；
- 最终 Context IDs；
- required evidence IDs；
- 实际估算 token 数；
- budget violation；
- 每个 stage latency；
- 证据为空的原因。

这样才能区分 Retriever 没召回、Assembler 丢证据和 Generator 没利用证据。

### P2：启用真实 Generator 评测

在 DeepEval 依赖、独立 evaluation model、版本、temperature 和凭据都明确后，再跑 Faithfulness、Answer Relevancy、Contextual Precision/Recall 和教学质量评估。

没有真实 Judge 时可以运行显式授权的 heuristic，但必须标 `DEGRADED`，不能把关键词重叠当成真实语义质量。

### P2：完成最小业务界面

优先做 Streamlit 双 Tab，而不是复杂前端：

- Tab 1：宏观洞察，SQL 聚合 Top 误区/考点/教学策略；
- Tab 2：自然语言问答，显示结构化答案、真实引用和完整原声展开；
- 所有结果显示数据版本、索引版本和审计状态；
- 不支持证据的回答必须明确拒答或显示 degraded。

### P3：面向 100 的工业化

只有 1.0 版本通过价值和质量验收后再做：

- 多租户与 ACL；
- 增量摄取、去重和幂等；
- 蓝绿索引、alias 和回滚；
- CI/CD 评测门禁；
- 线上反馈和 Bad Case 回流；
- 成本、延迟、错误率可观测；
- 私有化模型和 vLLM；
- 题库增强、导师培训、导师 QA、学情报告和学生端助教。

## 9. 绝对不要踩的坑

### 9.1 不要把降级伪装成成功

- 没有 LLM 不要用模板文本冒充 LLM 抽取；
- 没有 embedding 不要随机向量或硬编码答案；
- 不要把 `status="failed"` 的记录包装成 `success`；
- deterministic 只能在调用方显式指定时运行，并标记后端。

### 9.2 不要静默吞异常

捕获异常后必须满足其一：

- 真正修复并记录结构化日志；
- 包装上下文后重新抛出；
- 或返回显式的 `failed/degraded/unmeasured` 状态。

特别注意 `scripts/expand_knowledge_base.py` 当前捕获 LLM 抽取异常并跳过失败会话的行为。若产品决定允许部分成功，必须输出完整失败清单、覆盖率和状态；不能让批量任务看起来像 100% 成功。

### 9.3 不要把学生错误当成标准知识

- 学生文本只能进入错因诊断和证据槽位；
- Tutor 文本才能进入教学策略范式；
- 题目正确事实要来自题目元数据或权威事实表；
- 生成 Prompt 必须把这三类内容显式分隔。

### 9.4 不要用向量检索回答所有问题

“最常见的错误”“Top-N 考点”“出现次数”“不同学生重复出现”等问题需要 SQL 聚合；向量检索只负责语义案例发现。综合答案应是 SQL 统计 + 代表性证据 + LLM 解释。

### 9.5 不要把 Session 存在等同于 Citation 正确

必须检查完整四元组，而不是只检查 `session_id`。禁止对“后端刚从数据库查出来的引用”自动设置可信标记并跳过审计。

### 9.6 不要在没有 qrels 的情况下宣布 Recall/nDCG 提升

单目标 Session qrels 只能做回归诊断；正式指标要用多相关文档、graded relevance、holdout 和 slice。

### 9.7 不要把 deterministic embedding 当成生产语义能力

它是零成本可复现 baseline。任何“语义能力提升”都必须通过 Dense/Sparse/Reranker 的同集消融证明。

### 9.8 不要直接在源 Chroma 上做迁移或评测初始化

历史上 `PersistentClient` 即使只调用读取接口也可能修改 housekeeping 文件。评测应使用 SQLite `mode=ro&immutable=1` 或完整隔离副本；迁移先 dry-run，再复制到新目录，禁止覆盖源目录。

### 9.9 不要按 chunk 随机划分数据

同一 Session 的错因卡、策略卡和滑窗必须在同一 split，否则会发生严重数据泄漏，指标没有意义。

### 9.10 不要把 Eedi 结果写成 NBCOT 结论

Eedi 只验证数学辅导场景的管线和方法。客户数据到来后必须重新做隐私、taxonomy、抽取、qrels 和人工业务验收。

### 9.11 不要为了“全绿”降低硬门禁

评测系统中的 `FAILED`、`UNMEASURED`、`DEGRADED` 和退出码 2 都是有意义的状态。解决前置条件或修复系统，不要降低阈值、填默认值或篡改报告。

### 9.12 不要清理整个 `E:\PIAgent` 仓库

根仓库包含多个项目。任何删除、移动、重置操作都必须先解析精确目标，并且本任务没有授权清理其他项目。

## 10. 常用命令

在项目目录 `E:\PIAgent\10-projects\AgentLearn\Eedi-RAG` 执行。

```powershell
# 运行全量测试
.\.venv\Scripts\python.exe -m pytest -q

# 编译检查
.\.venv\Scripts\python.exe -m compileall -q src evals scripts tests

# 运行交互式 CLI
.\.venv\Scripts\python.exe scripts\interactive_cli.py

# 单次 Demo；默认 auto，需要真实 LLM_API_KEY 和 LLM_MODEL
.\.venv\Scripts\python.exe scripts\run_demo_query.py "学生在四舍五入题中常犯什么错误？"

# 当前 50-query 检索诊断
.\.venv\Scripts\python.exe -m evals.live_retrieval `
  --db data\db\tutoring_knowledge.duckdb `
  --chroma data\chroma `
  --output-root reports\eval `
  --run-id l1-retrieval-YYYYMMDD-NNN

# 当前 Storage 只读审计
.\.venv\Scripts\python.exe -m evals.live_storage `
  --db data\db\tutoring_knowledge.duckdb `
  --chroma data\chroma `
  --output-root reports\eval `
  --run-id l1-storage-YYYYMMDD-NNN
```

如果在 Windows 上遇到 pytest 临时目录权限问题，使用项目内临时根：

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp reports\runtime\pytest-l1 `
  tests\evals
```

运行真实扩容前必须显式配置 `LLM_API_KEY`、`LLM_MODEL`，可选 `LLM_BASE_URL`。不要把真实 key 写入代码、文档、报告或提交记录。

## 11. 继续工作时的推荐开场白

新对话可以直接使用：

> 请先读取项目根目录 `HANDOFF.md`、`AGENTS.md`、`docs/需求.md` 和 `docs/L1_Component_Evaluation_Construction_Guide.md`。当前任务按 HANDOFF 的 P0 顺序继续：先修复四元组引用审计和生成契约，再施工正式 graded qrels；保留当前工作树改动，不执行 reset/checkout/clean，不伪造成功状态。完成任何修改前先说明影响范围，完成后给出真实测试证据。

如继续当前最新主线，应优先读取第 14 节和
`docs/L1_Component_Evaluation_V2_Implementation_Plan.md`；Embedding 施工以
`Qwen/Qwen3-Embedding-0.6B` 为初期候选，先在隔离索引验证，再考虑生产切换。

## 12. 交接文档验证记录

初版 HANDOFF 写入前曾完成：

- 读取项目 `AGENTS.md`、业务需求、0→100 路线图、L0/L1 评测文档；
- 梳理 `src/` 核心数据流和 `evals/` 评测控制面；
- 读取当前 DuckDB 表计数和项目 artifact 状态；
- 运行 `pytest -q`：`111 passed, 1 failed`；
- 运行 `compileall`：通过；
- 确认当前唯一测试失败及其代码/测试契约冲突；
- 确认初版只新增 `HANDOFF.md`，未修改业务代码。

最后更新：2026-09-02。

本次更新重新核对了真实 run JSON、评测数据构建代码、生产 Chunk、Retriever 和 Assembler，并固定了 L1 六组件复核结论；新增 `docs/L1_Component_Evaluation_V2_Implementation_Plan.md`，未据此修改生产检索算法。

## 14. 最终会话交接：L1 v2 与 Embedding 选型（2026-09-02）

本节优先级高于前文旧的施工状态；旧报告和旧统计保留为审计历史，不得与本节口径混算。

### 14.1 我们正在做什么

项目目标是构建一个覆盖 L1 组件、L2 端到端链路、L3 业务效果的真实数据驱动评测体系，并用它指导教育辅导 RAG 的优化。当前使用 Eedi 数学辅导公开数据作为 NBCOT 真实数据到来前的技术代理，主链路为：

```text
清洗/脱敏 → LLM 知识卡抽取 → Chunk → DuckDB + Chroma
→ Query Expansion → Dense/RRF 检索 → Reranker/Assembler
→ Generator → 四元组引用审计
```

当前阶段聚焦 L1，并行推进 Embedding 选型；最终目标是将各子分支合并为一个可审查 PR，而不是把多个实验结果直接覆盖到生产目录。

### 14.2 已完成的工作

- L0/L1 评测控制面已实现，DeepEval 作为未来语义评测适配边界；确定性指标不调用 Judge。
- Storage L1 已通过：生产 DuckDB/Chroma 的 ID parity 100%、metadata contract 100%、orphan 0。
- 生产 Chroma 已完成受控 metadata 迁移；原始目录备份在 `data/chroma-backups/`，没有删除。
- L1 v2 已拆分 `grounding`、`chunking`、`chunk_structure`、`rewrite`、`fusion`、`assembler`。
- Chunk 结构已由生产实现核验：pointer/verbatim/boundary/Turn coverage 均 100%；结构 inflation 约 3.80，低于 4.0。
- Grounding 已使用真实 deterministic pipeline citation 和 retrieved evidence，不再把黄金证据复制成系统输出。
- Rewrite/Fusion 已记录 raw、misconception、strategy、curriculum、rewrite-only RRF、raw-inclusive RRF；完整 raw-inclusive RRF 相对 raw-only 有小幅正增益。
- Assembler 已记录真实 final evidence/prompt trace，并增加 `retriever_miss_rate` 与 `assembler_drop_rate` 分离指标。
- manifest 已加入 source/indexed/golden 三种计数、输入/输出 artifact hash、system config、embedding 和 qrels provenance。
- 当前 L1 v2 专项测试最近一次为 `39 passed`；全量测试历史结果为 `123 passed, 1 failed`，唯一失败是既有 `test_llm_expansion_extractor_uses_strict_shared_gate` 契约冲突，不能为本任务掩盖。

### 14.3 当前已固定的 Embedding 决策

初期采用 SiliconFlow OpenAI-compatible Embedding API：

```dotenv
EMBEDDING_PROVIDER=siliconflow
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

精确模型 ID 必须是 `Qwen/Qwen3-Embedding-0.6B`，不得带尾随空格，也不要误用视觉模型 `Qwen/Qwen3-VL-Embedding-8B`。

已完成 API 正常性与小规模质量验证：

| 模型 | 全部 30 Query/200 卡片 Recall@5 | MRR | nDCG@5 | 向量维度 | API 总时间 | 估算费用 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Embedding-0.6B | 1.000 | 1.000 | 1.000 | 1024 | 13.29 s | 约 $0.0043 |
| Qwen3-Embedding-4B | 1.000 | 1.000 | 1.000 | 2560 | 20.41 s | 约 $0.0085 |
| Qwen3-Embedding-8B | 1.000 | 1.000 | 1.000 | 4096 | 161.61 s | 约 $0.0170 |

正式报告：`reports/eval/embedding-api-benchmark/20260902-223930/report.json`。
该报告使用 30 个 Query、100 张 misconception 卡和 100 张 strategy 卡；三个模型都触顶，说明当前单目标 derived qrels 区分度不足。该结果支持“0.6B 是性价比最优初期候选”，不证明 0.6B 与 4B 在复杂真实业务中等价。

本次 benchmark 只读 DuckDB，没有打开或写入 Chroma；当前生产 artifact 组合 hash 仍为：

```text
711f1ed14e8f7014a2a57421665c08037fd8189c80bf922217209b7789718fa4
```

### 14.4 当前卡点

1. **Embedding API 已接入 Provider，但尚未切换生产 Retriever/Storage**：已实现 `src/embedding_provider.py` 的 OpenAI-compatible Provider，并完成独立 Qwen Dense index；生产默认仍保持原有 deterministic Chroma，不能把 benchmark 报告等同于已经切换生产。
2. **L1 v2 报告仍 BLOCKED**：真实 deterministic 基线的 Grounding、fallback-window retrieval 和 Assembler 最终证据保留率不足，这是当前算法事实，不应通过降低门槛解决。
3. **qrels 仍不是多相关人工 graded qrels**：当前 30 个 Query 各有一个主要文档正例，文档标签是 `derived_from_human_quote`；Recall/MRR/nDCG 适合相对比较，不适合宣布最终业务准确率。
4. **manifest 曾记录 `artifact_mutated_during_build=true`**：这是此前构建副本被 Chroma 客户端改写的审计事实；当前生产 artifact 只读复核通过，但后续所有构建必须在隔离副本执行且前后 hash 相等。
5. **当前没有 BM25/Sparse 和 Cross-Encoder**：所谓 `DualMetric` 只是同一 Dense 向量的 Cosine + L2，不能替代 BM25+Dense Hybrid。
6. **并行分支尚未合并**：当前仓库存在 `embedding-benchmark` 等分支及工作树用户改动；合并前必须逐 commit review，不能 reset/checkout 覆盖。

### 14.4.1 Qwen 0.6B 隔离索引与 L1 评测证据

已使用 SiliconFlow API 重建隔离 Chroma：

```text
reports/eval/embedding-qwen3-0.6b-20260902-233144/chroma
```

构建报告：
`reports/eval/embedding-qwen3-0.6b-20260902-233144/chroma.build.json`。
集合数量为 misconception 100、strategy 100、fallback window 719；向量维度 1024；
API 请求 31、输入/总 tokens 209374；DuckDB hash 未变化。

在该索引的隔离副本上完成 Retrieval、Fusion、Assembler L1：

```text
报告：reports/eval/l1-qwen3-0.6b-rfa-20260902-233144-rerun
运行：l1-qwen3-0.6b-rfa-20260902-233144-rerun
请求：420 次，重试 0，向量维度 1024
release_status：BLOCKED（执行成功，但质量/延迟门禁未通过）
```

关键聚合结果：

| 模块 | 指标 | 结果 | 说明 |
|---|---|---:|---|
| Retrieval | Candidate Recall@20 | 1.000 | 目标卡全部进入候选 |
| Retrieval | Recall@5 / MRR / nDCG@5 | 1.000 / 1.000 / 1.000 | 30 个单目标 qrels 上触顶，不能视为工业质量 |
| Retrieval | noise ratio@5 | 0.80 | 单目标 qrels 把其余候选视为噪声，需多相关 qrels 才能正式解释 |
| Retrieval | P95 latency | 685 ms | 超过当前 150 ms 门槛，API 逐 Query 调用是主要原因 |
| Fusion | raw-inclusive RRF MRR lift | -0.0389 | 当前 Raw Dense 已很强，RRF 在此小集合上反而扰动排序 |
| Assembler | retriever_miss_rate | 0.00 | Qwen 候选阶段未发生目标缺失 |
| Assembler | assembler_drop_rate | 0.60 | 目标进候选后仍有 60% 未被最终保留 |
| Assembler | final_evidence_recall | 0.40 | 最终证据覆盖不足 |
| Assembler | rerank_ndcg_gain | -0.1124 | 当前 MMR/选择逻辑在该候选分布下退化 |
| Assembler | token budget / evidence empty | 0 / 0 | 预算和非空证据正常 |

该评测只证明 Qwen 0.6B 在当前单目标 qrels 上能显著改善候选召回，不证明完整业务质量；生产 Chroma 未被修改，当前生产组合 hash 仍为
`711f1ed14e8f7014a2a57421665c08037fd8189c80bf922217209b7789718fa4`。

最近一次 Provider、索引脚本和 L1 评测回归测试：`44 passed`；生产 deterministic
回归（`tests/test_retriever.py` + `tests/test_storage_manager.py`）单独运行：`15 passed`。
Qwen L1 运行的 `BLOCKED` 主要来自 API 延迟、单目标 qrels 的 noise 口径、Fusion 轻微负增益和 Assembler 丢弃；不能通过降低门禁或回退 deterministic 来处理。

### 14.5 下一步计划

#### P0：接入并验证 0.6B API Dense

1. [x] 实现厂商无关的 `EmbeddingProvider`，支持 SiliconFlow OpenAI-compatible `/v1/embeddings`。
2. [x] API 失败、超时、维度错误、数量不一致显式失败；禁止静默切回 deterministic。
3. [x] 固定模型 ID、维度、batch 和调用 provenance；记录 usage tokens、request/retry 和耗时。
4. [x] 从 DuckDB 重建新的 Qwen 0.6B Chroma 隔离索引；未混用 128d deterministic 向量。
5. [x] 运行 Retrieval、Fusion、Assembler L1；结果仍为 BLOCKED，问题转入延迟、Fusion 和 Assembler 优化。
6. [ ] 通过正式 qrels 与完整 L1 门禁后，再做受控蓝绿切换；保留旧 Chroma 备份和回滚 hash。

#### P1：建立真正 Hybrid Retrieval

1. 在同一份 qrels 上实现 BM25-only、Dense-only、BM25 + Dense RRF、Multi-query Dense RRF、BM25 + Multi-query Dense RRF。
2. BM25 tokenizer 必须保留专业词、数字、小数、公式、选项和中文 bigram。
3. 记录每个 lane 的候选、rank、score、fusion rank，先看 Candidate Recall@20/@50，再看 Recall@5、MRR、nDCG 和延迟。
4. 只有候选池稳定后再加入 Cross-Encoder；固定候选池做 before/after paired evaluation。

#### P1：修正数据评价能力

1. qrels 增加多个相关文档、0/1/2/3 relevance、人工复核和 dev/holdout Session 隔离。
2. Grounding 分离 dataset quote integrity、真实 citation grounding、role accuracy 和 candidate authorization。
3. Assembler 继续分离 `retriever_miss` 与 `assembler_drop`，并按 misconception/strategy 槽位统计。
4. manifest 增加索引版本、模型 revision、provider、实际 indexed session、运行命令和 hash；任何 hash 变化都阻断正式报告。

#### P2：L2/L3

1. 统一生产 Trace：raw/expanded query、各路候选、RRF、Reranker、最终 Context、evidence、stage latency、错误/降级。
2. DeepEval 只在真实 Generator final Context 和独立 evaluation LLM 配置就绪后启用。
3. Eedi 结果只能作为技术代理；接入 NBCOT 前重新做 taxonomy、PII、qrels、人工业务验收和 ROI 评估。

### 14.6 绝对不要踩的坑

- 不要把 30 Query 全部 Recall@5=1.0 解释成模型已经达到工业质量；当前 qrels 过于简单且单目标。
- 不要把 API smoke test 当成生产索引已切换；必须重建独立 Chroma 并跑完整 L1。
- 不要把 `Qwen3-Embedding-0.6B` 与 `Qwen3-VL-Embedding-8B` 混用，也不要在模型 ID 末尾留空格。
- 不要把 Qwen 向量写入旧 deterministic 128d Chroma；不同模型/维度必须独立 index version。
- 不要把 `EMBEDDING_API_KEY_ENV` 字段中的真实密钥写入报告、日志或提交；长期配置应改为变量名引用，密钥只存在进程环境。
- 不要在 API 超时或依赖不可用时用 deterministic、随机向量、零向量或模板数据冒充成功。
- 不要把 DuckDB 的 `LIKE` 查询称作 BM25；当前系统之前没有 Sparse retrieval。
- 不要把单独 Rewrite Query 的负增益当成完整 Raw-inclusive RRF 的结论；必须保留 Raw、各 lane 和完整 Fusion 的 paired comparison。
- 不要把 Chunk retrieval 的 inflation 与 Chunk generation inflation 混为一谈；检索候选数不是切块膨胀率。
- 不要把 Retriever miss 算成 Assembler drop；目标不在候选池时，Assembler 无法凭空召回。
- 不要把 Session 存在或后端查到 Turn 标记为 Citation 正确；必须核验完整 `(session_id, turn_id, speaker, quote_text)` 四元组和候选授权。
- 不要用 `chromadb.PersistentClient` 对源 artifact 做“只读评测”；历史上它可能改写 housekeeping 文件。使用 immutable SQLite 或隔离副本。
- 不要在构建数据集或评测时覆盖源 artifact；构建前后 hash 必须相等，目标目录必须不存在或使用明确的新版本目录。
- 不要把当前生产工作树用户改动和子分支成果混在一个 commit；先逐文件 review，再按 PR 合并。
- 不要使用 `git reset --hard`、`git checkout --`、全仓库 `git clean`，尤其根仓库 `E:\PIAgent` 包含多个项目。

L1 最后一次优化和 L2 入场必须按
`docs/L1_Optimization_Closeout_L2_DeepEval_Entry_Plan.md` 执行；该文档把每个失败指标
分为必须修复、诊断/不适用或明确限制，并规定最终只能输出
`GO_TO_L2_TECHNICAL`、`BUSINESS_QUALITY_PASSED` 或带责任和复现命令的 `BLOCKED`。

### 14.7 继续工作的推荐开场白

> 请先读取 `HANDOFF.md` 第 14 节、`AGENTS.md`、`docs/L1_Component_Evaluation_V2_Implementation_Plan.md`，并检查当前分支与工作树。当前任务是实现并验证 `Qwen/Qwen3-Embedding-0.6B` 的 SiliconFlow OpenAI-compatible EmbeddingProvider：先写失败契约测试，再在隔离 artifact 重建 Dense index，运行同一套 L1 指标，记录模型/usage/latency/cost/hash；不得覆盖生产 Chroma、不得静默 fallback、不得把单目标 qrels 的触顶结果写成工业质量。完成后提供测试证据、报告路径、commit SHA 和 PR 合并说明。

## 13. L1 v2 P0 实施状态（2026-09-02）

已实施并验证的 P0 测量真实性工作：

- Grounding v2 从显式 `deterministic` 端到端系统运行采集真实 response citation，保留检索证据，并检查四元组与候选授权；不再把 `required_evidence` 复制成系统输出。
- Chunk 已拆为生产 `SlidingWindowChunker` 结构审计（100 个已索引 Session）和真实 `fallback_windows` 检索排名（30 个 Query）；不再使用 `hit + miss` oracle 排序。
- Rewrite/Fusion 保留按 intent 选择的 misconception/strategy lane，未标注 intent 保持为 `UNMEASURED`；Retriever trace 记录 raw、各 lane、rewrite-only RRF 和 raw-inclusive RRF。
- Assembler v2 采集真实 evidence/prompt trace，并按 misconception、strategy、fallback window 槽位出 precision/recall；trace 在写入 JSON 前会规范化数组/标量类型。
- 数据集 builder 需显式传入隔离的 `--db` 与 `--chroma` artifact，生成 `evals/datasets/l1-v2/` 的 manifest、queries、qrels、grounding、六组件 suite；报告自动生成 `summary.md`。

最新真实执行：

```text
python -m evals.cli --input evals\datasets\l1-v2\suite.json --output-root reports\eval\l1-v2-grounding
exit: 2
release_status: BLOCKED
SUCCESS: 1359
FAILED: 566
UNMEASURED: 213
```

该 BLOCKED 是正确结果：当前 deterministic 基线与黄金证据/qrels 仍有真实差距，且单目标 golden qrels 尚未升级为人工多相关 graded qrels。不能据此调整生产排序，更不能将结果写成通过。

验证：`pytest -q` 为 `123 passed, 1 failed`；唯一失败仍是第 6.4 节列出的 `test_llm_expansion_extractor_uses_strict_shared_gate`，没有修改相关实现。`compileall src evals scripts tests` 通过。评测使用 `reports/runtime/l1-v2-artifacts-20260902-154700/` 隔离副本；源 DB 与 Chroma 复核 hash 未变。

## 14. L1 Optimization Closeout 执行结果（2026-09-03）

本轮按 `docs/L1_Optimization_Closeout_L2_DeepEval_Entry_Plan.md` 执行 O1→O6，并将所有新实验写入不可覆盖的版本化目录：

- O1 冻结输入：`evals/datasets/l1-closeout-20260903-015400/`，30 Query、30 derived qrels；按 source session 隔离为 dev=24、holdout=6，`session_leakage_count=0`。
- Qwen 候选索引：`reports/eval/embedding-qwen3-0.6b-closeout-20260903-014700/chroma`，100/100/719，Chroma SHA256=`38c7b7076c4d8b787d5ac3474def6876c22316cbebae3667079fea289fef86f3`，DB+Chroma combined=`6bb387b4b9698009fffca6193a35d02b98b0424d017fc4add166f89428e9d67a`。
- O2：LLM 生成契约改为完整 `(session_id, turn_id, speaker, quote_text)`；审计不再信任 `verifiable_in_duckdb` 或按 Session 自动取前六轮。
- O3：misconception/strategy 独立选槽；raw-query rank boost 修复 RRF 归一化饱和导致的候选误选；Assembler latency 只测装配阶段。
- O4：embedding provider 支持按 provider/model/index_version/query_text 的成功结果缓存；同一 Query 多 lane 合并批量请求；记录 request/cache/retry/token/P50/P95。
- O5：fallback 增加对白区数字/公式 exact channel；Fusion 采用可审计 `raw_first`，保留 rewrite-only/raw-inclusive 对照 trace。
- O6 最新 run：`reports/eval/l1-closeout-qwen-20260903-025500/`，状态计数 `SUCCESS=2110 / FAILED=344 / UNMEASURED=262 / ERROR=0`；closeout 决策为 `BLOCKED`，详见 `closeout.md`、`closeout.json`。

当前硬阻塞及责任模块：

1. Grounding `citation_recall=0.5889 < 0.95`：最终生成仍未覆盖全部 required quote，责任模块为 Generator/引用选择；下一步需基于真实最终 evidence 做可解释的引用选择和人工复核，不能从黄金 qrels 反向复制。
2. Chunk retrieval `turn_recall_at_3=0.7944 < 0.90`、`turn_mrr=0.8461 < 0.85`：dev 分片分别为 0.7569/0.8076，责任模块为 fallback window Query/候选排序；holdout 为 0.9444/1.0，需扩大独立 qrels 后再调参。
3. Retrieval `latency_p95=443.8ms > 300ms`：责任模块为 API/Chroma 查询链路；provider 记录了 60 requests、390 cache hits、0 retries，但上游延迟存在长尾，不能把 token/cache 命中伪称达标。

Storage、Chunk Structure、单槽 Retrieval 质量、Assembler retention/evidence/budget/latency 和 selected Fusion non-regression 均已通过；derived qrels 的 Top-5 noise、无标签 Rewrite intent、未适用 fallback slot 均已降为诊断或 `UNMEASURED`，没有参与硬门禁。

本轮曾因误用 `PersistentClient` 读取旧 Qwen 目录导致 housekeeping hash 从计划值变化；该目录未再作为基线，已新建上述隔离索引并保留旧目录作为事故证据。所有最终 run 评测后源 hash 保持不变。

由于 L1 决策为 `BLOCKED` 且 DeepEval 专用 Judge 未配置，L2 未启动；禁止输出 Gold/Real 语义分数或 `GO_TO_L2_TECHNICAL`。

## 15. Qwen Embedding 生产接入边界（2026-09-03）

已补充 `src/embedding_provider.py` 的 `EmbeddingIndexContract`：持久化索引必须记录并匹配 provider、model、dimension、index_version；缺失或不一致时显式抛错，禁止把 Qwen Query 向量发送到旧 deterministic 索引。Qwen provider 默认契约为 `Qwen/Qwen3-Embedding-0.6B`、1024 维、`qwen3-embedding-0.6b-v1`。

`DualEngineStorageManager` 支持 `embedding_index_version` 校验；`scripts/rebuild_qwen_embedding_index.py` 将完整契约写入三个 Chroma collection metadata 和 build provenance；两个 Qwen L1 评测脚本会校验索引契约。`src/cli.py` 新增 `--db-path`、`--chroma-dir`、`--embedding-backend {deterministic,siliconflow}`，默认仍为 deterministic，Qwen 只能显式选择并指向匹配的新索引，例如：

```powershell
python -m src.cli --embedding-backend siliconflow --chroma-dir reports\eval\embedding-qwen3-0.6b-...\chroma
```

本次未调用真实 API、未修改 `data/chroma`、未提交 commit。验证：相关测试 `26 passed`，`compileall src scripts` 通过，`git diff --check` 通过。后续若重建 Qwen 索引，必须使用新目录并保留 build report；生产切换前先停止 CLI，执行 Storage L1 和完整 Retrieval/Fusion/Assembler L1，固定 hash 后再做蓝绿切换并保留回滚副本。

## 16. L2/L3 闭环与三层总报告（2026-09-03）

### 16.1 L2 DeepEval 闭环

已在项目 `.venv` 安装并锁定 `deepeval==4.2.0`（`pyproject.toml` 的 `evaluation` optional dependency）。新增 `scripts/run_l2_deepeval.py` 与 `evals/deepeval_adapter.py`：

```text
raw query → deterministic expansion → Qwen retrieval → raw_first fusion
→ Assembler → Real final Context / Gold Context
→ production LLM generator (mimo-v2.5)
→ quadruple citation audit
→ DeepEval Judge (mimo-v2.5-pro, OpenAI-compatible)
```

L2 数据集：`evals/datasets/l2-golden30-20260903-060000/`，30 cases、dev=24、holdout=6、Session leakage=0，包含 `queries.jsonl`、`expected_answers.jsonl`、`expected_evidence.jsonl`、`split_manifest.json` 和 manifest。

最新真实 smoke 报告：`reports/eval/l2-deepeval-smoke-20260903-060000/`。

- Gold/Real 各 2 cases，4 条真实 trace、24 个指标（含 deterministic citation audit）；`ERROR=0`。
- Judge readiness=`SUCCESS`，DeepEval=`4.2.0`，model=`mimo-v2.5-pro`；API key 只通过 `LLM_API_KEY` 环境变量读取，未写入报告。
- Gold：Answer Relevancy `1.000`、Contextual Precision `1.000`、Contextual Recall `0.833`、Faithfulness `0.875`、Pedagogical GEval `0.600`、Citation audit `0.833`。
- Real：Answer Relevancy `1.000`、Contextual Precision `1.000`、Contextual Recall `1.000`、Faithfulness `0.750`、Pedagogical GEval `0.600`、Citation audit `0.250`。
- artifact hash 前后未变；release decision=`BLOCKED_L1_PRECONDITION`。这表示 L2 链路和 Judge 可运行，但 L1 尚未达到 `GO_TO_L2_TECHNICAL`，且 smoke 样本不足以签署语义发布门禁。

Gold/Real 解释遵循固定口径：Gold 高、Real 低表示上游检索/装配损失；两者都低则需要检查 Generator 或数据。当前结果支持继续修复引用覆盖和教学适配性，不支持把 smoke 分数写成生产质量。

### 16.2 L3 业务/教研闭环

已新增 `scripts/run_l3_business_eval.py` 与 `tests/test_l3_business_eval.py`。L3 数据集：`evals/datasets/l3-business-proxy-20260903-051500/`，30 个 Eedi 技术代理 cases（dev=24、holdout=6、student_insight=18、tutor_intervention=12），包含业务任务、接受标准、required evidence quads、qrels 和 split manifest。

最新报告：`reports/eval/l3-business-proxy-20260903-051500/`。

| L3 指标 | 全量 | Dev | Holdout | 状态 |
|---|---:|---:|---:|---|
| Candidate Recall@20 | 1.0000 | 1.0000 | 1.0000 | `DIAGNOSTIC_PROXY` |
| Recall@5 | 1.0000 | 1.0000 | 1.0000 | `DIAGNOSTIC_PROXY` |
| MRR | 1.0000 | 1.0000 | 1.0000 | `DIAGNOSTIC_PROXY` |
| Final document recall | 1.0000 | 1.0000 | 1.0000 | `DIAGNOSTIC_PROXY` |
| Final evidence recall | 0.6111 | 0.6319 | 0.5278 | `DIAGNOSTIC_PROXY` |
| Actionability | 1.0000 | 1.0000 | 1.0000 | `DIAGNOSTIC_PROXY` |

真实业务指标均明确为 `UNMEASURED`：`researcher_adoption`、`task_completion`、`learning_gain`、`online_feedback`。Eedi 数学数据只验证技术链路，不代表 NBCOT 或客户业务效果；当前 qrels 仍由人工引用派生，不是独立多文档 graded judgments。

### 16.3 最终三层交付报告

总报告：`reports/eval/l123-analysis-20260903-063000/`，包含 `report.md`、`report.json`，并在 JSON 中嵌入 L1/L2/L3 manifest、raw data 文件路径和 SHA-256。

最终交付状态固定为：

```text
L1_BLOCKED_L2_VALIDATED_L3_BUSINESS_UNMEASURED
```

含义：

1. L1 已完成可审计 closeout，但 Grounding citation recall、fallback Chunk Turn Recall、Retrieval P95 仍未过门禁；不能发布或切换生产索引。
2. L2 Gold/Real + DeepEval 真实闭环已验证，Judge 可用且无 ERROR，但当前为 2-case smoke，发布受 L1 前置阻塞。
3. L3 已完成 Eedi proxy 的召回/证据/行动性测评；真实教研员采纳、任务完成、学习增益和线上反馈没有数据，不能宣称业务质量通过。

三层总报告的原始证据入口：

- [L1 closeout](/E:/PIAgent/10-projects/AgentLearn/Eedi-RAG/reports/eval/l1-closeout-qwen-20260903-025500/closeout.md)
- [L2 DeepEval smoke](/E:/PIAgent/10-projects/AgentLearn/Eedi-RAG/reports/eval/l2-deepeval-smoke-20260903-060000/report.md)
- [L3 proxy report](/E:/PIAgent/10-projects/AgentLearn/Eedi-RAG/reports/eval/l3-business-proxy-20260903-051500/report.md)
- [L1/L2/L3 analysis](/E:/PIAgent/10-projects/AgentLearn/Eedi-RAG/reports/eval/l123-analysis-20260903-063000/report.md)

## 17. L2/L3 最新复核（2026-09-03）

为修正 L2 trace 的 split 映射和 citation audit 记录，已用新 run ID 重跑 2-case smoke：

- L2 最新数据集：`evals/datasets/l2-golden30-20260903-060000/`。
- L2 最新报告：`reports/eval/l2-deepeval-smoke-20260903-060000/`。
- 4 条真实 Gold/Real trace、24 个指标、`ERROR=0`、Judge readiness=`SUCCESS`，源 Qwen combined hash 前后仍为 `6bb387b4b9698009fffca6193a35d02b98b0424d017fc4add166f89428e9d67a`。
- Gold 指标：Answer Relevancy 1.000、Citation audit 0.833、Contextual Precision 1.000、Contextual Recall 0.833、Faithfulness 0.875、Pedagogical GEval 0.600。
- Real 指标：Answer Relevancy 1.000、Citation audit 0.250、Contextual Precision 1.000、Contextual Recall 1.000、Faithfulness 0.750、Pedagogical GEval 0.600。
- release decision=`BLOCKED_L1_PRECONDITION`；该 smoke 只证明 L2 闭环可运行，不构成完整 30-case 语义发布基线。

L3 最新全量 proxy 仍为 `reports/eval/l3-business-proxy-20260903-051500/`：30 cases，Candidate Recall@20/Recall@5/MRR/final document recall/actionability 均 1.0，final evidence recall 0.6111；researcher adoption、task completion、learning gain、online feedback 全部 `UNMEASURED`。

最终三层报告已更新为：`reports/eval/l123-analysis-20260903-071500/`，决策=`L1_BLOCKED_L2_VALIDATED_L3_BUSINESS_UNMEASURED`。该报告包含三层 manifest、raw observations/traces/cases 路径与 SHA-256、aggregate/case/slice 指标和组件优化结论；其中 L2 证据文字已与最新 24 条指标 raw report 对齐。

最终验证：全量 `pytest`=`151 passed, 1 failed`；唯一失败仍是既有 `test_llm_expansion_extractor_uses_strict_shared_gate` 扩容异常契约冲突；`compileall` 和 `git diff --check` 通过。`.venv` `pip check` 仍报告可选 `huggingface-hub` 要求 `click>=8.4.2` 与 DeepEval `click<8.4.0` 的依赖冲突；不影响已验证的 L2 import/runner，但后续应通过独立环境或兼容依赖约束解决。

## 18. Turn 级召回根因排查（2026-09-03）

本轮新增冻结的技术代理数据集、分级 qrels、人工复核队列和根因报告，用于区分卡片生成、卡片排序、窗口检索/排序、Assembler 与 Generator 的责任边界：

- Turn/card graded 数据集：`evals/datasets/turn-card-graded-provisional-20260903-110000/`，30 cases、21,570 条窗口 qrel、100 张卡片 qrel；标签为 0/1/2/3 的**规则派生 provisional labels**，`needs_human_review=true`，不能当作最终业务真值。
- 人工复核队列：`evals/datasets/graded-qrels-review-queue-20260903-112500/`，`review_queue.jsonl` 共 1,206 行（card=600、window=606；HIGH=323、MEDIUM=883），全部 `PENDING`。队列包含 Top-20 候选及 provisional relevance≥2 行。
- 根因报告：`reports/eval/turn-recall-root-cause-20260903-120000/`（`report.md`/`report.json`），绑定上述数据集、L1 closeout 和 L2 report 的 SHA-256；源 Qwen combined artifact hash 未变化，`artifact_mutated=false`。
- 三层总报告已重新生成：`reports/eval/l123-analysis-20260903-123000/`，在原 L1/L2/L3 证据之外嵌入 `turn_root_cause`、qrels 限制和卡片排序/指针重叠统计；交付决策仍为 `L1_BLOCKED_L2_VALIDATED_L3_BUSINESS_UNMEASURED`。

### 18.1 证据链结论

1. **窗口生成不是首因**：生产滑窗的全量 union 对 30/30 cases 覆盖全部 required Turn，`window_generation_coverage=1.0000`。因此“原文被 Chunking 丢失”目前没有证据。
2. **卡片生成/source_turn_ids 是首要上游损失**：角色正确的卡片指针召回均值仅 `0.6833`（14/30 cases 有指针缺失）；Student Insight=`0.6111`，Tutor Intervention=`0.7917`。典型缺失见 Session 23、14、206、59，要求的角色 Turn 没进入对应卡片指针。
3. **卡片 Embedding/排序不是已证实的主因**：卡片 graded Recall@3=`0.8333`，5 个卡片 Top-3 miss 全部与指针缺失重叠（`5/5`），无“指针完整但单独排序 miss”的案例。没有指针的证据，向量相似度无法补回。
4. **窗口检索排序存在独立次因**：6/30 cases 指针完整但 Turn coverage@3<1（Window Retrieval Rank）；例如 Session 258、77、42 在 Top-3 未覆盖全部 required Turns，但 Top-5 可覆盖。当前 Turn coverage@3=`0.7944`、@5=`0.9000`，dev=`0.7569`、holdout=`0.9444`。
5. **Assembler 当前不是文档级首因**：L1 的候选 retention/final evidence recall 等文档/槽位指标为 1.0；Turn 覆盖仍受上游指针和窗口排序输入约束。需要在最终 Context 上补充 Turn 级 trace，才能排除更细的装配丢失。
6. **Generator 是下游引用/回答质量损失点**：已发出的 citation 精度/角色/quote 约 `0.9889`、authorization=`1.0`，但 L1 citation recall 仅 `0.5889`，L2 Real citation audit=`0.25`；说明“引用通常是真的”与“覆盖了所有必要证据”是两件事。L2 Gold/Real smoke 的 Answer Relevancy 均 `1.0`，不能掩盖 citation coverage 和教学表达不足（Pedagogical GEval=`0.6`）。

### 18.2 当前优先级与限制

- P0：对 provisional 0/1/2/3 qrels 做人工多相关复核，优先 level-2/3 和 Top-20 候选；复核完成前不得把 Recall/nDCG 写成业务准确率。
- P1：对 14 个 pointer-loss case 重跑卡片抽取，要求按角色枚举**全部** claim-supporting Turns，并对 quote/source_turn_ids 做逐字和角色契约校验。
- P1：对 6 个 pointer-complete/rank-loss case 增加 Turn-aware reranker 或 exact-query channel，在同一冻结 qrels 上做 dev/holdout 对照。
- P1：Generator citation planner 必须覆盖 required evidence set；证据不足时显式报告 partial/degraded，不得从 gold qrels 反向复制 citation。
- 上述结论是 Eedi 技术代理诊断，不是 NBCOT 或真实业务结论；qrels、Assembler Turn trace 和真实业务反馈仍未完成。

## 19. 卡片 source_turn_ids 小规模修复评测（2026-09-03）

### 19.1 方案决策

暂不直接更换模型。当前首要缺陷是抽取契约只要求“引用真实”，没有要求“穷举同一论证链的全部直接证据”；因此先保留 `mimo-v2.5`，将抽取提示升级为 `evidence-completeness-v2`：按角色逐轮扫描，覆盖诊断/追问/脚手架/结果等直接证据，并在输出前做完整性自检。现有 JSON schema、角色校验、逐字 grounding gate 和 Fail-Fast 重试保持不变。

### 19.2 A/B 结果

评测脚本：`scripts/run_card_extraction_recall_eval.py`。样本为 6 个非 fallback pointer-loss sessions（23、14、206、33、287、469），baseline 直接读取与 Qwen 隔离索引同源的只读 `data/db/tutoring_knowledge.duckdb`，没有写入生产 DB/Chroma。

报告：`reports/eval/card-extraction-recall-small-20260903-133000/`。

| 指标 | 旧提示/卡片 | evidence-completeness-v2 | 变化 |
|---|---:|---:|---:|
| 角色正确指针召回（6 cases） | 0.1667 | 0.7222 | +0.5556 |
| misconception 子集（4 cases） | 0.1250 | 0.5833 | +0.4583 |
| strategy 子集（2 cases） | 0.2500 | 1.0000 | +0.7500 |
| 成功/失败 | — | 6/0 | 全部通过严格 schema/grounding |
| 改善/持平/下降 | — | 6/0/0 | 无下降案例 |

结果说明：同模型、只改抽取契约即可显著提高 Turn 指针覆盖，当前没有“必须换模型”的证据。但部分输出仍会纳入较宽的同角色 Turn，且 Session 14/206 仍漏目标 Turn；因此下一步应增加二次 evidence-audit/最小充分证据判定，并补做 pointer precision/noise，而不是直接把本次 `0.7222` 宣布为最终达标。

### 19.3 验证与下一步

- 本轮测试：抽取、graded qrels、复核队列和三层报告相关测试共 `28 passed`；`compileall`、`git diff --check` 通过。
- 全量 100 卡尚未重抽，生产 DB/Chroma 未修改；当前改动只影响后续真实抽取调用的提示词。
- 若二次 evidence-audit 在更大 dev 集上仍低于目标，再对候选模型做同数据、同 schema、同 qrels 的 A/B；模型切换前必须记录成本、延迟、grounding 失败率和指针 precision。

## 20. 二次 evidence-audit 状态（2026-09-03）

已实现 `evidence_audit=True` 的第二遍严格审计：首轮卡片通过后，模型只返回两组最小充分 Turn ID；审计结果经 schema、存在性、角色和去重校验后，才与首轮指针合并；审计失败抛出 `LLMExtractionError`，不会静默沿用或伪造指针。

本地契约测试已通过，但真实 provider 评测暂未完成：

- `reports/eval/card-extraction-recall-audit-small-20260903-140000/`：3-case 运行在 provider 响应体读取阶段长时间无返回，已安全中止，状态 `UNMEASURED`。
- `reports/eval/card-extraction-recall-audit-small-20260903-143000/`：同样在首轮/审计调用阶段挂起，已安全中止，状态 `UNMEASURED`。
- `reports/eval/card-extraction-recall-audit-probe-20260903-150000/`：1-case、60 秒 probe 仍无法取得可解析响应，状态 `UNMEASURED`。

因此当前决策是：**不把 evidence-audit 接入正式扩容，也不据此更换模型**。已验证的 `evidence-completeness-v2` 继续作为正式抽取提示；待 LLM provider 响应稳定后，再用相同 6-case/14-case、显式超时和冻结 qrels 复测。只有在第二遍取得真实、可审计且无明显 precision 退化的结果后，才将 `extract_piu_with_llm` 显式开启该 pass，并先写入 staging、重新建索引和回归 L1。

### 20.1 直接替换结果

已按当前授权将正式扩容代码路径替换为 `evidence-completeness-v2`（同一 `mimo-v2.5`），并在日志中固定记录 `prompt_version`；`evidence_audit` 保持显式 `false`。同时，扩容抽取失败不再被转换为 `failed` 后跳过，改为立即抛错，禁止部分成功卡片写入。

注意：现有 `data/db/tutoring_knowledge.duckdb`、`data/chroma` 和 100 张历史卡仍是旧快照，尚未执行全量重抽取或覆盖。全量更新必须在 provider 稳定后写入 staging、完成全量成功与 Turn 指针审计，再进行蓝绿切换；本次没有伪装成“数据已经全部替换”。

## 21. 100 场历史卡全量重抽取与生产切换（2026-09-03）

### 21.1 全量重抽取

新增 `scripts/reextract_cards_staging.py`，从生产 DB 只读读取 100 个 `intervention_id`，使用 `mimo-v2.5`、`evidence-completeness-v2`、`temperature=0.0`、严格角色/逐字 grounding；采用 2 workers 并发，自纠 `max_retries=4`。会话级失败会被记录并继续处理，遍历结束后只有失败清单为空才创建 staging DB/Chroma。

最终 staging：`reports/eval/card-reextraction-staging-20260903-200000/`。

- 100/100 sessions 成功，0 failures；200 张卡片、719 个滑动窗口、2,335 个 Turn。
- staging snapshot：DuckDB `100/2335/100/100/719`；Chroma `100/100/719`。
- staging prompt/model：`evidence-completeness-v2` / `mimo-v2.5`；`evidence_audit=false`。
- staging DB SHA-256=`272ea013e792952aedee0a5aed38d9563fad7ced46281200b060306ce99dc184`。

期间曾有两次失败 staging 尝试，均未写入 DB/Chroma：

- `card-reextraction-staging-20260903-163000`：Session 33 在 2 retries 后角色指针失败。
- `card-reextraction-staging-20260903-180000`：Session 100 在 4 retries 后因弯撇号 `don’t`/ASCII `don't` grounding 差异失败；随后补充了仅针对排版标点的 NFKC 归一化及测试。

### 21.2 蓝绿切换与备份

已将新 deterministic staging DB/Chroma 切换到生产路径：

- 当前生产 DB：`data/db/tutoring_knowledge.duckdb`，SHA-256=`272ea013e792952aedee0a5aed38d9563fad7ced46281200b060306ce99dc184`。
- 当前生产 Chroma：`data/chroma`；与 staging Chroma 逐集合计数一致，最近一次评测前后生产 combined hash=`de05478b086eead4b7a7f4b58b52f418f1450cb8766ce4274dd73c33e9e3e693`（Chroma SQLite 打开后可能产生内部 metadata hash 变化，评测前必须重新记录实际 hash；切换初始 hash=`9b8cacfd...` 仅作历史记录）。
- 旧 DB 备份：`data/db/tutoring_knowledge.pre-card-reextract-20260903-220000.duckdb`，SHA-256=`09ef4e7d201f1f61034d5a7dd41345f4a122be01bfffae6713a7aae724b4a3ad`。
- 旧 Chroma 备份：`data/chroma-backups/card-reextract-20260903-220000/`。
- Qwen 评测索引保留在：`reports/eval/card-reextraction-qwen-index-20260903-203000/chroma/`，未替换生产 deterministic Chroma。

### 21.3 重抽后评测

同一 30-case、21,570 window qrels、card graded qrels 的评测产物：

- Qwen staging Turn/card 评测：`reports/eval/card-reextraction-turn-graded-20260903-210000/`。
- 生产 deterministic 评测：`reports/eval/card-reextraction-turn-graded-production-20260903-221000/`。
- 两者指标一致；相对旧生产 deterministic：card Recall@3=`0.3333→0.4667`、Recall@20=`0.6667→0.8000`、MRR=`0.3167→0.3833`、nDCG@5=`0.3000→0.3200`、角色正确 pointer recall=`0.6833→0.8722`。Card precision@5=`0.1200→0.1000`，说明指针覆盖提升伴随少量噪声，需后续最小充分证据审计。
- Qwen 版本 card Recall@3/@5/@20、MRR 均为 `1.0000`，nDCG@5=`0.9395`。
- Turn coverage@3=`0.7944`、@5=`0.9000` 未变化，因为本次只重抽卡片，滑动窗口未改变。
- 根因报告：`reports/eval/card-reextraction-turn-root-cause-20260903-223500/`；pointer-loss cases 从 14 降至 7，`NO_UPSTREAM_TURN_LOSS_AT_K3` 从 10 增至 15，card rank miss=`0`。
- 完整 L1 Qwen 组件评测：`reports/eval/l1-card-reextraction-qwen-full-20260903-223000/`；citation recall=`0.8056`（旧=`0.5889`），但仍低于 0.95；Turn recall@3=`0.7944`、Turn MRR=`0.8467` 仍未过门禁，整体 `BLOCKED`。

### 21.4 当前边界

生产卡库已经是全量新抽取版本，但 L1 尚未整体通过，不能宣称发布质量已达标。Generator citation coverage、窗口 Turn 排序、Assembler Turn 级 trace 和 provisional qrels 人工复核仍需继续；若需回滚，可使用上述带时间戳的 DB/Chroma 备份，禁止直接删除当前生产数据。

## 22. BM25 + Dense 与 Qwen Reranker（2026-09-04）

### 22.1 检索实现

当前 `src/retriever.py` 已从 Dense-only 扩展为真实 BM25 + Dense：每个 Chroma collection 启动时构建内存 BM25 index，取 BM25 与 Dense 候选 union，再按归一化分数融合：

```text
Dense_Hybrid = alpha * Cosine_Norm + (1 - alpha) * L2_Similarity
Final = (1 - bm25_weight) * Dense_Hybrid + bm25_weight * BM25_Normalized
```

默认 `retrieval_mode=bm25_dense`、`bm25_weight=0.35`；仍保留 `retrieval_mode=dense` 回归模式。候选会记录 `bm25_raw_score`、`bm25_score`、`dense_hybrid_score` 和 `retrieval_channels`。这不是 BM25 与 Dense 的 RRF，而是 lexical/dense 候选合并后的加权排序；多视角 RRF 仍在更上层用于多个查询视角融合。

### 22.2 Top-K sweep

脚本：`scripts/run_bm25_dense_sweep.py`；数据：30 cases、21,570 provisional window qrels、同一 dev/holdout split；BM25 weight=`0.35`。

生产 deterministic 结果：

| Candidate cutoff | Recall@cutoff | Card Recall@cutoff | Turn coverage@cutoff | nDCG@5 | MRR |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.6050 | 0.9333 | 0.6667 | 0.6073 | 0.6844 |
| 10 | 0.6906 | 0.9667 | 0.7444 | 0.6073 | 0.6892 |
| 15 | 0.7239 | 0.9667 | 0.7556 | 0.6073 | 0.6892 |
| 20 | 0.7433 | 1.0000 | 0.7889 | 0.6073 | 0.6912 |

同口径 Dense-only 对照为 Recall@5/10/15/20=`0.2961/0.3389/0.3706/0.3817`，Card Recall@20=`0.8000`，Turn coverage@20=`0.5000`。BM25+Dense 对精确数字、短语和对白词面命中带来明确收益。

Qwen embedding 隔离索引结果：Recall@5/10/15/20=`0.6550/0.7739/0.8072/0.8183`，Turn coverage=`0.7444/0.8000/0.8444/0.8444`，Card Recall 全部=`1.0000`。若只按最大 Recall，Top-20 最高；但 Top-15 与 Top-20 Recall 差仅 `0.0111`，Turn coverage 相同，因此按“最大 Recall 容忍 0.02、选择最小池”规则，正式选择 Top-15 作为 reranker candidate pool。

报告：

- deterministic sweep：`reports/eval/bm25-dense-sweep-production-20260903-232000/summary.md`
- Dense-only 对照：`reports/eval/dense-only-sweep-production-20260903-233000/summary.md`
- Qwen sweep：`reports/eval/bm25-dense-sweep-qwen-20260904-020000/summary.md`
- 更新后的容忍度选择 sweep（selected Top-15）：`reports/eval/bm25-dense-sweep-qwen-20260904-040000/summary.md`

### 22.3 Qwen3-Reranker-0.6B

新增 `src/reranker_provider.py`，调用 SiliconFlow `/v1/rerank`，模型为 `Qwen/Qwen3-Reranker-0.6B`；新增 `PedagogicalGoldAssembler(model_reranker=...)`，模型重排发生在 MMR/槽位装配之前。reranker 输入包含卡片文本和已回溯的直接证据 Turn，避免只优化卡片语义而忽略证据。

完整 30-case 结果（BM25+Dense Top-15、Qwen embedding）：

- provider success/failure=`30/0`，retry=`0`，reranker latency P50=`511.9ms`、P95=`1080.5ms`。
- rerank 前 candidate pool Card MRR=`0.9333`、Card Turn coverage=`0.8722`。
- rerank 后 Card Recall@5/10/15/20 均=`1.0000`，Card MRR=`0.9833`，nDCG@5=`0.9236`，Precision@5=`0.3600`。
- rerank 后 Card Turn coverage=`0.8722`，与 pool 相同；说明 reranker 改善了卡片排序，但没有增加卡片本身缺失的证据 Turn。

报告：`reports/eval/qwen-reranker-full-qwen-top15-20260904-030000/summary.md`；逐 case 顺序和分数在 `results.jsonl`。Top-20 对照报告为 `reports/eval/qwen-reranker-full-qwen-20260904-023000/summary.md`，Top-15 的最终 Card Recall/MRR/证据覆盖不退化，reranker P50/P95 从 `511.9/1080.5ms` 降至 `379.4/759.3ms`。2-case probe 和 deterministic reranker 对照也保留在同目录邻近版本中。

### 22.4 运行方式与边界

CLI 已增加（启用 reranker 时 pipeline 自动至少抓取 Top-15）：

```powershell
python -m src.cli --retrieval-mode bm25_dense --bm25-weight 0.35 --reranker-backend siliconflow
```

`--reranker-backend siliconflow` 是显式外部 API 开关；未配置时仍使用确定性 MMR。BM25 index 是进程内从 Chroma 文档构建的，不改变生产 artifact。当前最终 L1 仍不能发布：qrels 需要人工复核，窗口 Turn coverage 和 Generator citation coverage 仍未达到门禁；reranker 分数不能替代引用审计或回答质量评测。

## 23. 生产双 Chunk 路线与 L1 全量对照（2026-09-04）

### 23.1 生产可选开关

已在 `src/cli.py`、`src/rag_pipeline.py`、`src/reranker.py`、`src/retriever.py` 和 `evals/adapters/assembler.py` 接入显式 `chunk_strategy`：

```powershell
python -m src.cli --chunk-strategy card
python -m src.cli --chunk-strategy fallback
```

`card` 检索 misconception/strategy 卡片并通过 `source_turn_ids` 回溯原文；`fallback` 直接检索 `fallback_windows`，将真实窗口作为 Context 和 citation evidence。默认仍是 `card`，没有因为一次评测自动切换生产默认值。

### 23.2 两套完整 L1 运行

统一脚本：`scripts/run_l1_qwen_full_eval.py`，同一 Qwen artifact、30-case、split、provisional qrels 和 hash 保护，覆盖 Storage、Chunk Structure、Chunking、Grounding、Rewrite、Fusion、Retrieval、Assembler。

- Card：`reports/eval/l1-card-strategy-20260904-090000/`，`SUCCESS=2009 / FAILED=461 / UNMEASURED=263`，`release_status=BLOCKED`。
- Fallback：`reports/eval/l1-fallback-strategy-20260904-110000/`，`SUCCESS=1317 / FAILED=543 / UNMEASURED=878`，`release_status=BLOCKED`；Rewrite/Fusion 因 content-first 生产链路不执行而显式标记 `UNMEASURED/NOT_APPLICABLE`。
- 两次运行的 Qwen combined hash 均为 `c47318847151b90023f2dceddc7ab60b265725f546bc633b8d8f6e211c1408bb`，评测前后未改变。
- 详细解释和专项 Top-15 + reranker 对照见 `docs/Chunk_Strategy_L1_Comparison_20260904.md`。

### 23.3 结论

L1 门禁结果不支持“fallback 全面优于 card”：无 reranker 的当前 L1 中，card 的目标定位、Turn MRR、Assembler retention 和延迟更好；fallback 的 selection precision 更高但真实窗口排序和 P95 仍是硬问题。专项同一 Top-15 + Qwen reranker 对照则显示 fallback 的**全部 required Turn evidence recall=0.8222**，明显高于 card 的 `0.4889`，而 card 的目标角色证据 recall=`0.8556` 略高于 fallback 的 `0.8222`。这说明两条路线优化目标不同：

- `card`：短 Context、语义摘要集中、首个目标卡命中强，适合摘要/诊断。
- `fallback`：保留连续真实对白、绕过卡片指针遗漏、直接引用更完整，适合证据型问答。

当前建议：保留双开关；证据型问答显式选择 fallback，摘要/诊断优先 card。下一步优先优化 fallback 的候选召回/P95、Generator citation recall 和人工多相关 graded qrels，完成前不得宣称 L1 通过或 NBCOT 业务质量已证明。

## 24. 卡片语义与原文索引解耦、增量录入（2026-09-04）

本轮新增：

- `src/evidence_index.py`：从真实 `CleanedSession.turns` 构建 6-turn/3-step 重叠逻辑链窗口，强制 union 覆盖整场会话；不调用 LLM，不生成原文或 Turn ID。
- `bind_card_to_evidence_index()`：misconception 绑定所有非 noise 学生 Turn，strategy 绑定所有非 noise 导师 Turn；忽略 LLM 返回的 source pointers，并写入 `source_index_ids`、`evidence_binding_policy`、`evidence_index_version`、`evidence_index_hash`。
- `src/extraction_cache.py`：按完整 session hash + Prompt/model/schema/temperature/binding/index contract 做 success-only 原子缓存；损坏、失败、契约不匹配按 miss 处理。
- `scripts/incremental_card_ingest.py`：支持 `changed-only`、`audit-only`、`full`、指定 `--session-ids`、有限并发和可恢复失败 manifest。`audit-only` 不调用语义 LLM；每次还会独立写出 `evidence-index-<run_id>.jsonl`。
- 旧 `reextract_cards_staging.py` 与 `expand_knowledge_base.py` 已切到 `semantic_only=True → deterministic evidence binding`，避免旧全量入口继续信任 LLM 指针。
- 运行手册：`docs/Incremental_Card_Ingestion_Runbook.md`。

典型增量命令：

```powershell
python scripts/incremental_card_ingest.py --db data/db/tutoring_knowledge.duckdb --cache-dir reports/cache/card-extraction --mode changed-only --workers 4
python scripts/incremental_card_ingest.py --db data/db/tutoring_knowledge.duckdb --cache-dir reports/cache/card-extraction --mode audit-only --session-ids 14,206,469
```

首次运行仍需处理全部 cache miss；同一输入第二次运行应 `llm_processed_count=0`。修改 BM25、RRF、Reranker、Generator 或仅重绑原文时不需要重新调用 100 场 LLM。语义缓存 key 不包含 evidence binding policy/index version；只改原文索引规则会复用语义卡并重新绑定，不会触发 LLM。修改 Prompt/model/schema 时只重新处理契约不匹配的会话。当前实现仍只写隔离 cache/staging JSONL，不自动 promotion 到生产 DB/Chroma；promotion 必须在 hash、Storage parity、L1 和人工复核通过后显式执行。

本次历史卡片确定性回填与 card-specific 评测：`scripts/rebind_card_evidence_staging.py` 在 `reports/eval/card-evidence-rebind-staging-20260904-001500/` 生成 100 场/200 张卡片 staging，`llm_calls=0`；`scripts/eval_card_evidence_rebind.py` 在 `reports/eval/card-evidence-rebind-card-metrics-20260904-003000/` 对 30 cases 做同口径前后对照。Card evidence Recall@20 从 `0.8722` 提升到 `1.0000`，Assembler final Context evidence recall 从 `0.6556` 提升到 `0.7667`；但指针均值 `128.2→242.1`、evidence precision 下降，当前是 recall-first 结果，仍需最小充分证据筛选和人工 qrels 复核。生产 DB/Chroma 未 promotion。

### 24.1 Card-level vs logical-evidence-level Rerank（2026-09-04）

按用户要求，对同一 Qwen Embedding Top-20 卡片池、同一 30-case/split、同一 BM25+Dense 配置和真实 Qwen3-Reranker 做了两种重排单位的 A/B：

- `card_N`：重排 20 张卡片，选择 N=`1/2/3` 张；再展开卡片证据。
- `logical_evidence_M`：从这 20 张卡片展开真实 W6/S3 逻辑链窗口，重排原文窗口，选择 M=`3/5/8/10` 个证据单元。

报告：`reports/eval/card-vs-evidence-rerank-20260904-033000/`。

| Variant | Target-role Recall | All-required Recall | Evidence precision | Context tokens | Reranker input tokens | Reranker mean ms |
|---|---:|---:|---:|---:|---:|---:|
| card_1 | 0.9667 | 0.5833 | 0.1739 | 476 | 10,338 | 666 |
| card_2 | 1.0000 | 0.6056 | 0.0819 | 957 | 10,338 | 666 |
| card_3 | 1.0000 | 0.6056 | 0.0538 | 1,451 | 10,338 | 666 |
| logical_evidence_3 | 0.9389 | 0.9000 | 0.1191 | 817 | 38,974 | 2,582 |
| logical_evidence_5 | **0.9833** | **0.9556** | 0.0827 | **1,300** | 38,974 | 2,582 |
| logical_evidence_8 | 1.0000 | 1.0000 | 0.0551 | 2,028 | 38,974 | 2,582 |

Dev/holdout 稳定性：`logical_evidence_5` 的 all-required Recall=`0.9444/1.0000`，Context=`1314.6/1242.5 tokens`；`logical_evidence_3` 为 Recall=`0.8889/0.9444`。因此按“回复全面 + token≤1500”的约束，数据选择 `logical_evidence_5` 作为证据型问答的首选；`card_1` 保留为低延迟/短摘要路线。`logical_evidence_8/10` 虽 Recall=1.0，但平均 Context 超过 1500，不作为默认。

重要实现差距：当前 `PedagogicalGoldAssembler` 的生产 card 路径仍是“先重排卡片、每槽位 MMR 选 1 张、再按 session/turn 排证据”，没有原样消费 logical-evidence Reranker 的排序结果。下一步改造必须增加 `rerank_unit=logical_evidence` 和 `evidence_selection_count=5`，让 Assembler 只做去重/预算截断并保留 Reranker 顺序；最终回复全面性需再用 L2 citation/answer-completeness 评测确认。Reranker 输入 token 约为 card route 的 `3.77x`，均值延迟约 `3.9x`，不能在没有延迟预算决策的情况下无条件替换所有请求。

该实现差距已关闭：`PedagogicalGoldAssembler` 新增 `rerank_unit=card|logical_evidence` 和 `evidence_selection_count`；`DualMetricRetriever.expand_card_candidates_to_evidence_units()` 从 DuckDB 真实 Turn 构建 W6/S3 窗口；card pipeline 在 evidence-level 模式注入这些 units；Assembler 按 Reranker rank 最多选 5 个逻辑链单元，预算不足时停止、保序去重而不做破坏性截断。CLI 新增 `--rerank-unit`、`--evidence-selection-count` 和 `--reranker-pool-size`，默认池仍为 15；若要复现本次 Top-20 A/B，需显式设为 20。真实 staging smoke 已通过 `AUDITED_100_VERIFIED`（实际 4 个 units、20 条引用、1390 tokens、无截断，总耗时约 7.7s）。数据选择仍为：证据型问答用 `logical_evidence_5` 上限，低延迟摘要用 `card_1`；生产默认未自动切换。

### 24.2 三路 A/B/C：Context-Injected Child Rerank（2026-09-04）

按用户要求完成同口径三路真实 Qwen Reranker 评测：A=`Card-1`、B=`Logical-evidence-5`、C=`Context-injected Child-3/5/8`。报告：`reports/eval/card-evidence-child-rerank-20260904-110000/`，详细分析：`docs/Card_LogicalEvidence_ContextInjectedChild_Eval_20260904.md`。

结果：B 的 all-required Recall=`0.9556`，明显优于 A=`0.5833` 和 C-3/5/8=`0.4611/0.6278/0.7667`；C-3/5/8 的 mean Rerank=`1726ms`，介于 A=`683ms` 与 B=`2388ms`，但没有换来相应召回提升。额外 Parent Top-5 检查显示目标 Session `30/30` 都在 Parent Top-5，因此 C 的损失发生在 Child 排序而不是 Parent 初检；平均 Child 候选 `123.2` 个，短 required Turn 被同 Session 普通邻近 Turn 淹没。C-8 还有 `16/30` cases 超过 1500-token budget。

决策：当前不把 C 接入生产。继续使用 B 的 logical-evidence 路线（最多 Evidence-5，Assembler 按预算实际选择 4～5 个）处理证据型问答，A Card-1 处理低延迟摘要。C 保留为实验候选；下一版必须减少 Child 候选池并增加 coverage-aware selection，不能仅靠 Parent 摘要注入宣称 Recall 提升。

### 24.3 Anchored Context-Injected Child 复测（2026-09-04）

按用户指定的 `Top-20 Cards → Card Rerank → Parent Top-3/5 → source_turn_ids ±1 Child → per-session quota → coverage-aware selection → Assembler` 重新做全矩阵评测。报告：`reports/eval/anchored-context-injected-child-20260904-130000/`，分析：`docs/Anchored_Context_Injected_Child_Eval_20260904.md`。

结果最佳为 `Parent3 / quota5 / Child5`：all-required Recall=`0.7056`（dev=`0.7153`、holdout=`0.6667`）、Precision=`0.1659`、Context=`1151.9 tokens`；仍低于 B Logical-evidence-5 的 Recall=`0.9556`。Parent Top-5 目标 Session=`30/30`，所以损失发生在 Child 排序/选择，而不是 Parent 初检。当前 coverage-aware 只覆盖“任意 anchor”，无法识别 required anchor；Child±1 重叠窗口仍被普通邻接 Turn 淹没。C 继续保留为实验候选，不接入生产默认。

### 24.4 Anchored Logical Windows 独立 A/B（2026-09-04）

按用户同意的独立 A/B 口径完成真实 Qwen 评测：

```text
Baseline: Top-20 Cards → Card Rerank → 全部 W6/S3 Logical Windows → Window Rerank → W4/W5 → Assembler
Anchored: Top-20 Cards → Card Rerank → Parent Top-3/5 → 仅保留覆盖 source_turn_ids 的 W6/S3 → Window Rerank → W4/W5 → Assembler
```

报告：`reports/eval/anchored-logical-windows-ab-20260904-152000/`；详细分析：`docs/Anchored_Logical_Windows_Eval_20260904.md`。运行状态为 `MEASURED`，30/30 cases、180 条变体记录、0 case 失败；源 DB/Chroma 前后 artifact hash 一致。

结果矩阵（最终 `Assembler context.evidence_turns` 指标）：

| Variant | Final Context Recall | Dev / Holdout | Final Precision | Candidate windows | Candidate reduction | Window Rerank mean | Context tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline W4/W5 | `0.7889` | `0.7500 / 0.9444` | `0.1699` | `154.4` | `0%` | `2182ms` | `1324.7` |
| Anchored Parent-3 W4/W5 | `0.7500` | `0.7431 / 0.7778` | `0.1556` | `21.2` | `86.2%` | `366ms` | `1282.6` |
| Anchored Parent-5 W4/W5 | `0.7278` | `0.7292 / 0.7222` | `0.1539` | `34.8` | `77.3%` | `543ms` | `1275.9` |

Anchored 候选窗口 union Recall 全部为 `1.0000`，与 Baseline 相同；Parent Session Recall 也为 `1.0000`。因此锚定过滤没有直接删除 required Turn，Recall 损失发生在 Window Rerank 与 Assembler 预算选择。Parent-3 相对 Baseline 将 Window Reranker 输入减少 `72.3%`、均值延迟减少 `83.2%`，但最终 Recall 低 `3.89` 个百分点；Parent-5 成本更高且 Recall 再低 `2.22` 个百分点。1500-token 预算下 Assembler 平均实际只装入约 `3.0` 个窗口，所以 W5 请求没有提高最终 Context Recall。

决策：Anchored Logical Windows 当前不接入生产默认；可作为显式低延迟实验开关，但必须接受 provisional Recall 损失。下一轮应优先优化 Assembler 的预算选择/窗口去冗余，并用多相关 graded qrels 与 L2/DeepEval 验证，不应继续仅增加 Parent Top-K。

### 24.5 生产默认切换为 Anchored Parent-3/W5（2026-09-04）

用户确认 4～6pp 的 provisional Recall 损失可接受，以换取超过 80% 的窗口 Reranker 输入与延迟下降。生产代码现已接入两段式 Anchored 路线：

```text
Embedding/BM25+Dense Top-20 Cards
→ Card Rerank（每个语义 lane）
→ 总 Parent Top-3（双 lane 至少各 1 张）
→ source_turn_ids 锚定 W6/S3 Logical Windows
→ 注入 Parent metadata 的 Window Rerank
→ Window Top-5
→ Assembler（1500-token budget）
```

实现位置：`src/retriever.py` 的 `expand_anchored_logical_windows()`、`src/reranker.py` 的 `anchored_logical_window` 模式、`src/rag_pipeline.py` 的两段 Rerank 编排，以及 `src/cli.py` 的默认解析。启用真实 `SiliconFlow` Reranker 且未指定 `--rerank-unit` 时，CLI 默认 `anchored_logical_window`、`reranker_pool_size=20`、总 `parent_card_count=3`（两个非空语义 lane 各保留至少 1 张，再按分数补足）、`evidence_selection_count=5`；无外部 Reranker 的 deterministic 本地运行仍默认 `card`。旧 `card` 和 `logical_evidence` 路线均可显式回退。

Anchored 生产模式要求显式配置模型 Reranker；缺少模型时会抛出受检错误，不会静默退回或伪造“已重排”状态。父卡元数据只用于窗口 Reranker 输入，最终 citation 仍由真实 Turn 审计授权。

回归证据：Anchored 生产新增测试与既有 RAG/Reranker/Retriever 测试共 `45 passed`；完整报告见 `docs/Anchored_Logical_Windows_Eval_20260904.md` 和 `reports/eval/anchored-logical-windows-ab-20260904-152000/`。生产 DB/Chroma 未被重写。

### 24.6 Assembler 重叠感知优化（2026-09-04）

用户确认两套路线被 1500-token 预算卡住的主要原因是 W6/S3 窗口重叠重复消耗 Token。本轮仅优化 Assembler：同一 `(session_id, turn_id)` 在最终 Prompt 中只渲染一次，按新增 Turn 的增量 Token 判断预算，同时保留窗口 rank 和真实 Turn provenance。

最终真实 Qwen 复测报告：`reports/eval/anchored-logical-windows-assembler-final-20260904-134055/`；状态 `MEASURED`，30/30 cases、180 条变体记录、0 case 失败，源 artifact hash 未变化。

| Variant | Target-role Recall | Final All-required Recall | Dev / Holdout | Evidence Precision | Context Tokens | 实际装入窗口 | Window Input Tokens | 平均 Rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline W5 | `0.9500` | **`0.9222`** | `0.9028 / 1.0000` | `0.1342` | `1277.1` | `4.83` | `22862.3` | `2018ms` |
| Anchored Parent-3 W5 | `0.9056` | **`0.9000`** | `0.9028 / 0.8889` | `0.1340` | `1257.3` | `4.87` | **`6354.9`** | **`372ms`** |
| Anchored Parent-5 W5 | `0.9222` | `0.9000` | `0.9028 / 0.8889` | `0.1344` | `1252.7` | `4.87` | `10218.2` | `548ms` |

相对优化前，Baseline W5 Final Recall `0.7889→0.9222`，Anchored Parent-3 W5 `0.7500→0.9000`；平均实际窗口由约 `3` 提升到约 `4.8`，两者均无预算违规。Anchored Parent-3 相对 Baseline 仍低 `2.22pp` Recall，但 Window Reranker 输入减少 `72.2%`、均值延迟减少 `81.6%`，Precision 基本持平。最终生产默认继续为真实 SiliconFlow Reranker 下的 Anchored Parent-3/W5；最大召回需求可显式回退 logical-evidence W5。

### 24.7 L2 DeepEval Anchored Parent-3/W5 全量回归（2026-09-04）

L2 runner 已改为复用生产 `prepare_context()`，并通过 4 个可恢复 shard 完成 30 Gold + 30 Real 全量 DeepEval。合并报告：`reports/eval/l2-anchored-parent3-w5-full-20260904-160000/`；专项分析：`docs/L2_DeepEval_Anchored_Full_20260904.md`。

运行状态：Judge `mimo-v2.5-pro` 可用，360 条指标记录，源 artifact hash `b45860c8…a8aeb9` 前后不变；报告 release decision 为 `BLOCKED_L1_PRECONDITION`，因为历史 L1 closeout 仍为 `BLOCKED`。

| Track | Answer Relevancy | Contextual Precision | Contextual Recall | Faithfulness | Pedagogical GEval | Citation Audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gold | `0.9583` | `0.9667` | `0.6867` | `0.6972` | `0.5033` | `0.7111` |
| Real | `0.9400` | `0.9550` | `0.8022` | `0.7528` | `0.5600` | `0.2222` |

Real 的相关性和上下文排序已较好，但 Citation Audit `0.2222`（0/30 达到 0.95）、Faithfulness `0.7528`、Contextual Recall `0.8022` 和 Pedagogical GEval `0.5600` 均未达首轮门槛。全量 6 条 `generation/audit CitationAuditError`：Gold 1、Real 5；主要是 Generator 复制 `[Turn N]` 标记进 quote_text 或只输出部分 required 引用。结论：Assembler 上下文优化已验证有效，但 L2 的主要剩余瓶颈是 Generator 引用契约、忠实度和教学表达，不应宣称“全线飘绿”。下一步固定 Anchored Parent-3/W5，专项修复 Generator citation schema/prompt，并用多相关 graded qrels 和教研员 rubric 校准 DeepEval。

L2 runner 现支持 `--case-start/--max-cases` 分片、每 case checkpoint 和 `--resume`，并新增 `scripts/merge_l2_deepeval_reports.py` 做 artifact/route/case 完整性校验后合并。由于全量远端调用存在长尾，后续必须优先使用分片运行；不能把串行长时间无输出视为成功。

### 24.8 Generator-first v2 与 v1 对照（2026-09-05）

已提交基线 commit：`2040f15`（L1 Anchored 与 L2 baseline）。Generator v2 实现位于 `src/generator_contract.py`、`src/rag_pipeline.py` 和 `scripts/run_l2_deepeval.py`，包含 Prompt v2、evidence ID 物化、claim-level 校验、Gold Context v2 和 rubric v2。v2 全量合并报告：`reports/eval/l2-generator-v2-full-20260905-000000/`；Gold/Real 各 30 case，主要均值为 Gold Faithfulness `0.9241`、Pedagogical `0.9286`，Real Faithfulness `0.7704`、Pedagogical `0.9231`，但 Citation Audit 为 `0.5690/0.4877`，Real Contextual Recall 为 `0.7404`，尚未通过 L2 门禁。

恢复 API 余额后已启动新的 v1 分片：`l2-generator-v1-shard1..4-20260905-0200`，参数固定为 Anchored Parent-3/W5、BM25+Dense、Gold Context v2、Generator contract v1、Judge `mimo-v2.5`。当前已持久化 12/30 case（1–3、9–11、17–19、24–26）；由于后续 DeepEval 远端长请求超过等待窗口，任务已停止，checkpoint 保留，可用 `--resume` 继续。完成前不得把此前 `20260905-0100-partial` 的 402 结果用于选型。trace 现在额外保存 `generator_claims`，便于后续按 claim 分析引用覆盖。

当前 release 仍为 `BLOCKED_L1_PRECONDITION`；远端 API 的 402/connection error 必须按 ERROR 记录，禁止用默认回答或估计分数补齐。此前曾有凭据误打印到工具输出，需轮换对应密钥；文档与报告不保存密钥值。

Gold 对齐修复已完成并提交：Gold Context v2 现在按 required Turn 生成去重节点，DeepEval 按指标分别使用核心答案/结构化答案和多节点上下文；trace/report 增加 `generator_core_answer`、`claim_coverage`、`gold_alignment_warnings`、节点数与上下文长度。30-case 本地结构审计 required Turn 缺失为 `0`、ground-truth 泄漏为 `0`；Gold smoke `l2-gold-alignment-smoke-20260905-0901` 的核心语义指标为 `1.00/1.00/0.90/0.90`，但 Citation Audit 为 `0.50`，尚未构成全量通过。完整本地回归：`235 passed`（后续新增诊断测试需再次执行）。
