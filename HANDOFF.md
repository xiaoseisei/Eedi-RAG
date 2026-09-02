# Eedi-RAG 项目交接文档

> 面向完全没有上下文的新对话。先读本文件，再读“必读文件”即可恢复项目背景、当前状态和施工顺序。

## 0. 交接摘要

- **工作区**：`E:\PIAgent\10-projects\AgentLearn\Eedi-RAG`
- **根 Git 仓库**：`E:\PIAgent`（注意：根仓库包含多个项目，不能对整个仓库执行清理或重置）
- **当前分支**：`master`
- **当前任务**：在真实教育辅导需求场景下，使用 Hugging Face 的 Eedi Tutoring Dialogues 作为公开代理数据，完成一个从数据治理、知识抽取、RAG 构建、检索与证据优化到业务落地的 0→1→100 完整路线。
- **本次交付**：新增 L1 v2 具体实施说明书，并更新本文件以固定六组件结论；没有据此修改业务代码、没有安装依赖、没有重置已有改动。
- **当前判断**：架构方向有竞争力，但当前质量基线仍属于诊断基线，不能宣称已经达到生产可用或已证明 NBCOT 业务价值。
- **当前首要方向**：先修正 Grounding、Chunking、Rewrite/Fusion、Assembler 的 L1 v2 测量真实性，再建立正式 graded qrels，随后在隔离索引上优化 BM25+Dense；不要先继续堆 Prompt或调整 MMR 参数。

## 1. 必读文件和阅读顺序

建议新对话按以下顺序阅读：

1. `AGENTS.md`：工程真实性、Fail-Fast、禁止伪造执行和状态的项目守则。
2. `docs/需求.md`：外部业务需求原文及其针对当前项目的总结。
3. `docs/Eedi_RAG_Full_Lifecycle_Roadmap_0_to_100.md`：当前 0→1→100 路线图和架构目标。
4. `docs/L0_DeepEval_L1_Component_Baseline.md`：评测设计、质量门禁及历史证据。
5. `docs/L1_Component_Evaluation_Construction_Guide.md`：当前 L1 评测施工状态、artifact 保护和后续顺序。
6. `docs/L1_Component_Evaluation_V2_Implementation_Plan.md`：L1 v2 指标口径修正、Hybrid Retrieval 和验收施工说明书。
7. 本文件的“当前真实状态”和“下一步计划”。

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
