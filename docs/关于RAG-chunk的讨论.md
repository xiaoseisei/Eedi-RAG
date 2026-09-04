# Eedi-RAG 对话分块（Chunk）策略演进与重构讨论专题

> **文档定位**：记录关于 Eedi-RAG 项目数据层现状、现有 Chunk 策略短板测算、开源/学术界教育 RAG 方案调研、业务需求回归分析以及“按需即时生成卡片 (On-Demand JIT)”全新切块架构的完整讨论与决策纪要。

---

## 一、 当前项目数据层全景与客观数据盘点

### 1.1 数据生命周期与多层架构划分 (L0 ~ L5)
项目数据流围绕非结构化师生辅导对话向高密度教研知识资产演进，整体划分为 6 层：
*   **L0 原始公开代理数据层 (Raw / Ingestion Layer)**：Hugging Face 开源的 `Eedi/Question-Anchored-Tutoring-Dialogues-2k` 数据集，包含辅导对话、题目选项元数据与三级学科考纲。
*   **L1 融合清洗与会话层 (Data Fusion & Normalization)**：经 `src/data_fusion.py` 执行 LaTeX 公式保护区隔离、PII 隐私脱敏、状态机碎句压缩（Turn Compression）及双向有效教学门禁后的自包含完整会话实体。
*   **L2 知识蒸馏与切块层 (Distillation & Chunking)**：由 `src/extract_knowledge.py` 与 `src/chunker.py` 承载，原设计为双基座架构（主线 LLM 双卡片 + 兜底滑动窗口）。
*   **L3 双引擎持久化存储层 (Dual-Engine Storage)**：由 `src/storage_manager.py` 统一管理的“确定性关系底表 (DuckDB) + 物理隔离语义向量库 (ChromaDB)”。
*   **L4 评测基准与金标测试层 (Evaluation & Benchmarks)**：30 条带 `(session_id, turn_id)` 逐字真实原声引用的金标教研问答集 (`golden_test_set.json`) 与 Ragas 评测报告。
*   **L5 本地轻量验证沙盒 (Sample & Sandbox)**：`data/sample/` 微型离线回归测试集。

### 1.2 业务领域本体模型契约 (`src/models.py`)
*   `Question` & `SubjectHierarchy`：题目题干（含 LaTeX 公式）、选项映射（A/B/C/D）及三级学科树（Subject > Topic > Subtopic）。
*   `DialogueTurn`：包含 `turn_id`、说话人角色 (`tutor`/`student`)、清洗文本、原始碎句字面 `raw_messages` 及官方教学动作预测标签 `talk_moves`。
*   `CleanedSession`：完整的教学交互全景单元（PIU），将题目事实、考纲与多轮时序问答整合为单一大宽对象。
*   `StudentMisconceptionProfile`：学术化错因命名、误选选项、深层思维机理、困惑触发词、逐字原声引用及轮次 ID。
*   `TutorStrategyProfile`：阶段教学目标、教学法分类（Socratic / Scaffolding / Counter_Example 等）、破局灵魂一问（Key Aha Question）、脚手架引导步骤链及转化成效。
*   `Chunk`：统一的切块数据契约实体，承载 `chunk_id`、`chunk_type`、`content`、`metadata`、`source_turn_ids`。
*   `DialogueCitation`：真实师生对白引用凭据，带 DuckDB 真伪校验字段。

### 1.3 现存物理数据规模实测数据表（真实环境 Census）

| 存储载体 / 文件路径 | 数据类型 / 格式 | 真实规模统计 (Lines / Rows / Items) | 存储占用 | 业务角色说明 |
| :--- | :--- | :--- | :--- | :--- |
| `data/anchored-dialogues/train.csv` | 原始对白 (CSV) | **55,892** 行 | 3.45 MB | 师生交互时序记录，含 `TalkMovePrediction` |
| `data/anchored-dialogues/test.csv` | 原始对白 (CSV) | **13,522** 行 | 843 KB | 评估对白切分数据 |
| `data/dq-question-metadata.csv` | 考题元数据 (CSV) | **15,272** 行 | 885 KB | 题干片段（含 LaTeX 公式）与选项标签 |
| `data/dialogue-subjects.csv` | 考纲知识树 (CSV) | **9,035** 行 | 399 KB | 三级学科考纲层级节点 |
| `data/cleaned_sessions.jsonl` | 融合清洗会话 (JSONL) | **1,576** 场全量辅导 (100% 达标) | 9.70 MB | 覆盖 868 道考题，共 37,186 轮对话 |
| `data/db/tutoring_knowledge.duckdb` | 关系底表 (DuckDB) | • `tutoring_sessions`: **100** 行<br>• `session_dialogue_turns`: **2,335** 行<br>• `misconception_chunks`: **100** 行<br>• `tutor_strategy_chunks`: **100** 行<br>• `sliding_window_chunks`: **719** 行 | 1.8 MB | 事实底表，负责 SQL 聚合与 `[Turn N]` 行级逐字真伪审计 |
| `data/chroma/` | 语义向量索引 (ChromaDB) | • `student_misconceptions`: **100** items<br>• `tutor_strategies`: **100** items<br>• `fallback_windows`: **719** items | 目录存储 | 多路物理隔离语义检索库 |
| `data/golden_test_set.json` | 金标评测集 (JSON) | **30** 个金标教研问答用例 | 33.9 KB | 带原声逐字引用的真值基准 |
| `data/authoring_catalog.json` | 精编教研编目 (JSON) | **100** 场深度知识卡片目录 | 196 KB | 结构化教学资产样板 |

---

## 二、 当前 Chunk 策略的致命短板与成本测算

### 2.1 短板 1：主线知识卡片切块 (`KnowledgeDistilledChunker`) 的前置阻塞与成本失控
*   **机制瓶颈**：原设计在数据切块入库前，强制要求调用 `src/extract_knowledge.py` 进行结构化抽取。抽取过程采用**两阶段机制**：
    1. 第一阶段：发送超长 Prompt（含考纲树、题干选项、20~40 轮完整师生对白），让 LLM 生成包含两张大卡片的大 JSON；
    2. 第二阶段：发起 `build_evidence_audit_prompt` 证据审计请求，核验 `source_turn_ids` 的充分性与噪声；
    3. 自纠机制：若格式或逐字原声校验失败，最多自纠重试 3 次。
*   **全量成本测算**：
    *   全量有 **1,576 场** 会话，若每场平均触发 2.5 次 LLM 调用，全量需要发起 **约 3,900 ~ 5,000 次 API 请求**；
    *   Token 消耗高达 **500 万 ~ 800 万 Tokens**；
    *   单场处理 5~15 秒，串行需 **3 ~ 5 小时**；即使并发也极易遭遇商业 API 限频（RPM/TPM）、网络超时中断或计费超标；
    *   **架构脆弱性**：卡片格式一旦微调（例如增加字段或修改 Prompt），全量 1,576 场必须推翻重跑，时间与金钱成本不可承受。

### 2.2 短板 2：兜底滑动窗口切块 (`SlidingWindowChunker`, W6_S3) 的切片膨胀与冗余
*   **参数设定**：窗口大小 6 轮，步长 3 轮（重合度 50%）。
*   **切片量爆炸**：单场 24 轮会话平均切出 7~8 个窗口块。全量 1,576 场如果全切滑动窗口，将产生 **~11,500 个 Chunks**；
*   **检索质量劣化**：步长过小导致相邻 Chunk 存在 50% 文本重复，向量检索召回时大量挤占 Top-K 配额（召回同质化片段），增加了下游 Reranker 的计算负担；纯规则滑动窗口容易在教学动作中途硬性截断，造成因果上下文撕裂。

### 2.3 根源反思：入库时过度工程化 (Eager Pre-computation) 与分块悖论 (Chunking Paradox)
旧方案将“**切块入库**”与“**深度知识蒸馏**”强行捆绑，陷入了典型的**分块悖论**：
*   想让大模型看懂长因果链 ➔ 把 Chunk 做大 ➔ 向量检索语义稀释、模糊；
*   想让向量检索精准 ➔ 把 Chunk 切细 ➔ 大模型缺乏前后文、产生幻觉；
*   试图通过离线预制卡片解决 ➔ 带来高昂的离线抽取费用和极其僵化的更新成本。

---

## 三、 开源与学术界教育 RAG 分块前沿方案调研

针对教育大模型、智能辅导系统（ITS）与对话型 RAG 场景，开源社区与前沿学术界的实践提供了明确的参考：

### 3.1 核心共识与反模式
*   **反模式**：**在教育对话场景中，固定长度/滑动窗口切块表现最差**。对话是非线性、强因果依赖的，固定切块必然割裂“学生困惑”与“导师追问”的教学因果链（Context Fragmentation）。
*   **共识**：**教学对话的切块必须保持“教学动作（Pedagogical Move）”与“问答交互（Exchange）”的语义完整性**。

### 3.2 四大主流开源与学术模式

| 策略模式 | 代表项目 / 文献来源 | 核心实现机制 | 适用场景与工程收益 |
| :--- | :--- | :--- | :--- |
| **模式 1：教学动作与交互对驱动切块 (Dialogue-Act Chunking)** | • *Domain-Adapted Retrieval for In-Context Annotation of Pedagogical Dialogue Acts* (Lee et al., 2026, 基于 Eedi / TalkMoves)<br>• Google DeepMind / LearnLM 辅导研究 | 针对带有教学动作（`<Press for Accuracy>`, `<Press for Reasoning>`, `<Revoicing>`）或角色转换节点，提取 **“学生困惑 + 导师追问”的最小交互对（Exchange Pair，2~4 轮）** | **0 API 费用，毫秒级规则处理**。语义高度聚焦，检索时直接命中金牌提问与学生关键卡点。 |
| **模式 2：父子层级切块 (Parent-Child / Small-to-Big)** | • LlamaIndex `ParentDocumentRetriever`<br>• LangChain Parent-Child 模式<br>• Socratic Tutoring (KITE, AITEE) | **Child Chunk（小块，50~150 tokens）**：专供高灵敏度向量检索（如单独抽取的核心问题句或错选句）；<br>**Parent Chunk（大块）**：完整考题事实与前后 8~15 轮真实对白。通过外键从关系底表秒级关联获取 | **攻克分块悖论**：向量检索既能灵敏命中微观细节，LLM 生成又能拿到完整的因果大上下文，大幅降低幻觉。 |
| **模式 3：考题与干扰项锚定库 (Question-Centric Bank)** | • 开源库 **`labelbank`** (Kaggle Eedi Misconception 银牌方案)<br>• NUS **MathRAG** (数学推理与公式验证) | 不以时间流切块，而是以 **“考题 (Question) + 错误选项 (Distractor)” 为物理主键**。聚合针对同一考题的多场辅导记录，形成高密度知识锚点 | 全量 1,576 场辅导仅对应 **868 道考题**，切块总量天然收敛至千级以内，索引极其精炼。 |
| **模式 4：自适应教学阶段分段 (Phase-based Segmentation)** | • 华东师范大学 **EduChat**<br>• Coursera / edX 在线教学切片 | 顺应辅导“三段式”节律：<br>1. 题意审题与卡点暴露阶段<br>2. 启发破局与脚手架搭建阶段<br>3. 巩固推导与收尾阶段 | 消除固定滑窗 50% 的重叠冗余，单场仅产出 2~3 个自包含 Chunk，适合长文本 Embedding 模型。 |

---

## 四、 项目业务需求深度回归分析

系统必须满足两大考察维度与两项核心能力：

### 4.1 考察维度定义
*   **维度 A：关于学生端（学情与错题洞察）**
    *   *高频困惑*：学生在辅导中最常提出的问题是什么？
    *   *核心难点*：哪些概念、主题或题型卡壳时间最长？
    *   *共性误区*：哪些认知错误（Misconceptions）在不同学生身上反复出现？
    *   *错题模式*：学生在做题/考试时最容易犯的错误类型是什么？
    *   *困难学生画像*：基础薄弱学生是否存在共性思维卡点模式？
*   **维度 B：关于导师端（名师教学方法论与策略沉淀）**
    *   *核心解题套路*：经验丰富的导师反复传授哪些通用做题和思考策略？
    *   *启发式教学技巧*：遇到难懂概念，导师用了哪些通俗比喻、分步引导（Scaffolding）？
    *   *排除法与决策技巧*：在两个相似选项中纠结时，导师如何引导做出抉择？
    *   *教研内容改进*：哪些信息可以直接用来优化教材、微课与题库解析？

### 4.2 核心能力与计算范式冲突分析
*   **能力 1：非结构化对话的规律提炼 (Macro Aggregation / OLAP)**
    *   *本质*：要求跨多篇、跨会话的批量分析与全局统计（高频、共性、卡壳最长、反复出现）；
    *   *技术局限*：纯靠向量检索 Top-K 只能抽样举出局部个案，无法给出全局统计真相；若用 LLM 逐篇抽卡再合并，成本与延迟不可承受。
*   **能力 2：基于真实数据的 RAG 检索问答 (Micro Point Query / RAG)**
    *   *本质*：自然语言提问，检索出最匹配的个案证据并输出结构化解答（带 `[Turn N]` 行号与逐字原声，拒绝捏造）。
    *   *技术强项*：语义检索 + 向量库 + DuckDB 逐字审计是最佳路径。

---

## 五、 核心认知转变：为什么“入库前做卡片”是反模式？

### 5.1 灵魂追问：“那实际上其实还是需要 LLM 来做卡片呀？”
**结论：根本不需要在入库和 Chunk 阶段用 LLM 预先做卡片！**

### 5.2 离线预制卡片 (Eager) vs 按需即时生成 (On-Demand JIT)

| 比较维度 | 旧策略：入库前调用 LLM 预制卡片 (Eager) | 新策略：零成本入库 + 检索后按需生成卡片 (On-Demand JIT) |
| :--- | :--- | :--- |
| **切块与入库阶段** | 强制发起 **~4,000 次 LLM 调用**，消耗数百万 Token | **纯规则切块，0 次 LLM 调用，0 元 API 费** |
| **全量入库耗时** | **3 ~ 5 小时**（受网络、限流、校验重试制约） | **2 ~ 3 秒钟**（纯本地 Python 状态机与正则计算） |
| **卡片生成时机** | 入库时一次性预先批量生成（不管后续查不查） | **用户提出具体教研问题时，基于检索命中的真实对白动态拼装** |
| **冷数据开销** | 80% 从未被查询的会话也被迫产生抽取费用 | **零浪费**：问什么提炼什么，单次查询仅调用 1 次 LLM |
| **业务迭代适应度** | 极差：修改 Prompt、增减字段需推翻全量数据重抽 | **极高**：只需修改下游生成 Prompt，全量数据底座无缝兼容 |

---

## 六、 最终架构选择：双轨驱动与“按需生成卡片”新策略

```
                            全量 1,576 场清洗辅导会话 (cleaned_sessions.jsonl)
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    【轨道 1：零成本确定性规则切块】                             【轨道 2：DuckDB 关系事实底表】
      (专供 能力 2: 微观 RAG 检索问答)                             (专供 能力 1: 宏观学情规律提炼)
                   │                                                           │
     ┌─────────────┴─────────────┐                                             │
     ▼                           ▼                                             │
【学生困惑块 (A面)】       【名师破局块 (B面)】                                 │
 - 注入: 考题+错误选项       - 注入: 考题+考纲                                 │
 - 规则截取: 学生暴露困惑     - 规则截取: 包含 <Press>/                        │
   的 2~4 轮核心问答对         <Revoicing> 的 2~4 轮启发对白                   │
     │                           │                                             │
     ▼                           ▼                                             ▼
向量集合: misconceptions    向量集合: strategies                SQL 确定性统计与分析引擎:
 (~1,576 块，0 费用)         (~1,576 块，0 费用)                  - 各考点卡壳轮次分布 (识别核心难点)
                                                                 - 错误选项被选率统计 (识别高频错题)
                                                                 - 导师教学动作矩阵 (沉淀解题套路)
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                  【教研工作台 / RAG 问答中台】
                   - 宏观问题 ➔ 查 DuckDB 聚合事实出具趋势报表与共性画像
                   - 微观问题 ➔ 查 向量库 + DuckDB 反查 [Turn N] 证据，即时生成教研指南卡片
```

### 6.1 轨道 1（微观 RAG）：考题与教学动作双锚定规则切块 (A/B 面)
*   **A 面：学生学情困惑切块 (`chunk_type='misconception_snippet'`)**：
    *   **规则**：注入 `[学科考纲] + [考题题干与选项]`，定位学生首次选错或明确表达求助（如 `not sure`, `don't understand`, 选错选项）的上下文，提取 2~4 轮紧凑问答对；
    *   **归宿**：写入 ChromaDB 的 `student_misconceptions` 集合；
    *   **成本与规模**：全量 1,576 块，生成耗时约 1 秒，**0 API 费用**。
*   **B 面：名师启发引导切块 (`chunk_type='strategy_snippet'`)**：
    *   **规则**：注入 `[学科考纲] + [考题原题]`，根据数据自带的官方教学动作标签，定位带有 `<Press for Reasoning>`, `<Press for Accuracy>`, `<Revoicing>` 的关键轮次，提取前后各 1~2 轮连续引导对白；
    *   **归宿**：写入 ChromaDB 的 `tutor_strategies` 集合；
    *   **成本与规模**：全量 1,576 块，生成耗时约 1 秒，**0 API 费用**。
*   **体量优化效果**：全量切块总数由滑动窗口的 11,500+ 骤降至 **~3,150 个**，降幅达 **73%**，Embedding 向量化时间与 ChromaDB 存储占用同步减少 70% 以上。

### 6.2 轨道 2（宏观规律）：DuckDB 关系底表确定性聚合 (OLAP)
回答宏观学情问题时，由 SQL 提供绝对真实、无幻觉的统计支撑：
*   **核心难点挖掘**：通过 `SELECT subject_path, AVG(total_turns), COUNT(*) FROM tutoring_sessions GROUP BY 1 ORDER BY 2 DESC`，直接找出辅导轮次显著偏高（如 $>30$ 轮）的考点，数据驱动锚定最卡壳知识点；
*   **高频错题模式**：统计 `options_json` 中各选项的被选率，自动识别出强干扰选项分布；
*   **名师套路沉淀**：统计 `<Press for Reasoning>` 与 `<Revoicing>` 在各导师和考点下的共现频率，归纳标准引导流。

### 6.3 问答端按需即时生成卡片 (On-Demand JIT Assembly)
*   **执行逻辑**：
    1. 教研人员提问：“四舍五入 5.4598 到 1 位小数时，学生为什么常选 5.45？”
    2. 向量库在 **0.02 秒** 内召回最相关的 2~3 个 A 面或 B 面规则切块；
    3. 根据切块元数据中的 `(session_id, source_turn_ids)`，秒级从 DuckDB `session_dialogue_turns` 底表中反查完整时序真实对白；
    4. **此时调用单次 LLM**，将题目事实、考纲与真实对白喂给模型，动态渲染出一张即时定制的《教研备课指南与名师破局锦囊》（严格带 `[Turn N]` 行号与引用）；
    5. 经过项目既有的逐字门禁（Audit Gate）核实，确保 100% 来源真实。

### 6.4 分级知识资产管理
*   **全量底座（100% 覆盖，0 成本）**：全量 1,576 场辅导均以规则切块 + DuckDB 存储，实现无死角全量检索与宏观统计。
*   **黄金样板（重点覆盖，深度蒸馏）**：维持现有的 100 场高质量卡片编目（`authoring_catalog.json`），仅对教研关注的核心典型错题按需进行离线深度提纯，作为标杆示例。

---

## 七、 实施落地路线建议

1. **第 1 步：实现零成本规则切块器 (`RuleBasedActionChunker`)**
   * 在 `src/chunker.py` 中新增基于角色与 `TalkMovePrediction` 的轻量切块器，产出 A 面（困惑片段）与 B 面（策略片段）；
2. **第 2 步：全量数据秒级切块与入库验证**
   * 对 `cleaned_sessions.jsonl` 执行全量 1,576 场无 LLM 规则切块，更新 DuckDB 与 ChromaDB 索引；
3. **第 3 步：DuckDB 宏观学情分析器构建**
   * 编写 SQL 聚合查询模块，支撑核心难点、高频错选与教学动作矩阵的秒级统计报表（满足能力 1）；
4. **第 4 步：在线按需生成卡片流程串联**
   * 确保 `src/rag_pipeline.py` 消费规则召回片段后，通过 DuckDB 证据回溯即时生成结构化教研卡片（满足能力 2）。
