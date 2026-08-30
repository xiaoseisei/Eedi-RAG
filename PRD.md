# 产品需求文档 (PRD): 基于真实辅导记录的智能教学洞察与苏格拉底式 RAG 系统 (Eedi-RAG)

---

## 1. 文档信息
- **项目名称**: Eedi-RAG (Educational Tutoring Dialogue Mining & Socratic RAG System)
- **版本**: v1.0.0
- **状态**: 规划与设计阶段 (Draft / Approved for PoC)
- **目标路径**: `E:\PIAgent\10-projects\AgentLearn\Eedi-RAG`

---

## 2. 背景与需求复述 (Background & Requirements)

### 2.1 项目背景
在 1v1 师生在线辅导（Tutoring）场景中，积累了大量真实的交互对话文字稿（Transcripts）。这些非结构化文本中沉淀了极具价值的机构专有知识：
1. **学生认知模式**：常见错因、思维盲区、易混淆概念、重考/困难学生的典型卡点。
2. **导师教学经验**：优秀导师如何将抽象概念拆解、如何通过苏格拉底式提问（Socratic Questioning）引导学生自主思考、如何辅助排除干扰选项。

### 2.2 核心目标
本项目旨在摆脱传统“直接把长文本/PDF 暴力切块灌入向量库”的粗暴做法，针对“多轮对话型教学数据”的特征，构建一套**非结构化辅导数据挖掘、多维知识库沉淀与启发式 RAG 交互原型系统**。

- **短期目标 (PoC 阶段)**：
  1. 完成 Eedi 2k 真实辅导对话与题目元数据的清洗、关联聚合与结构化抽取。
  2. 提取并建立两大核心知识库：**`学生认知误区库 (Misconceptions)`** 与 **`导师启发式教学策略库 (Pedagogical Strategies)`**。
  3. 搭建极简交互界面，实现高精度的“教学洞察查询”与“导师引导话术检索”。
- **长期目标 (Phase 2 & 3)**：
  1. 打造面向学生的**“苏格拉底式 AI 助教”**（模拟名师引导，拒绝直接报答案）。
  2. 实现新导师培训/教学质检（QA）评分与自动化错题微课生成。

---

## 3. 用户画像与核心场景 (User Personas & Use Cases)

| 角色 | 核心痛点 | 核心使用场景 |
| :--- | :--- | :--- |
| **教研主管 / 课程研发专家** | 缺乏客观数据支撑，不知道学生做某类题目时的最深层误区是什么。 | **【场景 A：教学洞察与题库诊断】**<br>提问：“在做四舍五入或代数题时，学生最常出现哪两类错误概念？”系统基于数百场真实辅导记录汇总归纳出模式。 |
| **新入职导师 / 辅导顾问** | 面对卡壳的学生不知道如何循序渐进地启发，容易直接告诉答案导致教学效果差。 | **【场景 B：名师策略与话术检索】**<br>提问：“当学生分不清两个相近选项时，资深导师常用的启发式第一问是什么？”系统检索真实案例并返回金牌话术。 |
| **备考学生 (未来扩展)** | 遇到错题直接看解析记不住，缺乏名师点拨。 | **【场景 C：苏格拉底式 AI 助教】**<br>学生上传错题和自己的错误理解，AI 助教调用名师引导策略，单步提问引导学生自我纠错。 |

---

## 4. 系统总体架构与数据流 (Architecture & Data Flow)

```
                       [Eedi 数据源 (2,000+ 对话)]
                       ├── anchored-dialogues (train/test)
                       ├── dq-question-metadata (题目与选项)
                       └── dialogue-subjects (学科知识体系)
                                   │
                                   ▼
                    【阶段一：数据关联与清洗流水线】
                       - 按 InterventionId 聚合多轮对话
                       - 清洗无意义口语废话、标准化角色
                                   │
                                   ▼
                   【阶段二：LLM 结构化认知与策略抽取】
                       - 抽取: 学生误区特征 (Misconceptions)
                       - 抽取: 导师分步引导链 (Scaffolding Steps)
                       - 抽取: 教学动作标签 (TalkMoves)
                                   │
                                   ▼
                   【阶段三：多维向量化与元数据知识库】
                  (ChromaDB / LanceDB 存储 + 混合检索)
                  ├── Collection 1: 学生错因与认知库
                  └── Collection 2: 导师启发话术策略库
                                   │
                                   ▼
                     【阶段四：RAG 检索与应用交互层】
                  ├── 1. 教学洞察分析引擎 (Insight Agent)
                  ├── 2. 导师策略检索接口 (Strategy Retriever)
                  └── 3. 苏格拉底式 AI 助教 (Socratic Tutor Agent)
                                   │
                                   ▼
                        【前端展示: Streamlit UI】
```

---

## 5. 功能需求详述 (Functional Requirements)

### 5.1 模块一：数据预处理与上下文关联 (Data Pipeline)
- **F-1.1 数据拼接**：通过 `InterventionId` 关联 `anchored-dialogues`、`dq-question-metadata` 与 `dialogue-subjects`，构建单次辅导的完整事实对象（包含：学科分类、原题干、正确/错误选项、多轮师生时间序发言）。
- **F-1.2 隐私与噪音过滤**：
  - 去除打招呼等开场白噪音（保留教学意图密集的对话段）。
  - 对学生姓名等敏感信息进行占位符替换（De-identification）。

### 5.2 模块二：教学知识抽取与结构化打标 (LLM-based Extraction)
采用结构化输出（JSON Schema）对每段辅导记录调用 LLM 进行知识蒸馏：
- **F-2.1 学生端抽取**：
  - `question_context`: 题目与考点
  - `error_choice`: 学生的错误选择
  - `core_misconception`: 学生的深层认知错误（如“误将小数点后位数与大小直接挂钩”）
  - `confusion_trigger`: 触发困惑的关键概念
- **F-2.2 导师端抽取**：
  - `pedagogical_goal`: 本轮辅导的教学目标
  - `scaffolding_sequence`: 导师拆解的引导步骤列表（Step 1, Step 2, Step 3...）
  - `socratic_questions`: 导师提出的启发性提问原句
  - `resolution_moment`: 学生产生“Aha moment”的关键转折点

### 5.3 模块三：多维向量化与分层检索体系 (Multi-Index RAG)
- **F-3.1 向量库构建**：使用轻量本地向量库（如 `ChromaDB` / `FAISS`），配合高效 Embedding 模型（如 `text-embedding-3-small` 或开源 `bge-small-en-v1.5`）。
- **F-3.2 路由检索机制 (Query Router)**：
  - 当查询倾向于“分析学生弱项/考点难点”时 $\rightarrow$ 路由至 **学生错因库** 检索并聚合。
  - 当查询倾向于“如何讲解/教学方法”时 $\rightarrow$ 路由至 **导师策略库** 检索真实话术。
  - 支持按 `Subject`（学科/知识点）和 `GradeLevel` 进行元数据过滤（Metadata Filtering）。

### 5.4 模块四：交互式应用界面 (Interactive UI)
采用 Python Streamlit 构建三栏式仪表板：
1. **教学洞察面板 (Teaching Insights)**：输入主题或题型，聚合统计并生成学生最易犯错的 Top-N 误区报告。
2. **导师话术检索器 (Tutor Playbook)**：输入教学困境，展示金牌导师的真实问答样例与引导步骤。
3. **AI 启发式对话测试台 (Socratic Tutor Playground)**：输入一道错题，让 AI 扮演导师，测试其能否模仿真实导师进行多轮启发式对话。

---

## 6. 非功能需求 (Non-Functional Requirements)

1. **防幻觉与可解释性 (Faithfulness & Grounding)**：
   - 每次生成洞察或推荐话术时，必须附带**原始数据来源依据（`InterventionId` 与真实对话原句引用）**，严禁模型凭空捏造学生错误。
2. **低成本与轻量化 (Cost & Efficiency)**：
   - 初始处理 2,000 条对话时采用批量批处理（Batching），控制 API Token 消耗。
   - 向量检索本地运行，无需依赖昂贵的托管云服务。
3. **数据安全与隐私合规 (Data Privacy)**：
   - 确保外部 API 调用不开启任何训练保留（Zero Data Retention）。

---

## 7. 实施里程碑与路线图 (Milestones & Roadmap)

| 阶段 | 周期 | 核心任务 | 交付物 |
| :--- | :--- | :--- | :--- |
| **M1: 数据流水线与探索** | Day 1-2 | 编写合并脚本，完成 2k 对话与题目的结构化聚合 | `parse_dataset.py` + 抽样分析 Notebook |
| **M2: LLM 知识抽取与向量库** | Day 3-4 | 设计 Extraction Prompt，批量提取误区与策略，写入 ChromaDB | `extract_knowledge.py` + `chroma_db/` |
| **M3: RAG 检索链与 Prompt 调优** | Day 5-6 | 实现双路由 RAG 检索器，编写防幻觉生成流水线 | `retriever.py` + `rag_engine.py` |
| **M4: Streamlit 交互原型** | Day 7 | 搭建可视化 Web 演示界面，完成端到端测试 | `app.py` + 完整使用演示文档 |

---

## 8. 验收标准与成功指标 (Acceptance Criteria & KPIs)

1. **数据覆盖度**：成功处理并结构化解析 $\ge 95\%$ 的 Eedi 辅导对话记录。
2. **检索相关性**：对常见学科考点（如代数、四舍五入、几何）的教学策略查询，Top-3 检索结果与目标考点强相关且包含明确的启发式问答。
3. **洞察准确性**：AI 生成的“学生易错总结”能准确溯源到至少 2 个真实的 `InterventionId` 对话案例。
4. **原型可用性**：Streamlit 界面可在本地一键启动，查询端到端延迟控制在 3 秒内（不含长生成）。
