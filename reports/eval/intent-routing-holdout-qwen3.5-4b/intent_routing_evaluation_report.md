# Eedi-RAG 意图识别分层架构 Holdout 独立评测审计报告

> **评测时间**: 2026-09-11T01:29:30.898690  
> **模型底座 (Tier-2)**: `Qwen/Qwen3.5-4B` (通过硅基流动 SiliconFlow 在线调用)  
> **Holdout 评测集**: `data/business/holdout_intent_eval_v1.jsonl` (共 48 条独立业务查询)  

---

## 1. 核心架构与 5 大指标矩阵严格区分

根据**工程诚实第一铁律**与**意图路由三态契约**，本系统将业务意图识别严格拆分为两层架构：
1. **Tier-1 确定性规则层**：仅当存在唯一强指向信号时直解（`RESOLVED`）；出现语义冲突或弱信号时主动弃权（`NEED_LLM`）；对越界请求精准拦截（`UNSUPPORTED`）；
2. **Tier-2 LLM 意图消歧层**：接管主动弃权样本，通过轻量级 LLM (`Qwen/Qwen3.5-4B`) 进行候选意图裁决；
3. **可用性与降级契约**：如果 LLM 未配置或调用失败，系统诚实返回 `UNMEASURED` 并阻断下游，绝不伪装成 `UNSUPPORTED` 或虚构默认意图。

### 5 维核心指标矩阵总览

| 指标层级 | 具体评测指标 | 实测数值 | 期望标准 | 状态与工程评价 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Tier-1 规则精确率** | **Rule Precision** | **100.0%** (14例直解) | 100% | ✅ 完美达成 (规则直解绝对准确，零误解) |
| **1. Tier-1 规则覆盖率** | **Coverage Rate** | **29.17%** | 20% ~ 40% | ✅ 优良 (约 1/3 高频典型模板无需网络 IO 极速直解) |
| **1. Tier-1 主动弃权率** | **Abstention Rate** | **50.0%** (24例移交) | 50% ~ 70% | ✅ 规范 (模糊及多意图查询主动移交 LLM 消歧) |
| **1. Tier-1 不支持拦截率**| **Unsupported Rate** | **20.83%** (10例拦截) | 15% ~ 25% | ✅ 精确 (拦截率100%命中越界评选/因果题) |
| **1. Tier-1 路由契约符合度**| **Route Alignment** | **91.67%** | ≥ 90% | ✅ 达标 (完全符合三态路由规范预期) |
| **2. Tier-2 LLM 消歧率** | **Disambiguation Acc** | **58.33%** (14/24) | ≥ 60% | ⚠️ 需优化 (Qwen3.5-4B 在多意图和口语上的裁决率) |
| **2. Tier-2 API 故障率** | **API Failure Rate** | **0.0%** (0例失败) | 0.0% | ✅ 健壮 (0 故障) |
| **3. 端到端系统准确率** | **E2E System Accuracy** | **79.17%** (38/48) | ≥ 75% | ✅ 优良 (级联系统总正确率) |
| **4. 契约未测量率** | **UNMEASURED Rate** | **0.0%** (0例阻塞) | ≤ 5% (配置完整时) | ✅ 正常 (0 阻塞) |
| **5. 性能延迟** | **Average Latency** | 规则: **0.601 ms** / LLM: **2556.95 ms** | < 1ms / < 3s | ✅ 规则极速微秒级，LLM 稳定可控 |

---

## 2. 数据集分类分层表现明细 (Category Breakdown)

| 类别 (Category) | 样本数 | Tier-1 RESOLVED | Tier-1 NEED_LLM | Tier-1 UNSUPPORTED | 路由符合度 | Tier-2 消歧命中率 | UNMEASURED | 端到端正确率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CLEAR_RULE` | 12 | 11 | 1 | 0 | 91.7% | 100.0% | 0 | **100.0%** |
| `CONFLICT_NEED_LLM` | 12 | 3 | 9 | 0 | 75.0% | 44.4% | 0 | **58.3%** |
| `PARAPHRASE_NEED_LLM` | 14 | 0 | 14 | 0 | 100.0% | 64.3% | 0 | **64.3%** |
| `OUT_OF_BOUNDS_UNSUPPORTED` | 10 | 0 | 0 | 10 | 100.0% | 100.0% | 0 | **100.0%** |

---

## 3. 多意图冲突与边界样本专项人工复核表 (Human Review Matrix)

本节针对 **多意图冲突样本 (`CONFLICT_NEED_LLM`)** 与 **边界口语化样本 (`PARAPHRASE_NEED_LLM`)** 提供逐条审计与人工复核标记，清晰暴露大模型与 Gold 标签的交叠逻辑：

| ID | 查询原句 (Query) | 类别 | 候选意图 (Candidates) | LLM 主预测 (Primary) | LLM 次预测 (Secondary) | Gold 预期 | 人工复核标记与评注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `holdout-013` | 重考学生最常问什么？ | `CONFLICT_NEED_LLM` | `STUDENT_COHORT_PATTERNS, STUDENT_FREQUENT_QUESTIONS` | `STUDENT_FREQUENT_QUESTIONS` | `-` | `STUDENT_FREQUENT_QUESTIONS` | ✅ 多意图语义冲突经 LLM 成功消歧 (判定为 STUDENT_FREQUENT_QUESTIONS)。 |
| `holdout-014` | 导师在辅导这道高频错题时，应该怎么引导学生理解误区？ | `CONFLICT_NEED_LLM` | `TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | `-` | `TUTOR_STRATEGY_PATTERNS` | ✅ 多意图语义冲突经 LLM 成功消歧 (判定为 TUTOR_STRATEGY_PATTERNS)。 |
| `holdout-015` | 基础薄弱学生在考试错题模式中主要集中在哪些题型？ | `CONFLICT_NEED_LLM` | `STUDENT_COHORT_PATTERNS` | `STUDENT_COHORT_PATTERNS` | `-` | `STUDENT_COHORT_PATTERNS` | ✅ 多意图语义冲突经 LLM 成功消歧 (判定为 STUDENT_COHORT_PATTERNS)。 |
| `holdout-016` | 针对反复出现的误区，导师通常采用哪些教学策略进行干预？ | `CONFLICT_NEED_LLM` | `RECURRING_MISCONCEPTIONS, TUTOR_STRATEGY_PATTERNS` | `RECURRING_MISCONCEPTIONS` | `TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | ⚠️ ⚠️ 人工复核建议: 查询兼具复合意图，LLM 主意图为 RECURRING_MISCONCEPTIONS，次意图为 TUTOR_STRATEGY_PATTERNS(匹配Gold)。属于主次意图合理重叠。 |
| `holdout-017` | 如何根据学生最常问的高频问题优化教材的内容设计？ | `CONFLICT_NEED_LLM` | `CONTENT_IMPROVEMENT, STUDENT_FREQUENT_QUESTIONS` | `CONTENT_IMPROVEMENT` | `STUDENT_FREQUENT_QUESTIONS` | `CONTENT_IMPROVEMENT` | ✅ 多意图语义冲突经 LLM 成功消歧 (判定为 CONTENT_IMPROVEMENT)。 |
| `holdout-018` | 这道长方体表面积题目中，学生最常问的问题反映了什么思维卡点？ | `CONFLICT_NEED_LLM` | `STUDENT_FREQUENT_QUESTIONS, QUESTION_TUTORING` | `STUDENT_FREQUENT_QUESTIONS` | `-` | `QUESTION_TUTORING` | ⚠️ ❌ 消歧偏离: 候选意图 ['STUDENT_FREQUENT_QUESTIONS', 'QUESTION_TUTORING']，LLM 判定为 STUDENT_FREQUENT_QUESTIONS，偏离预期 Gold 意图 QUESTION_TUTORING。 |
| `holdout-019` | 基础薄弱学生群体在初一数学中最难理解的核心概念是什么？ | `CONFLICT_NEED_LLM` | `DIFFICULT_CONCEPTS, STUDENT_COHORT_PATTERNS` | `DIFFICULT_CONCEPTS` | `STUDENT_COHORT_PATTERNS` | `DIFFICULT_CONCEPTS` | ✅ 多意图语义冲突经 LLM 成功消歧 (判定为 DIFFICULT_CONCEPTS)。 |
| `holdout-020` | 分析历次考试错误模式中最难理解的几何证明考点 | `CONFLICT_NEED_LLM` | `DIFFICULT_CONCEPTS, EXAM_ERROR_PATTERNS` | `DIFFICULT_CONCEPTS` | `EXAM_ERROR_PATTERNS` | `EXAM_ERROR_PATTERNS` | ⚠️ ⚠️ 人工复核建议: 查询兼具复合意图，LLM 主意图为 DIFFICULT_CONCEPTS，次意图为 EXAM_ERROR_PATTERNS(匹配Gold)。属于主次意图合理重叠。 |
| `holdout-021` | 不同课堂学生反复出现的相同误区对题库内容改进有什么启发？ | `CONFLICT_NEED_LLM` | `CONTENT_IMPROVEMENT` | `CONTENT_IMPROVEMENT` | `-` | `CONTENT_IMPROVEMENT` | ✅ 多意图语义冲突经 LLM 成功消歧 (判定为 CONTENT_IMPROVEMENT)。 |
| `holdout-022` | 导师在面对学生考试错误模式时，常用怎样的提问方式打开思路？ | `CONFLICT_NEED_LLM` | `EXAM_ERROR_PATTERNS, TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | `EXAM_ERROR_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | ✅ 多意图语义冲突经 LLM 成功消歧 (判定为 TUTOR_STRATEGY_PATTERNS)。 |
| `holdout-023` | 多次重考学生与首考学生在最常提出的问题上有何异同？ | `CONFLICT_NEED_LLM` | `STUDENT_FREQUENT_QUESTIONS, STUDENT_COHORT_PATTERNS` | `STUDENT_FREQUENT_QUESTIONS` | `STUDENT_COHORT_PATTERNS` | `STUDENT_COHORT_PATTERNS` | ⚠️ ⚠️ 人工复核建议: 查询兼具复合意图，LLM 主意图为 STUDENT_FREQUENT_QUESTIONS，次意图为 STUDENT_COHORT_PATTERNS(匹配Gold)。属于主次意图合理重叠。 |
| `holdout-024` | 针对共性误区，导师在课堂上应该如何解开学生的认知纠结？ | `CONFLICT_NEED_LLM` | `RECURRING_MISCONCEPTIONS, QUESTION_TUTORING` | `QUESTION_TUTORING` | `-` | `TUTOR_STRATEGY_PATTERNS` | ⚠️ ❌ 消歧偏离: 候选意图 ['RECURRING_MISCONCEPTIONS', 'QUESTION_TUTORING']，LLM 判定为 QUESTION_TUTORING，偏离预期 Gold 意图 TUTOR_STRATEGY_PATTERNS。 |
| `holdout-025` | 孩子做题老是把负负得正搞混，背后的深层认知障碍是什么？ | `PARAPHRASE_NEED_LLM` | `全集` | `RECURRING_MISCONCEPTIONS` | `DIFFICULT_CONCEPTS` | `RECURRING_MISCONCEPTIONS` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (RECURRING_MISCONCEPTIONS)。 |
| `holdout-026` | 这道求未知数应用题，为什么很多孩子都卡在列方程的第一步？ | `PARAPHRASE_NEED_LLM` | `全集` | `STUDENT_FREQUENT_QUESTIONS` | `DIFFICULT_CONCEPTS` | `QUESTION_TUTORING` | ⚠️ ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 QUESTION_TUTORING (判定为 STUDENT_FREQUENT_QUESTIONS)。 |
| `holdout-027` | 在课堂上面对思维受阻的学生，授课老师通常怎样一步一步点拨到位？ | `PARAPHRASE_NEED_LLM` | `全集` | `TUTOR_STRATEGY_PATTERNS` | `-` | `TUTOR_STRATEGY_PATTERNS` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (TUTOR_STRATEGY_PATTERNS)。 |
| `holdout-028` | 二刷这道题的同学普遍会在哪些推导步骤上产生困惑？ | `PARAPHRASE_NEED_LLM` | `全集` | `STUDENT_FREQUENT_QUESTIONS` | `CONTENT_IMPROVEMENT` | `STUDENT_FREQUENT_QUESTIONS` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (STUDENT_FREQUENT_QUESTIONS)。 |
| `holdout-029` | 初一数学部分大家公认最啃不动、最吃力的知识模块是哪些？ | `PARAPHRASE_NEED_LLM` | `全集` | `DIFFICULT_CONCEPTS` | `STUDENT_FREQUENT_QUESTIONS` | `DIFFICULT_CONCEPTS` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (DIFFICULT_CONCEPTS)。 |
| `holdout-030` | 期末大考批卷时卷面上最普遍的丢分硬伤集中在哪几类？ | `PARAPHRASE_NEED_LLM` | `全集` | `EXAM_ERROR_PATTERNS` | `-` | `EXAM_ERROR_PATTERNS` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (EXAM_ERROR_PATTERNS)。 |
| `holdout-031` | 根据学生平时的答题痛点，教研组需要补充哪些针对性拓展习题？ | `PARAPHRASE_NEED_LLM` | `全集` | `CONTENT_IMPROVEMENT` | `-` | `CONTENT_IMPROVEMENT` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (CONTENT_IMPROVEMENT)。 |
| `holdout-032` | 平时成绩拔尖的孩子和跟不上的孩子在解题思路构思上有何本质差别？ | `PARAPHRASE_NEED_LLM` | `全集` | `STUDENT_COHORT_PATTERNS` | `DIFFICULT_CONCEPTS` | `STUDENT_COHORT_PATTERNS` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (STUDENT_COHORT_PATTERNS)。 |
| `holdout-033` | Show me what learners get stuck on the most during the fractions module. | `PARAPHRASE_NEED_LLM` | `全集` | `DIFFICULT_CONCEPTS` | `STUDENT_FREQUENT_QUESTIONS` | `DIFFICULT_CONCEPTS` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (DIFFICULT_CONCEPTS)。 |
| `holdout-034` | What are the common pitfalls that high schoolers stumble over in geometry proofs? | `PARAPHRASE_NEED_LLM` | `全集` | `RECURRING_MISCONCEPTIONS` | `DIFFICULT_CONCEPTS` | `EXAM_ERROR_PATTERNS` | ⚠️ ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 EXAM_ERROR_PATTERNS (判定为 RECURRING_MISCONCEPTIONS)。 |
| `holdout-035` | How do experienced instructors scaffold when pupils fail to grasp equivalent fractions? | `PARAPHRASE_NEED_LLM` | `全集` | `DIFFICULT_CONCEPTS` | `TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | ⚠️ ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 TUTOR_STRATEGY_PATTERNS (判定为 DIFFICULT_CONCEPTS)。 |
| `holdout-036` | What typical stumbling blocks do repeat test-takers encounter in quadratic equations? | `PARAPHRASE_NEED_LLM` | `全集` | `RECURRING_MISCONCEPTIONS` | `STUDENT_COHORT_PATTERNS` | `STUDENT_FREQUENT_QUESTIONS` | ⚠️ ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 STUDENT_FREQUENT_QUESTIONS (判定为 RECURRING_MISCONCEPTIONS)。 |
| `holdout-037` | Which topics in our syllabus require enhanced explanatory diagrams based on session data? | `PARAPHRASE_NEED_LLM` | `全集` | `CONTENT_IMPROVEMENT` | `DIFFICULT_CONCEPTS` | `CONTENT_IMPROVEMENT` | ✅ 口语化/同义改写样本经 LLM 准确捕获核心意图 (CONTENT_IMPROVEMENT)。 |
| `holdout-038` | 为什么在做这道等比数列选择题时，大家几乎全军覆没选了 A？ | `PARAPHRASE_NEED_LLM` | `全集` | `EXAM_ERROR_PATTERNS` | `STUDENT_FREQUENT_QUESTIONS` | `QUESTION_TUTORING` | ⚠️ ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 QUESTION_TUTORING (判定为 EXAM_ERROR_PATTERNS)。 |

---

## 4. 全量 Case-by-Case 执行审计日志 (Full Execution Log)

| ID | 查询原句 (Query) | 评测类别 | Tier-1 路由 | Tier-2 判定 | 最终意图 | Gold 意图 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `holdout-001` | QuestionId 1419 的选项 B 误区解析是什么？ | `CLEAR_RULE` | `RESOLVED` | - | `QUESTION_TUTORING` | `QUESTION_TUTORING` | ✅ PASS |
| `holdout-002` | 学生在计算 5/7 - 1/4 时为什么得出了 4/3？ | `CLEAR_RULE` | `RESOLVED` | - | `QUESTION_TUTORING` | `QUESTION_TUTORING` | ✅ PASS |
| `holdout-003` | 在计算 (-3)^2 - 4*2 这一题中，学生为什么会算成 -17？ | `CLEAR_RULE` | `RESOLVED` | - | `QUESTION_TUTORING` | `QUESTION_TUTORING` | ✅ PASS |
| `holdout-004` | 统计全库范围内学生最常问的问题清单 | `CLEAR_RULE` | `RESOLVED` | - | `STUDENT_FREQUENT_QUESTIONS` | `STUDENT_FREQUENT_QUESTIONS` | ✅ PASS |
| `holdout-005` | 在题库辅导记录中，提问密度最高的学生常问疑问有哪些？ | `CLEAR_RULE` | `NEED_LLM` | STUDENT_FREQUENT_QUESTIONS (conf=0.85) | `STUDENT_FREQUENT_QUESTIONS` | `STUDENT_FREQUENT_QUESTIONS` | ✅ PASS |
| `holdout-006` | 按学科全量统计哪些概念让学生最难理解？ | `CLEAR_RULE` | `RESOLVED` | - | `DIFFICULT_CONCEPTS` | `DIFFICULT_CONCEPTS` | ✅ PASS |
| `holdout-007` | 整个初中代数领域学生学习用时最长、核心难点是什么？ | `CLEAR_RULE` | `RESOLVED` | - | `DIFFICULT_CONCEPTS` | `DIFFICULT_CONCEPTS` | ✅ PASS |
| `holdout-008` | 分析学生反复出现的误区在历次教学中的发生率 | `CLEAR_RULE` | `RESOLVED` | - | `RECURRING_MISCONCEPTIONS` | `RECURRING_MISCONCEPTIONS` | ✅ PASS |
| `holdout-009` | 归纳期末统考中学生表现出的典型考试错误模式 | `CLEAR_RULE` | `RESOLVED` | - | `EXAM_ERROR_PATTERNS` | `EXAM_ERROR_PATTERNS` | ✅ PASS |
| `holdout-010` | 根据基础薄弱学生与重考者的多次重考表现分析其特征 | `CLEAR_RULE` | `RESOLVED` | - | `STUDENT_COHORT_PATTERNS` | `STUDENT_COHORT_PATTERNS` | ✅ PASS |
| `holdout-011` | 总结导师常用的引导策略体系 | `CLEAR_RULE` | `RESOLVED` | - | `TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | ✅ PASS |
| `holdout-012` | 基于课堂数据分析新增哪些教学内容最有助于填补知识盲区？ | `CLEAR_RULE` | `RESOLVED` | - | `CONTENT_IMPROVEMENT` | `CONTENT_IMPROVEMENT` | ✅ PASS |
| `holdout-013` | 重考学生最常问什么？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | STUDENT_FREQUENT_QUESTIONS (conf=0.95) | `STUDENT_FREQUENT_QUESTIONS` | `STUDENT_FREQUENT_QUESTIONS` | ✅ PASS |
| `holdout-014` | 导师在辅导这道高频错题时，应该怎么引导学生理解误区？ | `CONFLICT_NEED_LLM` | `RESOLVED` | - | `TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | ✅ PASS |
| `holdout-015` | 基础薄弱学生在考试错题模式中主要集中在哪些题型？ | `CONFLICT_NEED_LLM` | `RESOLVED` | - | `STUDENT_COHORT_PATTERNS` | `STUDENT_COHORT_PATTERNS` | ✅ PASS |
| `holdout-016` | 针对反复出现的误区，导师通常采用哪些教学策略进行干预？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | RECURRING_MISCONCEPTIONS (conf=0.95) | `RECURRING_MISCONCEPTIONS` | `TUTOR_STRATEGY_PATTERNS` | ❌ FAIL |
| `holdout-017` | 如何根据学生最常问的高频问题优化教材的内容设计？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | CONTENT_IMPROVEMENT (conf=0.95) | `CONTENT_IMPROVEMENT` | `CONTENT_IMPROVEMENT` | ✅ PASS |
| `holdout-018` | 这道长方体表面积题目中，学生最常问的问题反映了什么思维卡点？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | STUDENT_FREQUENT_QUESTIONS (conf=0.95) | `STUDENT_FREQUENT_QUESTIONS` | `QUESTION_TUTORING` | ❌ FAIL |
| `holdout-019` | 基础薄弱学生群体在初一数学中最难理解的核心概念是什么？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | DIFFICULT_CONCEPTS (conf=0.85) | `DIFFICULT_CONCEPTS` | `DIFFICULT_CONCEPTS` | ✅ PASS |
| `holdout-020` | 分析历次考试错误模式中最难理解的几何证明考点 | `CONFLICT_NEED_LLM` | `NEED_LLM` | DIFFICULT_CONCEPTS (conf=0.85) | `DIFFICULT_CONCEPTS` | `EXAM_ERROR_PATTERNS` | ❌ FAIL |
| `holdout-021` | 不同课堂学生反复出现的相同误区对题库内容改进有什么启发？ | `CONFLICT_NEED_LLM` | `RESOLVED` | - | `CONTENT_IMPROVEMENT` | `CONTENT_IMPROVEMENT` | ✅ PASS |
| `holdout-022` | 导师在面对学生考试错误模式时，常用怎样的提问方式打开思路？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | TUTOR_STRATEGY_PATTERNS (conf=0.85) | `TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | ✅ PASS |
| `holdout-023` | 多次重考学生与首考学生在最常提出的问题上有何异同？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | STUDENT_FREQUENT_QUESTIONS (conf=0.95) | `STUDENT_FREQUENT_QUESTIONS` | `STUDENT_COHORT_PATTERNS` | ❌ FAIL |
| `holdout-024` | 针对共性误区，导师在课堂上应该如何解开学生的认知纠结？ | `CONFLICT_NEED_LLM` | `NEED_LLM` | QUESTION_TUTORING (conf=0.95) | `QUESTION_TUTORING` | `TUTOR_STRATEGY_PATTERNS` | ❌ FAIL |
| `holdout-025` | 孩子做题老是把负负得正搞混，背后的深层认知障碍是什么？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | RECURRING_MISCONCEPTIONS (conf=0.92) | `RECURRING_MISCONCEPTIONS` | `RECURRING_MISCONCEPTIONS` | ✅ PASS |
| `holdout-026` | 这道求未知数应用题，为什么很多孩子都卡在列方程的第一步？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | STUDENT_FREQUENT_QUESTIONS (conf=0.85) | `STUDENT_FREQUENT_QUESTIONS` | `QUESTION_TUTORING` | ❌ FAIL |
| `holdout-027` | 在课堂上面对思维受阻的学生，授课老师通常怎样一步一步点拨到位？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | TUTOR_STRATEGY_PATTERNS (conf=0.95) | `TUTOR_STRATEGY_PATTERNS` | `TUTOR_STRATEGY_PATTERNS` | ✅ PASS |
| `holdout-028` | 二刷这道题的同学普遍会在哪些推导步骤上产生困惑？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | STUDENT_FREQUENT_QUESTIONS (conf=0.85) | `STUDENT_FREQUENT_QUESTIONS` | `STUDENT_FREQUENT_QUESTIONS` | ✅ PASS |
| `holdout-029` | 初一数学部分大家公认最啃不动、最吃力的知识模块是哪些？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | DIFFICULT_CONCEPTS (conf=0.92) | `DIFFICULT_CONCEPTS` | `DIFFICULT_CONCEPTS` | ✅ PASS |
| `holdout-030` | 期末大考批卷时卷面上最普遍的丢分硬伤集中在哪几类？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | EXAM_ERROR_PATTERNS (conf=0.95) | `EXAM_ERROR_PATTERNS` | `EXAM_ERROR_PATTERNS` | ✅ PASS |
| `holdout-031` | 根据学生平时的答题痛点，教研组需要补充哪些针对性拓展习题？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | CONTENT_IMPROVEMENT (conf=0.95) | `CONTENT_IMPROVEMENT` | `CONTENT_IMPROVEMENT` | ✅ PASS |
| `holdout-032` | 平时成绩拔尖的孩子和跟不上的孩子在解题思路构思上有何本质差别？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | STUDENT_COHORT_PATTERNS (conf=0.85) | `STUDENT_COHORT_PATTERNS` | `STUDENT_COHORT_PATTERNS` | ✅ PASS |
| `holdout-033` | Show me what learners get stuck on the most during the fractions module. | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | DIFFICULT_CONCEPTS (conf=0.85) | `DIFFICULT_CONCEPTS` | `DIFFICULT_CONCEPTS` | ✅ PASS |
| `holdout-034` | What are the common pitfalls that high schoolers stumble over in geometry proofs? | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | RECURRING_MISCONCEPTIONS (conf=0.92) | `RECURRING_MISCONCEPTIONS` | `EXAM_ERROR_PATTERNS` | ❌ FAIL |
| `holdout-035` | How do experienced instructors scaffold when pupils fail to grasp equivalent fractions? | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | DIFFICULT_CONCEPTS (conf=0.92) | `DIFFICULT_CONCEPTS` | `TUTOR_STRATEGY_PATTERNS` | ❌ FAIL |
| `holdout-036` | What typical stumbling blocks do repeat test-takers encounter in quadratic equations? | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | RECURRING_MISCONCEPTIONS (conf=0.85) | `RECURRING_MISCONCEPTIONS` | `STUDENT_FREQUENT_QUESTIONS` | ❌ FAIL |
| `holdout-037` | Which topics in our syllabus require enhanced explanatory diagrams based on session data? | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | CONTENT_IMPROVEMENT (conf=0.85) | `CONTENT_IMPROVEMENT` | `CONTENT_IMPROVEMENT` | ✅ PASS |
| `holdout-038` | 为什么在做这道等比数列选择题时，大家几乎全军覆没选了 A？ | `PARAPHRASE_NEED_LLM` | `NEED_LLM` | EXAM_ERROR_PATTERNS (conf=0.92) | `EXAM_ERROR_PATTERNS` | `QUESTION_TUTORING` | ❌ FAIL |
| `holdout-039` | 谁是平台里教得最好的导师？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-040` | 学生学完这个微课后下周考试能提升多少分？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-041` | 请给出因果效果来证明该引导策略对提分的直接促进作用 | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-042` | 在全校老师中评选最佳导师的量化指标有哪些？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-043` | 这道题在 NBCOT 考试中的考察要求是什么？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-044` | 哪位老师讲课效果最好且深受学生喜爱？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-045` | 平台里谁是最优秀的导师？请列出前三名 | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-046` | 只要学生刷完这组错因卡片，能否保证考试提高多少分？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-047` | 谁是整个学期辅导中最好的老师？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |
| `holdout-048` | 该引导话术相比传统讲法具体能提升多少分？ | `OUT_OF_BOUNDS_UNSUPPORTED` | `UNSUPPORTED` | - | `UNSUPPORTED` | `UNSUPPORTED` | ✅ PASS |

---

## 5. 差异案例详细归因与优化建议 (Discrepancy Attribution)

本次评测在 48 条独立 Holdout 样本中，共有 10 处结果与 Gold 意图标注存在差异。归因分析如下：

### 用例 `holdout-016` (CONFLICT_NEED_LLM)
- **原始查询**: "针对反复出现的误区，导师通常采用哪些教学策略进行干预？"
- **Tier-1 候选意图**: `['RECURRING_MISCONCEPTIONS', 'TUTOR_STRATEGY_PATTERNS']`
- **预测意图**: `RECURRING_MISCONCEPTIONS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `TUTOR_STRATEGY_PATTERNS`
- **LLM 次要预测**: `TUTOR_STRATEGY_PATTERNS`
- **LLM 置信度**: `0.95`
- **LLM 思考解释**: The query explicitly identifies 'recurring misconceptions' as the subject and asks for the specific strategies used to address them, making the misconception the primary focus while the strategy is the requested solution. Both elements are central to the question, warranting a secondary intent.
- **人工复核与归因**: ⚠️ 人工复核建议: 查询兼具复合意图，LLM 主意图为 RECURRING_MISCONCEPTIONS，次意图为 TUTOR_STRATEGY_PATTERNS(匹配Gold)。属于主次意图合理重叠。
- **数据集设计初衷**: 命中反复出现的误区与教学策略，提问核心为导师的干预手段。

### 用例 `holdout-018` (CONFLICT_NEED_LLM)
- **原始查询**: "这道长方体表面积题目中，学生最常问的问题反映了什么思维卡点？"
- **Tier-1 候选意图**: `['STUDENT_FREQUENT_QUESTIONS', 'QUESTION_TUTORING']`
- **预测意图**: `STUDENT_FREQUENT_QUESTIONS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `QUESTION_TUTORING`
- **LLM 次要预测**: `None`
- **LLM 置信度**: `0.95`
- **LLM 思考解释**: The query explicitly asks about the most common questions students ask regarding a specific math problem, which directly aligns with the intent of identifying frequent student inquiries. The focus on 'thinking bottlenecks' derived from these questions reinforces the primary classification rather than a general tutoring request.
- **人工复核与归因**: ❌ 消歧偏离: 候选意图 ['STUDENT_FREQUENT_QUESTIONS', 'QUESTION_TUTORING']，LLM 判定为 STUDENT_FREQUENT_QUESTIONS，偏离预期 Gold 意图 QUESTION_TUTORING。
- **数据集设计初衷**: 兼具长方体表面积具体题意与最常问短语，核心在于该特定题目的错因剖析。

### 用例 `holdout-020` (CONFLICT_NEED_LLM)
- **原始查询**: "分析历次考试错误模式中最难理解的几何证明考点"
- **Tier-1 候选意图**: `['DIFFICULT_CONCEPTS', 'EXAM_ERROR_PATTERNS']`
- **预测意图**: `DIFFICULT_CONCEPTS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `EXAM_ERROR_PATTERNS`
- **LLM 次要预测**: `EXAM_ERROR_PATTERNS`
- **LLM 置信度**: `0.85`
- **LLM 思考解释**: The query explicitly requests an analysis of the hardest-to-understand geometric proof concepts, making this the primary intent. The mention of 'exam error patterns' serves as the context or source for identifying these concepts, supporting a secondary intent.
- **人工复核与归因**: ⚠️ 人工复核建议: 查询兼具复合意图，LLM 主意图为 DIFFICULT_CONCEPTS，次意图为 EXAM_ERROR_PATTERNS(匹配Gold)。属于主次意图合理重叠。
- **数据集设计初衷**: 命中考试错误模式与最难理解，分析对象为考试中的试卷失分规律。

### 用例 `holdout-023` (CONFLICT_NEED_LLM)
- **原始查询**: "多次重考学生与首考学生在最常提出的问题上有何异同？"
- **Tier-1 候选意图**: `['STUDENT_FREQUENT_QUESTIONS', 'STUDENT_COHORT_PATTERNS']`
- **预测意图**: `STUDENT_FREQUENT_QUESTIONS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `STUDENT_COHORT_PATTERNS`
- **LLM 次要预测**: `STUDENT_COHORT_PATTERNS`
- **LLM 置信度**: `0.95`
- **LLM 思考解释**: The query explicitly asks for a comparison of the most frequently asked questions between two student groups, making frequent questions the primary focus. The requirement to analyze differences based on the student cohorts (first-time vs. retakers) establishes the secondary intent of cohort pattern analysis.
- **人工复核与归因**: ⚠️ 人工复核建议: 查询兼具复合意图，LLM 主意图为 STUDENT_FREQUENT_QUESTIONS，次意图为 STUDENT_COHORT_PATTERNS(匹配Gold)。属于主次意图合理重叠。
- **数据集设计初衷**: 对比两类学生群体的问题分布，核心意图为群体分层规律。

### 用例 `holdout-024` (CONFLICT_NEED_LLM)
- **原始查询**: "针对共性误区，导师在课堂上应该如何解开学生的认知纠结？"
- **Tier-1 候选意图**: `['RECURRING_MISCONCEPTIONS', 'QUESTION_TUTORING']`
- **预测意图**: `QUESTION_TUTORING` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `TUTOR_STRATEGY_PATTERNS`
- **LLM 次要预测**: `None`
- **LLM 置信度**: `0.95`
- **LLM 思考解释**: The query explicitly asks for specific instructional strategies for a tutor to resolve student confusion, which aligns directly with the tutoring intent. While it mentions misconceptions, the focus is on the method of resolution rather than the definition or analysis of the misconceptions themselves.
- **人工复核与归因**: ❌ 消歧偏离: 候选意图 ['RECURRING_MISCONCEPTIONS', 'QUESTION_TUTORING']，LLM 判定为 QUESTION_TUTORING，偏离预期 Gold 意图 TUTOR_STRATEGY_PATTERNS。
- **数据集设计初衷**: 命中共性误区与导师引导解题，核心在教学行为。

### 用例 `holdout-026` (PARAPHRASE_NEED_LLM)
- **原始查询**: "这道求未知数应用题，为什么很多孩子都卡在列方程的第一步？"
- **Tier-1 候选意图**: `[]`
- **预测意图**: `STUDENT_FREQUENT_QUESTIONS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `QUESTION_TUTORING`
- **LLM 次要预测**: `DIFFICULT_CONCEPTS`
- **LLM 置信度**: `0.85`
- **LLM 思考解释**: The query explicitly asks why students struggle with a specific step in solving a problem, indicating a frequent question about student difficulties. The mention of 'setting up equations' points to the underlying difficulty of the algebraic concept.
- **人工复核与归因**: ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 QUESTION_TUTORING (判定为 STUDENT_FREQUENT_QUESTIONS)。
- **数据集设计初衷**: 口语化卡在列方程第一步，特定题型错因探究，实为单题辅导。

### 用例 `holdout-034` (PARAPHRASE_NEED_LLM)
- **原始查询**: "What are the common pitfalls that high schoolers stumble over in geometry proofs?"
- **Tier-1 候选意图**: `[]`
- **预测意图**: `RECURRING_MISCONCEPTIONS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `EXAM_ERROR_PATTERNS`
- **LLM 次要预测**: `DIFFICULT_CONCEPTS`
- **LLM 置信度**: `0.92`
- **LLM 思考解释**: The query explicitly asks for 'pitfalls' and 'stumble over,' which directly maps to recurring misconceptions, while the context of 'geometry proofs' identifies the specific difficult concepts involved.
- **人工复核与归因**: ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 EXAM_ERROR_PATTERNS (判定为 RECURRING_MISCONCEPTIONS)。
- **数据集设计初衷**: 英文关于几何证明常见试卷陷阱，实为考试错题模式。

### 用例 `holdout-035` (PARAPHRASE_NEED_LLM)
- **原始查询**: "How do experienced instructors scaffold when pupils fail to grasp equivalent fractions?"
- **Tier-1 候选意图**: `[]`
- **预测意图**: `DIFFICULT_CONCEPTS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `TUTOR_STRATEGY_PATTERNS`
- **LLM 次要预测**: `TUTOR_STRATEGY_PATTERNS`
- **LLM 置信度**: `0.92`
- **LLM 思考解释**: The query explicitly asks about the pedagogical method of scaffolding for a specific mathematical concept (equivalent fractions), making the difficulty of the concept the primary focus. The mention of 'experienced instructors' and 'scaffold' indicates a secondary interest in the teaching strategies used to address this difficulty.
- **人工复核与归因**: ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 TUTOR_STRATEGY_PATTERNS (判定为 DIFFICULT_CONCEPTS)。
- **数据集设计初衷**: 英文关于教师如何搭建支架点拨学生，实为导师策略模式。

### 用例 `holdout-036` (PARAPHRASE_NEED_LLM)
- **原始查询**: "What typical stumbling blocks do repeat test-takers encounter in quadratic equations?"
- **Tier-1 候选意图**: `[]`
- **预测意图**: `RECURRING_MISCONCEPTIONS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `STUDENT_FREQUENT_QUESTIONS`
- **LLM 次要预测**: `STUDENT_COHORT_PATTERNS`
- **LLM 置信度**: `0.85`
- **LLM 思考解释**: The query specifically asks for 'stumbling blocks' encountered by 'repeat test-takers,' which directly targets common errors and misconceptions. The mention of 'repeat' test-takers implies a focus on patterns within a specific student cohort.
- **人工复核与归因**: ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 STUDENT_FREQUENT_QUESTIONS (判定为 RECURRING_MISCONCEPTIONS)。
- **数据集设计初衷**: 英文关于二刷考生最常遇到的疑惑阻碍，实为学生高频问题。

### 用例 `holdout-038` (PARAPHRASE_NEED_LLM)
- **原始查询**: "为什么在做这道等比数列选择题时，大家几乎全军覆没选了 A？"
- **Tier-1 候选意图**: `[]`
- **预测意图**: `EXAM_ERROR_PATTERNS` (路由来源: `llm_classifier`)
- **预期 Gold 意图**: `QUESTION_TUTORING`
- **LLM 次要预测**: `STUDENT_FREQUENT_QUESTIONS`
- **LLM 置信度**: `0.92`
- **LLM 思考解释**: The query explicitly asks about a specific incorrect answer choice (A) resulting in a high failure rate, which directly maps to exam error patterns. The mention of 'everyone' selecting the same wrong option also indicates a frequent question or common misconception.
- **人工复核与归因**: ❌ 语义理解偏差: 口语化/边界表达未能被正确归类至 QUESTION_TUTORING (判定为 EXAM_ERROR_PATTERNS)。
- **数据集设计初衷**: 口语化全军覆没选 A，探究特定题目选项误区，实为单题辅导。


---
*报告自动生成自 `scripts/evaluate_intent_routing_tiers.py`*
