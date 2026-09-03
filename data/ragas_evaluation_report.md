# 🎓 Eedi-RAG 30 题黄金基准集 RAGAS 全链路量化评测报告

> **评测时间**: 2026-08-31 | **测试集规模**: 30 题黄金基准 | **知识库规模**: 100 场全学科会话 (2,335 轮对白)

## 📊 一、 全局核心量化指标雷达 (Global Scorecard)

| 指标分类 | 评测指标 (Metric) | 测量得分 | 工业级达标线 (Target) | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **检索层** | **Context Recall (召回率)** | **52.5%** | ≥ 75.0% | ⚠️ 待调优 |
| **检索层** | **Context Precision (精准度)** | **60.0%** | ≥ 70.0% | ⚠️ 待调优 |
| **生成层** | **Faithfulness (事实忠实度)** | **82.8%** | ≥ 80.0% | ✅ 达标 |
| **生成层** | **Answer Relevance (回答相关度)** | **100.0%** | ≥ 75.0% | ✅ 达标 |
| **溯源层** | **Citation Accuracy (原声引用率)** | **22.2%** | ≥ 70.0% | ⚠️ 待调优 |
| **意图层** | **Intent Accuracy (意图分类准确率)** | **86.7%** | ≥ 85.0% | ✅ 达标 |
| **综合** | **Global RAGAS Score** | **0.5334** | ≥ 0.7500 | ⚠️ 良好 |

---

## 🔍 二、 30 题逐案诊断与根因分析明细表 (Case Breakdown)

| # | 提问意图 | 考点模块 | Recall | Precision | Faithfulness | Relevance | 根因诊断 (Root-Cause Diagnosis) |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :--- |
| 1 | `四舍五入 5.4598 到 1 位小数时，学生为什么会误...` | Rounding to Decimal Places | 1.00 | 0.33 | 0.81 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 2 | `当学生误以为任意两个数的最小公倍数就是它们的乘积时，名师...` | Mental Multiplication and Division | 1.00 | 0.17 | 0.91 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 3 | `学生将「四舍五入到最近 0.02」等同于「四舍五入到小数...` | Rounding to Decimal Places | 1.00 | 1.00 | 0.91 | 1.00 | 🟢 优秀基线 (Healthy Baseline): 检索精准无噪，回答切题深刻且事实忠实。 |
| 4 | `学生看到分数线 (54+58)/2 时，为什么会先做 5...` | Mental Multiplication and Division | 1.00 | 0.77 | 0.72 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 5 | `展开含负数系数的括号 -3(1-2p) 时，学生最容易在...` | Expanding Single Brackets | 0.25 | 0.42 | 0.92 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 6 | `学生在展开 4(2x+1)-(5x-9) 时，对第二个括...` | Expanding Single Brackets | 0.00 | 0.40 | 0.74 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 7 | `当学生把 2a-4(3-a) 展开为 2a-12-a 而...` | Expanding Single Brackets | 0.50 | 0.67 | 0.92 | 1.00 | 🟡 召回不充分 (Insufficient Coverage): 回答忠实但信息量单薄，需适当调大 Top-K 或增强多视角子查询。 |
| 8 | `不等式 -2m>12 中，学生为什么会得到 m>-6 而...` | Solving Linear Inequalities | 0.00 | 0.40 | 0.86 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 9 | `面对「深度随时间变化的图像对应哪种玻璃杯形状」这类实际问...` | Real Life Graphs | 0.00 | 0.40 | 0.83 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 10 | `「已知均值为 4，求缺失数」这类均值反推题中，学生卡在哪...` | Averages (mean, median, mode) from a List of Data | 0.33 | 1.00 | 0.75 | 1.00 | 🟡 召回不充分 (Insufficient Coverage): 回答忠实但信息量单薄，需适当调大 Top-K 或增强多视角子查询。 |
| 11 | `当数据列表中同时包含小数和分数时（如 0.4, 1/4,...` | Averages (mean, median, mode) from a List of Data | 0.33 | 1.00 | 0.78 | 1.00 | 🟡 召回不充分 (Insufficient Coverage): 回答忠实但信息量单薄，需适当调大 Top-K 或增强多视角子查询。 |
| 12 | `在分组数据频率表中找中位数所在区间时，学生为什么会把「频...` | Averages and Range from Grouped Data | 0.50 | 0.68 | 0.77 | 1.00 | 🟡 召回不充分 (Insufficient Coverage): 回答忠实但信息量单薄，需适当调大 Top-K 或增强多视角子查询。 |
| 13 | `求极差（Range）时，学生初始想到「用最大数除以最小数...` | Range and Interquartile Range from a List of Data | 0.00 | 0.40 | 0.82 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 14 | `当学生说「体积就是把所有面的面积加起来」时，说明他混淆了...` | Volume of Prisms | 1.00 | 1.00 | 0.83 | 1.00 | 🟢 优秀基线 (Healthy Baseline): 检索精准无噪，回答切题深刻且事实忠实。 |
| 15 | `学生选 10:12 表示「ten to twelve a...` | Volume of Prisms | 0.00 | 0.40 | 0.83 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 16 | `当学生在频率表中用「频率列的极差」替代「数据值列的极差」...` | Volume of Prisms | 1.00 | 0.46 | 0.91 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 17 | `面对分数运算顺序题 1/2+1/3×1/4，学生为什么会...` | BIDMAS | 0.33 | 1.00 | 0.79 | 1.00 | 🟡 召回不充分 (Insufficient Coverage): 回答忠实但信息量单薄，需适当调大 Top-K 或增强多视角子查询。 |
| 18 | `BIDMAS 题中 n+4÷5 与 (n+4)/5 的区...` | BIDMAS | 1.00 | 1.00 | 0.91 | 1.00 | 🟢 优秀基线 (Healthy Baseline): 检索精准无噪，回答切题深刻且事实忠实。 |
| 19 | `当学生对心算减法 204-36 先减个位再减十位而得到 ...` | Mental Addition and Subtraction | 0.00 | 0.40 | 0.83 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 20 | `学生读到「£6,537 was taken」时误解为减法...` | Mental Multiplication and Division | 0.50 | 0.64 | 0.68 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 21 | `学生做 32×65=2080 → 3.2×6.5=? 时...` | Place Value | 0.00 | 0.40 | 0.83 | 1.00 | 🔴 检索失效 (Retrieval Failure): 关键事实未召回且噪声严重，需优化向量表征与考纲实体注入。 |
| 22 | `在带分数加法 3⅜ + 1⅝ 中，学生能正确得出分子之和...` | Adding and Subtracting Fractions | 0.67 | 0.63 | 0.91 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 23 | `在异分母减法 5/7 - 1/4 中，学生已知公分母是 ...` | Adding and Subtracting Fractions | 0.67 | 1.00 | 0.83 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 24 | `对于质数判定题「100 之后的下一个质数是哪个」，学生选...` | Prime Numbers and Prime Factors | 0.67 | 0.58 | 0.91 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 25 | `学生在负数乘法中认为 p×(-q) 和 (-q)² 的结...` | Multiplying and Dividing Negative Numbers | 0.50 | 0.62 | 0.67 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 26 | `当速度问题要求将 40 分钟转换为小时的分数时，学生为什...` | Speed, Distance, Time | 0.33 | 0.91 | 0.73 | 1.00 | 🟡 召回不充分 (Insufficient Coverage): 回答忠实但信息量单薄，需适当调大 Top-K 或增强多视角子查询。 |
| 27 | `学生不知道「密度」的概念时，导师如何从「质量除以体积」的...` | Density | 1.00 | 0.12 | 0.91 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 28 | `在除以小数的题 0.2÷0.4 中，导师如何通过「分子分...` | Adding and Subtracting Negative Numbers | 0.50 | 0.40 | 0.82 | 1.00 | 🟡 召回不充分 (Insufficient Coverage): 回答忠实但信息量单薄，需适当调大 Top-K 或增强多视角子查询。 |
| 29 | `当学生误将 7.503 四舍五入到最近整数的结果写为 7...` | Rounding to Decimal Places | 0.67 | 0.40 | 0.91 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |
| 30 | `356,958 四舍五入到最近百位时，学生把百位上的 9...` | Rounding to the Nearest Whole (10, 100, etc) | 1.00 | 0.40 | 0.81 | 1.00 | 🟡 局部待调优 (Sub-optimal): 系统各指标均衡但存在调优空间。 |

---

## 💡 三、 调优建议与下阶段行动项

1. **学情原声引用进一步增强**：针对部分长尾考点，继续扩充 DuckDB 学生困惑检索模式。
2. **MMR 多样性参数校准**：当前 $\lambda=0.7$ 在保持相关性与多样性之间表现稳健，可作为 Baseline 固化。