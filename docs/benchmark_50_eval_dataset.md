# 2026 Eedi-RAG 全量 50 条真实教研基准评测数据集规范手册 (Benchmark 50 Dataset Spec)

## 一、 数据集架构与设计理念 (Design Principles)

本评测数据集面向 **K12 数学个性化辅导与教研诊断场景**，完整覆盖底层知识库中 10 个真实辅导场景（Session #10 ~ Session #80），总计构建 **50 条高仿真端到端真实教研提问**（每场会话 5 条，严格按 25 条【学情错因诊断】+ 25 条【名师启发策略】平衡分布）。

### 1. 核心设计原则
* **真实教研驱动**：模拟一线教师、教研员和辅导助教在面对学生卡点时的真实提问语境（涵盖口语化、概念泛化、反问启发、具体算式等多种表达形态）；
* **双轨物理隔离对齐**：清晰区分 `student_misconceptions`（学情诊断与错因机理）与 `tutor_strategies`（名师破局一问与启发脚手架）；
* **白盒证据强绑定**：每条 Query 均精确锚定真实会话中的 `source_turn_ids`，实现“卡片为引，原文为据”的 100% 可解释性与防幻觉验证。

---

## 二、 50 条全量教研评测数据集详表 (按 10 大教学场景分组)

```
====================================================================================================
📚 目录索引:
  1. [Session 10] 四舍五入 5.4598 到 1 位小数 (Decimals & Rounding) -------------------- #01 ~ #05
  2. [Session 14] 分式复合运算中的运算顺序 (Order of Operations & BIDMAS) ------------ #06 ~ #10
  3. [Session 23] 最小公倍数 (LCM) 与两数乘积混淆 (Factors, Multiples & Coprime) ------ #11 ~ #15
  4. [Session 29] 频数统计表中求极差 (Frequency Tables & Range) ---------------------- #16 ~ #20
  5. [Session 33] 非常规精度四舍五入到 0.02 倍数 (Rounding to Nearest 0.02) ----------- #21 ~ #25
  6. [Session 42] 负数乘方中负号归属与括号规则 (Negative Base Exponentiation) --------- #26 ~ #30
  7. [Session 59] 质数判断与末尾为 5 的合数排除 (Prime Numbers & Divisibility) -------- #31 ~ #35
  8. [Session 69] 一元一次不等式两边除以负数 (Linear Inequalities & Sign Reversal) ---- #36 ~ #40
  9. [Session 77] 三维长方体体积与表面积混淆 (Volume vs Surface Area of Cuboid) ------- #41 ~ #45
 10. [Session 80] 英语时间表达与表盘钟面识别 (Analog Clocks & '10 to 12') ------------- #46 ~ #50
====================================================================================================
```

---

### 场景 1：Session #10 | 小数四舍五入精度与数位截断

* **原题 ID (QID)**: `104614`
* **考纲路径**: `Mathematics > Number > Rounding and Estimating > Rounding to Decimal Places`
* **题目原题**: "What is 5.4598 rounded to 1 decimal place?" (选项: A: 5.4, B: 5.45, C: 5.46, D: 5.5)
* **真值错因卡**: 「四舍五入数位保留与小数点移位混淆」 (错误选项: B, 证据轮次: `[2, 4, 8]`)
* **真值策略卡**: 「[Scaffolding] Can you round 5.45 to one decimal place?」 (教学动作: `<Press for Accuracy>`, `<Keep Together>`, 证据轮次: `[3, 5, 9]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#01** | 错因诊断 | `student_misconceptions` | 四舍五入 5.4598 到 1 位小数时，学生为什么会误选 5.45？ | **【典型错选项归因】** 测试模型能否通过具体数值 `5.4598` 和错选 `5.45` 准确定位数位截断与精度混淆机理。 | `Session 10` (Turn 2,4,8) |
| **#02** | 名师策略 | `tutor_strategies` | 当学生把保留一位小数做成保留两位时，名师是如何用简单数字破局提问的？ | **【简化数值启发策略】** 评测对名师将复杂多位小数简化为 `5.45` 降低认知负荷的破局一问召回。 | `Session 10` (Turn 3,5) |
| **#03** | 错因诊断 | `student_misconceptions` | 在将 5.4598 近似为一位小数时，学生容易混淆十分位与百分位的原因是什么？ | **【数位概念辨析】** 评测对“十分位 (1dp)”与“百分位 (2dp)”术语混淆及数位对齐障碍的语义召回。 | `Session 10` (Turn 2,8) |
| **#04** | 名师策略 | `tutor_strategies` | 针对小数四舍五入时的数位截断误区，导师通过怎样的反问引导学生关注关键判定数字？ | **【关键位判定反问】** 考查对名师通过聚焦百分位关键数字 `5` 进行判定并进位的话术检索。 | `Session 10` (Turn 5,9) |
| **#05** | 错因诊断 | `student_misconceptions` | 面对四舍五入题目，学生为什么容易直接截断后面多余数字而不进行进位？ | **【算法认知缺陷】** 测试对“纯截断 (Truncation) 替代舍入 (Rounding)”深层认知机理的泛化召回。 | `Session 10` (Turn 4,8) |

---

### 场景 2：Session #14 | 分式复合运算中的运算优先级

* **原题 ID (QID)**: `107386`
* **考纲路径**: `Mathematics > Number > Fractions and Decimals > Order of Operations`
* **题目原题**: "Work out \( (54 + 58) \div 2 \)" (选项: A: 56, B: 83, C: 112, D: 28)
* **真值错因卡**: 「运算顺序混淆与分数符号认知不清」 (错误选项: B [54 + 29 = 83], 证据轮次: `[4, 6]`)
* **真值策略卡**: 「[Socratic_Questioning] what do you think this question is asking?」 (教学动作: `<Press for Reasoning>`, `<Revoicing>`, 证据轮次: `[3, 5]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#06** | 错因诊断 | `student_misconceptions` | 学生在做包含加法和除法的分式运算时容易犯什么运算顺序错误？ | **【四则运算优先级】** 评测模型能否识别学生将 `54 + (58 ÷ 2) = 83` 的常见先除后加错误顺序。 | `Session 14` (Turn 4,6) |
| **#07** | 名师策略 | `tutor_strategies` | 名师如何通过苏格拉底反问引导学生识别混合运算中的运算优先级？ | **【审题苏格拉底反问】** 考查对名师用开放式反问“这道题到底在问什么”促使学生重构算式结构的检索。 | `Session 14` (Turn 3,5) |
| **#08** | 错因诊断 | `student_misconceptions` | 在计算 (54+58)/2 这类算式时，学生为什么会先算除法或者遗漏括号内的求和？ | **【括号与除号作用域】** 评测对“忽视括号整体性/分子求和优先于分数线除法”认知障碍的捕获。 | `Session 14` (Turn 4) |
| **#09** | 名师策略 | `tutor_strategies` | 当学生对多步混合运算感到困惑时，导师采用什么脚手架步骤拆解计算过程？ | **【两步计算拆解脚手架】** 考查导师先引导算分子求和 `54+58=112` 再除以 `2` 的渐进式教学动作。 | `Session 14` (Turn 7,9) |
| **#10** | 名师策略 | `tutor_strategies` | 导师如何通过提示题目真实要求来帮学生理清复合算式的计算次序？ | **【题意重述与聚焦】** 测试通过 `Revoicing` 话术帮助学生重新锚定混合运算核心步骤的检索。 | `Session 14` (Turn 5,11) |

---

### 场景 3：Session #23 | 最小公倍数 (LCM) 与两数乘积特例泛化

* **原题 ID (QID)**: `147113`
* **考纲路径**: `Mathematics > Number > Multiples and Factors > LCM`
* **题目原题**: "Is the lowest common multiple of two numbers always equal to their product?" (选项: A: Always, B: Sometimes, C: Never)
* **真值错因卡**: 「将特定条件下的正确规律泛化为普遍规律」 (错误选项: C, 证据轮次: `[1, 7, 9, 11, 17, 19, 21, 25]`)
* **真值策略卡**: 「[Socratic_Questioning] Is there an example where it doesn't work?」 (教学动作: `<Press for Accuracy>`, `<Press for Reasoning>`, 证据轮次: `[4, 6, 14, 16, 18, 22, 24]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#11** | 错因诊断 | `student_misconceptions` | 学生为什么会误以为任意两个数的最小公倍数就是它们的乘积？ | **【特例过度泛化】** 评测将 3 和 5（互质数）乘积规律盲目推广到所有非互质数场景的错因召回。 | `Session 23` (Turn 1,7,9) |
| **#12** | 名师策略 | `tutor_strategies` | 当学生盲目认为两数相乘就是LCM时，老师用什么反例策略引导学生思考？ | **【反例教学法 (Counterexample)】** 考查名师通过追问反例（如 4 和 6 的 LCM 是 12 而非 24）破除思维定势。 | `Session 23` (Turn 4,14,16) |
| **#13** | 错因诊断 | `student_misconceptions` | 学生将 3 和 5 等互质数特例推广到所有数求最小公倍数时的思维定势是什么？ | **【互质性前置认知缺失】** 考查学生在未掌握互质（Coprime）概念前机械记忆乘积公式的认知盲区。 | `Session 23` (Turn 9,11) |
| **#14** | 名师策略 | `tutor_strategies` | 名师如何通过提问“是否存在不适用的例子”促使学生自主反思最小公倍数计算方法？ | **【自主反思提问】** 评测针对经典名师破局一问 `Is there an example where it doesn't work?` 的直接命中。 | `Session 23` (Turn 14,18) |
| **#15** | 错因诊断 | `student_misconceptions` | 求非互质数（如 4 和 6）的公倍数时，学生漏掉公因数导致结果偏大的根本原因？ | **【公因数忽略机理】** 测试对“未能提取公共质因数导致倍数成倍膨胀”数学本质机理的语义检索。 | `Session 23` (Turn 17,21) |

---

### 场景 4：Session #29 | 频数分布表中求极差 (Range from Frequency Table)

* **原题 ID (QID)**: `108130`
* **考纲路径**: `Mathematics > Statistics > Data Presentation > Frequency Tables`
* **题目原题**: "Find the range of the number of TVs from the frequency table." (选项: A: 3, B: 12, C: 4, D: 11)
* **真值错因卡**: 「范围计算中频率与数据值的混淆」 (错误选项: B, 证据轮次: `[4, 6]`)
* **真值策略卡**: 「[Socratic_Questioning] At least one person has to have given that answer... can you see those 7 people who have 1 TV?」 (教学动作: `<Revoicing>`, `<Press for Accuracy>`, 证据轮次: `[5, 7, 9]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#16** | 错因诊断 | `student_misconceptions` | 在看电视数量的频数统计表中，学生计算极差时为什么会混淆频数和数据本身？ | **【频数与变量值概念混淆】** 评测学生错误拿人数频数最大值（12）做极差计算（12-0=12）的诊断召回。 | `Session 29` (Turn 4,6) |
| **#17** | 名师策略 | `tutor_strategies` | 名师如何引导学生看懂频数表里的实际人数与对应电视机数量？ | **【现实情境具象化】** 考查名师用“这7个人拥有1台电视”的实际调查意义唤醒学生真实理解的话术检索。 | `Session 29` (Turn 5,7) |
| **#18** | 错因诊断 | `student_misconceptions` | 学生误将频数列表中的最大数值或零频数项作为极差极值计算的深层机理？ | **【空频数与极值伪关联】** 考查学生将 `Frequency=0` 的无效行当作数据最小值计算的典型错误剖析。 | `Session 29` (Turn 4) |
| **#19** | 名师策略 | `tutor_strategies` | 导师如何结合现实生活情境帮助学生区分样本频数与变量测量值？ | **【统计学量纲辨析教学】** 评测导师将表格抽象数据映射至真实生活实体（人与电视）的启发法。 | `Session 29` (Turn 7,9) |
| **#20** | 名师策略 | `tutor_strategies` | 在统计图表教学中，名师如何一步步引导学生定位数据集中的最大值与最小值？ | **【极差定位脚手架】** 考查导师先找最大电视数再找最小电视数的两步定位法。 | `Session 29` (Turn 5,9) |

---

### 场景 5：Session #33 | 非常规步长舍入 (0.02 倍数区间夹逼)

* **原题 ID (QID)**: `104615`
* **考纲路径**: `Mathematics > Number > Rounding and Estimating > Rounding to Decimal Places`
* **题目原题**: "What is 3.153 rounded to the nearest 0.02?" (选项: A: 3.15, B: 3.14, C: 3.16, D: 3.20)
* **真值错因卡**: 「四舍五入中‘四舍五入到最近0.02’与传统‘四舍五入到最近0.1/0.01’概念混淆」 (错误选项: B, 证据轮次: `[2, 4]`)
* **真值策略卡**: 「[Scaffolding] So what would the value above 3.153 be in the 0.02 times table?」 (教学动作: `<Press for Accuracy>`, `<Revoicing>`, 证据轮次: `[5, 7, 9]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#21** | 错因诊断 | `student_misconceptions` | 将 3.153 四舍五入到最近的 0.02 时，学生面临的主要概念卡点是什么？ | **【非十进制步长舍入卡点】** 评测面对 `0.02` 非标准倍数舍入时，学生对目标刻度不明确的错因。 | `Session 33` (Turn 2,4) |
| **#22** | 名师策略 | `tutor_strategies` | 针对四舍五入到 0.02 倍数这道难题，名师搭建了怎样的脚手架步骤？ | **【0.02倍数区间夹逼】** 考查导师引导寻找 3.153 两侧最近的 0.02 倍数（3.14 与 3.16）的脚手架检索。 | `Session 33` (Turn 5,7) |
| **#23** | 错因诊断 | `student_misconceptions` | 学生把“四舍五入到 0.02”误当作常规保留一位或两位小数时的认知障碍？ | **【传统算法机械套用】** 评测由于思维定势将问题退化为简单“看千分位3舍去得到3.15或3.14”的机理。 | `Session 33` (Turn 2) |
| **#24** | 名师策略 | `tutor_strategies` | 导师如何通过列举 3.14 与 3.16 的区间端点启发学生判断距离 3.153 最近的数值？ | **【距离度量启发法】** 考查名师对比 `|3.153-3.14|=0.013` 与 `|3.16-3.153|=0.007` 的临近判断策略。 | `Session 33` (Turn 7,9) |
| **#25** | 错因诊断 | `student_misconceptions` | 面对非标准步长的舍入问题，学生缺乏数轴区间概念导致盲目进位的心理机制？ | **【数轴几何直观缺失】** 考查学生缺乏在数轴上将目标数投射至离散刻度点的直观几何感知。 | `Session 33` (Turn 4) |

---

### 场景 6：Session #42 | 负数乘方中负号归属与括号作用域

* **原题 ID (QID)**: `89558`
* **考纲路径**: `Mathematics > Algebra > Algebraic Expressions > Powers and Indices`
* **题目原题**: "Which of the following statements is true for positive numbers p and q? Statement 1: \( p \times (-q) = (-p) \times q \); Statement 2: \( (-q)^2 = -q^2 \)" (选项: A: Statement 1 only, B: Statement 2 only, C: Both, D: Neither)
* **真值错因卡**: 「负数幂运算中负号处理规则混淆」 (错误选项: C, 证据轮次: `[12, 13]`)
* **真值策略卡**: 「[Socratic_Questioning] Does -p x q give us the same answer?」 (教学动作: `<Keep Together>`, `<Press for Accuracy>`, 证据轮次: `[3, 5, 7, 11, 13]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#26** | 错因诊断 | `student_misconceptions` | 学生对于 (-q)^2 和 -q^2 的负号位置与括号规则有什么深层认知盲区？ | **【括号与幂运算优先级】** 评测对 `(-q)^2 = q^2` 与 `-q^2 = -(q^2)` 符号作用域混淆的精确诊断。 | `Session 42` (Turn 12,13) |
| **#27** | 名师策略 | `tutor_strategies` | 辅导负数乘方时，导师如何通过追问 -p x q 帮助学生厘清负号归属？ | **【代数式乘法对称性验证】** 考查名师通过对比 `p*(-q) = -pq` 与 `(-p)*q = -pq` 建立符号守恒感知的策略。 | `Session 42` (Turn 3,5) |
| **#28** | 错因诊断 | `student_misconceptions` | 学生认为负数参与任何乘方运算结果都必定带负号的错误直觉是什么？ | **【符号传递直觉谬误】** 考查学生直觉认为“带负号的式子平方后依然是负数”的错误心理表征。 | `Session 42` (Turn 12) |
| **#29** | 名师策略 | `tutor_strategies` | 名师如何通过对比 (-p)*q 与 -(p*q) 的等价形式帮助学生建立代数式符号感知？ | **【等价形式转化法】** 评测名师通过展开代数相乘公式强化负因数移动法则的教学动作。 | `Session 42` (Turn 5,11) |
| **#30** | 名师策略 | `tutor_strategies` | 当学生混淆乘方优先级与相反数符号时，导师如何用具体代数展开式进行验证？ | **【展开式直观求证】** 考查名师引导写出 `(-q)*(-q)` 对比 `-(q*q)` 彻底消除歧义的脚手架步骤。 | `Session 42` (Turn 11,13) |

---

### 场景 7：Session #59 | 质数判定与末尾为 5 的整除性排除

* **原题 ID (QID)**: `103042`
* **考纲路径**: `Mathematics > Number > Factors, Multiples and Primes > Prime Numbers and Prime Factors`
* **题目原题**: "Which is the next prime number greater than 100?" (选项: A: 101, B: 103, C: 105, D: 107)
* **真值错因卡**: 「质数概念与倍数特征应用模糊」 (错误选项: C, 证据轮次: `[2, 6, 12]`)
* **真值策略卡**: 「[Socratic_Questioning] Why can we rule it out as it ends in a 5?」 (教学动作: `<Press for Accuracy>`, `<Press for Reasoning>`, 证据轮次: `[5, 11]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#31** | 错因诊断 | `student_misconceptions` | 学生判断 105 是否为质数时，为什么能背出定义却依然判断错误？ | **【抽象定义与具体判别脱节】** 评测学生能复述定义但面对三位数奇数时无法关联“尾数是5能被5整除”的脱节机理。 | `Session 59` (Turn 2,6) |
| **#32** | 名师策略 | `tutor_strategies` | 老师如何用‘末尾是5的数字特征’迅速启发学生排除合数 105？ | **【特征速算法破局】** 考查名师提问 `Why can we rule it out as it ends in a 5?` 引导排除选项 C 的话术。 | `Session 59` (Turn 5,11) |
| **#33** | 错因诊断 | `student_misconceptions` | 学生在判断较大奇数是否为质数时，忽视基本整除性规则的思维盲区是什么？ | **【试除法停滞盲区】** 考查学生盲目从 2、3 试除而遗漏个位数 5 整除特征的思维窄化。 | `Session 59` (Turn 6,12) |
| **#34** | 名师策略 | `tutor_strategies` | 导师如何引导学生观察 105 的个位数字并联想到 5 的倍数特征？ | **【个位观察引导】** 评测导师让学生聚焦个位数字建立整除性直觉的启发动作。 | `Session 59` (Turn 5) |
| **#35** | 错因诊断 | `student_misconceptions` | 为什么学生容易把非偶数的奇数（如 105）直觉性地误判为质数？ | **【奇数-质数等同混淆】** 考查学生由于“偶数除2外都是合数”产生“非偶数大概率是质数”的启发式认知偏差。 | `Session 59` (Turn 2,12) |

---

### 场景 8：Session #69 | 一元一次不等式两边除以负数翻转不等号

* **原题 ID (QID)**: `106909`
* **考纲路径**: `Mathematics > Algebra > Inequalities > Solving Linear Inequalities`
* **题目原题**: "Leo says \( -2m > 12 \) gives \( m > -6 \); Sophie says \( -2 + m > 12 \) gives \( m > 14 \). Who is correct?" (选项: A: Only Leo, B: Only Sophie, C: Both, D: Neither)
* **真值错因卡**: 「不等式两边除以负数时未反转符号，且解题目标混淆」 (错误选项: B, 证据轮次: `[10, 20, 22]`)
* **真值策略卡**: 「[Scaffolding] Super :) What does that give us ? :0 (引导计算 -12 ÷ -2 并引出翻转符号)」 (教学动作: `<Press for Accuracy>`, `<Revoicing>`, 证据轮次: `[9, 11, 15, 21]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#36** | 错因诊断 | `student_misconceptions` | 解不等式两边同时除以负数（如 -12 ÷ -2）时，学生常漏掉哪个核心操作？ | **【不等号反转遗忘】** 评测对“两边除以负系数必须翻转不等号方向（> 变 <）”规则遗漏的诊断。 | `Session 69` (Turn 10,20) |
| **#37** | 名师策略 | `tutor_strategies` | 导师如何分步引导学生计算 -12 ÷ -2 并引出翻转不等号的关键规则？ | **【分步求解加规则引出】** 考查导师先让学生完成数值运算再引入不等号翻转定理的渐进教学法。 | `Session 69` (Turn 9,11,15) |
| **#38** | 错因诊断 | `student_misconceptions` | 学生在求解 -2m > 12 时，将等式移项法则生搬硬套到不等式的认知误区？ | **【等式定势在不等式中机械迁移】** 考查将等式保持符号不变的惯性直接迁移至负系数不等式的错因机理。 | `Session 69` (Turn 20,22) |
| **#39** | 名师策略 | `tutor_strategies` | 导师如何通过带正负号的数值除法逐步引出“除以负数不等号方向改变”的核心定理？ | **【负数除法推导脚手架】** 考查将抽象不等式定理拆解为基础有理数除法的教学步骤。 | `Session 69` (Turn 11,15) |
| **#40** | 名师策略 | `tutor_strategies` | 名师如何用肯定鼓励加渐进提问推动学生推进解题？ | **【情感肯定与渐进提问协同】** 评测通过 `Super :)` 肯定学生局部正确并即时追问推进的互动策略。 | `Session 69` (Turn 9,21) |

---

### 场景 9：Session #77 | 三维长方体体积计算公式与表面积混淆

* **原题 ID (QID)**: `131463`
* **考纲路径**: `Mathematics > Geometry and Measure > Volume and Surface Area > Volume of Prisms`
* **题目原题**: "What is the volume of this cuboid (dimensions 10cm x 4cm x 5cm)?" (选项: A: 19 cm³, B: 50 cm³, C: 200 cm³, D: 220 cm²)
* **真值错因卡**: 「体积与表面积概念混淆」 (错误选项: D [表面积=220], 证据轮次: `[3, 5]`)
* **真值策略卡**: 「[Analogy] Remember we multiply the length by the width by the height to find the volume of a cuboid This question wants the volume not the area :)」 (教学动作: `<Revoicing>`, `<Press for Accuracy>`, 证据轮次: `[4, 8, 10, 12]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#41** | 错因诊断 | `student_misconceptions` | 在计算长方体体积时，学生为什么容易把体积公式和表面积/周长混在一起？ | **【空间容积与外表面积混淆】** 评测学生误用表面积公式 `2*(10*4+10*5+4*5)=220` 替代体积 `10*4*5=200` 的诊断。 | `Session 77` (Turn 3,5) |
| **#42** | 名师策略 | `tutor_strategies` | 名师如何用长宽高连乘公式（length x width x height）快速纠偏体积计算？ | **【长宽高公式提醒纠偏】** 考查名师直接指出“题目求的是体积不是面积”并重申连乘公式的策略。 | `Session 77` (Turn 4,8) |
| **#43** | 错因诊断 | `student_misconceptions` | 面对长方体三维尺寸（如 10cm, 4cm, 5cm），学生将加法与乘法混淆导致结果错误的原因？ | **【棱长求和与体积相乘混淆】** 考查学生将三维尺寸简单相加（如 `10+4+5=19` 误选 A）的量纲错误分析。 | `Session 77` (Turn 3) |
| **#44** | 名师策略 | `tutor_strategies` | 导师如何通过强调“这道题求的是体积不是面积”来明确几何计算目标？ | **【几何目标重新锚定】** 评测通过语言澄清消除几何量度歧义的教学话术。 | `Session 77` (Turn 4,10) |
| **#45** | 错因诊断 | `student_misconceptions` | 学生对于“空间容积”与“外表平面积”物理意义区分不清的深层机理？ | **【三维空间感与二维平面感脱节】** 考查学生在立体几何中缺乏“填充内部”与“包裹外部”直观物理表征的错因。 | `Session 77` (Turn 5) |

---

### 场景 10：Session #80 | 英语时间表达中的分钟-小时倒装与钟面识别

* **原题 ID (QID)**: `75997`
* **考纲路径**: `Mathematics > Geometry and Measure > Time > Clocks and Analog Time`
* **题目原题**: "Which clock shows ten to twelve at night?" (选项: A: 10:12, B: 12:50, C: 00:50, D: 23:50)
* **真值错因卡**: 「时间表述中的分钟-小时倒装与12/24小时制混淆」 (错误选项: A [10:12], 证据轮次: `[4]`)
* **真值策略卡**: 「[Socratic_Questioning] Which time does B show?」 (教学动作: `<Press for Accuracy>`, `<Revoicing>`, `<Keep Together>`, 证据轮次: `[5, 7, 9, 10, 11]`)

| ID | 评测类别 | 目标集合 | 用户教研提问 (Raw Query) | 中文注释与评测考点 | 真值锚点与证据 |
|:---:|:---:|:---:|---|---|---|
| **#46** | 错因诊断 | `student_misconceptions` | 学生为什么会把英语时间 '10 to 12' 误解为 10点12分（10:12）？ | **【英语时间倒装词序直译】** 评测将 `ten to twelve` 按字面字序误译为 `10:12`（而非差10分到12点即 11:50）的机理。 | `Session 80` (Turn 4) |
| **#47** | 名师策略 | `tutor_strategies` | 导师如何通过指向选项时钟（Which time does B show）引导学生识别正确钟面？ | **【选项反推排除启发法】** 考查名师先让学生认读 B 选项（12:50）含义，再自主反推正确选项 D（23:50）的策略。 | `Session 80` (Turn 5,7) |
| **#48** | 错因诊断 | `student_misconceptions` | 学生在阅读英式时间表达 'X to Y' 时，因直译词序导致的时分倒装错误？ | **【跨语言句式迁移障碍】** 考查学生缺乏“to 表示朝向下一小时差量”语义理解的深层原因剖析。 | `Session 80` (Turn 4) |
| **#49** | 名师策略 | `tutor_strategies` | 导师如何通过逐个排除错误钟面选项来降低学生的认知负荷？ | **【逐项排除法】** 评测导师引导学生逐个分析 B、C、D 选项将复杂题目降维的教学动作。 | `Session 80` (Turn 7,9,11) |
| **#50** | 名师策略 | `tutor_strategies` | 名师如何通过提问具体的表盘时间（如 11:50）引导学生关联 '差10分到12点' 的概念？ | **【具体时间锚定法】** 考查名师将 24 小时制数字 `23:50` 与生活用语 `10 to 12 at night` 桥接的教学策略。 | `Session 80` (Turn 9,11) |

---

## 三、 数据集质量与覆盖度审计 (Dataset Quality & Coverage Audit)

1. **学科知识点覆盖度**: 100% 覆盖数与代数（小数、分式、负数幂、质数、不等式）、统计与概率（频数表与极差）、图形与几何（长方体体积、时钟与时间）；
2. **提问句式多样性**: 涵盖“为什么误选”、“深层认知盲区”、“名师如何破局”、“脚手架步骤”、“反例追问”等 8 种高频教研句型；
3. **可复现性保证**: 评测脚本已沉淀至 [scripts/eval_50_queries.py](file:///E:/PIAgent/10-projects/AgentLearn/Eedi-RAG/scripts/eval_50_queries.py)，可随时一键全量复测。
