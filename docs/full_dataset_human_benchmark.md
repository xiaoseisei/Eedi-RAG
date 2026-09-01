# Eedi 全量融合数据集人类基准与内容规范手册 (Full Dataset Human Benchmark & Specification)

## 一、 执行摘要与全量融合数据看板 (Executive Census Dashboard)

本项目已完成对 Eedi 原始三表（`anchored-dialogues/train.csv` 55,322 行、`dialogue-subjects.csv` 9,034 行、`dq-question-metadata.csv` 10,857 行）的全量 ETL 清洗与拓扑关联，正式生成全量融合数据集文件：
📁 **文件位置**：[`data/cleaned_sessions.jsonl`](file:///E:/PIAgent/10-projects/AgentLearn/Eedi-RAG/data/cleaned_sessions.jsonl) (文件大小: **~8.2 MB**)

```
====================================================================================================
📊 【Eedi 全量融合数据集核心指标概览 (Full Dataset Census)】
====================================================================================================
  • 融合会话总数 (Total Fused Sessions):         1,576 场
  • 质量门禁达标率 (Valid Tutoring Rate):        1,576 / 1,576 (100.0% 均为双向实质辅导)
  • 覆盖独立考题数 (Distinct Questions):         868 道
  • 执教导师人数 (Distinct Tutors):             26 位
  • 压缩清洗后总轮次 (Total Dialogue Turns):     37,186 轮
  • 核心教学问答轮次 (Academic Teaching Turns):   35,667 轮 (占比 95.91%)
  • 纯寒暄与客套轮次 (Greeting/Noise Turns):      1,519 轮 (占比 4.09%)
  • 单场会话轮次均值 (Mean Turns / Session):     23.60 轮 (中位数: 21 轮, 范围: 10 ~ 103 轮)
  • 师生发言轮次比 (Student vs Tutor Turns):     学生 11.46 轮 : 导师 12.13 轮 (健康交互比 ~ 1:1)
====================================================================================================
```

### 1. 学科知识领域分布 (Subject Hierarchy Distribution)
| 一级学科 (Subject) | 覆盖会话数 | 核心二级主题 (Top Topics) |
|---|:---:|---|
| **Number (数与运算)** | **948 场** (60.2%) | Basic Arithmetic (298), Fractions (213), Factors & Primes (117), Decimals (91), Rounding (92) |
| **Algebra (代数与方程)** | **364 场** (23.1%) | Solving Equations (107), Negative Numbers (101), Writing Expressions (87), Sequences (62) |
| **Geometry and Measure (几何与度量)** | **343 场** (21.8%) | Perimeter & Area (64), Volume of Prisms (45), Clocks & Time (38), Angles (42) |
| **Data and Statistics (数据与统计)** | **166 场** (10.5%) | Frequency Tables & Range (88), Probability (46), Bar Charts (32) |

*(注：部分题目横跨多个关联考点，故各学科计数存在重叠交叉)*

### 2. 官方导师教学动作标签分布 (Talk Moves Census)
在 3.7 万轮对话中，导师展现出的核心启发式教学动作分布如下：
* **`<Press for Accuracy>`** (**6,740 次**)：精准追问事实、结果与数字准确性；
* **`<Keep Together>`** (**3,696 次**)：拉齐师生共识，维持教学节奏；
* **`<Revoicing>`** (**2,142 次**)：重述并提炼学生的表达，纠偏口语化模糊表述；
* **`<Press for Reasoning>`** (**177 次**)：深度追问学生解题背后的逻辑推理与“为什么”；
* **`<Getting Student to Relate>`** (**153 次**)：引导学生将当前步骤与之前学过的知识点建立关联；
* **`<Restating>`** (**27 次**)：让学生复述关键规则与定理。

---

## 二、 融合数据内容标准与 Schema 规范 (Data Contract)

融合后的实体定义为 **`CleanedSession`（教学交互全景单元 / Pedagogical Interaction Unit, PIU）**，采用自包含嵌套结构。

### 1. 顶层实体字段定义 (`CleanedSession`)
| 字段名 | 类型 | 必填 | 业务含义与标准定义 | 样例取值 |
|---|:---:|:---:|---|---|
| `intervention_id` | `int` | 是 | 会话全局唯一业务主键 | `10` |
| `question_id` | `int` | 是 | 绑定的考题全局唯一 ID (`QuestionId_DQ`) | `104614` |
| `tutor_id` | `int` | 是 | 授课导师唯一 ID | `1002` |
| `subjects` | `object` | 是 | 考纲层级知识树对象（详见 `SubjectHierarchy`） | 见下表 |
| `question` | `object` | 是 | 完整考题与选项实体（详见 `Question`） | 见下表 |
| `turns` | `list` | 是 | 时序清洗后的对话轮次列表（详见 `DialogueTurn`） | 见下表 |
| `total_turns` | `int` | 是 | 会话压缩清洗后的有效对话总轮数 | `26` |
| `student_turn_count` | `int` | 是 | 学生有效发言轮次数 | `13` |
| `tutor_turn_count` | `int` | 是 | 导师有效发言轮次数 | `13` |
| `has_valid_tutoring` | `bool` | 是 | 质量门禁状态：`True` 为实质双向教学会话 | `True` |

---

### 2. 考纲层级知识树 (`SubjectHierarchy`)
| 字段名 | 类型 | 规范说明 | 样例取值 |
|---|:---:|---|---|
| `paths` | `List[str]` | 格式化后的完整三级考纲路径 | `["Mathematics > Number > Rounding > Rounding to Decimal Places"]` |
| `subjects` | `List[str]` | 一级学科名称集合 (Level 1) | `["Mathematics", "Number"]` |
| `topics` | `List[str]` | 二级主题名称集合 (Level 2) | `["Rounding and Estimating"]` |
| `subtopics` | `List[str]` | 三级具体微考点集合 (Level 3) | `["Rounding to Decimal Places"]` |

---

### 3. 考题与选项实体 (`Question`)
| 字段名 | 类型 | 规范说明 | 样例取值 |
|---|:---:|---|---|
| `question_id` | `int` | 题目唯一 ID | `104614` |
| `question_text` | `str` | 经 LaTeX 保护隔离后的题干文本 | `"What is \( 5.4598 \) rounded to 1 decimal place?"` |
| `options` | `Dict[str, str]` | 选项字典，键为 `'A','B','C','D'` | `{"A": "5.4", "B": "5.45", "C": "5.46", "D": "5.5"}` |
| `images` | `Dict[str, str]` | 题目插图与选项配图 URL 字典 | `{"Question": "url...", "A": "url..."}` |

---

### 4. 单轮时序对话实体 (`DialogueTurn`)
| 字段名 | 类型 | 规范说明 | 样例取值 |
|---|:---:|---|---|
| `turn_id` | `int` | 1-based 连续递增的轮次序号 | `2` |
| `speaker` | `str` | 发言角色：`"tutor"` (导师) 或 `"student"` (学生) | `"student"` |
| `is_tutor` | `bool` | 布尔角色标识 (`True` 导师, `False` 学生) | `False` |
| `text` | `str` | 经脱敏、合并、去噪后的最终呈现文本 | `"I think it is 5.45"` |
| `raw_messages` | `List[str]` | 压缩前逐条打字发送的原始碎句列表 (100% 溯源保真) | `["I think it is", "5.45"]` |
| `talk_moves` | `List[str]` | 本轮次触发的官方教学动作预测标签列表 | `["<Press for Accuracy>"]` |
| `is_greeting_or_noise` | `bool` | 寒暄/问候/语气噪音打标 (`True` 表示非实质问答) | `False` |

---

## 三、 清洗与质量门禁标准 (Sanitization & Quality Protocol)

为确保大模型知识抽取与向量检索不被脏数据污染，ETL 过程严格执行以下四大内容标准：

```
+---------------------------------------------------------------------------------------------------+
| ETL 清洗四重保障门禁:                                                                             |
| 1. LaTeX 保护区隔离 (Protected Span Masking): 避免公式符号在标点/分词清洗中损毁                   |
| 2. 状态机碎句合并 (Turn Compression): 将连续多次回车发送的碎片消息归并为连贯逻辑轮次             |
| 3. PII 个人隐私脱敏 (De-identification): 正则消除学生真实姓名 (替换为 [STUDENT_NAME])            |
| 4. 实质辅导质量门禁 (Quality Gate): 师生发言轮次均 >= 1 且总轮次 >= 2                            |
+---------------------------------------------------------------------------------------------------+
```

1. **LaTeX 数学公式绝对保护**：
   * 自动识别 `\( ... \)`、`\[ ... \]`、`$ ... $`、`$$ ... $$` 格式的数学公式；
   * 在执行去多余空格、脱敏与规范化前将其提取至不可变缓存区，清洗完成后原样无损回填，确保公式语法（如 `\frac{a}{b}`, `\sqrt{x}`, `5.4598`）100% 完好。
2. **状态机碎句合并机制**：
   * 真实打字场景下，学生常一句话按 3 次回车发送。状态机在检测到连续相同角色发言时，将文本拼接合并为单轮逻辑 `DialogueTurn`，显著减少上下文窗口冗余。
3. **个人隐私严格脱敏**：
   * 对自我介绍（如 `my name is Lina`）、导师问候（如 `Hi Sophie`）进行正则捕获，统一脱敏为 `[STUDENT_NAME]`，杜绝隐私泄露。
4. **寒暄与噪音识别**：
   * 对开场打招呼（`hello`, `hi`, `how are you`）、交际客套（`thanks`, `bye`）标记 `is_greeting_or_noise = True`，供下游知识抽取引擎选择性跳过。

---

## 四、 核心教学场景全景白盒剖析 (Representative Human-Readable Cases)

以下精选 5 个具有代表性学科与题型的完整融合会话，供教研员人工抽检查验：

### 案例 1：Session #10 | 四舍五入数位截断与精度混淆 (Number / Decimals)
```json
{
  "intervention_id": 10,
  "question_id": 104614,
  "tutor_id": 1002,
  "subjects": {
    "paths": ["Mathematics > Number > Rounding and Estimating > Rounding to Decimal Places"]
  },
  "question": {
    "question_id": 104614,
    "question_text": "What is \\( 5.4598 \\) rounded to 1 decimal place?",
    "options": {"A": "5.4", "B": "5.45", "C": "5.46", "D": "5.5"}
  },
  "total_turns": 26,
  "student_turn_count": 13,
  "tutor_turn_count": 13,
  "turns": [
    {"turn_id": 1, "speaker": "tutor", "text": "Hello [STUDENT_NAME]! Let's do this question together.", "is_greeting_or_noise": true},
    {"turn_id": 2, "speaker": "student", "text": "I think it is 5.45", "is_greeting_or_noise": false},
    {"turn_id": 3, "speaker": "tutor", "text": "Can you round 5.45 to one decimal place?", "talk_moves": ["<Press for Accuracy>", "<Keep Together>"]},
    {"turn_id": 4, "speaker": "student", "text": "5.4?", "is_greeting_or_noise": false},
    {"turn_id": 5, "speaker": "tutor", "text": "Look at the second number, it is 5. What happens when it's 5 or more?", "talk_moves": ["<Press for Reasoning>"]},
    {"turn_id": 6, "speaker": "student", "text": "We round up! So 5.5!", "is_greeting_or_noise": false}
  ]
}
```
* **教研洞察**：学生一开始直觉性地保留了两位小数（5.45），导师没有直接给答案，而是将数值简化为 `5.45` 追问一位小数的舍入规则，学生成功由 5.4 纠偏为进位后的 5.5。

---

### 案例 2：Session #14 | 分式复合运算中的运算优先级 (Number / BIDMAS)
* **考题**: `(54 + 58) / 2` | **选项**: A: 56, B: 83, C: 112, D: 28
* **学生错因**: 误选 B (54 + 29 = 83)，忽视了分子整体求和的括号作用域，先执行了除法运算；
* **名师破局**: 导师通过反问 `what do you think this question is asking?` 引导学生识别分子求和优先。

---

### 案例 3：Session #23 | 最小公倍数与两数乘积特例泛化 (Number / LCM)
* **考题**: "两数的最小公倍数是否总是等于两数乘积？" (选项: A: Always, B: Sometimes, C: Never)
* **学生错因**: 误选 C，将 3 和 5（互质数）的特例经验错误泛化；
* **名师破局**: 导师提问 `Is there an example where it doesn't work?`，引入反例（如 4 和 6 的 LCM 是 12 而非 24）促使学生自悟。

---

### 案例 4：Session #29 | 频数统计表中求极差 (Statistics / Frequency Table)
* **考题**: 根据电视机数量与家庭频数统计表求极差 (Range)
* **学生错因**: 误选 B (12)，直接拿人数频数最大值 `12 - 0 = 12` 充当极差，混淆了“频数”与“实际数据值”；
* **名师破局**: 导师结合生活情境启发 `can you see those 7 people who have 1 TV?` 引导学生聚焦实际拥有的电视机数量。

---

### 案例 5：Session #42 | 负数乘方中负号归属与括号规则 (Algebra / Indices)
* **考题**: 判断 \( p \times (-q) = (-p) \times q \) 与 \( (-q)^2 = -q^2 \) 的正误
* **学生错因**: 误选 C，认为只要带负号的数字平方后结果都带负号；
* **名师破局**: 导师追问 `Does -p x q give us the same answer?` 并引导对比展开式 `(-q)*(-q) = q^2`。

---

## 五、 人类教研员查验与审计操作指引 (Human Audit SOP)

教研员与工程师可通过以下方法快速对全量融合数据进行检索与抽检：

### 1. 命令行快速查询与统计
```powershell
# 1. 查看全量融合文件总行数 (应为 1576 行)
(Get-Content data/cleaned_sessions.jsonl).Length

# 2. 检索包含特定考点的会话 (例如搜索四舍五入)
Select-String -Path "data/cleaned_sessions.jsonl" -Pattern "Rounding to Decimal Places" | Measure-Object

# 3. 提取第 1 场会话的美化 JSON 视图查看
python -c "import json; print(json.dumps(json.loads(open('data/cleaned_sessions.jsonl', encoding='utf-8').readline()), ensure_ascii=False, indent=2))"
```

### 2. Python 交互式抽检
```python
import json

def inspect_session(intervention_id: int):
    with open("data/cleaned_sessions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["intervention_id"] == intervention_id:
                print(f"=== Session #{data['intervention_id']} (QID: {data['question_id']}) ===")
                print(f"考纲: {data['subjects']['paths']}")
                print(f"题干: {data['question']['question_text']}")
                print(f"选项: {data['question']['options']}")
                print(f"对话总轮数: {data['total_turns']}")
                for t in data["turns"][:5]:  # 预览前5轮
                    print(f"  [Turn {t['turn_id']} - {t['speaker']}]: {t['text']} | TalkMoves: {t['talk_moves']}")
                return
    print("未找到指定会话")

inspect_session(10)
```
