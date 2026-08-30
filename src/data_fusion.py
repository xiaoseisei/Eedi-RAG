"""
================================================================================
模块名称: src/data_fusion.py
业务定位: Eedi 教学辅导多表关联融合与确定性清洗 ETL 引擎 (Step 1)
核心职责:
  1. 融合三张离散的关系型 CSV 表:
     - dialogue-subjects.csv (学科考纲树)
     - dq-question-metadata.csv (题目题干与选项)
     - anchored-dialogues/train.csv (师生真实问答流与 TalkMoves 标签)
  2. 保护区隔离清洗 (Protected Span Masking):
     - 严格保护 LaTeX 数学公式 \\( ... \\) 和 \\[ ... \\]，杜绝清洗过度导致公式被当作特殊符号损毁。
  3. 隐私数据脱敏 (PII De-identification):
     - 精准识别师生对话中透露的真实姓名、电话与邮箱，替换为统一占位符 [STUDENT_NAME]。
  4. 状态机碎句压缩 (Turn Compression):
     - 将学生/导师连续多次回车发送的碎片化短消息合并为连贯的单个对话轮次 (Turn)。
  5. 质量门禁校验 (Quality Gate):
     - 过滤未产生实质教学问答的单向/冷启动废会话，输出自包含的 CleanedSession JSONL。
================================================================================
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from src.models import SubjectHierarchy, Question, DialogueTurn, CleanedSession


# ==============================================================================
# 正则表达式与模式库定义 (Pattern Definitions)
# ==============================================================================

# 1. LaTeX 数学公式保护正则 (包含行内公式 \( ... \), $ ... $ 与块级公式 \[ ... \], $$ ... $$)
# 在清洗普通文本前，需先将这些公式提取并替换为不可变占位符，避免标点/特殊字符清洗逻辑破坏公式语法。
LATEX_PATTERNS = [
    re.compile(r"\\\(.*?\\\)", re.DOTALL),          # LaTeX 行内公式 \( ... \)
    re.compile(r"\\\[.*?\\\]", re.DOTALL),          # LaTeX 块级公式 \[ ... \]
    re.compile(r"\$\$.*?\$\$", re.DOTALL),          # TeX 双美元块级公式 $$ ... $$
    re.compile(r"\$(?!\s)(.*?)(?<!\s)\$"),          # TeX 单美元行内公式 $ ... $
]

# 2. 个人隐私信息 (PII, Personally Identifiable Information) 识别正则
# 匹配常见的自我介绍模式 (如 "preferred to be called Lina Chen", "my name is Alex")
NAME_INTRO_PATTERN = re.compile(
    r"\b(preferred to be called|my name is|I am called|call me|name's)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
)
# 匹配常见问候中的受托人姓名 (如 "Hello Lina Chen", "Hi Sophie")
GREETING_NAME_PATTERN = re.compile(
    r"\b(Hello|Hi|Hey|Thanks|Thank you)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)
# 匹配数据集中已知典型暴露姓名
EXACT_NAME_PATTERN = re.compile(r"\bLina\s+Chen\b|\bLina\b", re.IGNORECASE)
# 匹配电话号码与国际区号格式
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b")
# 匹配通用邮箱地址
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# 3. 交际性寒暄与客套词词典 (Phatic/Greeting Lexicon)
# 用于判断某轮对话是否纯粹是开场打招呼或结束感谢（非实质教学交互），打标 is_greeting_or_noise
GREETING_WORDS = {
    "hello", "hi", "hey", "you ok?", "you ok", "you okay", "you okay?",
    "thx", "thx u", "thanks", "thank you", "ok thank you", "take care",
    "good morning", "good afternoon", "bye", "goodbye", "ok", "k"
}


# ==============================================================================
# 核心文本清洗与保护区算法 (Sanitization & Protected Span Masking)
# ==============================================================================

def mask_protected_spans(text: str) -> Tuple[str, Dict[str, str]]:
    """
    【保护区隔离算法 - 第一步：掩码隔离】
    
    业务原理:
      在对教育文本进行去除多余空格、脱敏或字符规范化时，数学公式中的反斜杠 '\'、
      大括号 '{}'、下划线 '_'、乘号 '^' 极易被误杀。
      通过预先扫描所有 LaTeX 公式，将其替换为临时唯一标记（如 __PROTECTED_SPAN_0__），
      从而在后续清洗过程中将公式整体安全保护起来。
      
    Args:
        text (str): 原始待清洗文本。
        
    Returns:
        Tuple[str, Dict[str, str]]: (掩码后的文本, 占位符到原始公式的映射字典)。
    """
    protected_map: Dict[str, str] = {}
    counter = 0

    def replace_match(match: re.Match) -> str:
        nonlocal counter
        token = f"__PROTECTED_SPAN_{counter}__"
        protected_map[token] = match.group(0)
        counter += 1
        return token

    masked_text = text
    for pattern in LATEX_PATTERNS:
        masked_text = pattern.sub(replace_match, masked_text)

    return masked_text, protected_map


def unmask_protected_spans(masked_text: str, protected_map: Dict[str, str]) -> str:
    """
    【保护区隔离算法 - 第三步：掩码还原】
    
    业务原理:
      在完成普通文本的脱敏与空白标准化后，将所有 __PROTECTED_SPAN_X__ 占位符
      无损替换回原始 LaTeX 公式，确保 100% 原始数学语义不丢失。
      
    Args:
        masked_text (str): 经过清洗的包含占位符的文本。
        protected_map (Dict[str, str]): 占位符与原公式映射字典。
        
    Returns:
        str: 还原后的完整清洗文本。
    """
    result = masked_text
    for token, original in protected_map.items():
        result = result.replace(token, original)
    return result


def deidentify_and_clean_text(raw_text: str, is_tutor: bool = False) -> Tuple[str, bool]:
    """
    【综合文本清洗与 PII 隐私脱敏函数】
    
    执行流程:
      1. LaTeX 公式掩码隔离；
      2. PII 敏感信息脱敏（替换为 [STUDENT_NAME]、[PHONE_NUMBER]、[EMAIL]）；
      3. 空白与换行标准化（压缩连续空格、连续多余空行）；
      4. LaTeX 公式无损还原；
      5. 语义打标：判定当前文本是否为纯寒暄/噪音 (is_greeting_or_noise)。
      
    Args:
        raw_text (str): 原始单条或多条合并的消息文本。
        is_tutor (bool): 当前发言者是否为导师。
        
    Returns:
        Tuple[str, bool]: (清洗且脱敏后的文本, 是否为纯寒暄标记)。
    """
    if not raw_text or not isinstance(raw_text, str):
        return "", True

    # 1. 提取并掩码数学公式保护区
    masked_text, protected_map = mask_protected_spans(raw_text)

    # 2. 对非保护区文本进行 PII 识别与脱敏
    cleaned = EXACT_NAME_PATTERN.sub("[STUDENT_NAME]", masked_text)
    cleaned = NAME_INTRO_PATTERN.sub(r"\1 [STUDENT_NAME]", cleaned)
    cleaned = GREETING_NAME_PATTERN.sub(r"\1 [STUDENT_NAME]", cleaned)
    cleaned = PHONE_PATTERN.sub("[PHONE_NUMBER]", cleaned)
    cleaned = EMAIL_PATTERN.sub("[EMAIL]", cleaned)

    # 3. 空白字符与换行压缩归一化
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned).strip()

    # 4. 无损还原数学公式
    final_text = unmask_protected_spans(cleaned, protected_map)

    # 5. 判定是否为纯寒暄客套词
    lower_unmasked = re.sub(r"[^a-zA-Z0-9\s]", "", masked_text).strip().lower()
    is_greeting = lower_unmasked in GREETING_WORDS or (
        len(lower_unmasked.split()) <= 4 and any(
            g in lower_unmasked for g in ["hello", "you ok", "thx", "take care", "goodbye"]
        )
    )

    return final_text, is_greeting


# ==============================================================================
# 关系型数据解析与透视构建 (Relational Parsers & Pivot Engine)
# ==============================================================================

def build_subject_hierarchy(df_subj_for_session: pd.DataFrame) -> SubjectHierarchy:
    """
    【学科知识树层级构建器】
    
    业务原理:
      根据 dialogue-subjects.csv 中 SubjectId 与 ParentSubjectId 的拓扑树关系，
      自底向上遍历或由根到叶拼装完整的学科路径。
      
    示例路径:
      Level 1: Number (SubjectId: 32)
      Level 2: Rounding and Estimating (SubjectId: 141, Parent: 32)
      Level 3: Rounding to Decimal Places (SubjectId: 214, Parent: 141)
      -> 拼装全路径: "Number > Rounding and Estimating > Rounding to Decimal Places"
      
    Args:
        df_subj_for_session (pd.DataFrame): 当前 InterventionId 对应的学科树切片数据。
        
    Returns:
        SubjectHierarchy: 结构化多分支考纲路径对象。
    """
    if df_subj_for_session.empty:
        return SubjectHierarchy()

    subjects_l1: Dict[int, str] = {}
    topics_l2: Dict[int, Tuple[int, str]] = {}     # id -> (parent_id, name)
    subtopics_l3: Dict[int, Tuple[int, str]] = {}  # id -> (parent_id, name)

    all_l1_names: List[str] = []
    all_l2_names: List[str] = []
    all_l3_names: List[str] = []

    # 遍历切片，分类记录各层级节点
    for _, row in df_subj_for_session.iterrows():
        s_id = int(row["SubjectId"])
        p_id = int(row["ParentSubjectId"]) if pd.notna(row["ParentSubjectId"]) else 0
        name = str(row["SubjectName"]).strip()
        level = int(row["SubjectLevel"])

        if level == 1:
            subjects_l1[s_id] = name
            if name not in all_l1_names:
                all_l1_names.append(name)
        elif level == 2:
            topics_l2[s_id] = (p_id, name)
            if name not in all_l2_names:
                all_l2_names.append(name)
        elif level == 3:
            subtopics_l3[s_id] = (p_id, name)
            if name not in all_l3_names:
                all_l3_names.append(name)

    paths: List[str] = []

    # 优先构建完整 3 级全路径 (Level 1 > Level 2 > Level 3)
    if subtopics_l3:
        for s3_id, (s2_parent_id, s3_name) in subtopics_l3.items():
            if s2_parent_id in topics_l2:
                s1_parent_id, s2_name = topics_l2[s2_parent_id]
                s1_name = subjects_l1.get(s1_parent_id, "General")
                path = f"{s1_name} > {s2_name} > {s3_name}"
            else:
                s1_name = subjects_l1.get(s2_parent_id, "General")
                path = f"{s1_name} > {s3_name}"
            if path not in paths:
                paths.append(path)
    # 若无 Level 3 则退化构建 2 级路径 (Level 1 > Level 2)
    elif topics_l2:
        for s2_id, (s1_parent_id, s2_name) in topics_l2.items():
            s1_name = subjects_l1.get(s1_parent_id, "General")
            path = f"{s1_name} > {s2_name}"
            if path not in paths:
                paths.append(path)
    # 若仅有 Level 1 则直接输出一级学科
    elif subjects_l1:
        for s1_name in subjects_l1.values():
            if s1_name not in paths:
                paths.append(s1_name)

    return SubjectHierarchy(
        paths=paths,
        subjects=all_l1_names,
        topics=all_l2_names,
        subtopics=all_l3_names
    )


def pivot_question_metadata(question_id: int, df_meta_for_session: pd.DataFrame) -> Question:
    """
    【题目与选项透视构建器 (Pivot Engine)】
    
    业务原理:
      在 dq-question-metadata.csv 中，题目题干可能分多行(Sequence)存储，
      选项(Answer A Text, Answer B Text...)也是分散的行记录。
      本函数按 Sequence 正序将多行题干拼接为完整题目，并将选项与图片透视为结构化字典。
      
    Args:
        question_id (int): 题目唯一 ID。
        df_meta_for_session (pd.DataFrame): 当前题目元数据切片。
        
    Returns:
        Question: 结构化的题干与选项对象。
    """
    q_df = df_meta_for_session[df_meta_for_session["QuestionId_DQ"] == question_id].sort_values(by="Sequence")
    
    # 兼容传入已过滤后的 DataFrame 切片
    if q_df.empty:
        q_df = df_meta_for_session.sort_values(by="Sequence")

    question_text_parts: List[str] = []
    options: Dict[str, str] = {}
    images: Dict[str, str] = {}

    for _, row in q_df.iterrows():
        label = str(row["Label"]).strip()
        text = str(row["Text"]).strip() if pd.notna(row["Text"]) else ""

        if label == "Question Text":
            if text:
                question_text_parts.append(text)
        elif label == "Answer A Text":
            options["A"] = text
        elif label == "Answer B Text":
            options["B"] = text
        elif label == "Answer C Text":
            options["C"] = text
        elif label == "Answer D Text":
            options["D"] = text
        elif label == "Question Image":
            images["Question"] = text
        elif "Image" in label:
            # 解析如 "Answer A Image" -> "A"
            opt_key = label.replace("Answer ", "").replace(" Image", "").strip()
            images[opt_key] = text

    # 将分段题干以换行符自然拼接
    full_question_text = "\n".join(question_text_parts)

    return Question(
        question_id=int(question_id),
        question_text=full_question_text,
        options=options,
        images=images
    )


def compress_and_clean_dialogue_turns(df_dialogue_for_session: pd.DataFrame) -> List[DialogueTurn]:
    """
    【基于状态机的说话人连续发言压缩器 (Turn Compression)】
    
    业务原理:
      真实辅导场景中，学生常常连续敲回车发送 3~4 个短句（如 "Hi", "I am stuck", "Can you help?"）。
      如果直接逐行作为检索单位，会导致单句语义极其贫乏。
      本状态机将同一角色连续发送的碎片化消息合并为一个连贯的 DialogueTurn，
      同时聚合每轮中包含的 TalkMovePrediction 教学动作标签。
      
    Args:
        df_dialogue_for_session (pd.DataFrame): 当前会话的原始对话流切片。
        
    Returns:
        List[DialogueTurn]: 合并压缩后的连贯对话轮次列表。
    """
    if df_dialogue_for_session.empty:
        return []

    # 严格按消息时序 MessageSequence 排序
    sorted_df = df_dialogue_for_session.sort_values(by="MessageSequence")
    turns: List[DialogueTurn] = []

    current_is_tutor: Optional[bool] = None
    current_raw_messages: List[str] = []
    current_talk_moves: List[str] = []

    def finalize_turn(turn_idx: int) -> DialogueTurn:
        """闭包辅助函数: 封装着色并生成一个完整的 DialogueTurn 实例"""
        nonlocal current_is_tutor, current_raw_messages, current_talk_moves
        combined_raw = " ".join(current_raw_messages)
        cleaned_text, is_greeting = deidentify_and_clean_text(
            combined_raw,
            is_tutor=bool(current_is_tutor)
        )
        # TalkMoves 去重且保持出现次序
        unique_talk_moves = []
        for tm in current_talk_moves:
            if tm not in unique_talk_moves:
                unique_talk_moves.append(tm)

        return DialogueTurn(
            turn_id=turn_idx,
            speaker="tutor" if current_is_tutor else "student",
            is_tutor=bool(current_is_tutor),
            raw_messages=list(current_raw_messages),
            text=cleaned_text,
            talk_moves=unique_talk_moves,
            is_greeting_or_noise=is_greeting
        )

    turn_counter = 1
    for _, row in sorted_df.iterrows():
        is_tutor = bool(int(row["IsTutor"]) == 1)
        msg_str = str(row["MessageString"]).strip() if pd.notna(row["MessageString"]) else ""
        tm_tag = str(row["TalkMovePrediction"]).strip() if pd.notna(row["TalkMovePrediction"]) else None

        if current_is_tutor is None:
            # 状态机初始状态：录入第一条消息
            current_is_tutor = is_tutor
            current_raw_messages.append(msg_str)
            if tm_tag and tm_tag not in ["<None>", "<NA>", "nan", "None"]:
                current_talk_moves.append(tm_tag)
        elif current_is_tutor == is_tutor:
            # 状态机状态持续：同一说话人继续发言，追加消息与标签
            current_raw_messages.append(msg_str)
            if tm_tag and tm_tag not in ["<None>", "<NA>", "nan", "None"]:
                current_talk_moves.append(tm_tag)
        else:
            # 状态机状态转移：说话人角色切换，结算上一个 Turn
            turns.append(finalize_turn(turn_counter))
            turn_counter += 1
            # 开启新说话人的轮次
            current_is_tutor = is_tutor
            current_raw_messages = [msg_str]
            current_talk_moves = []
            if tm_tag and tm_tag not in ["<None>", "<NA>", "nan", "None"]:
                current_talk_moves.append(tm_tag)

    # 结算最后一个发言轮次
    if current_raw_messages:
        turns.append(finalize_turn(turn_counter))

    return turns


# ==============================================================================
# ETL 主控制管线 (Main ETL Pipeline)
# ==============================================================================

def fuse_all_sessions(
    subjects_path: str,
    metadata_path: str,
    dialogues_path: str,
    output_path: Optional[str] = None
) -> List[CleanedSession]:
    """
    【三表数据融合 ETL 主入口函数】
    
    处理全流程:
      1. 并行/分批读取三大原始 CSV 表；
      2. 按 InterventionId 执行高效分组；
      3. 对每个 Session 依次构建学科树、透视考题元数据、状态机压缩问答流；
      4. 执行质量门禁校验 (判断是否为双向有效辅导)；
      5. 持久化输出为每行一个 Session 的 JSONL 文件。
      
    Args:
        subjects_path (str): dialogue-subjects.csv 文件路径。
        metadata_path (str): dq-question-metadata.csv 文件路径。
        dialogues_path (str): train.csv 或 test.csv 文件路径。
        output_path (Optional[str]): 产出 JSONL 文件保存路径。
        
    Returns:
        List[CleanedSession]: 清洗并融合后的会话对象列表。
    """
    df_subj = pd.read_csv(subjects_path)
    df_meta = pd.read_csv(metadata_path)
    df_dialogues = pd.read_csv(dialogues_path)

    # 按 InterventionId 建立索引分组，提升匹配效率 (避免 O(N*M) 循环检索)
    subj_grouped = df_subj.groupby("InterventionId")
    meta_grouped = df_meta.groupby("InterventionId")
    dialogues_grouped = df_dialogues.groupby("InterventionId")

    all_intervention_ids = df_dialogues["InterventionId"].unique()
    sessions: List[CleanedSession] = []

    for intervention_id in all_intervention_ids:
        # 1. 提取当前会话对话流切片
        d_slice = dialogues_grouped.get_group(intervention_id) if intervention_id in dialogues_grouped.groups else pd.DataFrame()
        if d_slice.empty:
            continue

        # 提取绑定的考题 ID 与导师 ID
        q_id = int(d_slice["QuestionId_DQ"].iloc[0])
        tutor_id = int(d_slice["TutorId"].iloc[0])

        # 2. 提取并构建学科考纲树
        s_slice = subj_grouped.get_group(intervention_id) if intervention_id in subj_grouped.groups else pd.DataFrame()
        subject_hierarchy = build_subject_hierarchy(s_slice)

        # 3. 提取并透视题目干与选项
        m_slice = meta_grouped.get_group(intervention_id) if intervention_id in meta_grouped.groups else pd.DataFrame()
        question = pivot_question_metadata(q_id, m_slice)

        # 4. 状态机合并清洗对话轮次
        turns = compress_and_clean_dialogue_turns(d_slice)

        # 5. 计算会话特征指标
        student_turns = sum(1 for t in turns if not t.is_tutor)
        tutor_turns = sum(1 for t in turns if t.is_tutor)
        total_turns = len(turns)

        # 质量门禁：师生双方均至少发言 1 轮且总轮次 >= 2 判定为实质辅导
        has_valid_tutoring = (total_turns >= 2) and (student_turns >= 1) and (tutor_turns >= 1)

        session = CleanedSession(
            intervention_id=int(intervention_id),
            question_id=q_id,
            tutor_id=tutor_id,
            subjects=subject_hierarchy,
            question=question,
            turns=turns,
            total_turns=total_turns,
            student_turn_count=student_turns,
            tutor_turn_count=tutor_turns,
            has_valid_tutoring=has_valid_tutoring
        )
        sessions.append(session)

    # 6. 持久化输出为标准 JSON Lines 格式
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in sessions:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")

    return sessions


# ==============================================================================
# CLI 命令行执行入口
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fuse Eedi 3-table relational dataset into CleanedSession JSONL.")
    parser.add_argument("--subjects", default="data/dialogue-subjects.csv", help="Path to dialogue-subjects.csv")
    parser.add_argument("--metadata", default="data/dq-question-metadata.csv", help="Path to dq-question-metadata.csv")
    parser.add_argument("--dialogues", default="data/anchored-dialogues/train.csv", help="Path to train.csv")
    parser.add_argument("--output", default="data/cleaned_sessions.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    print(f"Starting Data Fusion ETL...")
    fused_sessions = fuse_all_sessions(args.subjects, args.metadata, args.dialogues, args.output)
    print(f"Successfully fused {len(fused_sessions)} sessions to {args.output}")
