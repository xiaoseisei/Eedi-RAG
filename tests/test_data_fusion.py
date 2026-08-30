"""
================================================================================
测试套件: tests/test_data_fusion.py
定位与规范: 针对 Step 1 三表关联融合 ETL 模块的 TDD 单元测试套件
测试目标:
  1. 验证单分支与多分支学科树 (SubjectHierarchy) 递归路径拼装的正确性；
  2. 验证题目多 Sequence 行拼接与选项 A/B/C/D 透视 (Question Pivot)；
  3. 验证文本清洗过程中 LaTeX 数学公式 \\( ... \\) 零损伤保护与 PII 姓名脱敏；
  4. 验证状态机说话人碎句压缩 (Turn Compression) 与 TalkMoves 标签对齐；
  5. 验证端到端三表融合输出 JSONL 格式的完整性。
================================================================================
"""

import os
import json
import pytest
import pandas as pd
from src.models import SubjectHierarchy, Question, DialogueTurn, CleanedSession
from src.data_fusion import (
    build_subject_hierarchy,
    pivot_question_metadata,
    compress_and_clean_dialogue_turns,
    deidentify_and_clean_text,
    fuse_all_sessions,
)


def test_build_subject_hierarchy_single_branch():
    """
    【测试用例 1：单分支学科考纲路径拼装】
    
    测试意图:
      验证标准 3 级学科节点 (Level 1 Subject -> Level 2 Topic -> Level 3 Subtopic)
      能否正确解析为以 ' > ' 分隔的标准全路径字符串。
    """
    # 模拟单个 3 级知识树切片
    df_subj = pd.DataFrame([
        {"SubjectId": 32, "ParentSubjectId": 3, "SubjectName": "Number", "SubjectLevel": 1, "SubjectType": "Subject"},
        {"SubjectId": 141, "ParentSubjectId": 32, "SubjectName": "Rounding and Estimating", "SubjectLevel": 2, "Topic": "Topic"},
        {"SubjectId": 214, "ParentSubjectId": 141, "SubjectName": "Rounding to Decimal Places", "SubjectLevel": 3, "Subtopic": "Subtopic"},
    ])
    hierarchy = build_subject_hierarchy(df_subj)
    
    assert isinstance(hierarchy, SubjectHierarchy)
    assert len(hierarchy.paths) == 1
    assert hierarchy.paths[0] == "Number > Rounding and Estimating > Rounding to Decimal Places"
    assert "Number" in hierarchy.subjects
    assert "Rounding and Estimating" in hierarchy.topics
    assert "Rounding to Decimal Places" in hierarchy.subtopics


def test_build_subject_hierarchy_multi_branch():
    """
    【测试用例 2：多分支复杂学科知识树拼装】
    
    测试意图:
      一个题目可能同时涉及多个一级学科或多个二级主题（例如既涉及算术又涉及几何）。
      验证函数能正确识别并拆解出多条平行的完整学科路径。
    """
    df_subj = pd.DataFrame([
        {"SubjectId": 32, "ParentSubjectId": 3, "SubjectName": "Number", "SubjectLevel": 1, "SubjectType": "Subject"},
        {"SubjectId": 71, "ParentSubjectId": 3, "SubjectName": "Geometry and Measure", "SubjectLevel": 1, "SubjectType": "Subject"},
        {"SubjectId": 144, "ParentSubjectId": 32, "SubjectName": "Basic Arithmetic", "SubjectLevel": 2, "SubjectType": "Topic"},
        {"SubjectId": 272, "ParentSubjectId": 71, "SubjectName": "Similarity and Congruency", "SubjectLevel": 2, "SubjectType": "Topic"},
        {"SubjectId": 204, "ParentSubjectId": 144, "SubjectName": "Mental Multiplication and Division", "SubjectLevel": 3, "SubjectType": "Subtopic"},
        {"SubjectId": 90, "ParentSubjectId": 272, "SubjectName": "Length Scale Factors", "SubjectLevel": 3, "SubjectType": "Subtopic"},
    ])
    hierarchy = build_subject_hierarchy(df_subj)
    assert len(hierarchy.paths) == 2
    assert "Number > Basic Arithmetic > Mental Multiplication and Division" in hierarchy.paths
    assert "Geometry and Measure > Similarity and Congruency > Length Scale Factors" in hierarchy.paths


def test_pivot_question_metadata_multi_sequence():
    """
    【测试用例 3：多 Sequence 题干与选项透视 (Question Pivot)】
    
    测试意图:
      在真实数据中，复杂题干通常由 2~3 个 Sequence 片段组成。
      验证透视引擎能够按时序拼接题干文本，并正确将 Answer A/B/C/D 填充至 options 字典。
    """
    df_meta = pd.DataFrame([
        {"QuestionId_DQ": 101, "Sequence": 1, "Label": "Question Text", "Text": "Line 1 of question."},
        {"QuestionId_DQ": 101, "Sequence": 2, "Label": "Question Text", "Text": "Line 2 of question."},
        {"QuestionId_DQ": 101, "Sequence": 3, "Label": "Answer A Text", "Text": "Option A value"},
        {"QuestionId_DQ": 101, "Sequence": 4, "Label": "Answer B Text", "Text": "Option B value"},
        {"QuestionId_DQ": 101, "Sequence": 5, "Label": "Answer C Text", "Text": "Option C value"},
        {"QuestionId_DQ": 101, "Sequence": 6, "Label": "Answer D Text", "Text": "Option D value"},
    ])
    question = pivot_question_metadata(101, df_meta)
    
    assert isinstance(question, Question)
    assert question.question_id == 101
    assert "Line 1 of question.\nLine 2 of question." in question.question_text
    assert question.options.get("A") == "Option A value"
    assert question.options.get("B") == "Option B value"
    assert question.options.get("C") == "Option C value"
    assert question.options.get("D") == "Option D value"


def test_deidentify_and_clean_text_preserves_latex():
    r"""
    【测试用例 4：LaTeX 公式保护与 PII 隐私脱敏验证】
    
    测试意图:
      1. 验证数学公式 \( \mathbf{5 . 4 5 9 8} \) 中的特殊符号不被清洗破坏；
      2. 验证学生姓名 (Lina Chen) 被替换为 [STUDENT_NAME]；
      3. 验证联系电话被清洗替换。
    """
    raw = r"Hello Lina Chen, what is \( \mathbf{5 . 4 5 9 8} \) rounded to 1 decimal place? Call me at 13800000000."
    cleaned, is_greeting = deidentify_and_clean_text(raw, is_tutor=True)
    
    # 核心数学公式必须保持 100% 原始字面零损伤
    assert r"\( \mathbf{5 . 4 5 9 8} \)" in cleaned
    # 真实姓名与电话必须被精准脱敏
    assert "Lina Chen" not in cleaned
    assert "[STUDENT_NAME]" in cleaned
    assert "13800000000" not in cleaned


def test_compress_and_clean_dialogue_turns():
    """
    【测试用例 5：状态机说话人碎句压缩 (Turn Compression)】
    
    测试意图:
      1. 将同一说话人连续发出的 2~3 条碎句压缩为 1 个对话轮次；
      2. 验证轮次序号 (Turn ID) 从 1 开始按顺序严格递增；
      3. 验证各轮次的 TalkMoves 标签被正确归集至对应说话人轮次。
    """
    df_dialogue = pd.DataFrame([
        {"MessageSequence": 1, "IsTutor": 1, "MessageString": "Hello!", "TalkMovePrediction": "<None>"},
        {"MessageSequence": 2, "IsTutor": 1, "MessageString": "You OK?", "TalkMovePrediction": "<None>"},
        {"MessageSequence": 3, "IsTutor": 0, "MessageString": "Hi", "TalkMovePrediction": None},
        {"MessageSequence": 4, "IsTutor": 0, "MessageString": "I am stuck on this question", "TalkMovePrediction": None},
        {"MessageSequence": 5, "IsTutor": 0, "MessageString": "Can you help?", "TalkMovePrediction": None},
        {"MessageSequence": 6, "IsTutor": 1, "MessageString": "Sure! What is step 1?", "TalkMovePrediction": "<Press for Accuracy>"},
    ])
    turns = compress_and_clean_dialogue_turns(df_dialogue)
    
    # 6 条原始碎句应被合并为 3 轮连贯对话 (Tutor -> Student -> Tutor)
    assert len(turns) == 3
    
    # Turn 1: 导师开场寒暄
    assert turns[0].turn_id == 1
    assert turns[0].speaker == "tutor"
    assert len(turns[0].raw_messages) == 2
    assert "Hello!" in turns[0].text and "You OK?" in turns[0].text
    assert turns[0].is_greeting_or_noise is True

    # Turn 2: 学生求助 (3 句合并)
    assert turns[1].turn_id == 2
    assert turns[1].speaker == "student"
    assert len(turns[1].raw_messages) == 3
    assert "stuck on this question" in turns[1].text
    assert turns[1].talk_moves == []

    # Turn 3: 导师启发提问
    assert turns[2].turn_id == 3
    assert turns[2].speaker == "tutor"
    assert "<Press for Accuracy>" in turns[2].talk_moves
    assert "What is step 1?" in turns[2].text


def test_fuse_all_sessions_sample():
    """
    【测试用例 6：端到端三表融合与 JSONL 输出验证】
    
    测试意图:
      在抽样测试集文件上完整跑通 ETL 流程，验证产出的数据结构、行数以及 JSONL 文件序列化。
    """
    sample_subj_path = "data/sample/dialogue-subjects-sample.csv"
    sample_meta_path = "data/sample/dq-question-metadata-sample.csv"
    sample_train_path = "data/sample/train-sample.csv"
    output_jsonl = "data/sample/cleaned_sessions_sample.jsonl"

    sessions = fuse_all_sessions(
        subjects_path=sample_subj_path,
        metadata_path=sample_meta_path,
        dialogues_path=sample_train_path,
        output_path=output_jsonl
    )

    # 1. 验证返回的 Python 对象
    assert len(sessions) == 10
    assert os.path.exists(output_jsonl)

    # 2. 深入校验第 10 号会话
    s10 = next(s for s in sessions if s.intervention_id == 10)
    assert s10.question_id == 104614
    assert len(s10.subjects.paths) >= 1
    assert "Rounding" in s10.subjects.paths[0]
    assert len(s10.turns) > 0
    assert s10.has_valid_tutoring is True

    # 3. 验证持久化 JSON Lines 文件的合规性
    with open(output_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 10
    assert lines[0]["intervention_id"] == 10
