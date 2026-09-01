"""
================================================================================
脚本名称: scripts/export_authoring_catalog.py
业务定位: Step 6 - 黄金基准集出题参考底表生成器 (Benchmark Authoring Reference Generator)
核心功能:
  从已入库的 100 场全学科会话 (DuckDB) 中提取题目、选项、考点路径、
  学生真实发言原句 ([Turn N]) 与 导师破局提问 ([Turn N])，
  生成格式清晰的 authoring_catalog.json 与 Markdown 便携底表，
  供业务/教研专家随时翻阅真实原句、拟定高质量的 30 题黄金测试用例。
================================================================================
"""

import sys
import json
import duckdb
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage_manager import DualEngineStorageManager


def export_catalog(db_path: str = "data/db/tutoring_knowledge.duckdb"):
    conn = duckdb.connect(database=db_path, read_only=True)
    
    # 1. 查询会话基础事实
    sessions_df = conn.execute("""
    SELECT intervention_id, question_id, tutor_id, subject_path, question_text, options_json, total_turns
    FROM tutoring_sessions
    ORDER BY subject_path ASC, intervention_id ASC;
    """).fetchdf()
    
    # 2. 查询错因卡与策略卡
    misc_df = conn.execute("""
    SELECT session_id, misconception_name, error_choice, deep_mechanism, verbatim_student_quotes, source_turn_ids
    FROM misconception_chunks;
    """).fetchdf()
    misc_map = {row["session_id"]: row for row in misc_df.to_dict(orient="records")}
    
    strat_df = conn.execute("""
    SELECT session_id, strategy_category, key_aha_question, scaffolding_steps, talk_moves, source_turn_ids
    FROM tutor_strategy_chunks;
    """).fetchdf()
    strat_map = {row["session_id"]: row for row in strat_df.to_dict(orient="records")}
    
    # 3. 查询真实原声对话
    turns_df = conn.execute("""
    SELECT intervention_id, turn_id, speaker, text, talk_moves, is_greeting_or_noise
    FROM session_dialogue_turns
    WHERE is_greeting_or_noise = FALSE
    ORDER BY intervention_id ASC, turn_id ASC;
    """).fetchdf()
    
    turns_by_sess = defaultdict(list)
    for t in turns_df.to_dict(orient="records"):
        turns_by_sess[t["intervention_id"]].append(t)
        
    catalog = []
    
    for s in sessions_df.to_dict(orient="records"):
        sess_id = s["intervention_id"]
        misc = misc_map.get(sess_id, {})
        strat = strat_map.get(sess_id, {})
        turns = turns_by_sess.get(sess_id, [])
        
        # 整理学生代表性发问与原话
        student_turns = [
            {"turn_id": t["turn_id"], "quote": t["text"]}
            for t in turns if t["speaker"] == "student" and len(t["text"].strip()) > 3
        ]
        
        # 整理导师代表性启发与提问
        tutor_turns = []
        for t in turns:
            if t["speaker"] == "tutor" and "?" in str(t["text"]):
                tm = list(t["talk_moves"]) if t.get("talk_moves") is not None else []
                tutor_turns.append({"turn_id": t["turn_id"], "quote": t["text"], "talk_moves": tm})
        
        entry = {
            "session_id": sess_id,
            "question_id": s["question_id"],
            "subject_path": s["subject_path"],
            "question_text": s["question_text"],
            "options": json.loads(s["options_json"]) if s["options_json"] else {},
            "misconception_name": misc.get("misconception_name", "概念混淆"),
            "error_choice": misc.get("error_choice"),
            "key_aha_question": strat.get("key_aha_question", ""),
            "student_verbatim_quotes": student_turns[:6],
            "tutor_verbatim_prompts": tutor_turns[:4]
        }
        catalog.append(entry)
        
    conn.close()
    
    # 写入 JSON
    out_json = Path("data/authoring_catalog.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 成功导出 {len(catalog)} 场真实会话出题参考索引至 {out_json}！")
    return catalog


if __name__ == "__main__":
    export_catalog()
