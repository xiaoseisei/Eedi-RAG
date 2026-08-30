import duckdb
import chromadb
import json

def to_serializable(val):
    if hasattr(val, "tolist"):
        return val.tolist()
    return str(val)

duck_conn = duckdb.connect("data/db/tutoring_knowledge.duckdb", read_only=True)

print("================== [1. DuckDB: tutoring_sessions 宏观事实表] ==================")
s_row = duck_conn.execute("SELECT intervention_id, question_id, tutor_id, subject_path, question_text, total_turns FROM tutoring_sessions WHERE intervention_id = 10;").fetchdf().to_dict(orient="records")[0]
print(json.dumps(s_row, indent=2, ensure_ascii=False, default=to_serializable))

print("\n================== [2. DuckDB: session_dialogue_turns 逐轮证据实录] ==================")
t_rows = duck_conn.execute("SELECT turn_id, speaker, text, talk_moves FROM session_dialogue_turns WHERE intervention_id = 10 AND turn_id IN (10, 16, 17, 18);").fetchdf().to_dict(orient="records")
for t in t_rows:
    print(f"[Turn {t['turn_id']}] [{t['speaker']}]: {t['text']} | TalkMoves: {t['talk_moves']}")

print("\n================== [3. DuckDB: misconception_chunks 错因结构化表] ==================")
m_row = duck_conn.execute("SELECT session_id, misconception_name, error_choice, deep_mechanism, verbatim_student_quotes, source_turn_ids FROM misconception_chunks WHERE session_id = 10;").fetchdf().to_dict(orient="records")[0]
print(json.dumps(m_row, indent=2, ensure_ascii=False, default=to_serializable))

print("\n================== [4. DuckDB: tutor_strategy_chunks 名师策略表] ==================")
strat_row = duck_conn.execute("SELECT session_id, strategy_category, key_aha_question, scaffolding_steps, talk_moves, source_turn_ids FROM tutor_strategy_chunks WHERE session_id = 10;").fetchdf().to_dict(orient="records")[0]
print(json.dumps(strat_row, indent=2, ensure_ascii=False, default=to_serializable))

print("\n================== [5. ChromaDB: student_misconceptions 向量集合] ==================")
chroma_client = chromadb.PersistentClient(path="data/chroma")
c_misc = chroma_client.get_collection("student_misconceptions")
m_sample = c_misc.get(ids=["session_10_misconception"], include=["documents", "metadatas", "embeddings"])
print("【Embedding Document (注入考纲与原题的增强文本)】:")
print(m_sample["documents"][0])
print("\n【Metadata】:")
print(json.dumps(m_sample["metadatas"][0], indent=2, ensure_ascii=False, default=to_serializable))
print(f"\n【Embedding Vector】: 维度={len(m_sample['embeddings'][0])}, 示例前5维={m_sample['embeddings'][0][:5]}")

print("\n================== [6. ChromaDB: tutor_strategies 向量集合] ==================")
c_strat = chroma_client.get_collection("tutor_strategies")
s_sample = c_strat.get(ids=["session_10_tutor_strategy"], include=["documents", "metadatas"])
print("【Embedding Document】:")
print(s_sample["documents"][0])
print("\n【Metadata】:")
print(json.dumps(s_sample["metadatas"][0], indent=2, ensure_ascii=False, default=to_serializable))

print("\n================== [7. ChromaDB: fallback_windows 滑动窗口集合] ==================")
c_win = chroma_client.get_collection("fallback_windows")
win_sample = c_win.get(limit=1, include=["documents", "metadatas"])
print("【滑动窗口 Document 片段】:")
print(win_sample["documents"][0][:200] + "...")
print("\n【Metadata】:")
print(json.dumps(win_sample["metadatas"][0], indent=2, ensure_ascii=False, default=to_serializable))
