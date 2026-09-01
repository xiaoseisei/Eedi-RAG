import json
import duckdb

conn = duckdb.connect('data/db/tutoring_knowledge.duckdb', read_only=True)
with open('data/golden_test_set.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)

print(f"Total golden cases: {len(cases)}")
for idx in [8, 9, 13, 15, 19, 21, 25, 27, 28]:
    c = cases[idx-1]
    p = c.get('subject_path', '')
    target_sessions = [q.get('session_id') for q in c.get('verbatim_grounding_quotes', [])]
    print(f"Case {idx} Q: {c['question'][:35]}...")
    print(f"  Target Session IDs: {target_sessions} | Subject Path: {p}")
    # Check if target sessions exist in DuckDB
    for s_id in target_sessions:
        if s_id:
            row = conn.execute("SELECT intervention_id, subject_path, question_text FROM tutoring_sessions WHERE intervention_id = ?", [s_id]).fetchall()
            print(f"    Sess #{s_id} in DB: {bool(row)}")
conn.close()
