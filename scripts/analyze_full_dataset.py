# -*- coding: utf-8 -*-
import json
from collections import Counter
import pandas as pd
import numpy as np

sessions = []
with open("data/cleaned_sessions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            sessions.append(json.loads(line))

total_sessions = len(sessions)
valid_sessions = [s for s in sessions if s.get("has_valid_tutoring", False)]
invalid_sessions = [s for s in sessions if not s.get("has_valid_tutoring", False)]

total_turns = sum(s["total_turns"] for s in sessions)
total_student_turns = sum(s["student_turn_count"] for s in sessions)
total_tutor_turns = sum(s["tutor_turn_count"] for s in sessions)

turn_counts = [s["total_turns"] for s in sessions]
student_turn_counts = [s["student_turn_count"] for s in sessions]
tutor_turn_counts = [s["tutor_turn_count"] for s in sessions]

distinct_questions = len(set(s["question_id"] for s in sessions))
distinct_tutors = len(set(s["tutor_id"] for s in sessions))

# 学科与考点统计
subject_counts = Counter()
topic_counts = Counter()
path_counts = Counter()
for s in sessions:
    for p in s.get("subjects", {}).get("paths", []):
        path_counts[p] += 1
    for subj in s.get("subjects", {}).get("subjects", []):
        subject_counts[subj] += 1
    for top in s.get("subjects", {}).get("topics", []):
        topic_counts[top] += 1

# 教学动作统计
talk_moves_counts = Counter()
for s in sessions:
    for t in s.get("turns", []):
        for tm in t.get("talk_moves", []):
            talk_moves_counts[tm] += 1

# 寒暄噪音轮次统计
greeting_turns = sum(1 for s in sessions for t in s.get("turns", []) if t.get("is_greeting_or_noise", False))
academic_turns = total_turns - greeting_turns

stats = {
    "total_sessions": total_sessions,
    "valid_sessions_count": len(valid_sessions),
    "valid_sessions_rate": len(valid_sessions) / total_sessions * 100.0,
    "invalid_sessions_count": len(invalid_sessions),
    "distinct_questions": distinct_questions,
    "distinct_tutors": distinct_tutors,
    "total_turns": total_turns,
    "academic_turns": academic_turns,
    "greeting_turns": greeting_turns,
    "greeting_rate": greeting_turns / total_turns * 100.0,
    "turns_per_session": {
        "mean": float(np.mean(turn_counts)),
        "median": float(np.median(turn_counts)),
        "min": int(np.min(turn_counts)),
        "max": int(np.max(turn_counts)),
        "p25": float(np.percentile(turn_counts, 25)),
        "p75": float(np.percentile(turn_counts, 75))
    },
    "student_turns_mean": float(np.mean(student_turn_counts)),
    "tutor_turns_mean": float(np.mean(tutor_turn_counts)),
    "top_subjects": subject_counts.most_common(10),
    "top_topics": topic_counts.most_common(10),
    "top_paths": path_counts.most_common(10),
    "top_talk_moves": talk_moves_counts.most_common(15)
}

print(json.dumps(stats, ensure_ascii=False, indent=2))
