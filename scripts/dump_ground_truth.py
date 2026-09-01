# -*- coding: utf-8 -*-
import json
from pathlib import Path

sessions = {json.loads(line)['intervention_id']: json.loads(line) for line in open('data/sample/cleaned_sessions_sample.jsonl', encoding='utf-8') if line.strip()}
pius = {json.loads(line)['session_id']: json.loads(line) for line in open('data/sample/extracted_pius_sample.jsonl', encoding='utf-8') if line.strip()}

for sid, p in sorted(pius.items()):
    s = sessions.get(sid, {})
    q = s.get('question', {})
    subj = s.get('subjects', {}).get('paths', [''])[0]
    misc = p.get('misconception', {})
    strat = p.get('tutor_strategy', {})
    print(f"=== Session {sid} (QID: {q.get('question_id')}) ===")
    print(f"Subject Path: {subj}")
    print(f"Question Text: {q.get('question_text')}")
    print(f"Options: {q.get('options')}")
    print(f"Misconception Name: {misc.get('misconception_name')}")
    print(f"Error Choice: {misc.get('error_choice')}")
    print(f"Deep Mechanism: {misc.get('deep_mechanism')}")
    print(f"Confusion Triggers: {misc.get('confusion_triggers')}")
    print(f"Misc Turns: {misc.get('source_turn_ids')}")
    print(f"Strategy Category: {strat.get('strategy_category')}")
    print(f"Key Aha Question: {strat.get('key_aha_question')}")
    print(f"Scaffolding Steps: {strat.get('scaffolding_steps')}")
    print(f"Talk Moves: {strat.get('talk_moves')}")
    print(f"Resolution Outcome: {strat.get('resolution_outcome')}")
    print(f"Strat Turns: {strat.get('source_turn_ids')}")
    print()
