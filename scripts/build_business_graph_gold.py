"""Build database-grounded business-graph Gold v1 without consulting SUT rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = ROOT / "data/db/tutoring_knowledge.duckdb"
GRAPH_DB = ROOT / "reports/staging/session-card-graph-v1/graph-taxonomy-verify-2-20260908.duckdb"
TAXONOMY = ROOT / "data/business/graph-taxonomy-v1.json"
WINDOW_SIZE = 7
WINDOW_STEP = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _turns(conn: duckdb.DuckDBPyConnection, session_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT turn_id, speaker, text, is_greeting_or_noise FROM session_dialogue_turns "
        "WHERE intervention_id=? ORDER BY turn_id", [session_id]
    ).fetchall()
    return [{"turn_id": int(t), "speaker": str(s), "exact_quote": str(text), "is_noise": bool(noise)} for t, s, text, noise in rows]


def _windows(conn: duckdb.DuckDBPyConnection, session_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT window_start_turn, window_end_turn, source_turn_ids FROM sliding_window_chunks "
        "WHERE session_id=? ORDER BY window_start_turn", [session_id]
    ).fetchall()
    return [{
        "session_id": session_id,
        "window_policy": "W6_S3",
        "window_id": f"session_{session_id}_win_{int(start)}_{int(end)}",
        "source_turn_ids": [int(x) for x in (ids or [])],
    } for start, end, ids in rows]


def _w7_windows(turns: list[dict[str, Any]], session_id: int) -> list[dict[str, Any]]:
    ids = [int(item["turn_id"]) for item in turns]
    result = []
    for offset in range(0, len(ids), WINDOW_STEP):
        selected = ids[offset : offset + WINDOW_SIZE]
        if not selected:
            continue
        result.append({
            "session_id": session_id,
            "window_policy": "W7_S3",
            "window_id": f"session_{session_id}_win_{selected[0]}_{selected[-1]}",
            "source_turn_ids": selected,
        })
        if offset + WINDOW_SIZE >= len(ids):
            break
    return result


def _card(conn: duckdb.DuckDBPyConnection, table: str, session_id: int) -> dict[str, Any]:
    row = conn.execute(f"SELECT * FROM {table} WHERE session_id=?", [session_id]).fetchone()
    if row is None:
        raise ValueError(f"missing {table} session {session_id}")
    columns = [item[1] for item in conn.execute(f"PRAGMA table_info('{table}')").fetchall()]
    data = {name: _json(value) for name, value in zip(columns, row)}
    data["card_type"] = "MISCONCEPTION" if table == "misconception_chunks" else "TUTOR_STRATEGY"
    data["card_id"] = data["chunk_id"]
    data["source_session_id"] = int(data["session_id"])
    return data


def build_catalog(source_db: Path, graph_db: Path, taxonomy: Path) -> dict[str, Any]:
    source = duckdb.connect(str(source_db), read_only=True)
    graph = duckdb.connect(str(graph_db), read_only=True)
    try:
        source_hash = sha256(source_db)
        manifest = graph.execute("SELECT * FROM graph_manifest WHERE build_status='READY'").fetchall()
        if len(manifest) != 1 or str(manifest[0][1]).lower() != source_hash.lower():
            raise ValueError("no exactly-one READY graph manifest matches source DB hash")
        sessions = source.execute(
            "SELECT intervention_id, question_id, subject_path, question_text FROM tutoring_sessions "
            "ORDER BY intervention_id"
        ).fetchall()
        catalog_sessions = []
        card_quote_warnings: list[dict[str, Any]] = []
        for sid, qid, subject, question in sessions:
            sid = int(sid)
            turns = _turns(source, sid)
            cards = [_card(source, "misconception_chunks", sid), _card(source, "tutor_strategy_chunks", sid)]
            turn_by_id = {item["turn_id"]: item for item in turns}
            for card in cards:
                expected_speaker = "student" if card["card_type"] == "MISCONCEPTION" else "tutor"
                for turn_id in card.get("source_turn_ids") or []:
                    turn = turn_by_id.get(int(turn_id))
                    if turn is None or turn["speaker"] != expected_speaker:
                        raise ValueError(f"invalid {card['card_id']} source pointer or role")
                quotes = card.get("verbatim_student_quotes") or []
                if quotes and not all(str(q) in {t["exact_quote"] for t in turns} for q in quotes):
                    # Existing Card quote summaries are not authoritative. Keep the
                    # warning in the independent catalog, but never promote these
                    # strings into Gold; selected evidence always uses raw Turns.
                    card_quote_warnings.append({"card_id": card["card_id"], "session_id": sid})
            graph_nodes = graph.execute(
                "SELECT node_id,node_type,label,source_turn_ids FROM graph_nodes WHERE source_session_id=? ORDER BY node_id", [sid]
            ).fetchall()
            graph_edges = graph.execute(
                "SELECT source_node_id,target_node_id,edge_type,source_turn_ids FROM graph_edges WHERE source_session_id=? ORDER BY edge_id", [sid]
            ).fetchall()
            catalog_sessions.append({
                "session_id": sid,
                "question_id": int(qid),
                "subject_path": subject or "",
                "question_text": question or "",
                "turns": turns,
                "cards": cards,
                "windows_w6_s3": _windows(source, sid),
                "windows_w7_s3": _w7_windows(turns, sid),
                "graph_nodes": [{"node_id": n, "node_type": typ, "label": label, "source_turn_ids": _json(ids or [])} for n, typ, label, ids in graph_nodes],
                "graph_edges": [{"source_node_id": s, "target_node_id": t, "edge_type": e, "source_turn_ids": _json(ids or [])} for s, t, e, ids in graph_edges],
            })
        return {
            "schema_version": "business-graph-authoring-catalog/v1",
            "source_db_sha256": source_hash,
            "graph_db_sha256": sha256(graph_db),
            "taxonomy_sha256": sha256(taxonomy),
            "graph_version": str(manifest[0][0]),
            "window_config": {"window_size": WINDOW_SIZE, "step": WINDOW_STEP, "policy": "W7_S3"},
            "session_count": len(catalog_sessions),
            "card_quote_warning_count": len(card_quote_warnings),
            "card_quote_warnings": card_quote_warnings,
            "sessions": catalog_sessions,
        }
    finally:
        source.close()
        graph.close()


def _positive(case_id: str, category: str, query: str, intent: str, scope: str, computation: str,
              perspectives: list[str], session: dict[str, Any], card_type: str = "MISCONCEPTION",
              anchor_turn: int | None = None, plan: str = "SOCRATIC_TUTORING", lever: str = "student_misconception_card_rag") -> dict[str, Any]:
    card = next(item for item in session["cards"] if item["card_type"] == card_type)
    if anchor_turn is None:
        source_ids = {int(item) for item in (card.get("source_turn_ids") or [])}
        eligible = [item for item in session["turns"] if item["turn_id"] in source_ids and not item["is_noise"] and len(item["exact_quote"].strip()) >= 12 and not re.match(r"(?i)^(hi|hello|good|hey|how are you|i'?m great|im great)\b", item["exact_quote"].strip())]
        anchor_turn = int((eligible[0] if eligible else next(item for item in session["turns"] if item["turn_id"] in source_ids))["turn_id"])
    anchor = anchor_turn
    window = next((item for item in session["windows_w7_s3"] if anchor in item["source_turn_ids"]), None)
    if window is None:
        raise ValueError(f"no W7 window contains anchor {session['session_id']}:{anchor}")
    window = {**window, "anchor_turn_ids": [anchor]}
    required_turns = [{k: v for k, v in {**item, "session_id": session["session_id"]}.items() if k != "is_noise"} for item in session["turns"] if item["turn_id"] in window["source_turn_ids"] and item["turn_id"] == anchor]
    return {
        "case_id": case_id, "category": category, "query": query,
        "expected_router": {"intent": intent, "scope": scope, "computation_scope": computation, "perspectives": perspectives},
        "expected_plan": {"plan_type": plan, "selected_lever": {"index": 1, "id": lever}},
        "expected_sql_scope": {"filters": {"question_id": session["question_id"]}, "keywords": ["question_id", "session_id", "misconception_chunks"] if card_type == "MISCONCEPTION" else ["question_id", "session_id", "tutor_strategy_chunks"]},
        "expected_graph_paths": [{"nodes": ["SESSION", "MISCONCEPTION_EVENT" if card_type == "MISCONCEPTION" else "TUTOR_STRATEGY_EVENT", "TURN"], "edges": ["HAS_MISCONCEPTION" if card_type == "MISCONCEPTION" else "HAS_STRATEGY", "EVIDENCED_BY"]}],
        "expected_card_evidence": [{"card_type": card_type, "card_id": card["card_id"], "source_session_id": session["session_id"]}],
        "expected_window_evidence": [window],
        "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "max_parent_cards": 5, "required_window_ids": [window["window_id"]], "required_sections": ["学生错因卡" if card_type == "MISCONCEPTION" else "导师策略卡", "7轮课堂上下文", "数据来源和限制"]},
        "expected_citation_contract": {"citation_source": "AUTHORIZED_WINDOW_ONLY", "required_evidence_turns": required_turns},
        "expected_status": "SUCCESS",
        "ideal_answer": {"required_claims": [str(card.get("misconception_name") or card.get("pedagogical_goal") or "必须基于真实课堂证据回答")], "forbidden_claims": ["该方法一定有效", "学生分数得到提升"]},
    }


def build_cases(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {item["session_id"]: item for item in catalog["sessions"]}
    s10 = by_id[10]
    cases = [
        _positive("micro-student-error-001", "MICRO_STUDENT_ERROR", "学生为什么会把 5.4598 四舍五入成 5.45？", "QUESTION_TUTORING", "QUESTION", "MICRO", ["STUDENT"], s10, anchor_turn=10),
        _positive("micro-tutor-method-001", "MICRO_TUTOR_METHOD", "导师如何引导学生理解 5.4598 应该如何四舍五入？", "QUESTION_TUTORING", "QUESTION", "MICRO", ["TUTOR"], s10, card_type="TUTOR_STRATEGY", anchor_turn=9, plan="SOCRATIC_TUTORING", lever="tutor_strategy_card_rag"),
        _positive("micro-class-review-001", "MICRO_CLASS_REVIEW", "请回顾这节关于 5.4598 四舍五入的课堂。", "QUESTION_TUTORING", "CASE", "MICRO", ["STUDENT", "TUTOR"], s10, anchor_turn=6),
        _positive("micro-question-reasoning-001", "MICRO_QUESTION_OPTIONS_REASONING", "这道题的选项和学生推理过程反映了什么？", "QUESTION_TUTORING", "QUESTION", "MICRO", ["STUDENT", "TUTOR"], s10, anchor_turn=8),
    ]
    macro = [
        ("macro-student-faq-001", "MACRO_STUDENT_FREQUENT_QUESTIONS", "学生最常提出的问题是什么？", "STUDENT_FREQUENT_QUESTIONS", "STUDENT", ["STUDENT"]),
        ("macro-difficult-concept-001", "MACRO_DIFFICULT_CONCEPT", "哪些概念最难理解？", "DIFFICULT_CONCEPTS", "SUBJECT", ["STUDENT", "TUTOR"]),
        ("macro-recurring-misconception-001", "MACRO_RECURRING_MISCONCEPTION", "哪些学生误区在不同课堂中反复出现？", "RECURRING_MISCONCEPTIONS", "CORPUS", ["STUDENT"]),
        ("macro-exam-error-001", "MACRO_EXAM_ERROR", "学生最常见的考试错误模式是什么？", "EXAM_ERROR_PATTERNS", "CORPUS", ["STUDENT"]),
        ("macro-tutor-strategy-001", "MACRO_TUTOR_STRATEGY", "导师最常使用哪些教学策略？", "TUTOR_STRATEGY_PATTERNS", "CORPUS", ["TUTOR"]),
        ("macro-content-improvement-001", "MACRO_CONTENT_IMPROVEMENT", "哪些内容最值得优化为教材或题库解析？", "CONTENT_IMPROVEMENT", "CORPUS", ["STUDENT", "TUTOR"]),
        ("macro-cross-session-cooccurrence-001", "MACRO_CROSS_SESSION_COOCCURRENCE", "学生误区和导师策略在不同课堂中如何共现？", "CONTENT_IMPROVEMENT", "CORPUS", ["STUDENT", "TUTOR"]),
    ]
    for cid, category, query, intent, scope, perspectives in macro:
        operation = "misconception_strategy_pairs" if "CROSS_SESSION" in category else {"STUDENT_FREQUENT_QUESTIONS": "frequent_student_questions", "DIFFICULT_CONCEPTS": "difficult_concepts", "RECURRING_MISCONCEPTIONS": "recurring_misconceptions", "EXAM_ERROR_PATTERNS": "exam_error_patterns", "TUTOR_STRATEGY_PATTERNS": "tutor_strategy_patterns", "CONTENT_IMPROVEMENT": "content_opportunities"}.get(intent, "misconception_strategy_pairs")
        cases.append({
            "case_id": cid, "category": category, "query": query,
            "expected_router": {"intent": intent, "scope": scope, "computation_scope": "HYBRID", "perspectives": perspectives},
            "expected_plan": {"plan_type": "CONTENT_BRIEF" if intent == "CONTENT_IMPROVEMENT" else "ANALYTICS_WITH_EVIDENCE", "selected_lever": {"index": 2, "id": "graph_sql_with_evidence"}},
            "expected_sql_scope": {"filters": {}, "keywords": ["canonical_label", "COUNT(DISTINCT source_session_id)", "session_share", operation]},
            "expected_graph_paths": ([{"nodes": ["MISCONCEPTION_CONCEPT", "MISCONCEPTION_EVENT", "TUTOR_STRATEGY_EVENT"], "edges": ["TYPE_OF", "CO_OCCURS_WITH"]}] if "CROSS_SESSION" in category else [{"nodes": ["SESSION", "MISCONCEPTION_EVENT", "MISCONCEPTION_CONCEPT"], "edges": ["HAS_MISCONCEPTION", "TYPE_OF"]}]),
            "expected_card_evidence": [], "expected_window_evidence": [],
            "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "max_parent_cards": 5, "required_window_ids": [], "required_sections": ["宏观统计事实", "排名与占比", "数据来源和限制"]},
            "expected_citation_contract": {"citation_source": "AUTHORIZED_WINDOW_ONLY", "required_evidence_turns": []},
            "expected_status": "SUCCESS",
            "ideal_answer": {"required_claims": ["给出全库真实排名", "给出 distinct_session_count 和 session_share", "声明数据范围"], "forbidden_claims": ["该方法一定有效", "学生分数得到提升", "因果有效"]},
        })
    cases.extend([
        {"case_id": "mixed-macro-graph-evidence-001", "category": "MIXED_MACRO_GRAPH_MICRO", "query": "统计学生最常见的舍入误区，并给出课堂原声证据。", "expected_router": {"intent": "RECURRING_MISCONCEPTIONS", "scope": "CORPUS", "computation_scope": "HYBRID", "perspectives": ["STUDENT"]}, "expected_plan": {"plan_type": "ANALYTICS_WITH_EVIDENCE", "selected_lever": {"index": 2, "id": "graph_sql_with_evidence"}}, "expected_sql_scope": {"filters": {}, "keywords": ["canonical_label", "COUNT(DISTINCT source_session_id)", "session_share"]}, "expected_graph_paths": [{"nodes": ["SESSION", "MISCONCEPTION_EVENT", "MISCONCEPTION_CONCEPT", "TURN"], "edges": ["HAS_MISCONCEPTION", "TYPE_OF", "EVIDENCED_BY"]}], "expected_card_evidence": [{"card_type": "MISCONCEPTION", "card_id": s10["cards"][0]["card_id"], "source_session_id": 10}], "expected_window_evidence": [{**next(w for w in s10["windows_w7_s3"] if 10 in w["source_turn_ids"]), "anchor_turn_ids": [10]}], "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "max_parent_cards": 5, "required_window_ids": [next(w for w in s10["windows_w7_s3"] if 10 in w["source_turn_ids"])["window_id"]], "required_sections": ["宏观统计事实", "图关系", "7轮课堂上下文", "数据来源和限制"]}, "expected_citation_contract": {"citation_source": "AUTHORIZED_WINDOW_ONLY", "required_evidence_turns": [{**t, "session_id": 10} for t in s10["turns"] if t["turn_id"] == 10]}, "expected_status": "SUCCESS", "ideal_answer": {"required_claims": ["给出跨 Session 统计", "提供课堂原声证据"], "forbidden_claims": ["该策略导致成绩提升", "因果有效"]}},
        {"case_id": "mixed-misconception-intervention-001", "category": "MIXED_MISCONCEPTION_INTERVENTION", "query": "学生的舍入误区是什么，导师如何干预？", "expected_router": {"intent": "QUESTION_TUTORING", "scope": "CASE", "computation_scope": "MICRO", "perspectives": ["STUDENT", "TUTOR"]}, "expected_plan": {"plan_type": "SOCRATIC_TUTORING", "selected_lever": {"index": 1, "id": "student_misconception_card_rag"}}, "expected_sql_scope": {"filters": {"question_id": 104614}, "keywords": ["question_id", "session_id", "misconception_chunks", "tutor_strategy_chunks"]}, "expected_graph_paths": [{"nodes": ["SESSION", "MISCONCEPTION_EVENT", "TUTOR_STRATEGY_EVENT"], "edges": ["HAS_MISCONCEPTION", "HAS_STRATEGY", "CO_OCCURS_WITH"]}], "expected_card_evidence": [{"card_type": "MISCONCEPTION", "card_id": s10["cards"][0]["card_id"], "source_session_id": 10}, {"card_type": "TUTOR_STRATEGY", "card_id": s10["cards"][1]["card_id"], "source_session_id": 10}], "expected_window_evidence": [{**next(w for w in s10["windows_w7_s3"] if 10 in w["source_turn_ids"]), "anchor_turn_ids": [9, 10]}], "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "max_parent_cards": 5, "required_window_ids": [next(w for w in s10["windows_w7_s3"] if 10 in w["source_turn_ids"])["window_id"]], "required_sections": ["学生错因卡", "导师策略卡", "7轮课堂上下文"]}, "expected_citation_contract": {"citation_source": "AUTHORIZED_WINDOW_ONLY", "required_evidence_turns": [{**t, "session_id": 10} for t in s10["turns"] if t["turn_id"] in {9, 10}]}, "expected_status": "SUCCESS", "ideal_answer": {"required_claims": ["学生错因", "导师干预"], "forbidden_claims": ["一定有效", "成绩提升"]}},
        {"case_id": "mixed-cross-session-pattern-001", "category": "MIXED_CROSS_SESSION_PATTERN", "query": "跨课堂比较相同学生误区及导师策略。", "expected_router": {"intent": "CONTENT_IMPROVEMENT", "scope": "CORPUS", "computation_scope": "HYBRID", "perspectives": ["STUDENT", "TUTOR"]}, "expected_plan": {"plan_type": "CONTENT_BRIEF", "selected_lever": {"index": 2, "id": "graph_sql_with_evidence"}}, "expected_sql_scope": {"filters": {}, "keywords": ["source_session_id", "canonical_label", "CO_OCCURS_WITH"]}, "expected_graph_paths": [{"nodes": ["MISCONCEPTION_CONCEPT", "MISCONCEPTION_EVENT", "TUTOR_STRATEGY_EVENT"], "edges": ["TYPE_OF", "CO_OCCURS_WITH"]}], "expected_card_evidence": [], "expected_window_evidence": [], "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "required_window_ids": [], "required_sections": ["跨 Session 统计", "数据来源和限制"]}, "expected_citation_contract": {"citation_source": "AUTHORIZED_WINDOW_ONLY", "required_evidence_turns": []}, "expected_status": "SUCCESS", "ideal_answer": {"required_claims": ["明确比较范围"], "forbidden_claims": ["存在因果效果"]}},
        {"case_id": "negative-no-evidence-001", "category": "NO_EVIDENCE", "query": "请给出当前库中不存在的题目 999999 的课堂证据。", "expected_router": {"intent": "QUESTION_TUTORING", "scope": "QUESTION", "computation_scope": "MICRO", "perspectives": ["STUDENT", "TUTOR"]}, "expected_plan": {"plan_type": "SOCRATIC_TUTORING", "selected_lever": {"index": 1, "id": "student_misconception_card_rag"}}, "expected_sql_scope": {"filters": {"question_id": 999999}, "keywords": ["question_id"]}, "expected_graph_paths": [], "expected_card_evidence": [], "expected_window_evidence": [], "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "required_window_ids": [], "required_sections": ["数据来源和限制"]}, "expected_status": "NO_EVIDENCE", "ideal_answer": {"required_claims": ["未找到真实证据"], "forbidden_claims": ["推断学生误区"]}},
        {"case_id": "negative-unsupported-001", "category": "UNSUPPORTED", "query": "这些学生重考后分数是否提高？", "expected_router": {"intent": "STUDENT_COHORT_PATTERNS", "scope": "CORPUS", "computation_scope": "REJECT", "perspectives": ["STUDENT"]}, "expected_plan": {"plan_type": "REJECT", "selected_lever": {"index": 0, "id": "unsupported"}}, "expected_sql_scope": {"filters": {}, "keywords": ["student_key", "attempt_number"]}, "expected_graph_paths": [], "expected_card_evidence": [], "expected_window_evidence": [], "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "required_window_ids": [], "required_sections": ["数据来源和限制"]}, "expected_status": "UNSUPPORTED", "ideal_answer": {"required_claims": ["当前数据不支持重考后成绩变化判断"], "forbidden_claims": ["成绩提高", "成绩下降"]}},
        {"case_id": "negative-failed-001", "category": "FAILED", "query": "在图谱 manifest 与源库哈希不一致时执行该统计。", "expected_router": {"intent": "RECURRING_MISCONCEPTIONS", "scope": "CORPUS", "computation_scope": "HYBRID", "perspectives": ["STUDENT"]}, "expected_plan": {"plan_type": "ANALYTICS_WITH_EVIDENCE", "selected_lever": {"index": 2, "id": "graph_sql_with_evidence"}}, "expected_sql_scope": {"filters": {}, "keywords": ["source_artifact_hash", "graph_manifest"]}, "expected_graph_paths": [], "expected_card_evidence": [], "expected_window_evidence": [], "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "required_window_ids": [], "required_sections": ["失败原因", "数据来源和限制"]}, "expected_status": "FAILED", "ideal_answer": {"required_claims": ["必须阻断并报告哈希不一致"], "forbidden_claims": ["统计成功"]}},
    ])
    # Expand from distinct real sessions. The targets are coverage targets, not
    # permission to synthesize rows: a missing source session aborts the build.
    used_sessions = {int(ref["source_session_id"]) for case in cases for ref in case.get("expected_card_evidence", [])}
    available = [item for item in catalog["sessions"] if item["session_id"] not in used_sessions]
    targets = {
        "MICRO_STUDENT_ERROR": 6, "MICRO_TUTOR_METHOD": 6,
        "MICRO_CLASS_REVIEW": 6, "MICRO_QUESTION_OPTIONS_REASONING": 6,
        "MACRO_STUDENT_FREQUENT_QUESTIONS": 5, "MACRO_DIFFICULT_CONCEPT": 5,
        "MACRO_RECURRING_MISCONCEPTION": 5, "MACRO_EXAM_ERROR": 5,
        "MACRO_TUTOR_STRATEGY": 5, "MACRO_CONTENT_IMPROVEMENT": 5,
        "MACRO_CROSS_SESSION_COOCCURRENCE": 5,
        "MIXED_MACRO_GRAPH_MICRO": 5, "MIXED_MISCONCEPTION_INTERVENTION": 5,
        "MIXED_CROSS_SESSION_PATTERN": 5,
    }
    existing_counts = {category: sum(case["category"] == category for case in cases) for category in targets}
    cursor = 0
    for category, target in targets.items():
        while existing_counts[category] < target:
            if cursor >= len(available):
                raise ValueError(f"real source sessions insufficient for category {category}")
            session = available[cursor]
            cursor += 1
            if category == "MICRO_TUTOR_METHOD":
                cases.append(_positive(f"micro-tutor-method-{existing_counts[category] + 1:03d}", category, f"导师如何处理 {session['question_text'][:80]}？", "QUESTION_TUTORING", "QUESTION", "MICRO", ["TUTOR"], session, card_type="TUTOR_STRATEGY", lever="tutor_strategy_card_rag"))
            elif category.startswith("MACRO"):
                intent = "CONTENT_IMPROVEMENT" if "CONTENT" in category or "CROSS_SESSION" in category else "RECURRING_MISCONCEPTIONS" if "MISCONCEPTION" in category else "EXAM_ERROR_PATTERNS" if "EXAM" in category else "TUTOR_STRATEGY_PATTERNS" if "TUTOR" in category else "DIFFICULT_CONCEPTS" if "DIFFICULT" in category else "STUDENT_FREQUENT_QUESTIONS"
                operation = "misconception_strategy_pairs" if "CROSS_SESSION" in category else {"STUDENT_FREQUENT_QUESTIONS": "frequent_student_questions", "DIFFICULT_CONCEPTS": "difficult_concepts", "RECURRING_MISCONCEPTIONS": "recurring_misconceptions", "EXAM_ERROR_PATTERNS": "exam_error_patterns", "TUTOR_STRATEGY_PATTERNS": "tutor_strategy_patterns", "CONTENT_IMPROVEMENT": "content_opportunities"}.get(intent, "misconception_strategy_pairs")
                macro_queries = {
                    "MACRO_STUDENT_FREQUENT_QUESTIONS": ["全库学生最常问的数学问题 Top-5 是什么？", "按学科统计，学生提问密度最高的主题有哪些？", "哪些问题在学生辅导记录中出现次数最多？", "学生最常请求解释的是哪些数学知识点？"],
                    "MACRO_DIFFICULT_CONCEPT": ["历次辅导中，学生最常卡住的概念有哪些？", "哪些数学主题的困难信号最集中？", "按学科看，学生最难理解的内容是什么？", "哪些概念出现了最多次不确定和反复尝试？"],
                    "MACRO_RECURRING_MISCONCEPTION": ["跨课堂反复出现的学生误区有哪些？", "哪些认知错误在不同学生身上重复出现？", "全库最顽固的数学误区排名是什么？", "哪些错因值得优先纳入错题讲解？"],
                    "MACRO_EXAM_ERROR": ["学生在做题时最常见的错误模式是什么？", "哪些考试错误类型跨课堂出现最多？", "从真实辅导记录看，学生最容易漏掉哪些条件？", "学生选择错误选项的主要模式有哪些？"],
                    "MACRO_TUTOR_STRATEGY": ["在真实辅导记录里，导师最常采用哪几类教学策略？", "哪些提问方式在辅导记录中出现最多？", "导师通常如何拆解学生的数学卡点？", "真实课堂里最常见的启发式教学动作是什么？"],
                    "MACRO_CONTENT_IMPROVEMENT": ["哪些高频误区最值得改进教材和题库解析？", "哪些内容缺口适合补充例题或选项对比？", "从学生错误和导师策略看，教材应优先改哪里？", "哪些知识点需要增加更清晰的教学材料？"],
                    "MACRO_CROSS_SESSION_COOCCURRENCE": ["哪些学生误区和导师策略经常在同一课堂共现？", "跨会话看，哪些错因通常伴随哪些教学策略？", "导师的哪些策略最常用于处理特定学生误区？", "不同课堂中学生错因与导师干预的共现关系是什么？"],
                }[category]
                query = macro_queries[existing_counts[category] - 1]
                cases.append({"case_id": f"{category.lower().replace('_', '-')}-{existing_counts[category] + 1:03d}", "category": category, "query": query, "expected_router": {"intent": intent, "scope": "CORPUS", "computation_scope": "HYBRID", "perspectives": ["TUTOR"] if "TUTOR" in category else ["STUDENT"]}, "expected_plan": {"plan_type": "CONTENT_BRIEF" if intent == "CONTENT_IMPROVEMENT" else "ANALYTICS_WITH_EVIDENCE", "selected_lever": {"index": 2, "id": "graph_sql_with_evidence"}}, "expected_sql_scope": {"filters": {}, "keywords": ["canonical_label", "COUNT(DISTINCT source_session_id)", "session_share", operation]}, "expected_graph_paths": ([{"nodes": ["MISCONCEPTION_CONCEPT", "MISCONCEPTION_EVENT", "TUTOR_STRATEGY_EVENT"], "edges": ["TYPE_OF", "CO_OCCURS_WITH"]}] if "CROSS_SESSION" in category else []), "expected_card_evidence": [], "expected_window_evidence": [], "expected_prompt": {"rerank_unit": "WINDOW", "deduplicate_overlapping_windows": True, "max_parent_cards": 5, "required_window_ids": [], "required_sections": ["宏观统计事实", "排名与占比", "数据来源和限制"]}, "expected_citation_contract": {"citation_source": "AUTHORIZED_WINDOW_ONLY", "required_evidence_turns": []}, "expected_status": "SUCCESS", "ideal_answer": {"required_claims": ["给出全库真实排名", "给出 distinct_session_count 和 session_share", "声明数据范围"], "forbidden_claims": ["该方法一定有效", "学生分数得到提升", "因果有效"]}})
            else:
                intent = "QUESTION_TUTORING" if category.startswith("MICRO") or category == "MIXED_MISCONCEPTION_INTERVENTION" else ("CONTENT_IMPROVEMENT" if "CONTENT" in category or "CROSS_SESSION" in category else "RECURRING_MISCONCEPTIONS")
                plan = "SOCRATIC_TUTORING" if intent == "QUESTION_TUTORING" else "CONTENT_BRIEF" if intent == "CONTENT_IMPROVEMENT" else "ANALYTICS_WITH_EVIDENCE"
                perspectives = ["STUDENT", "TUTOR"] if category.startswith("MIXED") else ["STUDENT"]
                cases.append(_positive(f"{category.lower().replace('_', '-')}-{existing_counts[category] + 1:03d}", category, f"围绕真实题目“{session['question_text'][:80]}”进行{category.replace('_', ' ').lower()}分析。", intent, "CASE" if intent == "QUESTION_TUTORING" else "CORPUS", "MICRO" if intent == "QUESTION_TUTORING" else "HYBRID", perspectives, session, card_type="MISCONCEPTION", plan=plan, lever="graph_sql_with_evidence" if plan != "SOCRATIC_TUTORING" else "student_misconception_card_rag"))
            existing_counts[category] += 1
    for status, query_template in [("NO_EVIDENCE", "请给出题目 {n} 的真实课堂证据。"), ("UNSUPPORTED", "这些学生的长期学习增益是否提高？"), ("FAILED", "在源库与图谱版本不一致时执行统计。")]:
        base = next(case for case in cases if case["expected_status"] == status)
        for index in range(2, 4):
            clone = json.loads(json.dumps(base))
            clone["case_id"] = f"negative-{status.lower()}-{index:03d}"
            clone["query"] = {
                "NO_EVIDENCE": ["请给出题目 900002 的真实课堂证据。", "请查找题目 900003 的真实学生原话。"],
                "UNSUPPORTED": ["这些学生的长期学习增益是否提高？", "哪些学生的考试分数在重考后提升最多？"],
                "FAILED": ["在源库与图谱版本不一致时执行统计。", "当 READY graph manifest 缺失时执行统计。"],
            }[status][index - 2]
            cases.append(clone)
    for case in cases:
        case.setdefault("expected_citation_contract", {
            "citation_source": "AUTHORIZED_WINDOW_ONLY",
            "required_evidence_turns": [],
        })
    return cases


def validate_cases(cases: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    """Validate authored expectations against the independent source catalog."""
    sessions = {int(item["session_id"]): item for item in catalog["sessions"]}
    case_ids: set[str] = set()
    for case in cases:
        if case["case_id"] in case_ids:
            raise ValueError(f"duplicate Gold case ID: {case['case_id']}")
        case_ids.add(case["case_id"])
        status = case["expected_status"]
        if status != "SUCCESS":
            if case["expected_card_evidence"] or case["expected_window_evidence"]:
                raise ValueError(f"non-success case contains evidence: {case['case_id']}")
            continue
        for card_ref in case["expected_card_evidence"]:
            sid = int(card_ref["source_session_id"])
            session = sessions.get(sid)
            if session is None:
                raise ValueError(f"Gold references missing session {sid}")
            if not any(item["card_id"] == card_ref["card_id"] and item["card_type"] == card_ref["card_type"] for item in session["cards"]):
                raise ValueError(f"Gold references missing card {card_ref['card_id']}")
        for window in case["expected_window_evidence"]:
            session = sessions.get(int(window["session_id"]))
            if session is None or window["window_policy"] != "W7_S3":
                raise ValueError(f"invalid Gold Window policy/session: {case['case_id']}")
            matches = [item for item in session["windows_w7_s3"] if item["window_id"] == window["window_id"]]
            if len(matches) != 1 or matches[0]["source_turn_ids"] != window["source_turn_ids"]:
                raise ValueError(f"Gold Window is not generated from source Turns: {case['case_id']}")
            if not set(window.get("anchor_turn_ids", [])) <= set(window["source_turn_ids"]):
                raise ValueError(f"Gold Window anchor is outside Window: {case['case_id']}")
        for evidence in case["expected_citation_contract"].get("required_evidence_turns", []):
            session = sessions.get(int(evidence["session_id"]))
            if session is None:
                raise ValueError(f"Gold citation references missing session: {case['case_id']}")
            found = next((t for t in session["turns"] if t["turn_id"] == int(evidence["turn_id"])), None)
            if found is None or found["speaker"] != evidence["speaker"] or found["exact_quote"] != evidence["exact_quote"]:
                raise ValueError(f"Gold citation is not an exact authoritative Turn: {case['case_id']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/business")
    parser.add_argument("--source-db", type=Path, default=SOURCE_DB)
    parser.add_argument("--graph-db", type=Path, default=GRAPH_DB)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    args = parser.parse_args()
    catalog = build_catalog(args.source_db.resolve(strict=True), args.graph_db.resolve(strict=True), args.taxonomy.resolve(strict=True))
    cases = build_cases(catalog)
    validate_cases(cases, catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "business-graph-gold-v1.authoring-catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gold_path = args.output_dir / "business-graph-gold-v1.jsonl"
    gold_path.write_text("".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    coverage = {}
    for case in cases:
        coverage[case["category"]] = coverage.get(case["category"], 0) + 1
    (args.output_dir / "business-graph-gold-v1.coverage.json").write_text(json.dumps({"case_count": len(cases), "categories": coverage}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "business-graph-gold-manifest/v1",
        "dataset_version": "business-graph-gold-v1",
        "case_count": len(cases),
        "source_db_sha256": catalog["source_db_sha256"],
        "graph_db_sha256": catalog["graph_db_sha256"],
        "taxonomy_sha256": catalog["taxonomy_sha256"],
        "catalog_sha256": sha256(args.output_dir / "business-graph-gold-v1.authoring-catalog.json"),
        "gold_sha256": sha256(gold_path),
        "window_config": catalog["window_config"],
        "coverage": coverage,
        "truth_policy": "positive evidence is independently validated from source DB; no SUT output is used to author Gold",
    }
    (args.output_dir / "business-graph-gold-v1.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "catalog_sessions": catalog["session_count"], "source_db_sha256": catalog["source_db_sha256"], "graph_db_sha256": catalog["graph_db_sha256"], "output": str(gold_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
