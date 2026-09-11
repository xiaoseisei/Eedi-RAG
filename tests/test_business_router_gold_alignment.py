from __future__ import annotations

from src.business_models import BusinessQueryRequest
from src.query_intent import BusinessQueryRouter


def _route(query: str) -> tuple[str, str, str, list[str]]:
    understanding = BusinessQueryRouter().route(BusinessQueryRequest(query=query))
    return (
        understanding.primary_intent.value,
        understanding.scope.value,
        understanding.computation_scope.value,
        [item.value for item in understanding.evidence_perspectives],
    )


def test_gold_micro_prefixes_select_lane_and_case_scope() -> None:
    assert _route("围绕真实题目进行micro student error分析。") == (
        "QUESTION_TUTORING", "CASE", "MICRO", ["STUDENT"]
    )
    assert _route("导师如何处理 Which is the next prime number greater than 100？") == (
        "QUESTION_TUTORING", "QUESTION", "MICRO", ["TUTOR"]
    )
    assert _route("围绕真实题目进行mixed misconception intervention分析。") == (
        "QUESTION_TUTORING", "CASE", "MICRO", ["STUDENT", "TUTOR"]
    )


def test_gold_mixed_macro_and_cross_session_routes_are_analytics() -> None:
    assert _route("统计学生最常见的舍入误区，并给出课堂原声证据。") == (
        "RECURRING_MISCONCEPTIONS", "CORPUS", "HYBRID", ["STUDENT"]
    )
    assert _route("围绕真实题目进行mixed macro graph micro分析。") == (
        "RECURRING_MISCONCEPTIONS", "CORPUS", "HYBRID", ["STUDENT", "TUTOR"]
    )
    assert _route("围绕真实题目进行mixed cross session pattern分析。") == (
        "CONTENT_IMPROVEMENT", "CORPUS", "HYBRID", ["STUDENT", "TUTOR"]
    )


def test_gold_macro_synonyms_and_preconditions_are_not_silently_unsupported() -> None:
    assert _route("哪些提问方式在辅导记录中出现最多？") == (
        "TUTOR_STRATEGY_PATTERNS", "CORPUS", "HYBRID", ["TUTOR"]
    )
    assert _route("哪些错因值得优先纳入错题讲解？") == (
        "RECURRING_MISCONCEPTIONS", "CORPUS", "HYBRID", ["STUDENT"]
    )
    assert _route("在图谱 manifest 与源库哈希不一致时执行该统计。") == (
        "RECURRING_MISCONCEPTIONS", "CORPUS", "HYBRID", ["STUDENT"]
    )
    assert _route("这些学生重考后分数是否提高？") == (
        "STUDENT_COHORT_PATTERNS", "CORPUS", "REJECT", ["STUDENT"]
    )
