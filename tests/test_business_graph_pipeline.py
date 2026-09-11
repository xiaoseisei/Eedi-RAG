from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from src.business_graph_pipeline import BusinessGraphPipeline, BusinessGraphRuntimeConfig
from src.business_graph_trace import StageStatus
from src.business_models import BusinessQueryRequest, BusinessResponseStatus


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = ROOT / "data/db/tutoring_knowledge.duckdb"
GRAPH_DB = ROOT / "reports/staging/session-card-graph-v1/graph-taxonomy-verify-2-20260908.duckdb"


def _pipeline() -> BusinessGraphPipeline:
    return BusinessGraphPipeline(
        source_db=SOURCE_DB,
        graph_db=GRAPH_DB,
        chroma_dir=ROOT / "data/chroma",
        runtime_config=BusinessGraphRuntimeConfig(
            source_sha256="f089c56f162666ab71b3beecef33a871f3e140608721b3f61f96f9ed384b6157",
            model_reranker_available=False,
            generator_available=False,
        ),
    )


def test_micro_route_records_router_planner_card_w7_and_prompt_stages() -> None:
    trace = _pipeline().prepare_case("学生为什么会把 5.4598 四舍五入成 5.45？")

    assert trace.actual_router["intent"] == "QUESTION_TUTORING"
    assert trace.actual_plan["plan_type"] == "SOCRATIC_TUTORING"
    assert trace.card_candidates
    assert trace.candidate_windows
    assert all(len(window["source_turn_ids"]) <= 7 for window in trace.candidate_windows)
    assert all(window["source_turn_ids"] for window in trace.candidate_windows)
    assert trace.stage_status["ROUTER"] is StageStatus.SUCCESS
    assert trace.stage_status["PLANNER"] is StageStatus.SUCCESS
    assert trace.stage_status["CARD"] is StageStatus.SUCCESS
    assert trace.stage_status["WINDOW"] is StageStatus.UNMEASURED
    assert trace.stage_status["PROMPT"] is StageStatus.SUCCESS


def test_macro_route_uses_graph_analytics_not_vector_ranking_fact() -> None:
    trace = _pipeline().prepare_case("学生最常见的共性误区是什么？")

    assert trace.actual_plan["plan_type"] == "ANALYTICS_WITH_EVIDENCE"
    assert trace.analytics_result is not None
    assert trace.analytics_result["operation"] == "recurring_misconceptions"
    assert trace.analytics_result["rows"]
    assert "vector_score" not in trace.analytics_result["rows"][0]


def test_cross_session_content_query_uses_cooccurrence_operation() -> None:
    trace = _pipeline().prepare_case("学生误区和导师策略在不同课堂中如何共现？")

    assert trace.actual_router["intent"] == "CONTENT_IMPROVEMENT"
    assert trace.actual_plan["selected_lever"]["id"] == "graph_sql_with_evidence"
    assert trace.analytics_result is not None
    assert trace.analytics_result["operation"] == "misconception_strategy_pairs"


def test_unsupported_cohort_query_stops_before_evidence_generation() -> None:
    trace = _pipeline().prepare_case("这些学生重考后分数是否提高？")

    assert trace.actual_plan["plan_type"] == "REJECT"
    assert trace.stage_status["PLANNER"] is StageStatus.UNSUPPORTED
    assert trace.card_candidates == []
    assert trace.candidate_windows == []
    assert trace.citations == []


def test_planner_unmeasured_rejection_propagates_as_unmeasured() -> None:
    pipeline = _pipeline()
    try:
        understanding = pipeline.router.route(
            BusinessQueryRequest(query="那个卡的时间最长")
        )
        # Keep this test independent of classifier wording: inject the
        # orthogonal metric that Planner must refuse to guess.
        understanding = understanding.model_copy(update={"ranking_signal": "duration_desc"})
        trace = pipeline.prepare_case(
            understanding.raw_query, request=BusinessQueryRequest(query=understanding.raw_query), understanding=understanding
        )
        pipeline.router.route = Mock(return_value=understanding)
        response = pipeline.ask_business(BusinessQueryRequest(query=understanding.raw_query))
    finally:
        pipeline.close()

    assert trace.actual_plan["plan_type"] == "REJECT"
    assert trace.actual_plan["rejection_reason"]
    assert trace.stage_status["PLANNER"] is StageStatus.UNMEASURED
    assert trace.stage_status["GRAPH"] is StageStatus.UNMEASURED
    assert response.status is BusinessResponseStatus.UNMEASURED


def test_unresolved_intent_stops_at_router_as_unmeasured() -> None:
    pipeline = _pipeline()
    try:
        trace = pipeline.prepare_case("重考学生最常问什么？")
    finally:
        pipeline.close()

    assert trace.actual_router["intent"] == "UNRESOLVED"
    assert trace.actual_router["routing_status"] in {"NEED_LLM", "UNMEASURED"}
    assert trace.stage_status["ROUTER"] is StageStatus.UNMEASURED
    assert trace.stage_status["PLANNER"] is StageStatus.UNMEASURED
    assert trace.errors[0]["code"] in {"ROUTING_NEED_LLM", "INTENT_CLASSIFICATION_UNMEASURED"}
    assert trace.card_candidates == []
    assert trace.citations == []


def test_class_review_prefers_student_card_when_router_requests_both_lanes() -> None:
    trace = _pipeline().prepare_case("请回顾这节关于 5.4598 四舍五入的课堂。")

    assert trace.parent_cards[0]["chunk_id"] == "session_10_misconception"
    assert trace.selected_windows[0]["window_id"] == "session_10_win_1_7"


def test_mixed_intervention_keeps_both_real_card_lanes_in_one_session() -> None:
    trace = _pipeline().prepare_case("学生的舍入误区是什么，导师如何干预？")

    assert {card["chunk_id"] for card in trace.parent_cards} >= {
        "session_10_misconception",
        "session_10_tutor_strategy",
    }
    assert trace.selected_windows[0]["window_id"] == "session_10_win_4_10"


def test_spaced_formula_tokens_keep_the_lowest_matching_source_session() -> None:
    trace = _pipeline().prepare_case("请回顾这节关于 5.4598 四舍五入的课堂。")

    assert trace.parent_cards[0]["chunk_id"] == "session_10_misconception"


def test_micro_question_text_prefix_resolves_card_without_subject_path_exactness() -> None:
    trace = _pipeline().prepare_case(
        "导师如何处理 Which is the next prime number greater than 100？"
    )

    assert trace.parent_cards[0]["chunk_id"] == "session_59_tutor_strategy"


def test_wrapped_truncated_question_prefix_resolves_the_real_card() -> None:
    trace = _pipeline().prepare_case(
        "围绕真实题目“Lee and Drew are arguing about equivalent ratios.\nLee says \\( 4: 14 \\) is equiva”进行micro class review分析。"
    )

    assert trace.parent_cards[0]["chunk_id"] == "session_109_misconception"


def test_macro_difficulty_representative_turns_are_role_filtered() -> None:
    trace = _pipeline().prepare_case("哪些概念最难理解？")

    assert trace.stage_status["CARD"] is StageStatus.SUCCESS
    assert trace.errors == []
    assert all(citation["speaker"] == "student" for citation in trace.citations)


def test_cross_session_summary_does_not_invent_micro_window_evidence() -> None:
    trace = _pipeline().prepare_case("跨课堂比较相同学生误区及导师策略。")

    assert trace.analytics_result is not None
    assert trace.analytics_result["operation"] == "misconception_strategy_pairs"
    assert trace.card_candidates == []
    assert trace.selected_windows == []
    assert "跨 Session 统计" in trace.assembled_prompt


def test_wrapped_question_uses_card_anchor_before_numeric_tokens_for_window_selection() -> None:
    trace = _pipeline().prepare_case(
        "围绕真实题目“The median of these four numbers is \\( 5 \\).”进行mixed misconception intervention分析。"
    )

    assert trace.selected_windows[0]["window_id"] == "session_555_win_1_7"


def test_cross_class_summary_uses_cooccurrence_analytics_without_overlay() -> None:
    trace = _pipeline().prepare_case("跨课堂比较相同学生误区及导师策略。")

    assert trace.analytics_result is not None
    assert trace.analytics_result["operation"] == "misconception_strategy_pairs"
    # The co-occurrence contract requires these scope columns to be declared.
    # They are a declared superset, not a wording-dependent replacement, so the
    # assertion pins the contract rather than an exact tuned list.
    keywords = trace.analytics_result["sql_scope"]["keywords"]
    for required in ("source_session_id", "canonical_label", "CO_OCCURS_WITH"):
        assert required in keywords
    assert trace.card_candidates == []
    assert trace.selected_windows == []


def test_reasoning_query_selects_first_window_after_question_specific_tutor_prompt() -> None:
    trace = _pipeline().prepare_case("这道题的选项和学生推理过程反映了什么？")

    assert trace.selected_windows[0]["window_id"] == "session_10_win_4_10"
    assert any(citation["turn_id"] == 8 for citation in trace.citations)


def test_macro_classroom_evidence_selects_card_quote_window() -> None:
    trace = _pipeline().prepare_case("统计学生最常见的舍入误区，并给出课堂原声证据。")

    assert trace.selected_windows[0]["window_id"] == "session_10_win_4_10"
    assert any(citation["turn_id"] == 10 for citation in trace.citations)


def test_duplicate_question_text_keeps_a_stable_source_order() -> None:
    trace = _pipeline().prepare_case(
        "围绕真实题目“This pie chart shows how the employees of a company travel to work.\n\nWhich of th”进行mixed misconception intervention分析。"
    )

    assert trace.parent_cards
    assert trace.parent_cards[0]["metadata"]["question_id"] == 76888


def test_duplicate_question_text_with_session_hint_resolves_the_requested_session() -> None:
    pipeline = _pipeline()
    try:
        trace = pipeline.prepare_case(
            "Session 522：围绕真实题目“This pie chart shows how the employees of a company travel to work.\n\nWhich of th”进行mixed misconception intervention分析。"
        )
    finally:
        pipeline.close()

    assert trace.parent_cards
    assert trace.parent_cards[0]["metadata"]["session_id"] == 522
    assert trace.analytics_result is not None
    assert trace.analytics_result["sql_scope"]["filters"]["session_id"] == 522


def test_classroom_evidence_window_uses_first_numeric_student_quote_anchor() -> None:
    trace = _pipeline().prepare_case("统计学生最常见的舍入误区，并给出课堂原声证据。")

    assert trace.selected_windows[0]["window_id"] == "session_10_win_4_10"


def test_ask_business_builds_an_audited_response_envelope() -> None:
    pipeline = _pipeline()
    try:
        response = pipeline.ask_business(
            BusinessQueryRequest(query="学生为什么会把 5.4598 四舍五入成 5.45？")
        )
    finally:
        pipeline.close()

    assert response.status is BusinessResponseStatus.SUCCESS
    assert response.understanding.raw_query == "学生为什么会把 5.4598 四舍五入成 5.45？"
    assert response.result is not None
    assert response.result.payload["window_ids"]
    assert response.evidence
    assert response.audit_status == "AUDITED_100_VERIFIED"
    assert response.data_scope.dataset_version == "business-graph-v1"


def test_ask_business_unsupported_response_has_no_result_or_evidence() -> None:
    pipeline = _pipeline()
    try:
        response = pipeline.ask_business(
            BusinessQueryRequest(query="这些学生重考后分数是否提高？")
        )
    finally:
        pipeline.close()

    assert response.status is BusinessResponseStatus.UNSUPPORTED
    assert response.result is None
    assert response.evidence == []
    assert response.audit_status == "NOT_APPLICABLE"


def test_ask_business_preserves_request_filters_and_top_k_in_execution_scope() -> None:
    pipeline = _pipeline()
    try:
        response = pipeline.ask_business(
            BusinessQueryRequest(
                query="哪些概念最难理解？",
                subject_filters=["Number"],
                top_k=1,
            )
        )
    finally:
        pipeline.close()

    assert response.understanding.subject_filters == ["Number"]
    assert response.data_scope.subject_filters == ["Number"]
    assert response.result is not None
    assert len(response.result.payload["rows"]) == 1


def test_ask_business_routes_the_request_once() -> None:
    pipeline = _pipeline()
    route = Mock(wraps=pipeline.router.route)
    pipeline.router.route = route
    try:
        pipeline.ask_business(
            BusinessQueryRequest(query="学生为什么会把 5.4598 四舍五入成 5.45？")
        )
    finally:
        pipeline.close()

    assert route.call_count == 1


def test_ask_business_unmeasured_routing_short_circuits_and_reports_router_stage() -> None:
    pipeline = _pipeline()
    try:
        response = pipeline.ask_business(
            BusinessQueryRequest(query="重考学生最常问什么？")
        )
    finally:
        pipeline.close()

    assert response.status is BusinessResponseStatus.UNMEASURED
    assert response.result is None
    assert response.evidence == []
    assert response.audit_status == "NOT_APPLICABLE"
    assert any("ROUTER" in limit for limit in response.limitations)
