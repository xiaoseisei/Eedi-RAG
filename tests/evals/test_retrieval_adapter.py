from __future__ import annotations

from evals.adapters.retrieval import capture_legacy_retrieval_cases


class _FakeRetriever:
    def retrieve_misconceptions(self, query_text, *, top_k, fetch_evidence):
        assert fetch_evidence is False
        return [{"chunk_id": "session_10_misconception"}]

    def retrieve_strategies(self, query_text, *, top_k, fetch_evidence):
        assert fetch_evidence is False
        return [{"chunk_id": "session_23_tutor_strategy"}]

    def retrieve_multi_perspective_rrf(self, raw_query, *, top_k_each, fetch_evidence):
        assert fetch_evidence is False
        return {
            "misconceptions": [{"chunk_id": "session_10_misconception"}],
            "strategies": [{"chunk_id": "session_23_tutor_strategy"}],
        }


def test_legacy_retrieval_adapter_keeps_target_as_qrels_not_sut_input() -> None:
    cases = capture_legacy_retrieval_cases(_FakeRetriever(), mode="raw", limit=1)

    assert len(cases) == 1
    assert cases[0].qrels == {"session_10_misconception": 3}
    assert cases[0].ranked_ids == ["session_10_misconception"]
    assert cases[0].slices["mode"] == "raw"

