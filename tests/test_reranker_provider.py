from __future__ import annotations

import pytest

from src.reranker_provider import SiliconFlowQwen3Reranker


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.payload)


def test_reranker_parses_and_orders_siliconflow_results() -> None:
    client = _Client({"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]})
    provider = SiliconFlowQwen3Reranker(api_key="test", client=client, max_retries=0)
    results = provider.rerank("q", ["a", "b"], top_n=2)
    assert [result.index for result in results] == [1, 0]
    assert results[0].document == "b"
    assert client.calls[0][0].endswith("/rerank")
    assert provider.telemetry["request_count"] == 1


def test_reranker_rejects_invalid_result_indices() -> None:
    client = _Client({"results": [{"index": 3, "relevance_score": 0.9}]})
    provider = SiliconFlowQwen3Reranker(api_key="test", client=client, max_retries=0)
    with pytest.raises(Exception):
        provider.rerank("q", ["a"], top_n=1)


def test_reranker_requires_non_blank_inputs() -> None:
    provider = SiliconFlowQwen3Reranker(api_key="test", client=_Client({"results": []}), max_retries=0)
    with pytest.raises(ValueError):
        provider.rerank("", ["a"])
    with pytest.raises(ValueError):
        provider.rerank("q", [""])
