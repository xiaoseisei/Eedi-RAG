import json
import os
from types import SimpleNamespace

import pytest

from src.business_models import (
    BusinessQueryRequest,
    QueryIntent,
    RouteConfidence,
)
from src.intent_classifier import (
    IntentClassification,
    OpenAICompatibleIntentClassifier,
)
from src.query_intent import BusinessQueryRouter


class FakeCompletions:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        content = json.dumps(self.payload, ensure_ascii=False)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, payload: dict):
        self.chat = SimpleNamespace(completions=FakeCompletions(payload))


def test_clear_rule_route_does_not_call_second_layer():
    client = FakeClient(
        {
            "primary_intent": "CONTENT_IMPROVEMENT",
            "secondary_intent": None,
            "confidence": 0.95,
            "rationale": "unused",
        }
    )
    classifier = OpenAICompatibleIntentClassifier(client, model="test")
    result = BusinessQueryRouter(classifier).route(
        BusinessQueryRequest(query="学生在辅导中最常提出的问题是什么？")
    )
    assert result.primary_intent == QueryIntent.STUDENT_FREQUENT_QUESTIONS
    assert result.route_source == "deterministic"
    assert client.chat.completions.calls == 0


def test_low_confidence_route_uses_second_layer_and_preserves_contract():
    client = FakeClient(
        {
            "primary_intent": "CONTENT_IMPROVEMENT",
            "secondary_intent": "TUTOR_STRATEGY_PATTERNS",
            "confidence": 0.91,
            "rationale": "The query asks for intervention-oriented content changes.",
        }
    )
    result = BusinessQueryRouter(
        OpenAICompatibleIntentClassifier(client, model="test")
    ).route(BusinessQueryRequest(query="哪些做法值得纳入后续教学？"))
    assert result.primary_intent == QueryIntent.CONTENT_IMPROVEMENT
    assert result.secondary_intents == [QueryIntent.TUTOR_STRATEGY_PATTERNS]
    assert result.route_source == "llm_classifier"
    assert result.confidence == RouteConfidence.HIGH
    assert client.chat.completions.calls == 1


def test_generic_keyword_without_strong_intent_uses_second_layer():
    client = FakeClient(
        {
            "primary_intent": "TUTOR_STRATEGY_PATTERNS",
            "secondary_intent": None,
            "confidence": 0.88,
            "rationale": "The wording asks about teaching practice.",
        }
    )
    result = BusinessQueryRouter(
        OpenAICompatibleIntentClassifier(client, model="test")
    ).route(BusinessQueryRequest(query="导师在课堂里通常怎么做？"))
    assert result.primary_intent == QueryIntent.TUTOR_STRATEGY_PATTERNS
    assert result.route_source == "llm_classifier"
    assert client.chat.completions.calls == 1


def test_classifier_rejects_intent_outside_candidate_set():
    client = FakeClient(
        {
            "primary_intent": "UNSUPPORTED",
            "secondary_intent": None,
            "confidence": 0.9,
            "rationale": "invalid candidate",
        }
    )
    classifier = OpenAICompatibleIntentClassifier(client, model="test")
    with pytest.raises(RuntimeError, match="strict validation"):
        classifier.classify("opaque query", [QueryIntent.CONTENT_IMPROVEMENT])


def test_unresolved_rule_route_is_not_reported_as_unsupported():
    result = BusinessQueryRouter().route(
        BusinessQueryRequest(query="重考学生最常问什么？")
    )

    assert result.primary_intent == QueryIntent.UNRESOLVED
    assert result.routing_status in {"NEED_LLM", "UNMEASURED"}
    assert result.ambiguity_detected is True


def test_intent_classifier_from_env_reuses_siliconflow_connection(monkeypatch):
    client = FakeClient(
        {
            "primary_intent": "CONTENT_IMPROVEMENT",
            "secondary_intent": None,
            "confidence": 0.9,
            "rationale": "test",
        }
    )
    monkeypatch.setenv("EMBEDDING_API_KEY_ENV", "TEST_SILICONFLOW_KEY")
    monkeypatch.setenv("TEST_SILICONFLOW_KEY", "secret-for-test")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("INTENT_MODEL", "Qwen/Qwen3.5-4B")

    classifier = OpenAICompatibleIntentClassifier.from_env(client=client)

    assert classifier.model == "Qwen/Qwen3.5-4B"
    assert classifier.client is client


def test_try_from_env_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("INTENT_MODEL", raising=False)
    # Ensure try_from_env returns None without raising
    result = OpenAICompatibleIntentClassifier.try_from_env(model=None)
    assert result is None


def test_try_from_env_raises_when_configured_but_key_missing(monkeypatch):
    monkeypatch.setenv("INTENT_MODEL", "Qwen/Qwen3.5-4B")
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("INTENT_API_KEY_ENV", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY_ENV", raising=False)

    # When configured, errors MUST NOT be swallowed into None
    with pytest.raises(ValueError, match="Intent classifier API key is not configured"):
        OpenAICompatibleIntentClassifier.try_from_env()


def test_try_from_env_succeeds_when_properly_configured(monkeypatch):
    client = FakeClient(
        {
            "primary_intent": "CONTENT_IMPROVEMENT",
            "secondary_intent": None,
            "confidence": 0.9,
            "rationale": "test",
        }
    )
    monkeypatch.setenv("INTENT_MODEL", "Qwen/Qwen3.5-4B")
    classifier = OpenAICompatibleIntentClassifier.try_from_env(client=client)
    assert classifier is not None
    assert classifier.model == "Qwen/Qwen3.5-4B"

