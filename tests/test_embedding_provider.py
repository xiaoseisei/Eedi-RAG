from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embedding_provider import (
    EmbeddingIndexContract,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingFunction,
    validate_index_contract,
)


def _response(vectors: list[list[float]], usage: object | None = None) -> object:
    return SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=vector) for i, vector in enumerate(vectors)],
        usage=usage,
    )


class _Embeddings:
    def __init__(self, response: object | Exception, *, adapt_count: bool = True) -> None:
        self.response = response
        self.adapt_count = adapt_count
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        if self.adapt_count and len(kwargs["input"]) != len(self.response.data):
            return _response([[1.0, 0.0] for _ in kwargs["input"]])
        return self.response


def test_provider_batches_and_validates_vectors() -> None:
    embeddings = _Embeddings(_response([[1.0, 0.0], [0.0, 1.0]]))
    provider = OpenAICompatibleEmbeddingFunction(
        api_key="secret",
        base_url="https://example.test/v1",
        model="Qwen/Qwen3-Embedding-0.6B",
        batch_size=2,
        client=SimpleNamespace(embeddings=embeddings),
    )
    assert [vector.tolist() for vector in provider(["a", "b", "c"])] == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert len(embeddings.calls) == 2
    assert embeddings.calls[0]["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert provider.dimension == 2


def test_provider_rejects_response_count_mismatch() -> None:
    embeddings = _Embeddings(_response([[1.0, 0.0]]), adapt_count=False)
    provider = OpenAICompatibleEmbeddingFunction(
        api_key="secret", base_url="https://example.test/v1", model="model", max_retries=0,
        client=SimpleNamespace(embeddings=embeddings),
    )
    with pytest.raises(EmbeddingProviderError, match="count mismatch"):
        provider(["a", "b"])


def test_provider_retries_then_raises_without_fallback() -> None:
    embeddings = _Embeddings(RuntimeError("upstream unavailable"))
    provider = OpenAICompatibleEmbeddingFunction(
        api_key="secret", base_url="https://example.test/v1", model="model", max_retries=1,
        retry_backoff_seconds=0,
        client=SimpleNamespace(embeddings=embeddings),
    )
    with pytest.raises(EmbeddingProviderError, match="after 2 attempts"):
        provider(["a"])
    assert len(embeddings.calls) == 2


def test_from_env_accepts_legacy_direct_key_without_exposing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_KEY_ENV", "sk-test-secret")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    provider = OpenAICompatibleEmbeddingFunction.from_env(client=SimpleNamespace())
    assert provider.model == "Qwen/Qwen3-Embedding-0.6B"
    assert provider.api_key == "sk-test-secret"


def test_index_contract_rejects_old_or_mismatched_index() -> None:
    contract = EmbeddingIndexContract("openai_compatible:Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", 1024, "qwen3-embedding-0.6b-v1")
    with pytest.raises(EmbeddingProviderError, match="contract mismatch"):
        validate_index_contract({"embedding_model": "Qwen/Qwen3-Embedding-0.6B"}, contract)


def test_provider_contract_requires_explicit_index_version() -> None:
    provider = OpenAICompatibleEmbeddingFunction(
        api_key="secret", base_url="https://example.test/v1", model="model",
        expected_dimension=2, client=SimpleNamespace(embeddings=_Embeddings(_response([[1.0, 0.0]]))),
    )
    with pytest.raises(EmbeddingProviderError, match="index_version"):
        provider.contract()


def test_provider_caches_only_successful_vectors_with_full_contract_key() -> None:
    shared_cache: dict[str, list[float]] = {}
    embeddings = _Embeddings(_response([[1.0, 0.0]]))
    client = SimpleNamespace(embeddings=embeddings)
    provider = OpenAICompatibleEmbeddingFunction(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
        index_version="index-v1",
        expected_dimension=2,
        client=client,
        cache=shared_cache,
    )

    assert [v.tolist() for v in provider(["same query", "same query"])] == [[1.0, 0.0], [1.0, 0.0]]
    assert [v.tolist() for v in provider(["same query"])] == [[1.0, 0.0]]
    assert len(embeddings.calls) == 1
    assert provider.usage.cache_hit_count == 2

    other_version = OpenAICompatibleEmbeddingFunction(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
        index_version="index-v2",
        expected_dimension=2,
        client=client,
        cache=shared_cache,
    )
    other_version(["same query"])
    assert len(embeddings.calls) == 2


def test_provider_does_not_cache_failed_requests() -> None:
    shared_cache: dict[str, list[float]] = {}
    provider = OpenAICompatibleEmbeddingFunction(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model-a",
        index_version="index-v1",
        max_retries=0,
        client=SimpleNamespace(embeddings=_Embeddings(RuntimeError("upstream unavailable"))),
        cache=shared_cache,
    )

    with pytest.raises(EmbeddingProviderError):
        provider(["do not cache me"])

    assert shared_cache == {}
