from __future__ import annotations

"""OpenAI-compatible embedding providers used by the RAG index.

The provider deliberately has no deterministic or zero-vector fallback.  An
embedding failure must stop index construction or retrieval rather than create
an apparently healthy but semantically invalid index.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, MutableMapping

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@dataclass(frozen=True)
class EmbeddingUsage:
    request_count: int = 0
    retry_count: int = 0
    input_count: int = 0
    cache_hit_count: int = 0
    input_tokens: int = 0
    total_tokens: int = 0
    request_latency_ms: tuple[float, ...] = ()


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding request cannot produce a valid vector batch."""


@dataclass(frozen=True)
class EmbeddingIndexContract:
    """Identity that must match on both document and query embedding paths."""

    provider: str
    model: str
    dimension: int
    index_version: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip() or not self.index_version.strip():
            raise ValueError("embedding contract provider, model and index_version must be non-blank")
        if self.dimension <= 0:
            raise ValueError("embedding contract dimension must be positive")


def validate_index_contract(metadata: dict[str, Any] | None, contract: EmbeddingIndexContract) -> None:
    """Fail closed when a Chroma index was built by a different embedding contract."""
    metadata = metadata or {}
    expected = {
        "embedding_provider": contract.provider,
        "embedding_model": contract.model,
        "embedding_dimension": contract.dimension,
        "embedding_index_version": contract.index_version,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise EmbeddingProviderError(f"embedding index contract mismatch: {mismatches}")


class OpenAICompatibleEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible adapter for an OpenAI ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        batch_size: int = 32,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        client: Any | None = None,
        expected_dimension: int | None = None,
        index_version: str | None = None,
        cache: MutableMapping[str, list[float]] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("embedding API key must not be blank")
        if not base_url.strip():
            raise ValueError("embedding base_url must not be blank")
        if not model.strip() or model != model.strip():
            raise ValueError("embedding model must be non-blank and must not contain surrounding whitespace")
        if timeout_seconds <= 0:
            raise ValueError("embedding timeout_seconds must be positive")
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        if max_retries < 0:
            raise ValueError("embedding max_retries cannot be negative")
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("embedding expected_dimension must be positive")
        if index_version is not None and (not index_version.strip() or index_version != index_version.strip()):
            raise ValueError("embedding index_version must be non-blank and have no surrounding whitespace")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._client = client
        self._usage = EmbeddingUsage()
        self._dimension: int | None = None
        self.expected_dimension = expected_dimension
        self.index_version = index_version
        self._cache: MutableMapping[str, list[float]] = cache if cache is not None else {}

    @classmethod
    def from_env(cls, *, client: Any | None = None) -> "OpenAICompatibleEmbeddingFunction":
        """Build from environment without logging or returning secret values.

        ``EMBEDDING_API_KEY_ENV`` is intended to contain an environment
        variable name.  For backwards compatibility with the current local
        ``.env`` it may also contain the key value itself; that value is used
        only in memory and never emitted by this module.
        """

        key_ref = os.getenv("EMBEDDING_API_KEY_ENV", "SILICONFLOW_API_KEY").strip()
        api_key = os.getenv(key_ref) if key_ref and not key_ref.startswith("sk-") else key_ref
        if not api_key:
            raise EmbeddingProviderError(
                "embedding API key is not configured; set EMBEDDING_API_KEY_ENV to an env var name "
                "or configure the referenced variable"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("EMBEDDING_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL),
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_QWEN3_EMBEDDING_MODEL),
            timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            client=client,
        )

    def name(self) -> str:
        return f"openai_compatible:{self.model}"

    @property
    def dimension(self) -> int | None:
        return self._dimension or self.expected_dimension

    def contract(self, *, index_version: str | None = None) -> EmbeddingIndexContract:
        version = index_version or self.index_version
        if not version:
            raise EmbeddingProviderError("embedding index_version is required for an index contract")
        dimension = self.dimension
        if dimension is None:
            raise EmbeddingProviderError("embedding dimension is unknown until a real vector response is received")
        return EmbeddingIndexContract(self.name(), self.model, dimension, version)

    @property
    def usage(self) -> EmbeddingUsage:
        return self._usage

    @property
    def telemetry(self) -> dict[str, Any]:
        latencies = sorted(self._usage.request_latency_ms)

        def percentile(fraction: float) -> float | None:
            if not latencies:
                return None
            index = max(0, min(len(latencies) - 1, int((len(latencies) - 1) * fraction + 0.999999)))
            return round(latencies[index], 3)

        return {
            **self._usage.__dict__,
            "request_latency_p50_ms": percentile(0.50),
            "request_latency_p95_ms": percentile(0.95),
        }

    def _cache_key(self, text: str) -> str:
        identity = {
            "provider": self.name(),
            "model": self.model,
            "index_version": self.index_version or "",
            "query_text": text,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def __call__(self, input: Documents) -> Embeddings:
        texts = [str(item) for item in input]
        if not texts:
            return []

        positions_by_key: dict[str, list[int]] = {}
        text_by_key: dict[str, str] = {}
        for index, item in enumerate(texts):
            key = self._cache_key(item)
            positions_by_key.setdefault(key, []).append(index)
            text_by_key.setdefault(key, item)

        vectors_by_key: dict[str, list[float]] = {}
        cache_hits = 0
        missing_keys: list[str] = []
        for key, positions in positions_by_key.items():
            cached = self._cache.get(key)
            if cached is not None:
                vectors_by_key[key] = list(cached)
                cache_hits += len(positions)
            else:
                missing_keys.append(key)
                # Repeated identical inputs in the same call are deduplicated.
                cache_hits += max(0, len(positions) - 1)

        if cache_hits:
            self._usage = EmbeddingUsage(
                request_count=self._usage.request_count,
                retry_count=self._usage.retry_count,
                input_count=self._usage.input_count,
                cache_hit_count=self._usage.cache_hit_count + cache_hits,
                input_tokens=self._usage.input_tokens,
                total_tokens=self._usage.total_tokens,
                request_latency_ms=self._usage.request_latency_ms,
            )

        for offset in range(0, len(missing_keys), self.batch_size):
            batch_keys = missing_keys[offset : offset + self.batch_size]
            batch_vectors = self._embed_batch([text_by_key[key] for key in batch_keys])
            for key, vector in zip(batch_keys, batch_vectors):
                copied = list(vector)
                vectors_by_key[key] = copied
                # Cache only vectors returned by a fully validated successful request.
                self._cache[key] = copied

        vectors = [list(vectors_by_key[self._cache_key(item)]) for item in texts]
        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                f"embedding provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingProviderError("openai package is required for API embeddings") from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))
            try:
                response = self._get_client().embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float",
                )
                vectors = self._validate_response(response, expected_count=len(texts))
                usage = getattr(response, "usage", None)
                self._usage = EmbeddingUsage(
                    request_count=self._usage.request_count + 1,
                    retry_count=self._usage.retry_count + attempt,
                    input_count=self._usage.input_count + len(texts),
                    cache_hit_count=self._usage.cache_hit_count,
                    input_tokens=self._usage.input_tokens + int(getattr(usage, "prompt_tokens", 0) or 0),
                    total_tokens=self._usage.total_tokens + int(getattr(usage, "total_tokens", 0) or 0),
                    request_latency_ms=self._usage.request_latency_ms + (
                        round((time.perf_counter() - started) * 1000.0, 3),
                    ),
                )
                return vectors
            except Exception as exc:  # wrapped and re-raised after bounded retries
                last_error = exc
        self._usage = EmbeddingUsage(
            request_count=self._usage.request_count + 1,
            retry_count=self._usage.retry_count + self.max_retries,
            input_count=self._usage.input_count + len(texts),
            cache_hit_count=self._usage.cache_hit_count,
            input_tokens=self._usage.input_tokens,
            total_tokens=self._usage.total_tokens,
            request_latency_ms=self._usage.request_latency_ms + (
                round((time.perf_counter() - started) * 1000.0, 3),
            ),
        )
        raise EmbeddingProviderError(
            f"embedding request failed after {self.max_retries + 1} attempts for model {self.model}: {last_error}"
        ) from last_error

    def _validate_response(self, response: Any, *, expected_count: int) -> list[list[float]]:
        data = list(getattr(response, "data", None) or [])
        if len(data) != expected_count:
            raise EmbeddingProviderError(
                f"embedding response count mismatch: expected {expected_count}, got {len(data)}"
            )
        indexed = sorted(data, key=lambda item: int(getattr(item, "index", 0)))
        indices = [int(getattr(item, "index", -1)) for item in indexed]
        if indices != list(range(expected_count)):
            raise EmbeddingProviderError(
                f"embedding response indexes are not contiguous: expected 0..{expected_count - 1}, got {indices}"
            )
        vectors: list[list[float]] = []
        for item in indexed:
            raw = list(getattr(item, "embedding", None) or [])
            if not raw:
                raise EmbeddingProviderError("embedding response contained an empty vector")
            vector = [float(value) for value in raw]
            if not all(value == value and abs(value) != float("inf") for value in vector):
                raise EmbeddingProviderError("embedding response contained NaN or infinite values")
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise EmbeddingProviderError(
                    f"embedding dimension mismatch: expected {self._dimension}, got {len(vector)}"
                )
            if self.expected_dimension is not None and len(vector) != self.expected_dimension:
                raise EmbeddingProviderError(
                    f"embedding dimension mismatch: expected {self.expected_dimension}, got {len(vector)}"
                )
            vectors.append(vector)
        return vectors


class SiliconFlowQwen3EmbeddingFunction(OpenAICompatibleEmbeddingFunction):
    """Named provider for the initial SiliconFlow Qwen3 text embedding model."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", DEFAULT_SILICONFLOW_BASE_URL)
        kwargs.setdefault("model", DEFAULT_QWEN3_EMBEDDING_MODEL)
        kwargs.setdefault("expected_dimension", 1024)
        kwargs.setdefault("index_version", "qwen3-embedding-0.6b-v1")
        super().__init__(**kwargs)
