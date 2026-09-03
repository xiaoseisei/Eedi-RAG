from __future__ import annotations

"""SiliconFlow Qwen3-Reranker provider with strict response validation."""

import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_RERANKER_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


@dataclass(frozen=True)
class RerankUsage:
    request_count: int = 0
    retry_count: int = 0
    document_count: int = 0
    latency_ms: tuple[float, ...] = ()


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float
    document: str


class RerankerProviderError(RuntimeError):
    """Raised when the reranker cannot return a valid ordered result."""


class SiliconFlowQwen3Reranker:
    """OpenAI-compatible SiliconFlow ``/v1/rerank`` adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_RERANKER_BASE_URL,
        model: str = DEFAULT_QWEN3_RERANKER_MODEL,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("reranker API key must not be blank")
        if not base_url.strip():
            raise ValueError("reranker base_url must not be blank")
        if not model.strip() or model != model.strip():
            raise ValueError("reranker model must be non-blank and have no surrounding whitespace")
        if timeout_seconds <= 0:
            raise ValueError("reranker timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("reranker max_retries cannot be negative")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._client = client
        self._usage = RerankUsage()

    @classmethod
    def from_env(cls, *, client: Any | None = None) -> "SiliconFlowQwen3Reranker":
        key_ref = os.getenv("RERANKER_API_KEY_ENV", os.getenv("EMBEDDING_API_KEY_ENV", "SILICONFLOW_API_KEY")).strip()
        api_key = os.getenv(key_ref) if key_ref and not key_ref.startswith("sk-") else key_ref
        if not api_key:
            raise RerankerProviderError(
                "reranker API key is not configured; set RERANKER_API_KEY_ENV or reuse EMBEDDING_API_KEY_ENV"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("RERANKER_BASE_URL", os.getenv("EMBEDDING_BASE_URL", DEFAULT_RERANKER_BASE_URL)),
            model=os.getenv("RERANKER_MODEL", DEFAULT_QWEN3_RERANKER_MODEL),
            timeout_seconds=float(os.getenv("RERANKER_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("RERANKER_MAX_RETRIES", "2")),
            client=client,
        )

    def name(self) -> str:
        return f"siliconflow:{self.model}"

    @property
    def usage(self) -> RerankUsage:
        return self._usage

    @property
    def telemetry(self) -> dict[str, Any]:
        latencies = sorted(self._usage.latency_ms)
        if not latencies:
            p50 = p95 = None
        else:
            p50 = latencies[int((len(latencies) - 1) * 0.50)]
            p95 = latencies[int((len(latencies) - 1) * 0.95)]
        return {
            **self._usage.__dict__,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        }

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise RerankerProviderError("httpx is required for the SiliconFlow reranker") from exc
            self._client = httpx.Client(timeout=self.timeout_seconds)
        return self._client

    def _request(self, query: str, documents: Sequence[str], top_n: int) -> Any:
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "return_documents": False,
        }
        response = self._get_client().post(
            f"{self.base_url}/rerank",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        if hasattr(response, "json"):
            return response.json()
        if isinstance(response, dict):
            return response
        raise RerankerProviderError("reranker response has no JSON payload")

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not str(query).strip():
            raise ValueError("reranker query must not be blank")
        if not documents:
            return []
        if any(not str(document).strip() for document in documents):
            raise ValueError("reranker documents must not contain blank text")
        requested_top_n = len(documents) if top_n is None else int(top_n)
        if requested_top_n <= 0 or requested_top_n > len(documents):
            raise ValueError("reranker top_n must be in [1, len(documents)]")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                payload = self._request(str(query), [str(document) for document in documents], requested_top_n)
                raw_results = payload.get("results") if isinstance(payload, dict) else None
                if not isinstance(raw_results, list) or len(raw_results) < requested_top_n:
                    raise RerankerProviderError("reranker response results are missing or shorter than top_n")
                parsed: list[RerankResult] = []
                seen: set[int] = set()
                for item in raw_results[:requested_top_n]:
                    if not isinstance(item, dict):
                        raise RerankerProviderError("reranker result item must be an object")
                    index = int(item["index"])
                    score = float(item["relevance_score"])
                    if index < 0 or index >= len(documents) or index in seen:
                        raise RerankerProviderError(f"invalid or duplicate reranker result index: {index}")
                    if score != score or score in {float("inf"), float("-inf")}:
                        raise RerankerProviderError("reranker relevance_score must be finite")
                    seen.add(index)
                    parsed.append(RerankResult(index=index, relevance_score=score, document=str(documents[index])))
                elapsed = round((time.perf_counter() - started) * 1000.0, 3)
                self._usage = RerankUsage(
                    request_count=self._usage.request_count + 1,
                    retry_count=self._usage.retry_count,
                    document_count=self._usage.document_count + len(documents),
                    latency_ms=self._usage.latency_ms + (elapsed,),
                )
                return parsed
            except Exception as exc:
                last_error = exc if isinstance(exc, Exception) else Exception(str(exc))
                if attempt >= self.max_retries:
                    break
                self._usage = RerankUsage(
                    request_count=self._usage.request_count,
                    retry_count=self._usage.retry_count + 1,
                    document_count=self._usage.document_count,
                    latency_ms=self._usage.latency_ms,
                )
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RerankerProviderError(
            f"reranker failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error
