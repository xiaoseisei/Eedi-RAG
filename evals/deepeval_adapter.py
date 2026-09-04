from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from evals.contracts import DeepEvalCasePayload, DeepEvalReadiness, DeepEvalSettings, MetricStatus


class DeepEvalUnavailableError(RuntimeError):
    pass


class OpenAICompatibleDeepEvalModel:
    """Minimal DeepEvalBaseLLM adapter for an explicit OpenAI-compatible judge."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str | None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model_name.strip() or not api_key.strip():
            raise ValueError("DeepEval judge model and API key are required")
        if timeout_seconds <= 0:
            raise ValueError("DeepEval judge timeout_seconds must be positive")
        try:
            from deepeval.models.base_model import DeepEvalBaseLLM
        except ImportError as exc:
            raise DeepEvalUnavailableError("DeepEval is not installed") from exc

        class _Adapter(DeepEvalBaseLLM):
            def __init__(self, outer: "OpenAICompatibleDeepEvalModel") -> None:
                self._outer = outer
                super().__init__(outer.model_name)

            def load_model(self):
                return self

            def generate(self, prompt: str, schema: Any | None = None, **kwargs: Any) -> str:
                return self._outer._generate(prompt, schema=schema, **kwargs)

            async def a_generate(self, prompt: str, schema: Any | None = None, **kwargs: Any) -> str:
                return await asyncio.to_thread(self._outer._generate, prompt, schema=schema, **kwargs)

            def get_model_name(self, *args: Any, **kwargs: Any) -> str:
                return self._outer.model_name

        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = float(timeout_seconds)
        self._adapter = _Adapter(self)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DeepEvalUnavailableError("openai package is required for the DeepEval judge") from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout_seconds,
        )

    @property
    def model(self) -> Any:
        return self._adapter

    def _generate(self, prompt: str, *, schema: Any | None = None, **kwargs: Any) -> str:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**request)
        content = getattr(getattr(response, "choices", [None])[0], "message", None)
        text = getattr(content, "content", None)
        if not text:
            raise RuntimeError("DeepEval judge returned empty content")
        return str(text)


def make_deepeval_model(settings: DeepEvalSettings, *, environment: Mapping[str, str] | None = None) -> Any:
    """Construct an explicit judge model only when readiness is satisfied."""
    readiness = probe_deepeval(settings, environ=environment)
    if readiness.status is not MetricStatus.SUCCESS:
        raise DeepEvalUnavailableError(readiness.error_message or "DeepEval judge is not ready")
    env = os.environ if environment is None else environment
    api_key = env.get(settings.api_key_env, "")
    raw_timeout = env.get("DEEPEVAL_TIMEOUT_SECONDS", "60")
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise DeepEvalUnavailableError("DEEPEVAL_TIMEOUT_SECONDS must be a positive number") from exc
    return OpenAICompatibleDeepEvalModel(
        model_name=settings.evaluation_model,
        api_key=api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
        timeout_seconds=timeout_seconds,
    ).model


def probe_deepeval(
    settings: DeepEvalSettings,
    *,
    environ: Mapping[str, str] | None = None,
    module_finder: Callable[[str], Any] = importlib.util.find_spec,
) -> DeepEvalReadiness:
    """Probe readiness without importing DeepEval or exposing credential values."""

    environment = os.environ if environ is None else environ
    installed = module_finder("deepeval") is not None
    credentials_configured = bool(environment.get(settings.api_key_env, "").strip())
    version: str | None = None
    reasons: list[str] = []
    if installed:
        try:
            version = importlib.metadata.version("deepeval")
        except importlib.metadata.PackageNotFoundError:
            reasons.append("DeepEval module is importable but package metadata is missing")
    else:
        reasons.append("DeepEval is not installed")
    if not credentials_configured:
        reasons.append(f"evaluation credential environment variable {settings.api_key_env} is not configured")

    ready = installed and credentials_configured and not reasons
    return DeepEvalReadiness(
        status=MetricStatus.SUCCESS if ready else MetricStatus.UNMEASURED,
        installed=installed,
        credentials_configured=credentials_configured,
        model_configured=True,
        deepeval_version=version,
        evaluation_provider=settings.evaluation_provider,
        evaluation_model=settings.evaluation_model,
        score=None,
        error_message=None if ready else "; ".join(reasons),
    )


def build_deepeval_test_case(payload: DeepEvalCasePayload) -> Any:
    """Lazily construct an LLMTestCase; never changes the system under test."""

    if importlib.util.find_spec("deepeval") is None:
        raise DeepEvalUnavailableError("DeepEval is not installed; semantic evaluation remains UNMEASURED")
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(**payload.to_test_case_kwargs())
