from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from collections.abc import Callable, Mapping
from typing import Any

from evals.contracts import DeepEvalCasePayload, DeepEvalReadiness, DeepEvalSettings, MetricStatus


class DeepEvalUnavailableError(RuntimeError):
    pass


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

