"""Second-layer intent classification with a strict, injectable contract.

The classifier is deliberately limited to intent selection. Scope, evidence
perspectives, protected tokens, and unsupported capability checks remain
deterministic router responsibilities.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.business_models import QueryIntent


class IntentClassification(BaseModel):
    """The only model output accepted from a second-layer classifier."""

    model_config = ConfigDict(extra="forbid", strict=True)

    primary_intent: QueryIntent
    secondary_intent: QueryIntent | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2000)


class IntentClassifier(Protocol):
    def classify(
        self,
        query: str,
        candidates: Sequence[QueryIntent],
    ) -> IntentClassification:
        """Return a schema-validated classification or raise on failure."""


class OpenAICompatibleIntentClassifier:
    """Strict OpenAI-compatible adapter; no fallback or fabricated result."""

    def __init__(self, client: Any, *, model: str, max_retries: int = 2) -> None:
        if client is None:
            raise ValueError("client is required")
        if not model:
            raise ValueError("model is required")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.client = client
        self.model = model
        self.max_retries = max_retries

    @classmethod
    def from_env(
        cls,
        client: Any = None,
        *,
        model: str | None = None,
        max_retries: int = 2,
    ) -> OpenAICompatibleIntentClassifier:
        model = model or os.getenv("INTENT_MODEL")
        if not model:
            raise ValueError("INTENT_MODEL is not configured in environment")
        if client is None:
            import openai
            key_ref = os.getenv(
                "INTENT_API_KEY_ENV",
                os.getenv("EMBEDDING_API_KEY_ENV", "SILICONFLOW_API_KEY"),
            ).strip()
            api_key = os.getenv(key_ref) if key_ref and not key_ref.startswith("sk-") else key_ref
            if not api_key:
                raise ValueError(
                    "Intent classifier API key is not configured; set EMBEDDING_API_KEY_ENV or SILICONFLOW_API_KEY"
                )
            base_url = os.getenv("INTENT_BASE_URL", os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"))
            timeout = float(os.getenv("INTENT_TIMEOUT_SECONDS", "30"))
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        return cls(client=client, model=model, max_retries=max_retries)

    @classmethod
    def try_from_env(
        cls,
        client: Any = None,
        *,
        model: str | None = None,
        max_retries: int = 2,
    ) -> OpenAICompatibleIntentClassifier | None:
        """Attempt to initialize classifier from environment.

        Returns None ONLY when the classifier model is truly unconfigured in environment/arguments.
        If a model is specified or INTENT_MODEL is present in the environment, any configuration
        or initialization errors (such as missing API key, dependency error, or invalid parameter)
        will NOT be swallowed and will raise immediately to prevent pseudo-availability.
        """
        target_model = model or os.getenv("INTENT_MODEL")
        if not target_model:
            return None

        return cls.from_env(client=client, model=target_model, max_retries=max_retries)

    def classify(
        self,
        query: str,
        candidates: Sequence[QueryIntent],
    ) -> IntentClassification:
        candidate_values = [item.value for item in candidates if item != QueryIntent.UNSUPPORTED]
        if not candidate_values:
            candidate_values = [item.value for item in QueryIntent]
        system = (
            "You are a strict business-query intent classifier. "
            "Return JSON only with primary_intent, secondary_intent, confidence, rationale. "
            "Provide a concise rationale in 1-2 sentences. "
            "primary_intent and secondary_intent must be enum values from the candidate list. "
            "Choose secondary_intent=null when there is no clear secondary intent. "
            "Do not infer scope, evidence role, user role, or data availability. "
            "Follow these strict disambiguation rules:\n"
            "1. Primary vs Secondary: Choose the core requested action or final analytical output as primary_intent; context, subjects, or modifiers belong in secondary_intent.\n"
            "2. Single Question Priority: QUESTION_TUTORING strictly overrides all aggregate intents (EXAM_ERROR_PATTERNS, RECURRING_MISCONCEPTIONS, STUDENT_FREQUENT_QUESTIONS) whenever a specific single problem, question, equation, or option diagnosis is analyzed (e.g. '这道题', '这道选择题为什么选A', '这道长方体题目', '卡在第一步'), even if aggregate words like '大家全军覆没', '学生最常问的问题', or '很多孩子' appear.\n"
            "3. Tutor Strategy: When a query asks how a tutor/teacher should guide, scaffold, intervene, or teach (e.g. '导师如何引导', '怎么解开认知纠结', 'how instructors scaffold'), prefer TUTOR_STRATEGY_PATTERNS over misconception, difficulty, or tutoring labels.\n"
            "4. Student Cohorts: STUDENT_COHORT_PATTERNS applies ONLY when comparing two or more student groups (e.g. repeat test-takers vs first-timers, top vs struggling students) or profiling a cohort's overall traits. When a student group merely describes who is facing a difficulty (e.g. '基础薄弱学生最难理解的概念是什么') or where confusion occurs ('二刷同学在哪些步骤产生困惑'), prioritize the concept (DIFFICULT_CONCEPTS) or the frequent question (STUDENT_FREQUENT_QUESTIONS).\n"
            "5. Errors & Misconceptions: For exam/test errors, paper grading flaws, or formal pitfalls (e.g. 'pitfalls in geometry proofs', '丢分硬伤'), prefer EXAM_ERROR_PATTERNS. For recurring learning stumbling blocks or frequent inquiry questions, prefer STUDENT_FREQUENT_QUESTIONS. For deep cognitive conceptual confusions, prefer RECURRING_MISCONCEPTIONS."
        )
        user = json.dumps(
            {"query": query, "candidate_intents": candidate_values},
            ensure_ascii=False,
        )
        extra_body: dict[str, Any] = {}
        if "qwen" in self.model.lower():
            extra_body["enable_thinking"] = False

        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = self.client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or "{}"
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    if isinstance(payload.get("primary_intent"), str):
                        payload["primary_intent"] = QueryIntent(payload["primary_intent"])
                    if isinstance(payload.get("secondary_intent"), str):
                        payload["secondary_intent"] = QueryIntent(payload["secondary_intent"])
                result = IntentClassification.model_validate(payload)
                if result.primary_intent.value not in candidate_values:
                    raise ValueError("classifier returned an intent outside candidate set")
                if (
                    result.secondary_intent is not None
                    and result.secondary_intent.value not in candidate_values
                ):
                    raise ValueError("classifier returned a secondary intent outside candidate set")
                if result.secondary_intent == result.primary_intent:
                    raise ValueError("secondary_intent must differ from primary_intent")
                return result
            except Exception as exc:
                last_error = exc
        raise RuntimeError("intent classifier failed strict validation") from last_error


__all__ = [
    "IntentClassification",
    "IntentClassifier",
    "OpenAICompatibleIntentClassifier",
]
