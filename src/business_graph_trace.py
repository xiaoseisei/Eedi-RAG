"""Trace contract for the graph-business execution route.

The contract deliberately contains only observed runtime data.  Gold cases are
used by evaluators after execution and must never be forwarded to a runtime.
"""

from __future__ import annotations

from enum import Enum
import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StageStatus(str, Enum):
    """Explicit state for every route stage that is reached or short-circuited."""

    SUCCESS = "SUCCESS"
    NO_EVIDENCE = "NO_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"
    UNMEASURED = "UNMEASURED"


class BusinessGraphRuntimeTrace(BaseModel):
    """Observed, serializable result of one graph-business route execution."""

    model_config = ConfigDict(extra="forbid")

    actual_router: dict[str, Any]
    actual_plan: dict[str, Any]
    capabilities: dict[str, Any]
    analytics_result: dict[str, Any] | None
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    card_candidates: list[dict[str, Any]] = Field(default_factory=list)
    parent_cards: list[dict[str, Any]] = Field(default_factory=list)
    candidate_windows: list[dict[str, Any]] = Field(default_factory=list)
    ranked_windows: list[dict[str, Any]] = Field(default_factory=list)
    selected_windows: list[dict[str, Any]] = Field(default_factory=list)
    assembled_prompt: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    stage_status: dict[str, StageStatus] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)

    @field_validator("candidate_windows", "ranked_windows", "selected_windows")
    @classmethod
    def windows_are_only_window_ranking_units(
        cls, windows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        required = {"window_id", "session_id", "source_turn_ids", "anchor_turn_ids"}
        for window in windows:
            if "turn_id" in window:
                raise ValueError("Turn records cannot be ranking candidates")
            missing = sorted(required - set(window))
            if missing:
                raise ValueError(f"window record missing required fields: {', '.join(missing)}")
            if not isinstance(window["source_turn_ids"], list):
                raise ValueError("window source_turn_ids must be a list")
            if not isinstance(window["anchor_turn_ids"], list):
                raise ValueError("window anchor_turn_ids must be a list")
        return windows

    def to_json(self) -> str:
        """Return actual runtime trace JSON; this model has no Gold fields."""

        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


class CaseRuntime(Protocol):
    def ask_case(self, query: str) -> BusinessGraphRuntimeTrace: ...


def run_business_case(case: dict[str, Any], runtime: CaseRuntime) -> BusinessGraphRuntimeTrace:
    """Run one case without exposing its expected evidence or answer to the SUT."""

    query = case.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("business case requires a non-blank query")
    trace = runtime.ask_case(query)
    if not isinstance(trace, BusinessGraphRuntimeTrace):
        raise TypeError("runtime.ask_case must return BusinessGraphRuntimeTrace")
    return trace


__all__ = ["BusinessGraphRuntimeTrace", "CaseRuntime", "StageStatus", "run_business_case"]
