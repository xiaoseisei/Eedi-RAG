from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictContract(BaseModel):
    """Shared strict boundary: unknown fields are always rejected."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetricStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNMEASURED = "UNMEASURED"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class ComparisonOperator(str, Enum):
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"

    def passes(self, value: float, threshold: float) -> bool:
        if self is ComparisonOperator.GTE:
            return value >= threshold
        if self is ComparisonOperator.LTE:
            return value <= threshold
        return math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


class RunContext(StrictContract):
    run_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    dataset_version: str = Field(min_length=1)
    qrels_version: str = Field(min_length=1)
    system_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def config_matches_hash(self) -> "RunContext":
        if self.system_config:
            expected = canonical_system_config_hash(self.system_config)
            if expected != self.system_config_hash:
                raise ValueError("system_config_hash does not match canonical system_config")
        return self


class MetricObservation(StrictContract):
    schema_version: Literal["metric-observation/v1"] = "metric-observation/v1"
    run_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    case_id: str = Field(min_length=1)
    layer: Literal["L1"] = "L1"
    stage: str = Field(min_length=1)
    scope: Literal["case", "aggregate", "slice"]
    metric_name: str = Field(min_length=1)
    value: float | None
    threshold: float | None
    operator: ComparisonOperator | None
    status: MetricStatus
    hard_gate: bool
    evaluator_type: Literal["deterministic", "deepeval"]
    dataset_version: str = Field(min_length=1)
    qrels_version: str = Field(min_length=1)
    system_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    slices: dict[str, str] = Field(default_factory=dict)
    sample_size: int | None = Field(default=None, ge=0)
    passed_cases: int | None = Field(default=None, ge=0)
    measured_cases: int | None = Field(default=None, ge=0)
    unmeasured_cases: int | None = Field(default=None, ge=0)
    confidence_interval_low: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_interval_high: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("value", "threshold")
    @classmethod
    def finite_number_or_none(cls, value: float | None) -> float | None:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("metric numbers must be finite real values")
        return value

    @field_validator("slices")
    @classmethod
    def non_blank_slices(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not str(key).strip() or not str(item).strip() for key, item in value.items()):
            raise ValueError("slice keys and values must be non-blank")
        return value

    @model_validator(mode="after")
    def status_matches_measurement(self) -> "MetricObservation":
        measured = self.status in {MetricStatus.SUCCESS, MetricStatus.FAILED}
        if measured:
            if self.value is None or self.threshold is None or self.operator is None:
                raise ValueError(f"{self.status.value} requires value, threshold, and operator")
            passed = self.operator.passes(self.value, self.threshold)
            if self.status is MetricStatus.SUCCESS and not passed:
                raise ValueError("status=SUCCESS falsifies a threshold failure")
            if self.status is MetricStatus.FAILED and passed:
                raise ValueError("status=FAILED falsifies a threshold pass")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("measured observations cannot carry an error")
        elif self.status is MetricStatus.UNMEASURED:
            if self.value is not None:
                raise ValueError("UNMEASURED observations cannot carry a value")
            if not self.error_message or not self.error_message.strip():
                raise ValueError("UNMEASURED requires a reason in error_message")
        elif self.status in {MetricStatus.ERROR, MetricStatus.DEGRADED}:
            if not self.error_message or not self.error_message.strip():
                raise ValueError(f"{self.status.value} requires error_message")

        count_fields = (self.passed_cases, self.measured_cases, self.unmeasured_cases)
        if any(item is not None for item in count_fields):
            if any(item is None for item in count_fields):
                raise ValueError("passed/measured/unmeasured case counts must be supplied together")
            assert self.passed_cases is not None
            assert self.measured_cases is not None
            assert self.unmeasured_cases is not None
            if self.passed_cases > self.measured_cases:
                raise ValueError("passed_cases cannot exceed measured_cases")
            if self.sample_size is not None and self.measured_cases + self.unmeasured_cases != self.sample_size:
                raise ValueError("case counts must add up to sample_size")

        if (self.confidence_interval_low is None) != (self.confidence_interval_high is None):
            raise ValueError("both confidence interval bounds are required")
        if (
            self.confidence_interval_low is not None
            and self.confidence_interval_high is not None
            and self.confidence_interval_low > self.confidence_interval_high
        ):
            raise ValueError("confidence interval lower bound exceeds upper bound")
        return self


def observation_from_measurement(
    *,
    context: RunContext,
    case_id: str,
    stage: str,
    metric_name: str,
    value: float,
    threshold: float,
    operator: ComparisonOperator,
    hard_gate: bool,
    scope: Literal["case", "aggregate", "slice"] = "case",
    slices: dict[str, str] | None = None,
    sample_size: int | None = None,
    passed_cases: int | None = None,
    measured_cases: int | None = None,
    unmeasured_cases: int | None = None,
    confidence_interval: tuple[float, float] | None = None,
    evidence_refs: list[str] | None = None,
    evaluator_type: Literal["deterministic", "deepeval"] = "deterministic",
) -> MetricObservation:
    status = MetricStatus.SUCCESS if operator.passes(value, threshold) else MetricStatus.FAILED
    return MetricObservation(
        run_id=context.run_id,
        case_id=case_id,
        stage=stage,
        scope=scope,
        metric_name=metric_name,
        value=value,
        threshold=threshold,
        operator=operator,
        status=status,
        hard_gate=hard_gate,
        evaluator_type=evaluator_type,
        dataset_version=context.dataset_version,
        qrels_version=context.qrels_version,
        system_config_hash=context.system_config_hash,
        slices=slices or {},
        sample_size=sample_size,
        passed_cases=passed_cases,
        measured_cases=measured_cases,
        unmeasured_cases=unmeasured_cases,
        confidence_interval_low=confidence_interval[0] if confidence_interval else None,
        confidence_interval_high=confidence_interval[1] if confidence_interval else None,
        evidence_refs=evidence_refs or [],
    )


def unmeasured_observation(
    *,
    context: RunContext,
    case_id: str,
    stage: str,
    metric_name: str,
    threshold: float,
    operator: ComparisonOperator,
    hard_gate: bool,
    reason: str,
    scope: Literal["case", "aggregate", "slice"] = "case",
    slices: dict[str, str] | None = None,
    evaluator_type: Literal["deterministic", "deepeval"] = "deterministic",
    sample_size: int | None = None,
    evidence_refs: list[str] | None = None,
) -> MetricObservation:
    return MetricObservation(
        run_id=context.run_id,
        case_id=case_id,
        stage=stage,
        scope=scope,
        metric_name=metric_name,
        value=None,
        threshold=threshold,
        operator=operator,
        status=MetricStatus.UNMEASURED,
        hard_gate=hard_gate,
        evaluator_type=evaluator_type,
        dataset_version=context.dataset_version,
        qrels_version=context.qrels_version,
        system_config_hash=context.system_config_hash,
        slices=slices or {},
        sample_size=sample_size,
        evidence_refs=evidence_refs or [],
        error_type="UNMEASURED_PRECONDITION",
        error_message=reason,
    )


class EvidenceRef(StrictContract):
    session_id: int = Field(ge=1)
    turn_id: int = Field(ge=1)
    speaker: Literal["student", "tutor"]
    quote_text: str = Field(min_length=1)

    @field_validator("quote_text")
    @classmethod
    def quote_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("quote_text must not be blank")
        return value

    @property
    def turn_key(self) -> tuple[int, int]:
        return self.session_id, self.turn_id


class GroundingEvalCase(StrictContract):
    case_id: str = Field(min_length=1)
    output_citations: list[EvidenceRef]
    authoritative_turns: list[EvidenceRef] = Field(min_length=1)
    required_evidence: list[EvidenceRef] = Field(min_length=1)
    expected_speaker: Literal["student", "tutor"] | None = None
    quote_match_mode: Literal["exact", "substring"] = "substring"
    slices: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_turns_exist(self) -> "GroundingEvalCase":
        source_keys = {item.turn_key for item in self.authoritative_turns}
        missing = [item.turn_key for item in self.required_evidence if item.turn_key not in source_keys]
        if missing:
            raise ValueError(f"required evidence has no authoritative turn: {missing}")
        return self


class TurnKey(StrictContract):
    session_id: int = Field(ge=1)
    turn_id: int = Field(ge=1)


class ChunkRankedRecord(StrictContract):
    chunk_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    source_turn_ids: set[int]
    token_count: int = Field(ge=0)

    @field_validator("source_turn_ids")
    @classmethod
    def turn_ids_are_positive(cls, value: set[int]) -> set[int]:
        if any(turn_id < 1 for turn_id in value):
            raise ValueError("source turn IDs must be positive")
        return value


class ChunkingEvalCase(StrictContract):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    target_turns: list[TurnKey] = Field(min_length=1)
    ranked_chunks: list[ChunkRankedRecord]
    original_token_count: int = Field(gt=0)
    emitted_chunk_token_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    slices: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def chunk_case_is_consistent(self) -> "ChunkingEvalCase":
        if not self.query.strip():
            raise ValueError("query must not be blank")
        chunk_ids = [item.chunk_id for item in self.ranked_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("ranked chunks contain duplicate chunk IDs")
        target_keys = [(item.session_id, item.turn_id) for item in self.target_turns]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("target_turns contain duplicate session/turn pairs")
        if sum(item.token_count for item in self.ranked_chunks) > self.emitted_chunk_token_count:
            raise ValueError("ranked chunk tokens cannot exceed total emitted chunk tokens")
        return self


class RewriteEvalCase(StrictContract):
    case_id: str = Field(min_length=1)
    raw_query: str = Field(min_length=1)
    rewritten_query: str = Field(min_length=1)
    protected_tokens: set[str]
    critical_entities: set[str]
    intent_preserved: bool | None
    intent_label_source: Literal["human", "deepeval"] | None
    domain_injection_expected: bool
    injected_domain_context: str = ""
    raw_ranked_ids: list[str]
    rewritten_ranked_ids: list[str]
    qrels: dict[str, int]
    latency_ms: float = Field(ge=0.0)
    slices: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def rewrite_case_is_consistent(self) -> "RewriteEvalCase":
        if not self.raw_query.strip() or not self.rewritten_query.strip():
            raise ValueError("raw_query and rewritten_query must not be blank")
        for field_name, values in (
            ("protected_tokens", self.protected_tokens),
            ("critical_entities", self.critical_entities),
        ):
            if any(not item.strip() for item in values):
                raise ValueError(f"{field_name} cannot contain blank values")
            missing = [item for item in values if item.casefold() not in self.raw_query.casefold()]
            if missing:
                raise ValueError(f"{field_name} entries are absent from raw_query: {missing}")
        if (self.intent_preserved is None) != (self.intent_label_source is None):
            raise ValueError("intent_preserved and intent_label_source must be supplied together")
        for field_name, ranking in (
            ("raw_ranked_ids", self.raw_ranked_ids),
            ("rewritten_ranked_ids", self.rewritten_ranked_ids),
        ):
            if len(ranking) != len(set(ranking)):
                raise ValueError(f"{field_name} contains duplicate document IDs")
        _validate_qrels(self.qrels)
        return self


class IndexedRecord(StrictContract):
    record_id: str = Field(min_length=1)
    parent_id: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any]


class StorageCollectionSnapshot(StrictContract):
    name: str = Field(min_length=1)
    expected_ids: set[str]
    relational_records: list[IndexedRecord]
    index_records: list[IndexedRecord]
    required_metadata_fields: list[str]

    @model_validator(mode="after")
    def ids_are_unique_within_each_store(self) -> "StorageCollectionSnapshot":
        for store_name, records in (
            ("relational", self.relational_records),
            ("index", self.index_records),
        ):
            ids = [item.record_id for item in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {store_name} record IDs in {self.name}")
        if len(self.required_metadata_fields) != len(set(self.required_metadata_fields)):
            raise ValueError(f"duplicate metadata fields in {self.name}")
        return self


class StorageEvalSnapshot(StrictContract):
    case_id: str = Field(min_length=1)
    valid_parent_ids: set[int]
    collections: list[StorageCollectionSnapshot] = Field(min_length=1)
    additional_relational_records: list[IndexedRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def collection_names_are_unique(self) -> "StorageEvalSnapshot":
        names = [item.name for item in self.collections]
        if len(names) != len(set(names)):
            raise ValueError("storage collection names must be unique")
        return self


class RetrievalEvalCase(StrictContract):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    ranked_ids: list[str]
    qrels: dict[str, int]
    latency_ms: float = Field(ge=0.0)
    slices: dict[str, str] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def ranking_and_qrels_are_valid(self) -> "RetrievalEvalCase":
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if len(self.ranked_ids) != len(set(self.ranked_ids)):
            raise ValueError("ranked_ids contain duplicate document IDs")
        _validate_qrels(self.qrels)
        return self


class AssemblerEvalCase(StrictContract):
    case_id: str = Field(min_length=1)
    input_ranked_ids: list[str]
    final_context_ids: list[str]
    qrels: dict[str, int]
    required_evidence_ids: set[str] = Field(min_length=1)
    candidate_char_count: int = Field(ge=0)
    assembled_char_count: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    max_prompt_tokens: int = Field(gt=0)
    evidence_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    slices: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def assembler_case_is_consistent(self) -> "AssemblerEvalCase":
        if len(self.input_ranked_ids) != len(set(self.input_ranked_ids)):
            raise ValueError("input_ranked_ids contain duplicate document IDs")
        if len(self.final_context_ids) != len(set(self.final_context_ids)):
            raise ValueError("final_context_ids contain duplicate document IDs")
        if not set(self.final_context_ids).issubset(self.input_ranked_ids):
            raise ValueError("final_context_ids must be selected from input_ranked_ids")
        _validate_qrels(self.qrels)
        if not self.required_evidence_ids.issubset(self.qrels):
            raise ValueError("required_evidence_ids must be present in qrels")
        if any(self.qrels[item] <= 0 for item in self.required_evidence_ids):
            raise ValueError("required evidence must have positive relevance")
        if self.candidate_char_count == 0 and self.assembled_char_count > 0:
            raise ValueError("assembled characters cannot exist without candidate characters")
        return self


def _validate_qrels(qrels: dict[str, int]) -> None:
    if not qrels:
        raise ValueError("qrels must not be empty")
    for document_id, relevance in qrels.items():
        if not document_id.strip():
            raise ValueError("qrels document IDs must be non-blank")
        if isinstance(relevance, bool) or not isinstance(relevance, int) or not 0 <= relevance <= 3:
            raise ValueError("qrels relevance must be an integer from 0 to 3")
    if not any(relevance > 0 for relevance in qrels.values()):
        raise ValueError("qrels must contain at least one positive relevance judgment")


class DeepEvalSettings(StrictContract):
    evaluation_provider: str = Field(min_length=1)
    evaluation_model: str = Field(min_length=1)
    evaluation_model_version: str | None = None
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None


class DeepEvalReadiness(StrictContract):
    status: MetricStatus
    installed: bool
    credentials_configured: bool
    model_configured: bool
    deepeval_version: str | None
    evaluation_provider: str
    evaluation_model: str
    score: None = None
    error_message: str | None = None


class DeepEvalCasePayload(StrictContract):
    case_id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    actual_output: str = Field(min_length=1)
    expected_output: str = Field(min_length=1)
    final_retrieval_context: list[str] = Field(min_length=1)

    @field_validator("input", "actual_output", "expected_output")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DeepEval text fields must not be blank")
        return value

    @field_validator("final_retrieval_context")
    @classmethod
    def context_is_not_blank(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("final retrieval context cannot contain blank entries")
        return value

    def to_test_case_kwargs(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "retrieval_context": self.final_retrieval_context,
        }


class L1ComponentReport(StrictContract):
    schema_version: Literal["l1-component-report/v1"] = "l1-component-report/v1"
    run_id: str
    layer: Literal["L1"] = "L1"
    dataset_version: str
    qrels_version: str
    system_config_hash: str
    system_config: dict[str, Any] = Field(default_factory=dict)
    executed_modules: list[str]
    release_status: Literal["PASSED", "BLOCKED"]
    status_counts: dict[str, int]
    observations: list[MetricObservation]
    started_at: datetime
    completed_at: datetime

    @classmethod
    def from_observations(
        cls,
        *,
        context: RunContext,
        observations: list[MetricObservation],
        executed_modules: list[str],
        started_at: datetime | None = None,
    ) -> "L1ComponentReport":
        if not observations:
            raise ValueError("a component report requires at least one observation")
        if not executed_modules or len(executed_modules) != len(set(executed_modules)):
            raise ValueError("executed_modules must be non-empty and unique")
        for item in observations:
            if (
                item.run_id != context.run_id
                or item.dataset_version != context.dataset_version
                or item.qrels_version != context.qrels_version
                or item.system_config_hash != context.system_config_hash
            ):
                raise ValueError("observation context does not match report context")

        blocked = any(item.hard_gate and item.status is not MetricStatus.SUCCESS for item in observations)
        counts = {status.value: 0 for status in MetricStatus}
        for item in observations:
            counts[item.status.value] += 1
        now = datetime.now(UTC)
        return cls(
            run_id=context.run_id,
            dataset_version=context.dataset_version,
            qrels_version=context.qrels_version,
            system_config_hash=context.system_config_hash,
            system_config=context.system_config,
            executed_modules=executed_modules,
            release_status="BLOCKED" if blocked else "PASSED",
            status_counts=counts,
            observations=observations,
            started_at=started_at or now,
            completed_at=now,
        )


class L1ComponentSuiteSpec(StrictContract):
    schema_version: Literal["l1-component-suite/v1"] = "l1-component-suite/v1"
    context: RunContext
    storage: StorageEvalSnapshot | None = None
    grounding_cases: list[GroundingEvalCase] = Field(default_factory=list)
    chunking_cases: list[ChunkingEvalCase] = Field(default_factory=list)
    rewrite_cases: list[RewriteEvalCase] = Field(default_factory=list)
    retrieval_cases: list[RetrievalEvalCase] = Field(default_factory=list)
    assembler_cases: list[AssemblerEvalCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def at_least_one_module_is_present(self) -> "L1ComponentSuiteSpec":
        if not any(
            (
                self.storage,
                self.grounding_cases,
                self.chunking_cases,
                self.rewrite_cases,
                self.retrieval_cases,
                self.assembler_cases,
            )
        ):
            raise ValueError("component suite contains no executable module")
        if not self.context.system_config:
            raise ValueError("component suite requires an explicit system_config")
        return self


def canonical_system_config_hash(config: dict[str, Any]) -> str:
    import hashlib
    import json

    _reject_secret_config_keys(config)
    try:
        serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("system_config must be canonical JSON data") from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _reject_secret_config_keys(value: Any, path: str = "system_config") -> None:
    blocked = re.compile(r"(^|_)(token|secret|password|cookie|authorization|api_key)($|_)", re.IGNORECASE)
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if blocked.search(key_text) and not key_text.lower().endswith("_env"):
                raise ValueError(f"secret-like key is forbidden in report config: {path}.{key_text}")
            _reject_secret_config_keys(item, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_config_keys(item, f"{path}[{index}]")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
