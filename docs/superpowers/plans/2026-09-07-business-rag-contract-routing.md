# Business RAG Capability Upgrade Part 1: Contract and Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the explicit capability boundary for Eedi-RAG by modelling evidence perspective, macro/micro computation scope, business intent, data capabilities, and deterministic query plans without changing the existing retrieval result.

**Architecture:** Keep `EndToEndPedagogicalRAGPipeline.ask()` as the paired micro-RAG baseline. Add strict business request/response contracts, a two-axis query router, a capability registry, and a query planner beside it. This part produces an executable and evaluated plan for every query, but does not yet execute analytics or alter Chroma/DuckDB artifacts.

**Tech Stack:** Python 3.14, Pydantic 2, pytest 9, JSONL evaluation fixtures, existing Eedi-RAG models and evaluation contracts.

**Spec:** `docs/需求.md`; `docs/Query_Intent_Response_Executable_Design.md`

## Global Constraints

- Preserve the engineering-integrity rules in `AGENTS.md`: no fake execution, no swallowed exceptions, no false success status, and no permissive defaults at trust boundaries.
- Treat student/tutor perspective and macro/micro computation scope as independent axes.
- Keep `src.rag_pipeline.EndToEndPedagogicalRAGPipeline.ask()` unchanged as a paired baseline until Part 3 promotes `ask_business()`.
- Do not send Gold labels, qrels, or evaluation expected answers into the router, planner, retriever, assembler, or generator.
- Never infer a user identity or permission role from the query.
- Route count/rank/frequency terms such as “最常”, “Top-N”, “多少”, and “反复出现” to deterministic analytics, never to vector-only retrieval.
- Route current Eedi retaker/weak-foundation questions to `UNSUPPORTED` because the imported artifact lacks a verified student-to-session identity and longitudinal attempt/ability data.
- Preserve numbers, formulas, answer-option letters, LaTeX, and question IDs byte-for-byte in `protected_tokens`.
- Use strict Pydantic models with `extra="forbid"`; status, result, and audit fields must remain mutually consistent.
- Do not modify production DuckDB/Chroma artifacts in this part.

## Delivery Boundary

This is Part 1 of 3. It delivers:

```text
BusinessQueryRequest
  -> BusinessQueryRouter
  -> QueryUnderstanding (intent + scope + evidence perspectives)
  -> DataCapabilityRegistry
  -> BusinessQueryPlanner
  -> QueryPlan or REJECT
```

It deliberately does not deliver analytics events, SQL reports, representative evidence, LLM synthesis, or CLI promotion. Those are implemented in Parts 2 and 3.

## File Structure

```text
src/
├── business_models.py       # strict enums, request, understanding, plan, response shell
├── data_capabilities.py     # artifact-derived capability registry and missing-field reasons
├── query_intent.py          # deterministic router and optional bounded classifier protocol
└── query_planner.py         # understanding + capabilities -> executable plan

tests/
├── test_business_models.py
├── test_data_capabilities.py
├── test_query_intent.py
└── test_query_planner.py

evals/datasets/
└── query-router-v1.jsonl    # human-labelled dev/holdout routing cases

scripts/
└── evaluate_business_router.py
```

---

### Task 1: Add Strict Business Boundary Contracts

**Files:**
- Create: `src/business_models.py`
- Create: `tests/test_business_models.py`
- Reference: `src/models.py:416-470`

**Interfaces:**
- Consumes: Pydantic `BaseModel`, `ConfigDict`, `Field`, model validators.
- Produces: `EvidencePerspective`, `ComputationScope`, `QueryIntent`, `QueryScope`, `RouteConfidence`, `QueryPlanType`, `BusinessResponseStatus`, `DetailLevel`, `BusinessQueryRequest`, `QueryUnderstanding`, `DataCapabilitySnapshot`, `QueryPlan`, `DataScope`, `EvidencePointer`, and `BusinessResponseEnvelope`.

- [ ] **Step 1: Write strict model tests before creating the models**

```python
# tests/test_business_models.py
import pytest
from pydantic import ValidationError

from src.business_models import (
    BusinessQueryRequest,
    BusinessResponseEnvelope,
    BusinessResponseStatus,
    ComputationScope,
    DataCapabilitySnapshot,
    DataScope,
    DetailLevel,
    EvidencePerspective,
    QueryIntent,
    QueryPlan,
    QueryPlanType,
    QueryScope,
    QueryUnderstanding,
    RouteConfidence,
)


def _understanding() -> QueryUnderstanding:
    return QueryUnderstanding(
        raw_query="学生最常提出的问题是什么？",
        primary_intent=QueryIntent.STUDENT_FREQUENT_QUESTIONS,
        secondary_intents=[],
        scope=QueryScope.CORPUS,
        computation_scope=ComputationScope.HYBRID,
        evidence_perspectives=[EvidencePerspective.STUDENT],
        subject_filters=[],
        protected_tokens=[],
        requested_top_k=3,
        requires_aggregation=True,
        requires_case_retrieval=True,
        response_type="STUDENT_FAQ_REPORT",
        confidence=RouteConfidence.HIGH,
        route_source="deterministic",
    )


def test_business_request_is_strict_and_bounded() -> None:
    request = BusinessQueryRequest(
        query="列出 Top 10 高频问题",
        top_k=10,
        detail_level=DetailLevel.FULL,
    )
    assert request.top_k == 10
    with pytest.raises(ValidationError):
        BusinessQueryRequest(query="x", top_k=11)
    with pytest.raises(ValidationError):
        BusinessQueryRequest(query="x", unknown=True)


def test_success_requires_result_and_verified_audit() -> None:
    with pytest.raises(ValidationError, match="SUCCESS"):
        BusinessResponseEnvelope(
            query="学生最常提出的问题是什么？",
            understanding=_understanding(),
            status=BusinessResponseStatus.SUCCESS,
            summary="有结果",
            data_scope=DataScope(
                dataset_version="eedi-20260907",
                index_version="business-events-v1",
                total_sessions_considered=1576,
                matched_sessions=100,
            ),
            result=None,
            evidence=[],
            limitations=[],
            audit_status="NOT_APPLICABLE",
            trace_id="trace-1",
        )


def test_query_plan_rejects_vector_only_frequency_plan() -> None:
    understanding = _understanding()
    with pytest.raises(ValidationError, match="aggregation"):
        QueryPlan(
            understanding=understanding,
            plan_type=QueryPlanType.CASE_PLAYBOOK,
            analytics_operations=[],
            retrieval_perspectives=[EvidencePerspective.STUDENT],
            required_capabilities=[],
            response_type="STUDENT_FAQ_REPORT",
        )


def test_capability_snapshot_cannot_claim_available_with_missing_requirements() -> None:
    with pytest.raises(ValidationError, match="missing"):
        DataCapabilitySnapshot(
            capability="RETAKE_COHORT",
            available=True,
            coverage=1.0,
            missing_requirements=["attempt_number"],
            source_artifact="analytics.duckdb",
        )
```

- [ ] **Step 2: Run the model tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_models.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.business_models'`.

- [ ] **Step 3: Implement the enums and strict request/understanding/plan models**

```python
# src/business_models.py
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictBusinessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class EvidencePerspective(str, Enum):
    STUDENT = "STUDENT"
    TUTOR = "TUTOR"


class ComputationScope(str, Enum):
    MACRO = "MACRO"
    MICRO = "MICRO"
    HYBRID = "HYBRID"
    REJECT = "REJECT"


class QueryIntent(str, Enum):
    STUDENT_FREQUENT_QUESTIONS = "STUDENT_FREQUENT_QUESTIONS"
    DIFFICULT_CONCEPTS = "DIFFICULT_CONCEPTS"
    RECURRING_MISCONCEPTIONS = "RECURRING_MISCONCEPTIONS"
    EXAM_ERROR_PATTERNS = "EXAM_ERROR_PATTERNS"
    STUDENT_COHORT_PATTERNS = "STUDENT_COHORT_PATTERNS"
    TUTOR_STRATEGY_PATTERNS = "TUTOR_STRATEGY_PATTERNS"
    CONTENT_IMPROVEMENT = "CONTENT_IMPROVEMENT"
    QUESTION_TUTORING = "QUESTION_TUTORING"
    UNSUPPORTED = "UNSUPPORTED"


class QueryScope(str, Enum):
    CORPUS = "CORPUS"
    SUBJECT = "SUBJECT"
    CASE = "CASE"
    QUESTION = "QUESTION"


class RouteConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QueryPlanType(str, Enum):
    ANALYTICS_REPORT = "ANALYTICS_REPORT"
    ANALYTICS_WITH_EVIDENCE = "ANALYTICS_WITH_EVIDENCE"
    CASE_PLAYBOOK = "CASE_PLAYBOOK"
    CONTENT_BRIEF = "CONTENT_BRIEF"
    SOCRATIC_TUTORING = "SOCRATIC_TUTORING"
    REJECT = "REJECT"


class BusinessResponseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_EVIDENCE = "NO_EVIDENCE"
    UNSUPPORTED = "UNSUPPORTED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class DetailLevel(str, Enum):
    BRIEF = "BRIEF"
    STANDARD = "STANDARD"
    FULL = "FULL"


class BusinessQueryRequest(StrictBusinessModel):
    query: str = Field(min_length=1, max_length=2000)
    subject_filters: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=10)
    detail_level: DetailLevel = DetailLevel.STANDARD
    debug: bool = False


class QueryUnderstanding(StrictBusinessModel):
    raw_query: str = Field(min_length=1, max_length=2000)
    primary_intent: QueryIntent
    secondary_intents: list[QueryIntent] = Field(default_factory=list, max_length=1)
    scope: QueryScope
    computation_scope: ComputationScope
    evidence_perspectives: list[EvidencePerspective] = Field(max_length=2)
    subject_filters: list[str] = Field(default_factory=list)
    protected_tokens: list[str] = Field(default_factory=list)
    requested_top_k: int = Field(default=3, ge=1, le=10)
    requires_aggregation: bool
    requires_case_retrieval: bool
    response_type: str = Field(min_length=1)
    confidence: RouteConfidence
    route_source: Literal["deterministic", "llm_classifier"]


class DataCapabilitySnapshot(StrictBusinessModel):
    capability: str = Field(min_length=1)
    available: bool
    coverage: float = Field(ge=0.0, le=1.0)
    missing_requirements: list[str] = Field(default_factory=list)
    source_artifact: str = Field(min_length=1)

    @model_validator(mode="after")
    def availability_matches_requirements(self) -> "DataCapabilitySnapshot":
        if self.available and self.missing_requirements:
            raise ValueError("available capability cannot have missing requirements")
        return self


class DataCapabilityRegistry:
    @staticmethod
    def synthetic_snapshot(
        *, dialogue: bool, analytics: bool, retake: bool, ability: bool
    ) -> dict[str, DataCapabilitySnapshot]:
        """Create test-only snapshots; production inspect() always reads information_schema."""
        return {
            "DIALOGUE_EVIDENCE": DataCapabilitySnapshot(
                capability="DIALOGUE_EVIDENCE", available=dialogue,
                coverage=1.0 if dialogue else 0.0,
                missing_requirements=[] if dialogue else ["session_dialogue_turns"],
                source_artifact="synthetic",
            ),
            "BUSINESS_ANALYTICS": DataCapabilitySnapshot(
                capability="BUSINESS_ANALYTICS", available=analytics,
                coverage=1.0 if analytics else 0.0,
                missing_requirements=[] if analytics else ["business_analytics_manifest"],
                source_artifact="synthetic",
            ),
            "RETAKE_COHORT": DataCapabilitySnapshot(
                capability="RETAKE_COHORT", available=retake,
                coverage=1.0 if retake else 0.0,
                missing_requirements=[] if retake else ["student_key", "attempt_number"],
                source_artifact="synthetic",
            ),
            "ABILITY_COHORT": DataCapabilitySnapshot(
                capability="ABILITY_COHORT", available=ability,
                coverage=1.0 if ability else 0.0,
                missing_requirements=[] if ability else ["student_key", "historic_correctness"],
                source_artifact="synthetic",
            ),
        }


class QueryPlan(StrictBusinessModel):
    understanding: QueryUnderstanding
    plan_type: QueryPlanType
    analytics_operations: list[str] = Field(default_factory=list)
    retrieval_perspectives: list[EvidencePerspective] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    response_type: str = Field(min_length=1)
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_plan_semantics(self) -> "QueryPlan":
        if self.understanding.requires_aggregation and not self.analytics_operations:
            raise ValueError("aggregation query requires analytics operations")
        if self.plan_type == QueryPlanType.REJECT and not self.rejection_reason:
            raise ValueError("REJECT plan requires rejection_reason")
        if self.plan_type != QueryPlanType.REJECT and self.rejection_reason is not None:
            raise ValueError("executable plan cannot carry rejection_reason")
        return self
```

Add the following concrete temporary response shell in the same file. Part 3 Task 3 replaces `BusinessResultStub` with the full discriminated response union without changing the envelope fields.

```python
class DataScope(StrictBusinessModel):
    dataset_version: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    total_sessions_considered: int = Field(ge=0)
    matched_sessions: int = Field(ge=0)
    subject_filters: list[str] = Field(default_factory=list)
    other_filters: dict[str, str] = Field(default_factory=dict)
    unmeasured_metrics: list[str] = Field(default_factory=list)


class EvidencePointer(StrictBusinessModel):
    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    source_event_id: str = Field(min_length=1)
    source_card_id: str | None = None
    session_id: int = Field(ge=1)
    turn_id: int = Field(ge=1)
    speaker: Literal["student", "tutor"]
    exact_quote: str = Field(min_length=1)
    subject_path: str = Field(min_length=1)
    verified: bool


class BusinessResultStub(StrictBusinessModel):
    response_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class BusinessResponseEnvelope(StrictBusinessModel):
    schema_version: Literal["business-response/v1"] = "business-response/v1"
    query: str = Field(min_length=1)
    understanding: QueryUnderstanding
    status: BusinessResponseStatus
    summary: str = Field(min_length=1)
    data_scope: DataScope
    result: BusinessResultStub | None = None
    evidence: list[EvidencePointer] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    audit_status: Literal[
        "PENDING",
        "AUDITED_100_VERIFIED",
        "GROUNDING_DEGRADED",
        "NOT_APPLICABLE",
    ]
    trace_id: str = Field(min_length=1)
```

The envelope validator must enforce these exact invariants:

```python
@model_validator(mode="after")
def status_matches_payload(self) -> "BusinessResponseEnvelope":
    if self.status == BusinessResponseStatus.SUCCESS:
        if self.result is None:
            raise ValueError("SUCCESS requires a result")
        if self.evidence and self.audit_status != "AUDITED_100_VERIFIED":
            raise ValueError("SUCCESS with historical evidence requires verified audit")
    elif self.result is not None:
        raise ValueError("non-SUCCESS responses cannot carry a business result")
    if self.status == BusinessResponseStatus.UNSUPPORTED and self.audit_status != "NOT_APPLICABLE":
        raise ValueError("UNSUPPORTED must use NOT_APPLICABLE audit")
    return self
```

- [ ] **Step 4: Run the strict model tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_models.py -v
```

Expected: all tests pass; invalid extra fields, false-success envelopes, and vector-only frequency plans are rejected.

- [ ] **Step 5: Commit the contract boundary**

```powershell
git add src/business_models.py tests/test_business_models.py
git commit -m "feat(eedi-rag): add strict business query contracts"
```

---

### Task 2: Implement the Two-Axis Deterministic Query Router

**Files:**
- Create: `src/query_intent.py`
- Create: `tests/test_query_intent.py`
- Reference: `src/query_rewriter.py:241-285`

**Interfaces:**
- Consumes: `BusinessQueryRequest`, `QueryUnderstanding`, `QueryIntent`, `QueryScope`, `ComputationScope`, and `EvidencePerspective` from Task 1.
- Produces: `BusinessQueryRouter.route(request: BusinessQueryRequest) -> QueryUnderstanding` and `extract_protected_tokens(text: str) -> list[str]`.

- [ ] **Step 1: Write routing tests for business intent, computation scope, and evidence perspective**

```python
# tests/test_query_intent.py
import pytest

from src.business_models import (
    BusinessQueryRequest,
    ComputationScope,
    EvidencePerspective,
    QueryIntent,
    QueryScope,
)
from src.query_intent import BusinessQueryRouter


@pytest.mark.parametrize(
    ("query", "intent", "scope", "computation", "perspectives"),
    [
        (
            "学生在辅导中最常提出的问题是什么？列出10条",
            QueryIntent.STUDENT_FREQUENT_QUESTIONS,
            QueryScope.CORPUS,
            ComputationScope.HYBRID,
            [EvidencePerspective.STUDENT],
        ),
        (
            "导师反复使用哪些引导策略？",
            QueryIntent.TUTOR_STRATEGY_PATTERNS,
            QueryScope.CORPUS,
            ComputationScope.HYBRID,
            [EvidencePerspective.TUTOR],
        ),
        (
            "学生为什么把 5.4598 四舍五入成 5.45，导师如何引导？",
            QueryIntent.QUESTION_TUTORING,
            QueryScope.QUESTION,
            ComputationScope.MICRO,
            [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ),
        (
            "那些多次重考或基础薄弱的学生有什么共性模式？",
            QueryIntent.STUDENT_COHORT_PATTERNS,
            QueryScope.CORPUS,
            ComputationScope.HYBRID,
            [EvidencePerspective.STUDENT],
        ),
    ],
)
def test_router_models_two_independent_axes(
    query, intent, scope, computation, perspectives
) -> None:
    result = BusinessQueryRouter().route(BusinessQueryRequest(query=query, top_k=10))
    assert result.primary_intent == intent
    assert result.scope == scope
    assert result.computation_scope == computation
    assert result.evidence_perspectives == perspectives


def test_router_preserves_formula_and_requested_top_k() -> None:
    result = BusinessQueryRouter().route(
        BusinessQueryRequest(
            query="Top 5：为什么 (-q)^2 和 -q^2 容易混淆？",
            top_k=5,
        )
    )
    assert result.requested_top_k == 5
    assert "5" in result.protected_tokens
    assert "(-q)^2" in result.protected_tokens
    assert "-q^2" in result.protected_tokens


def test_router_does_not_infer_user_role() -> None:
    result = BusinessQueryRouter().route(
        BusinessQueryRequest(query="我是主管，学生最常问什么？")
    )
    assert result.primary_intent == QueryIntent.STUDENT_FREQUENT_QUESTIONS
    assert not hasattr(result, "user_role")
```

- [ ] **Step 2: Run the routing tests and observe the missing-router failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_query_intent.py -v
```

Expected: collection fails because `src.query_intent` does not exist.

- [ ] **Step 3: Implement normalized matching, protected-token extraction, and explicit rule scores**

Use NFKC normalization for matching only, retain `request.query` unchanged, and implement these exact categories:

```python
# src/query_intent.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.business_models import (
    BusinessQueryRequest,
    ComputationScope,
    EvidencePerspective,
    QueryIntent,
    QueryScope,
    QueryUnderstanding,
    RouteConfidence,
)

PROTECTED_TOKEN_RE = re.compile(
    r"\([^\r\n]{1,80}\)\^?\d*|[+-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|(?:选项|option)\s*[A-D]",
    re.IGNORECASE,
)

INTENT_SIGNALS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.STUDENT_FREQUENT_QUESTIONS: ("最常提出的问题", "最常问", "高频问题", "faq"),
    QueryIntent.DIFFICULT_CONCEPTS: ("最难", "卡壳", "难理解", "耗时最长", "困难主题"),
    QueryIntent.RECURRING_MISCONCEPTIONS: ("共性误区", "认知错误", "误解反复", "错误理解"),
    QueryIntent.EXAM_ERROR_PATTERNS: ("考试错误", "错误模式", "误选", "审题错误"),
    QueryIntent.STUDENT_COHORT_PATTERNS: ("多次重考", "重考者", "基础薄弱", "低基础"),
    QueryIntent.TUTOR_STRATEGY_PATTERNS: ("导师", "老师", "怎么引导", "教学策略", "脚手架"),
    QueryIntent.CONTENT_IMPROVEMENT: ("教材", "微课", "题库", "内容改进", "教育内容"),
    QueryIntent.QUESTION_TUTORING: ("这题", "怎么算", "为什么不能", "我选", "答案选项"),
}

MACRO_SIGNALS = ("最常", "多少", "top", "反复", "总体", "全部", "共性", "模式")
STUDENT_SIGNALS = ("学生", "困惑", "误区", "错误", "重考", "基础薄弱")
TUTOR_SIGNALS = ("导师", "老师", "教", "引导", "策略", "话术", "脚手架")
```

The router must score exact phrases `+3`, individual domain terms `+1`, matching scope `+2`, and conflicts `-2`. It must set:

```text
score >= 4 and lead >= 2 -> HIGH
score >= 3 and lead >= 1 -> MEDIUM
otherwise -> LOW
```

For macro intents that also require citations, set `ComputationScope.HYBRID`. For specific formulas/questions, set `MICRO`. Perspective order is always `[STUDENT, TUTOR]` when both are required so trace and evidence IDs remain deterministic.

- [ ] **Step 4: Run routing tests and the existing query-rewriter tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_query_intent.py tests/test_query_rewriter.py -v
```

Expected: both suites pass; the business router does not modify or replace `MultiPerspectiveQueryRewriter`.

- [ ] **Step 5: Commit the deterministic router**

```powershell
git add src/query_intent.py tests/test_query_intent.py
git commit -m "feat(eedi-rag): route business intent and evidence perspective"
```

---

### Task 3: Add Artifact-Derived Data Capability Gates

**Files:**
- Create: `src/data_capabilities.py`
- Create: `tests/test_data_capabilities.py`
- Reference: `src/storage_manager.py:246-325`
- Reference: `src/models.py:107-144`

**Interfaces:**
- Consumes: a DuckDB connection opened against the authoritative source and an optional analytics sidecar connection.
- Produces: `DataCapabilityRegistry.inspect() -> dict[str, DataCapabilitySnapshot]` and `DataCapabilityRegistry.require(names: list[str]) -> list[DataCapabilitySnapshot]`.

- [ ] **Step 1: Write capability tests against explicit in-memory schemas**

```python
# tests/test_data_capabilities.py
import duckdb
import pytest

from src.data_capabilities import DataCapabilityRegistry, MissingDataCapabilityError


def test_current_eedi_schema_exposes_dialogue_but_not_cohort_capabilities() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE tutoring_sessions(intervention_id BIGINT)")
    conn.execute(
        "CREATE TABLE session_dialogue_turns("
        "intervention_id BIGINT, turn_id INTEGER, speaker VARCHAR, text VARCHAR)"
    )
    registry = DataCapabilityRegistry(source_conn=conn, analytics_conn=None)
    snapshot = registry.inspect()
    assert snapshot["DIALOGUE_EVIDENCE"].available is True
    assert snapshot["RETAKE_COHORT"].available is False
    assert "attempt_number" in snapshot["RETAKE_COHORT"].missing_requirements
    assert snapshot["ABILITY_COHORT"].available is False


def test_require_raises_with_machine_readable_missing_fields() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE tutoring_sessions(intervention_id BIGINT)")
    registry = DataCapabilityRegistry(source_conn=conn, analytics_conn=None)
    with pytest.raises(MissingDataCapabilityError) as captured:
        registry.require(["RETAKE_COHORT"])
    assert captured.value.capability == "RETAKE_COHORT"
    assert "student_key" in captured.value.missing_requirements
```

- [ ] **Step 2: Run the capability tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_data_capabilities.py -v
```

Expected: collection fails because `src.data_capabilities` does not exist.

- [ ] **Step 3: Implement schema inspection without creating or mutating tables**

```python
# src/data_capabilities.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.business_models import DataCapabilitySnapshot


REQUIREMENTS = {
    "DIALOGUE_EVIDENCE": {
        "table": "session_dialogue_turns",
        "columns": {"intervention_id", "turn_id", "speaker", "text"},
    },
    "BUSINESS_ANALYTICS": {
        "table": "business_analytics_manifest",
        "columns": {"source_artifact_hash", "analytics_version", "status"},
    },
    "RETAKE_COHORT": {
        "table": "student_session_features",
        "columns": {"session_id", "student_key", "attempt_number", "measured_at"},
    },
    "ABILITY_COHORT": {
        "table": "student_session_features",
        "columns": {
            "session_id",
            "student_key",
            "historic_question_count",
            "historic_correctness",
            "measured_at",
        },
    },
}


class MissingDataCapabilityError(RuntimeError):
    def __init__(self, capability: str, missing_requirements: list[str]) -> None:
        self.capability = capability
        self.missing_requirements = missing_requirements
        super().__init__(
            f"missing data capability {capability}: {', '.join(missing_requirements)}"
        )
```

Use `information_schema.tables` and `information_schema.columns`; do not catch `Exception` and return an available/default state. Set coverage to `0.0` when the table or required columns are absent. Part 2 will calculate non-zero coverage from imported rows.

- [ ] **Step 4: Run capability tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_data_capabilities.py -v
```

Expected: both tests pass and no table is created as a side effect.

- [ ] **Step 5: Commit capability gates**

```powershell
git add src/data_capabilities.py tests/test_data_capabilities.py
git commit -m "feat(eedi-rag): expose strict data capability gates"
```

---

### Task 4: Implement the Business Query Planner

**Files:**
- Create: `src/query_planner.py`
- Create: `tests/test_query_planner.py`
- Reference: `docs/Query_Intent_Response_Executable_Design.md:314-344`

**Interfaces:**
- Consumes: `QueryUnderstanding` and `dict[str, DataCapabilitySnapshot]`.
- Produces: `BusinessQueryPlanner.plan(understanding, capabilities) -> QueryPlan`.

- [ ] **Step 1: Write planner tests that prohibit vector-only statistics and unsupported cohort claims**

```python
# tests/test_query_planner.py
from src.business_models import (
    BusinessQueryRequest,
    QueryPlanType,
)
from src.data_capabilities import DataCapabilityRegistry
from src.query_intent import BusinessQueryRouter
from src.query_planner import BusinessQueryPlanner


def _caps(dialogue=True, analytics=True, retake=False, ability=False):
    return DataCapabilityRegistry.synthetic_snapshot(
        dialogue=dialogue,
        analytics=analytics,
        retake=retake,
        ability=ability,
    )


def test_frequent_question_plan_uses_sql_then_student_evidence() -> None:
    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="学生最常提出的问题是什么？", top_k=10)
    )
    plan = BusinessQueryPlanner().plan(understanding, _caps())
    assert plan.plan_type == QueryPlanType.ANALYTICS_WITH_EVIDENCE
    assert plan.analytics_operations == ["student_frequent_questions"]
    assert [item.value for item in plan.retrieval_perspectives] == ["STUDENT"]


def test_current_eedi_retaker_question_is_rejected_before_retrieval_or_llm() -> None:
    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="重考者最常见的思维卡点是什么？")
    )
    plan = BusinessQueryPlanner().plan(understanding, _caps(retake=False))
    assert plan.plan_type == QueryPlanType.REJECT
    assert plan.analytics_operations == []
    assert plan.retrieval_perspectives == []
    assert "attempt_number" in plan.rejection_reason


def test_specific_question_uses_existing_micro_rag_without_analytics() -> None:
    understanding = BusinessQueryRouter().route(
        BusinessQueryRequest(query="为什么 (-q)^2 和 -q^2 不同？导师怎么引导？")
    )
    plan = BusinessQueryPlanner().plan(understanding, _caps(analytics=False))
    assert plan.plan_type == QueryPlanType.SOCRATIC_TUTORING
    assert plan.analytics_operations == []
```

- [ ] **Step 2: Run planner tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_query_planner.py -v
```

Expected: collection fails because `src.query_planner` does not exist.

- [ ] **Step 3: Implement an explicit plan table rather than nested heuristic fallthrough**

```python
# src/query_planner.py
from __future__ import annotations

from src.business_models import (
    EvidencePerspective,
    QueryIntent,
    QueryPlan,
    QueryPlanType,
    QueryUnderstanding,
)

PLAN_MATRIX = {
    QueryIntent.STUDENT_FREQUENT_QUESTIONS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["student_frequent_questions"],
        [EvidencePerspective.STUDENT],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "STUDENT_FAQ_REPORT",
    ),
    QueryIntent.DIFFICULT_CONCEPTS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["difficult_concepts"],
        [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "DIFFICULTY_REPORT",
    ),
    QueryIntent.RECURRING_MISCONCEPTIONS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["recurring_misconceptions"],
        [EvidencePerspective.STUDENT],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "MISCONCEPTION_REPORT",
    ),
    QueryIntent.EXAM_ERROR_PATTERNS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["exam_error_patterns"],
        [EvidencePerspective.STUDENT],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "EXAM_ERROR_PATTERN_REPORT",
    ),
    QueryIntent.TUTOR_STRATEGY_PATTERNS: (
        QueryPlanType.ANALYTICS_WITH_EVIDENCE,
        ["tutor_strategy_patterns"],
        [EvidencePerspective.TUTOR],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "TUTOR_STRATEGY_REPORT",
    ),
    QueryIntent.CONTENT_IMPROVEMENT: (
        QueryPlanType.CONTENT_BRIEF,
        ["content_opportunities"],
        [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ["BUSINESS_ANALYTICS", "DIALOGUE_EVIDENCE"],
        "CONTENT_IMPROVEMENT_BRIEF",
    ),
    QueryIntent.QUESTION_TUTORING: (
        QueryPlanType.SOCRATIC_TUTORING,
        [],
        [EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
        ["DIALOGUE_EVIDENCE"],
        "SOCRATIC_TURN",
    ),
}
```

For `STUDENT_COHORT_PATTERNS`, require `RETAKE_COHORT` when the query contains “重考”, require `ABILITY_COHORT` when it contains “基础薄弱/低基础”, and require both when both are present. Return `REJECT` with the exact missing fields supplied by `DataCapabilitySnapshot`; do not route to generic student misconceptions.

- [ ] **Step 4: Run model, router, capability, and planner tests together**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_business_models.py `
  tests/test_query_intent.py `
  tests/test_data_capabilities.py `
  tests/test_query_planner.py -v
```

Expected: all tests pass with deterministic plan results.

- [ ] **Step 5: Commit the query planner**

```powershell
git add src/query_planner.py tests/test_query_planner.py
git commit -m "feat(eedi-rag): plan macro analytics and micro evidence execution"
```

---

### Task 5: Freeze and Evaluate the Router Dataset

**Files:**
- Create: `evals/datasets/query-router-v1.jsonl`
- Create: `scripts/evaluate_business_router.py`
- Create: `tests/test_business_router_eval.py`
- Reference: `evals/contracts.py:12-99`

**Interfaces:**
- Consumes: human-labelled JSONL rows and `BusinessQueryRouter`.
- Produces: an immutable JSON report with primary-intent accuracy, scope accuracy, perspective accuracy, protected-token preservation, unsupported precision, latency P95, and per-case failures.

- [ ] **Step 1: Write dataset contract and evaluator tests**

```python
# tests/test_business_router_eval.py
import json

import pytest

from scripts.evaluate_business_router import RouterCase, evaluate_router_cases


def test_router_case_requires_human_label_and_valid_split() -> None:
    with pytest.raises(ValueError):
        RouterCase.model_validate(
            {
                "case_id": "router-001",
                "query": "学生最常问什么？",
                "primary_intent": "STUDENT_FREQUENT_QUESTIONS",
                "scope": "CORPUS",
                "evidence_perspectives": ["STUDENT"],
                "protected_tokens": [],
                "label_source": "generated",
                "split": "dev",
            }
        )


def test_router_evaluator_reports_failures_without_overwriting_them() -> None:
    cases = [
        RouterCase(
            case_id="router-001",
            query="学生最常问什么？",
            primary_intent="STUDENT_FREQUENT_QUESTIONS",
            scope="CORPUS",
            computation_scope="HYBRID",
            evidence_perspectives=["STUDENT"],
            protected_tokens=[],
            label_source="human",
            split="dev",
        )
    ]
    report = evaluate_router_cases(cases)
    assert report["case_count"] == 1
    assert report["metrics"]["primary_intent_accuracy"] == 1.0
    assert report["failures"] == []
```

- [ ] **Step 2: Run the evaluator tests and observe the missing-script failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_router_eval.py -v
```

Expected: collection fails because `scripts.evaluate_business_router` does not exist.

- [ ] **Step 3: Create the human-labelled routing dataset**

Author at least 135 rows: 15 for each of the nine `QueryIntent` values, with at least 20% English or mixed-language queries, 20 formula/number cases, 15 subject-scope cases, 15 question-scope cases, 10 compound-perspective cases, and 15 unsupported/data-capability cases. Every row uses this strict shape:

```json
{"case_id":"router-001","query":"学生在辅导中最常提出的问题是什么？","primary_intent":"STUDENT_FREQUENT_QUESTIONS","secondary_intents":[],"scope":"CORPUS","computation_scope":"HYBRID","evidence_perspectives":["STUDENT"],"subject_filters":[],"protected_tokens":[],"requested_top_k":3,"label_source":"human","split":"dev"}
{"case_id":"router-002","query":"Top 5: why do students confuse (-q)^2 with -q^2?","primary_intent":"RECURRING_MISCONCEPTIONS","secondary_intents":[],"scope":"SUBJECT","computation_scope":"HYBRID","evidence_perspectives":["STUDENT"],"subject_filters":["Algebra > Algebraic Expressions > Powers and Indices"],"protected_tokens":["5","(-q)^2","-q^2"],"requested_top_k":5,"label_source":"human","split":"holdout"}
{"case_id":"router-003","query":"那些多次重考或基础薄弱的学生有什么共同卡点？","primary_intent":"STUDENT_COHORT_PATTERNS","secondary_intents":[],"scope":"CORPUS","computation_scope":"HYBRID","evidence_perspectives":["STUDENT"],"subject_filters":[],"protected_tokens":[],"requested_top_k":3,"label_source":"human","split":"holdout"}
```

Use a deterministic `case_id`; never generate expected labels by calling the system under test. A human reviewer must sign off the JSONL in the pull request.

- [ ] **Step 4: Implement the evaluator and enforce release gates**

The evaluator must load JSONL with `RouterCase(extra="forbid", strict=True)`, time each local route, and write a new output directory rather than overwrite a previous report. Enforce:

```python
GATES = {
    "primary_intent_accuracy": 0.95,
    "scope_accuracy": 0.95,
    "computation_scope_accuracy": 0.95,
    "perspective_exact_match": 0.95,
    "protected_token_preservation": 1.0,
    "unsupported_precision": 1.0,
    "deterministic_router_p95_ms": 10.0,
}
```

Command interface:

```powershell
.venv\Scripts\python.exe scripts/evaluate_business_router.py `
  --dataset evals/datasets/query-router-v1.jsonl `
  --split holdout `
  --output reports/eval/business-router-v1-20260907
```

Expected: `report.json`, `report.md`, and `failures.jsonl` are created; exit code is non-zero if any gate fails.

- [ ] **Step 5: Run the Part 1 verification suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_business_models.py `
  tests/test_query_intent.py `
  tests/test_data_capabilities.py `
  tests/test_query_planner.py `
  tests/test_business_router_eval.py -v

.venv\Scripts\python.exe scripts/evaluate_business_router.py `
  --dataset evals/datasets/query-router-v1.jsonl `
  --split holdout `
  --output reports/eval/business-router-v1-20260907
```

Expected: tests pass; evaluator truthfully reports pass/fail against all gates.

- [ ] **Step 6: Commit the frozen dataset and evaluator**

```powershell
git add evals/datasets/query-router-v1.jsonl scripts/evaluate_business_router.py tests/test_business_router_eval.py
git commit -m "test(eedi-rag): freeze business router acceptance set"
```

## Part 1 Exit Gate

Part 1 is complete only when:

- Every supported query maps to an explicit intent, scope, computation scope, and ordered evidence perspective list.
- Frequency/ranking queries cannot produce a vector-only plan.
- Retaker/weak-foundation queries are recognized as cohort queries rather than silently mapped to generic misconceptions.
- Current Eedi artifacts produce `REJECT` for unavailable cohort capabilities with exact missing fields.
- The original `EndToEndPedagogicalRAGPipeline.ask()` behavior and existing query-rewriter tests remain unchanged.
- Router holdout metrics and all failures are persisted without being rewritten as success.
