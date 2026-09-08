# Business RAG Capability Upgrade Part 2: Analytics Events and Data Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned, auditable analytics sidecar that turns authoritative dialogue turns and existing card facts into countable student-question, difficulty, misconception, exam-error, and tutor-strategy events, while exposing longitudinal cohort analysis only when authorized data is actually present.

**Architecture:** The current DuckDB remains the immutable authority for sessions and turns, and the current Chroma collections remain the micro evidence index. A separate DuckDB analytics artifact stores derived events plus source hashes and derivation versions. SQL computes macro counts and rankings; every aggregate row retains representative `(session_id, turn_id)` pointers for Part 3 to materialize as micro evidence.

**Tech Stack:** Python 3.14, DuckDB, Pydantic 2, pandas already present in the project, pytest 9, SHA-256 artifact provenance.

**Spec:** `docs/需求.md`; `docs/Query_Intent_Response_Executable_Design.md`; `docs/superpowers/plans/2026-09-07-business-rag-contract-routing.md`

## Global Constraints

- Complete Part 1 before executing this plan; import its types instead of redefining query intent, scope, perspective, status, or capability models.
- Read the authoritative `data/db/tutoring_knowledge.duckdb` in read-only mode during analytics builds.
- Build analytics into a new staging file and promote it only after row-level evidence validation and source-hash validation pass.
- Do not add derived tables to the authoritative dialogue DuckDB in this phase.
- Every derived event must carry a deterministic ID, derivation version, source Session/Turn pointers, and an evidence verification state.
- `raw_question` and other quoted fields must be exact authoritative Turn text or an exact substring; normalized labels may never replace the raw evidence.
- Count coverage with `COUNT(DISTINCT session_id)` as the primary prevalence measure; expose event count separately.
- Do not infer `REPEATED_INCORRECT_ATTEMPT`, `STRATEGY_SWITCH`, `UNRESOLVED_ENDING`, retaker status, or weak-foundation status when their required source fields are absent. Report those metrics as `UNMEASURED` or capabilities as unavailable.
- Do not label an individual student as weak, careless, or low ability from a single dialogue.
- Current Eedi operation must not create a synthetic student identity from TutorId, InterventionId, message order, names, or text similarity.
- Longitudinal import accepts only pseudonymous student keys and explicitly authorized metadata; it rejects raw names, email addresses, phone numbers, cookies, and credentials.
- No LLM-generated number, percentage, rank, Session ID, Turn ID, or evidence quotation may be persisted as a deterministic aggregate fact.

## Delivery Boundary

This is Part 2 of 3. It produces a usable analytics artifact and deterministic report facts. It does not yet call the existing RAG pipeline or an answer generator.

```text
Authoritative DuckDB (read-only)
  -> deterministic event extraction/derivation
  -> row-level evidence audit
  -> analytics staging DuckDB
  -> manifest status=READY
  -> immutable analytics artifact
  -> AnalyticsRepository aggregate results
```

## File Structure

```text
src/
├── analytics_models.py       # strict event and aggregate result models
├── analytics_store.py        # sidecar DDL, provenance validation, read-only repository
├── event_extraction.py       # student-question and measured difficulty rules
├── event_derivation.py       # misconception/error/strategy derivation from existing cards
├── analytics.py              # deterministic aggregate queries
└── cohort_data.py            # authorized longitudinal import and cohort capability checks

scripts/
├── build_business_analytics.py
└── import_authorized_cohort_data.py

tests/
├── test_analytics_models.py
├── test_analytics_store.py
├── test_event_extraction.py
├── test_event_derivation.py
├── test_analytics.py
└── test_cohort_data.py

data/business/
└── cohort-definitions.schema.json
```

---

### Task 1: Define Event and Aggregate Result Contracts

**Files:**
- Create: `src/analytics_models.py`
- Create: `tests/test_analytics_models.py`
- Reference: `src/models.py:73-144`
- Reference: `src/models.py:161-289`

**Interfaces:**
- Consumes: no runtime state; models receive already parsed values.
- Produces: `QuestionFunction`, `DifficultySignalType`, `ExamErrorCategory`, `MeasurementStatus`, `StudentQuestionEvent`, `DifficultyEvent`, `MisconceptionEvent`, `ExamErrorEvent`, `TutorStrategyEvent`, `RepresentativeEvidenceRef`, `AggregateMetric`, `FAQAggregate`, `DifficultyAggregate`, `MisconceptionAggregate`, `ExamErrorAggregate`, and `TutorStrategyAggregate`.

- [ ] **Step 1: Write model tests for exact raw evidence and measured/unmeasured consistency**

```python
# tests/test_analytics_models.py
import pytest
from pydantic import ValidationError

from src.analytics_models import (
    DifficultyEvent,
    DifficultySignalType,
    MeasurementStatus,
    StudentQuestionEvent,
)


def test_student_question_event_requires_exact_source_pointer() -> None:
    event = StudentQuestionEvent(
        event_id="sqe-v1-10-4",
        session_id=10,
        turn_id=4,
        subject_path="Number > Rounding",
        raw_question="How do I start?",
        normalized_question="How do I start this type of question?",
        question_cluster_id="number-rounding:ask-next-step",
        question_function="ASK_NEXT_STEP",
        extraction_method="deterministic-question-v1",
        derivation_version="business-events-v1",
        evidence_verified=True,
    )
    assert event.turn_id == 4
    with pytest.raises(ValidationError):
        event.model_copy(update={"turn_id": 0}, deep=True).__class__.model_validate(
            {**event.model_dump(), "turn_id": 0}
        )


def test_unmeasured_difficulty_signal_cannot_store_zero_as_observation() -> None:
    with pytest.raises(ValidationError, match="UNMEASURED"):
        DifficultyEvent(
            event_id="diff-v1-10-unresolved",
            session_id=10,
            subject_path="Number > Rounding",
            signal_type=DifficultySignalType.UNRESOLVED_ENDING,
            measurement_status=MeasurementStatus.UNMEASURED,
            signal_value=0.0,
            source_turn_ids=[],
            derivation_version="business-events-v1",
        )
```

- [ ] **Step 2: Run the model tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_analytics_models.py -v
```

Expected: collection fails because `src.analytics_models` does not exist.

- [ ] **Step 3: Implement strict enums and event models**

```python
# src/analytics_models.py
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from src.business_models import StrictBusinessModel


class QuestionFunction(str, Enum):
    REQUEST_EXPLANATION = "REQUEST_EXPLANATION"
    REQUEST_CONFIRMATION = "REQUEST_CONFIRMATION"
    ASK_NEXT_STEP = "ASK_NEXT_STEP"
    COMPARE_OPTIONS = "COMPARE_OPTIONS"
    RECALL_RULE = "RECALL_RULE"
    ASK_DEFINITION = "ASK_DEFINITION"
    EXPRESS_UNCERTAINTY = "EXPRESS_UNCERTAINTY"
    OTHER = "OTHER"


class DifficultySignalType(str, Enum):
    REPEATED_HELP_REQUEST = "REPEATED_HELP_REQUEST"
    LONG_TUTORING_SEQUENCE = "LONG_TUTORING_SEQUENCE"
    REPEATED_INCORRECT_ATTEMPT = "REPEATED_INCORRECT_ATTEMPT"
    STRATEGY_SWITCH = "STRATEGY_SWITCH"
    UNRESOLVED_ENDING = "UNRESOLVED_ENDING"


class ExamErrorCategory(str, Enum):
    CONCEPTUAL = "CONCEPTUAL"
    PROCEDURAL = "PROCEDURAL"
    REPRESENTATION = "REPRESENTATION"
    LANGUAGE_INTERPRETATION = "LANGUAGE_INTERPRETATION"
    CONDITION_OMISSION = "CONDITION_OMISSION"
    OVERGENERALIZATION = "OVERGENERALIZATION"
    OPTION_COMPARISON = "OPTION_COMPARISON"
    UNCLASSIFIED = "UNCLASSIFIED"


class MeasurementStatus(str, Enum):
    MEASURED = "MEASURED"
    UNMEASURED = "UNMEASURED"


class StudentQuestionEvent(StrictBusinessModel):
    event_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    turn_id: int = Field(ge=1)
    subject_path: str = Field(min_length=1)
    raw_question: str = Field(min_length=1)
    normalized_question: str = Field(min_length=1)
    question_cluster_id: str = Field(min_length=1)
    question_function: QuestionFunction
    extraction_method: Literal["deterministic-question-v1"]
    derivation_version: Literal["business-events-v1"]
    evidence_verified: bool


class DifficultyEvent(StrictBusinessModel):
    event_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    subject_path: str = Field(min_length=1)
    signal_type: DifficultySignalType
    measurement_status: MeasurementStatus
    signal_value: float | None
    source_turn_ids: list[int]
    derivation_version: Literal["business-events-v1"]

    @model_validator(mode="after")
    def measurement_matches_value(self) -> "DifficultyEvent":
        if self.measurement_status == MeasurementStatus.UNMEASURED:
            if self.signal_value is not None or self.source_turn_ids:
                raise ValueError("UNMEASURED signal cannot carry a value or evidence")
        elif self.signal_value is None or not self.source_turn_ids:
            raise ValueError("MEASURED signal requires value and source turns")
        return self
```

Add these exact shared provenance and aggregate models next:

```python
class RepresentativeEvidenceRef(StrictBusinessModel):
    source_event_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    turn_ids: list[int] = Field(min_length=1)
    perspective: Literal["STUDENT", "TUTOR"]
    subject_path: str = Field(min_length=1)


class MisconceptionEvent(StrictBusinessModel):
    event_id: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    subject_path: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    error_category: ExamErrorCategory
    manifestation: str = Field(min_length=1)
    source_turn_ids: list[int] = Field(min_length=1)
    derivation_version: Literal["business-events-v1"]
    evidence_verified: bool


class ExamErrorEvent(StrictBusinessModel):
    event_id: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    subject_path: str = Field(min_length=1)
    pattern_id: str = Field(min_length=1)
    pattern_name: str = Field(min_length=1)
    error_category: ExamErrorCategory
    source_turn_ids: list[int] = Field(min_length=1)
    derivation_version: Literal["business-events-v1"]
    evidence_verified: bool


class TutorStrategyEvent(StrictBusinessModel):
    event_id: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    subject_path: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    strategy_name: str = Field(min_length=1)
    strategy_category: str = Field(min_length=1)
    source_turn_ids: list[int] = Field(min_length=1)
    derivation_version: Literal["business-events-v1"]
    evidence_verified: bool


class AggregateBase(StrictBusinessModel):
    rank: int = Field(ge=1)
    event_count: int = Field(ge=1)
    distinct_session_count: int = Field(ge=1)
    representative_refs: list[RepresentativeEvidenceRef] = Field(min_length=1)
```

Every aggregate-specific model extends `AggregateBase` and adds its canonical key, display name, category, and optional `session_share`. Aggregate rows never store generated summaries as source facts.

- [ ] **Step 4: Run the model tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_analytics_models.py -v
```

Expected: all model invariants pass.

- [ ] **Step 5: Commit analytics contracts**

```powershell
git add src/analytics_models.py tests/test_analytics_models.py
git commit -m "feat(eedi-rag): define auditable analytics event contracts"
```

---

### Task 2: Create the Versioned Analytics Sidecar Store

**Files:**
- Create: `src/analytics_store.py`
- Create: `tests/test_analytics_store.py`
- Create: `scripts/build_business_analytics.py`
- Reference: `src/knowledge_import.py:98-107`
- Reference: `scripts/import_full_knowledge_base.py:95-181`

**Interfaces:**
- Consumes: authoritative source DuckDB path, new analytics output path, `source_sha256()`.
- Produces: `AnalyticsStore.create_staging(path, manifest)`, `AnalyticsStore.open_ready(path, expected_source_hash)`, transactional `replace_events(...)`, and CLI build lifecycle with exit codes.

- [ ] **Step 1: Write sidecar creation and stale-artifact rejection tests**

```python
# tests/test_analytics_store.py
from pathlib import Path

import duckdb
import pytest

from src.analytics_store import AnalyticsManifest, AnalyticsStore


def test_analytics_store_records_source_hash_and_ready_state(tmp_path: Path) -> None:
    path = tmp_path / "business-analytics.duckdb"
    manifest = AnalyticsManifest(
        analytics_version="business-events-v1",
        source_artifact_hash="a" * 64,
        source_session_count=2,
        build_status="BUILDING",
    )
    store = AnalyticsStore.create_staging(path, manifest)
    store.mark_ready(row_counts={"student_question_events": 0})
    store.close()
    ready = AnalyticsStore.open_ready(path, expected_source_hash="a" * 64)
    assert ready.manifest.build_status == "READY"
    ready.close()


def test_analytics_store_rejects_stale_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "business-analytics.duckdb"
    store = AnalyticsStore.create_staging(
        path,
        AnalyticsManifest(
            analytics_version="business-events-v1",
            source_artifact_hash="a" * 64,
            source_session_count=2,
            build_status="BUILDING",
        ),
    )
    store.mark_ready(row_counts={})
    store.close()
    with pytest.raises(ValueError, match="source hash mismatch"):
        AnalyticsStore.open_ready(path, expected_source_hash="b" * 64)
```

- [ ] **Step 2: Run the sidecar tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_analytics_store.py -v
```

Expected: collection fails because `src.analytics_store` does not exist.

- [ ] **Step 3: Implement sidecar DDL and manifest validation**

Create these concrete models before implementing the store:

```python
class AnalyticsManifest(StrictBusinessModel):
    analytics_version: str = Field(min_length=1)
    source_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_session_count: int = Field(ge=0)
    build_status: Literal["BUILDING", "FAILED", "READY"]
    built_at_utc: datetime
    row_counts: dict[str, int] = Field(default_factory=dict)
    failure_reason: str | None = None

    @model_validator(mode="after")
    def status_matches_failure(self) -> "AnalyticsManifest":
        if self.build_status == "FAILED" and not self.failure_reason:
            raise ValueError("FAILED analytics manifest requires failure_reason")
        if self.build_status != "FAILED" and self.failure_reason is not None:
            raise ValueError("only FAILED analytics manifest may carry failure_reason")
        return self


class BusinessAnalyticsBuildResult(StrictBusinessModel):
    build_status: Literal["READY", "FAILED"]
    source_hash_before: str
    source_hash_after: str
    row_counts: dict[str, int]
    invalid_evidence_count: int = Field(ge=0)
    manifest_path: str = Field(min_length=1)
```

Use one-row manifest semantics and these tables:

```sql
CREATE TABLE business_analytics_manifest (
    analytics_version VARCHAR PRIMARY KEY,
    source_artifact_hash VARCHAR NOT NULL,
    source_session_count BIGINT NOT NULL,
    build_status VARCHAR NOT NULL CHECK (build_status IN ('BUILDING', 'FAILED', 'READY')),
    built_at_utc TIMESTAMP NOT NULL,
    row_counts_json VARCHAR NOT NULL
);

CREATE TABLE student_question_events (
    event_id VARCHAR PRIMARY KEY,
    session_id BIGINT NOT NULL,
    turn_id INTEGER NOT NULL,
    subject_path VARCHAR NOT NULL,
    raw_question VARCHAR NOT NULL,
    normalized_question VARCHAR NOT NULL,
    question_cluster_id VARCHAR NOT NULL,
    question_function VARCHAR NOT NULL,
    extraction_method VARCHAR NOT NULL,
    derivation_version VARCHAR NOT NULL,
    evidence_verified BOOLEAN NOT NULL
);

CREATE TABLE concept_difficulty_events (
    event_id VARCHAR PRIMARY KEY,
    session_id BIGINT NOT NULL,
    subject_path VARCHAR NOT NULL,
    signal_type VARCHAR NOT NULL,
    measurement_status VARCHAR NOT NULL,
    signal_value DOUBLE,
    source_turn_ids INTEGER[] NOT NULL,
    derivation_version VARCHAR NOT NULL
);

CREATE TABLE misconception_events (
    event_id VARCHAR PRIMARY KEY,
    source_chunk_id VARCHAR NOT NULL,
    session_id BIGINT NOT NULL,
    subject_path VARCHAR NOT NULL,
    cluster_id VARCHAR NOT NULL,
    canonical_name VARCHAR NOT NULL,
    error_category VARCHAR NOT NULL,
    manifestation VARCHAR NOT NULL,
    source_turn_ids INTEGER[] NOT NULL,
    derivation_version VARCHAR NOT NULL,
    evidence_verified BOOLEAN NOT NULL
);

CREATE TABLE exam_error_events (
    event_id VARCHAR PRIMARY KEY,
    source_chunk_id VARCHAR NOT NULL,
    session_id BIGINT NOT NULL,
    subject_path VARCHAR NOT NULL,
    pattern_id VARCHAR NOT NULL,
    pattern_name VARCHAR NOT NULL,
    error_category VARCHAR NOT NULL,
    source_turn_ids INTEGER[] NOT NULL,
    derivation_version VARCHAR NOT NULL,
    evidence_verified BOOLEAN NOT NULL
);

CREATE TABLE tutor_strategy_events (
    event_id VARCHAR PRIMARY KEY,
    source_chunk_id VARCHAR NOT NULL,
    session_id BIGINT NOT NULL,
    subject_path VARCHAR NOT NULL,
    strategy_id VARCHAR NOT NULL,
    strategy_name VARCHAR NOT NULL,
    strategy_category VARCHAR NOT NULL,
    source_turn_ids INTEGER[] NOT NULL,
    derivation_version VARCHAR NOT NULL,
    evidence_verified BOOLEAN NOT NULL
);

CREATE TABLE student_session_features (
    session_id BIGINT PRIMARY KEY,
    student_key VARCHAR NOT NULL,
    attempt_number INTEGER,
    historic_question_count INTEGER,
    historic_correctness DOUBLE,
    measured_at TIMESTAMP NOT NULL,
    source_system VARCHAR NOT NULL,
    authorization_reference VARCHAR NOT NULL,
    source_record_hash VARCHAR NOT NULL
);
```

`open_ready()` must use `duckdb.connect(path, read_only=True)`, require exactly one `READY` manifest row, verify `expected_source_hash`, and reject missing/duplicate/BUILDING/FAILED manifests.

- [ ] **Step 4: Implement safe staging build semantics**

The CLI accepts explicit absolute or repository-root-resolved paths:

```powershell
.venv\Scripts\python.exe scripts/build_business_analytics.py `
  --source-db data/db/tutoring_knowledge.duckdb `
  --output reports/staging/business-analytics/business-analytics.duckdb `
  --analytics-version business-events-v1
```

It must:

1. resolve and verify the source path exists;
2. hash the source before opening;
3. open source with `read_only=True`;
4. refuse to use the same resolved path for source and output;
5. create only the explicit staging output;
6. keep manifest `BUILDING` until all event audits complete;
7. mark `FAILED` with a concise non-secret failure reason on a controlled build failure;
8. re-hash the source and fail if it changed during the build;
9. mark `READY` only after row counts and evidence checks pass.

- [ ] **Step 5: Run sidecar tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_analytics_store.py -v
```

Expected: all tests pass; the source fixture remains unchanged.

- [ ] **Step 6: Commit the sidecar store skeleton**

```powershell
git add src/analytics_store.py scripts/build_business_analytics.py tests/test_analytics_store.py
git commit -m "feat(eedi-rag): add versioned analytics sidecar store"
```

---

### Task 3: Extract Auditable Student Question Events

**Files:**
- Create: `src/event_extraction.py`
- Create: `tests/test_event_extraction.py`
- Modify: `scripts/build_business_analytics.py`
- Reference: `src/storage_manager.py:734-794`

**Interfaces:**
- Consumes: authoritative session row plus ordered authoritative student/tutor Turn rows.
- Produces: `extract_student_question_events(session_id, subject_path, turns) -> list[StudentQuestionEvent]` and `derive_measured_difficulty_events(...) -> list[DifficultyEvent]`.

- [ ] **Step 1: Write event extraction tests with exact-quote evidence**

```python
# tests/test_event_extraction.py
from src.analytics_models import QuestionFunction
from src.event_extraction import extract_student_question_events


def test_extracts_student_questions_without_rewriting_raw_evidence() -> None:
    turns = [
        {"turn_id": 1, "speaker": "tutor", "text": "What have you tried?", "is_greeting_or_noise": False},
        {"turn_id": 2, "speaker": "student", "text": "I don't know how to start", "is_greeting_or_noise": False},
        {"turn_id": 3, "speaker": "student", "text": "Is it B?", "is_greeting_or_noise": False},
    ]
    events = extract_student_question_events(10, "Number > Fractions", turns)
    assert [item.turn_id for item in events] == [2, 3]
    assert events[0].raw_question == "I don't know how to start"
    assert events[0].question_function == QuestionFunction.ASK_NEXT_STEP
    assert events[1].question_function == QuestionFunction.REQUEST_CONFIRMATION
    assert all(item.evidence_verified for item in events)


def test_ignores_tutor_questions_and_greeting_noise() -> None:
    turns = [
        {"turn_id": 1, "speaker": "tutor", "text": "How are you?", "is_greeting_or_noise": True},
        {"turn_id": 2, "speaker": "student", "text": "hello", "is_greeting_or_noise": True},
    ]
    assert extract_student_question_events(10, "Number", turns) == []


def test_cluster_is_subject_plus_controlled_question_function() -> None:
    turns = [{"turn_id": 2, "speaker": "student", "text": "What do I do first?", "is_greeting_or_noise": False}]
    event = extract_student_question_events(10, "Number > Fractions", turns)[0]
    assert event.question_cluster_id == "number-fractions:ask-next-step"
```

- [ ] **Step 2: Run extraction tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_event_extraction.py -v
```

Expected: collection fails because `src.event_extraction` does not exist.

- [ ] **Step 3: Implement conservative question-event rules**

Recognize an event only when the student Turn contains `?` or one of these case-insensitive signals:

```python
QUESTION_RULES = (
    (QuestionFunction.ASK_NEXT_STEP, ("how do i start", "what do i do", "what next", "where do i start")),
    (QuestionFunction.REQUEST_CONFIRMATION, ("is it", "am i right", "is that right", "correct?")),
    (QuestionFunction.REQUEST_EXPLANATION, ("why", "how does", "can you explain", "don't understand")),
    (QuestionFunction.COMPARE_OPTIONS, ("option", "rather than", "difference between", "which answer")),
    (QuestionFunction.RECALL_RULE, ("what formula", "which rule", "do i multiply", "do i divide")),
    (QuestionFunction.ASK_DEFINITION, ("what is", "what does", "mean?")),
    (QuestionFunction.EXPRESS_UNCERTAINTY, ("i don't know", "not sure", "confused", "maybe")),
)
```

Priority follows the tuple order. If a Turn has `?` but matches no rule, use `OTHER`. `normalized_question` is the controlled display label for the matched function, not a fabricated quotation. `question_cluster_id` is `slug(subject_path) + ":" + question_function.casefold().replace("_", "-")`. Verify `raw_question == authoritative_turn["text"]` before setting `evidence_verified=True`.

- [ ] **Step 4: Add only currently measurable difficulty signals**

Implement:

```text
LONG_TUTORING_SEQUENCE: one measured row per session, signal_value=total non-noise turns,
                        source_turn_ids=all non-noise Turn IDs.
REPEATED_HELP_REQUEST: one measured row only when at least two student question events have
                       ASK_NEXT_STEP, REQUEST_EXPLANATION, or EXPRESS_UNCERTAINTY;
                       signal_value=count, source_turn_ids=those exact student Turn IDs.
```

Do not emit rows for `REPEATED_INCORRECT_ATTEMPT`, `STRATEGY_SWITCH`, or `UNRESOLVED_ENDING` in v1. Their absence is surfaced as `UNMEASURED` in `AnalyticsRepository`, not converted to zero.

- [ ] **Step 5: Wire extraction into the staging builder and audit every row**

For each `StudentQuestionEvent`, execute a parameterized source query by `(session_id, turn_id, speaker='student')` and require exact text equality. For each `DifficultyEvent`, require every `source_turn_id` exists in the same Session. Roll back the event batch and mark the build failed on the first mismatch.

- [ ] **Step 6: Run extraction and sidecar tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_analytics_models.py `
  tests/test_analytics_store.py `
  tests/test_event_extraction.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit question and difficulty extraction**

```powershell
git add src/event_extraction.py scripts/build_business_analytics.py tests/test_event_extraction.py
git commit -m "feat(eedi-rag): derive verified student question events"
```

---

### Task 4: Derive Misconception, Exam-Error, and Tutor-Strategy Events

**Files:**
- Create: `src/event_derivation.py`
- Create: `tests/test_event_derivation.py`
- Modify: `scripts/build_business_analytics.py`
- Reference: `src/storage_manager.py:280-312`

**Interfaces:**
- Consumes: authoritative `misconception_chunks`, `tutor_strategy_chunks`, and `session_dialogue_turns` rows.
- Produces: `derive_misconception_event(row, authoritative_turns)`, `derive_exam_error_event(...)`, and `derive_tutor_strategy_event(...)`.

- [ ] **Step 1: Write conservative classification and grounding tests**

```python
# tests/test_event_derivation.py
import pytest

from src.analytics_models import ExamErrorCategory
from src.event_derivation import (
    derive_exam_error_event,
    derive_misconception_event,
    derive_tutor_strategy_event,
)


def test_misconception_cluster_uses_subject_and_controlled_error_category() -> None:
    row = {
        "chunk_id": "session_10_misconception",
        "session_id": 10,
        "subject_path": "Algebra > Powers",
        "misconception_name": "括号范围混淆",
        "deep_mechanism": "学生忽略负号与括号的作用范围",
        "source_turn_ids": [2],
    }
    event = derive_misconception_event(
        row,
        authoritative_turns={2: {"speaker": "student", "text": "I thought it stayed negative"}},
    )
    assert event.error_category == ExamErrorCategory.CONCEPTUAL
    assert event.cluster_id == "algebra-powers:conceptual"
    assert event.evidence_verified is True


def test_unknown_error_category_stays_unclassified() -> None:
    row = {
        "chunk_id": "session_10_misconception",
        "session_id": 10,
        "subject_path": "Number",
        "misconception_name": "其他表现",
        "deep_mechanism": "没有足够信息进行稳定分类",
        "source_turn_ids": [2],
    }
    event = derive_exam_error_event(
        row,
        authoritative_turns={2: {"speaker": "student", "text": "maybe"}},
    )
    assert event.error_category == ExamErrorCategory.UNCLASSIFIED


def test_strategy_event_rejects_non_tutor_source_turn() -> None:
    row = {
        "chunk_id": "session_10_tutor_strategy",
        "session_id": 10,
        "subject_path": "Number",
        "strategy_category": "Scaffolding",
        "key_aha_question": "What should we do first?",
        "source_turn_ids": [2],
    }
    with pytest.raises(ValueError, match="tutor"):
        derive_tutor_strategy_event(
            row,
            authoritative_turns={2: {"speaker": "student", "text": "I don't know"}},
        )
```

- [ ] **Step 2: Run derivation tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_event_derivation.py -v
```

Expected: collection fails because `src.event_derivation` does not exist.

- [ ] **Step 3: Implement a declared, versioned error-category ruleset**

Apply first-match priority to normalized `misconception_name + deep_mechanism + error_choice`:

```python
ERROR_CATEGORY_RULES = (
    (ExamErrorCategory.OPTION_COMPARISON, ("选项", "option", "distractor", "两个答案")),
    (ExamErrorCategory.CONDITION_OMISSION, ("忽略条件", "漏掉", "忘记翻转", "omission")),
    (ExamErrorCategory.LANGUAGE_INTERPRETATION, ("题意", "语言", "误读", "phrase", "interpret")),
    (ExamErrorCategory.REPRESENTATION, ("图形", "图表", "表示", "坐标", "diagram")),
    (ExamErrorCategory.OVERGENERALIZATION, ("过度泛化", "一律", "总是", "套用", "generalization")),
    (ExamErrorCategory.PROCEDURAL, ("步骤", "运算顺序", "计算过程", "程序", "procedure")),
    (ExamErrorCategory.CONCEPTUAL, ("概念", "机理", "混淆", "理解", "concept")),
)
```

No match returns `UNCLASSIFIED`. The ruleset version is `error-taxonomy-v1` and must be recorded in the analytics manifest. This classifier does not claim human-reviewed diagnostic truth; the report must disclose `derivation_method="deterministic_rules"`.

- [ ] **Step 4: Derive stable aggregation keys**

Use:

```text
misconception cluster_id = slug(subject_path) + ':' + error_category.casefold()
exam pattern_id          = 'exam-error:' + error_category.casefold()
tutor strategy_id        = 'tutor-strategy:' + strategy_category.casefold()
```

Preserve the original `misconception_name`, `deep_mechanism`, `key_aha_question`, and strategy metadata as manifestations/source fields. Never group directly by unconstrained LLM prose.

- [ ] **Step 5: Audit roles and exact source pointers during derivation**

Require all misconception/error `source_turn_ids` to resolve to student Turns and all strategy `source_turn_ids` to resolve to tutor Turns. If a card has empty/missing IDs or a role mismatch, exclude nothing silently: raise an evidence validation error, roll back the staging transaction, and mark the build `FAILED`.

- [ ] **Step 6: Run derivation and existing card-grounding tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_event_derivation.py `
  tests/test_storage_manager.py::test_direct_card_ingest_enforces_evidence_roles `
  tests/test_evidence_index.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the derived event lanes**

```powershell
git add src/event_derivation.py scripts/build_business_analytics.py tests/test_event_derivation.py
git commit -m "feat(eedi-rag): derive misconception and strategy analytics events"
```

---

### Task 5: Implement Deterministic Macro Analytics

**Files:**
- Create: `src/analytics.py`
- Create: `tests/test_analytics.py`
- Reference: `src/storage_manager.py:761-842`

**Interfaces:**
- Consumes: a ready, read-only `AnalyticsStore` plus `subject_filters` and `top_k`.
- Produces: `AnalyticsRepository.student_frequent_questions(top_k: int, subject_filters: list[str]) -> tuple[list[FAQAggregate], DataScope]`, `.difficult_concepts(...) -> tuple[list[DifficultyAggregate], DataScope]`, `.recurring_misconceptions(...) -> tuple[list[MisconceptionAggregate], DataScope]`, `.exam_error_patterns(...) -> tuple[list[ExamErrorAggregate], DataScope]`, and `.tutor_strategy_patterns(...) -> tuple[list[TutorStrategyAggregate], DataScope]`.

- [ ] **Step 1: Write fixture truth tests for count, distinct-session share, and representative diversity**

```python
# tests/test_analytics.py
def test_faq_counts_distinct_sessions_not_repeated_turns(analytics_repo) -> None:
    rows = analytics_repo.student_frequent_questions(top_k=3, subject_filters=[])
    assert rows[0].question_cluster_id == "number-fractions:ask-next-step"
    assert rows[0].event_count == 3
    assert rows[0].distinct_session_count == 2
    assert rows[0].session_share == 2 / 3
    assert len({ref.session_id for ref in rows[0].representative_refs}) == 2


def test_empty_filter_returns_no_rows_and_real_data_scope(analytics_repo) -> None:
    rows, scope = analytics_repo.recurring_misconceptions(
        top_k=3,
        subject_filters=["Does Not Exist"],
    )
    assert rows == []
    assert scope.total_sessions_considered == 3
    assert scope.matched_sessions == 0


def test_difficulty_discloses_unmeasured_signals(analytics_repo) -> None:
    rows, scope = analytics_repo.difficult_concepts(top_k=3, subject_filters=[])
    assert "UNRESOLVED_ENDING" in scope.unmeasured_metrics
    assert "STRATEGY_SWITCH" in scope.unmeasured_metrics
```

- [ ] **Step 2: Run analytics tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_analytics.py -v
```

Expected: collection fails because `src.analytics` does not exist.

- [ ] **Step 3: Implement FAQ SQL with deterministic tie-breaking**

```sql
WITH scoped_sessions AS (
    SELECT DISTINCT session_id
    FROM student_question_events
    WHERE evidence_verified = TRUE
      AND ($subject_count = 0 OR subject_path IN (SELECT * FROM UNNEST($subjects)))
), ranked AS (
    SELECT
        question_cluster_id,
        question_function,
        MIN(normalized_question) AS canonical_question,
        COUNT(*) AS event_count,
        COUNT(DISTINCT session_id) AS distinct_session_count
    FROM student_question_events
    WHERE evidence_verified = TRUE
      AND ($subject_count = 0 OR subject_path IN (SELECT * FROM UNNEST($subjects)))
    GROUP BY question_cluster_id, question_function
)
SELECT *,
       distinct_session_count::DOUBLE /
       NULLIF((SELECT COUNT(*) FROM scoped_sessions), 0) AS session_share
FROM ranked
ORDER BY distinct_session_count DESC, event_count DESC, question_cluster_id ASC
LIMIT $top_k;
```

Use parameter binding; do not interpolate filter text into SQL. Fetch up to two representative events per cluster using `ROW_NUMBER() OVER (PARTITION BY question_cluster_id, session_id ORDER BY turn_id)` and then de-duplicate Session IDs.

- [ ] **Step 4: Implement the other four aggregate queries**

Apply the same deterministic conventions:

```text
difficult_concepts:
  group by subject_path;
  expose AVG(LONG_TUTORING_SEQUENCE), sessions with REPEATED_HELP_REQUEST,
  and the explicit list of unavailable signal types; do not collapse them into an opaque score.

recurring_misconceptions:
  group by cluster_id, canonical_name, error_category;
  count distinct sessions; choose up to 3 distinct-session manifestations.

exam_error_patterns:
  group by pattern_id, pattern_name, error_category;
  keep UNCLASSIFIED visible as its own category instead of redistributing it.

tutor_strategy_patterns:
  group by strategy_id, strategy_name, strategy_category;
  count distinct sessions; choose up to 2 distinct-session tutor evidence refs.
```

Every query returns a `DataScope` containing analytics/source versions, total source sessions, matched sessions, filters, and unmeasured metrics.

- [ ] **Step 5: Run analytics and storage tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_analytics.py `
  tests/test_analytics_store.py `
  tests/test_storage_manager.py::test_sql_analytics_and_group_by -v
```

Expected: exact fixture counts pass and existing storage analytics remain unchanged.

- [ ] **Step 6: Commit macro analytics**

```powershell
git add src/analytics.py tests/test_analytics.py
git commit -m "feat(eedi-rag): compute deterministic business analytics"
```

---

### Task 6: Add Strict Authorized Cohort Import and Conditional Analytics

**Files:**
- Create: `src/cohort_data.py`
- Create: `scripts/import_authorized_cohort_data.py`
- Create: `data/business/cohort-definitions.schema.json`
- Create: `tests/test_cohort_data.py`
- Modify: `src/data_capabilities.py`
- Modify: `src/analytics.py`

**Interfaces:**
- Consumes: an explicitly supplied authorized CSV/JSONL with pseudonymous student/session features and an explicitly supplied, versioned cohort definition.
- Produces: validated `student_session_features` rows, `RETAKE_COHORT`/`ABILITY_COHORT` capability coverage, and `AnalyticsRepository.student_cohort_patterns(...)` only when requirements pass.

- [ ] **Step 1: Write strict rejection and conditional-enablement tests**

```python
# tests/test_cohort_data.py
from pathlib import Path

import pytest

from src.cohort_data import CohortDefinition, import_authorized_cohort_rows


def test_current_dataset_without_student_mapping_keeps_cohort_disabled(capability_registry) -> None:
    snapshot = capability_registry.inspect()
    assert snapshot["RETAKE_COHORT"].available is False
    assert snapshot["ABILITY_COHORT"].available is False


def test_import_rejects_raw_identity_columns(tmp_path: Path, analytics_store) -> None:
    path = tmp_path / "cohort.csv"
    path.write_text(
        "session_id,student_key,attempt_number,email,measured_at,source_system,authorization_reference\n"
        "10,student-a,2,person@example.com,2026-09-01T00:00:00Z,client,approval-1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prohibited identity column"):
        import_authorized_cohort_rows(path, analytics_store)


def test_retaker_definition_requires_explicit_attempt_threshold() -> None:
    definition = CohortDefinition(
        cohort_id="retaker-v1",
        cohort_type="RETAKE",
        attempt_number_gte=2,
        historic_question_count_gte=None,
        historic_correctness_lt=None,
        definition_version="client-approved-v1",
    )
    assert definition.attempt_number_gte == 2
```

- [ ] **Step 2: Run cohort tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cohort_data.py -v
```

Expected: collection fails because `src.cohort_data` does not exist.

- [ ] **Step 3: Define a no-default cohort contract**

```python
# src/cohort_data.py
class CohortDefinition(StrictBusinessModel):
    cohort_id: str = Field(min_length=1)
    cohort_type: Literal["RETAKE", "ABILITY"]
    attempt_number_gte: int | None = Field(default=None, ge=2)
    historic_question_count_gte: int | None = Field(default=None, ge=1)
    historic_correctness_lt: float | None = Field(default=None, ge=0.0, le=1.0)
    definition_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def required_thresholds_are_explicit(self) -> "CohortDefinition":
        if self.cohort_type == "RETAKE" and self.attempt_number_gte is None:
            raise ValueError("RETAKE requires attempt_number_gte")
        if self.cohort_type == "ABILITY" and (
            self.historic_question_count_gte is None
            or self.historic_correctness_lt is None
        ):
            raise ValueError("ABILITY requires explicit history count and correctness thresholds")
        return self
```

There is no built-in threshold and no automatic “weak student” label. The client/business owner supplies and approves the definition version.

- [ ] **Step 4: Implement strict import into a new staged sidecar copy**

Accepted columns are exactly:

```text
session_id
student_key
attempt_number
historic_question_count
historic_correctness
measured_at
source_system
authorization_reference
```

Reject unexpected identity columns including `name`, `full_name`, `email`, `phone`, `address`, `school`, `cookie`, and `token`. Require every `session_id` to exist in the authoritative source DB, reject duplicate Session IDs, validate numeric ranges, hash each canonical input record into `source_record_hash`, and write only to a caller-specified staging analytics artifact.

- [ ] **Step 5: Update capability coverage and cohort analytics**

Compute capability coverage as:

```text
RETAKE_COHORT coverage = rows with student_key + attempt_number / authoritative session count
ABILITY_COHORT coverage = rows with student_key + historic_question_count + historic_correctness / authoritative session count
```

The registry marks a capability available only when:

```text
required columns exist
AND at least one eligible row exists
AND a matching explicit CohortDefinition is supplied for the query
```

`student_cohort_patterns()` filters Sessions using the approved definition, then aggregates existing misconception/difficulty events inside that explicit Session set. It reports member count, session count, metadata coverage, definition version, and representative evidence refs. It does not claim causality or compare outcomes unless a separately defined comparison cohort is supplied.

- [ ] **Step 6: Run cohort and capability tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_cohort_data.py `
  tests/test_data_capabilities.py `
  tests/test_analytics.py -v
```

Expected: synthetic authorized fixtures enable their declared capability; current-Eedi fixtures remain unavailable.

- [ ] **Step 7: Commit conditional cohort support**

```powershell
git add `
  src/cohort_data.py `
  src/data_capabilities.py `
  src/analytics.py `
  scripts/import_authorized_cohort_data.py `
  data/business/cohort-definitions.schema.json `
  tests/test_cohort_data.py
git commit -m "feat(eedi-rag): gate cohort analytics on authorized longitudinal data"
```

---

### Task 7: Build and Audit the Full Current-Eedi Analytics Artifact

**Files:**
- Modify: `scripts/build_business_analytics.py`
- Create: `tests/test_business_analytics_build.py`
- Create at runtime: `reports/staging/business-analytics-v1/manifest.json`
- Create at runtime: `reports/staging/business-analytics-v1/business-analytics.duckdb`

**Interfaces:**
- Consumes: current authoritative production DuckDB, all Part 2 derivation functions.
- Produces: one `READY` staging analytics artifact or a truthful non-zero exit with `FAILED` manifest.

- [ ] **Step 1: Write an end-to-end build test against a small authoritative fixture**

```python
# tests/test_business_analytics_build.py
def test_build_creates_ready_manifest_with_verified_event_counts(source_db, tmp_path) -> None:
    output = tmp_path / "business-analytics.duckdb"
    result = build_business_analytics(source_db, output, "business-events-v1")
    assert result.build_status == "READY"
    assert result.row_counts["student_question_events"] > 0
    assert result.invalid_evidence_count == 0
    assert result.source_hash_before == result.source_hash_after
```

- [ ] **Step 2: Run the build test and observe the missing orchestration failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_analytics_build.py -v
```

Expected: failure because the full `build_business_analytics()` orchestration has not been implemented.

- [ ] **Step 3: Implement the transactional full-build sequence**

Use this exact order:

```text
1. source hash before
2. source schema/count audit
3. BUILDING manifest
4. student question events
5. measured difficulty events
6. misconception events
7. exam error events
8. tutor strategy events
9. per-table primary-key and evidence-pointer audit
10. source hash after
11. row-count manifest
12. READY status
```

Any exception must roll back the active table transaction, write `FAILED` to the staging manifest when possible, return a non-zero CLI exit code, and preserve the original exception chain.

- [ ] **Step 4: Run the full Part 2 test suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_analytics_models.py `
  tests/test_analytics_store.py `
  tests/test_event_extraction.py `
  tests/test_event_derivation.py `
  tests/test_analytics.py `
  tests/test_cohort_data.py `
  tests/test_business_analytics_build.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Build the current full artifact without cohort metadata**

Run:

```powershell
.venv\Scripts\python.exe scripts/build_business_analytics.py `
  --source-db data/db/tutoring_knowledge.duckdb `
  --output reports/staging/business-analytics-v1/business-analytics.duckdb `
  --manifest reports/staging/business-analytics-v1/manifest.json `
  --analytics-version business-events-v1
```

Expected observable evidence:

```text
source sessions = 1576
source dialogue turns = 37186
invalid evidence pointers = 0
manifest status = READY
RETAKE_COHORT = unavailable
ABILITY_COHORT = unavailable
```

If current counts differ because the authoritative artifact legitimately changed, the manifest records the actual counts and hashes; do not edit the report to force the expected historic numbers.

- [ ] **Step 6: Inspect aggregate facts without invoking an LLM**

Run:

```powershell
.venv\Scripts\python.exe scripts/build_business_analytics.py `
  --inspect reports/staging/business-analytics-v1/business-analytics.duckdb `
  --top-k 10
```

Expected: deterministic FAQ, difficulty, misconception, exam-error, and tutor-strategy tables plus explicit unmeasured-signal and cohort-capability notices.

- [ ] **Step 7: Commit builder/test changes, not generated production artifacts**

```powershell
git add scripts/build_business_analytics.py tests/test_business_analytics_build.py
git commit -m "feat(eedi-rag): build audited business analytics artifact"
```

## Part 2 Exit Gate

Part 2 is complete only when:

- The authoritative dialogue DuckDB hash is identical before and after the build.
- The analytics sidecar manifest is `READY`, source-hash bound, and contains real row counts.
- Every persisted event passes Session/Turn/role/text authorization checks.
- FAQ and pattern rankings use `COUNT(DISTINCT session_id)` and deterministic tie-breaking.
- Difficulty output identifies unavailable signals as `UNMEASURED` instead of zero.
- Current Eedi remains unable to claim retaker/weak-foundation cohorts.
- Synthetic authorized cohort fixtures prove the future data adapter, while real current execution stays disabled without approved inputs.
