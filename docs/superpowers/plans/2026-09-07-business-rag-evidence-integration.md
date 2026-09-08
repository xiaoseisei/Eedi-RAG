# Business RAG Capability Upgrade Part 3: Macro Analytics and Micro Evidence Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate deterministic macro analytics with the existing student/tutor micro evidence RAG so business questions return correctly computed rankings plus auditable representative dialogue, while specific case questions continue to use BM25/Dense/RRF and Parent-5/W7.

**Architecture:** A new `BusinessRAGPipeline` coordinates the Part 1 router/planner, Part 2 analytics repository, and the existing RAG components. Macro queries run SQL first and authorize representative Session/Turn pointers; only then does an evidence bridge materialize bounded W7 dialogue context. Micro queries call the existing dual-lane retriever and Parent-5/W7 path directly. Backend code owns counts, ranks, statuses, and exact citations; the LLM may write only bounded narrative fields from authorized facts and evidence.

**Tech Stack:** Python 3.14, DuckDB, ChromaDB, BM25, existing OpenAI-compatible embedding/reranker/generator providers, Pydantic 2, pytest 9, existing L1/L2 evaluation infrastructure.

**Spec:** `docs/需求.md`; `docs/Query_Intent_Response_Executable_Design.md`; `docs/superpowers/plans/2026-09-07-business-rag-contract-routing.md`; `docs/superpowers/plans/2026-09-07-business-rag-analytics-assets.md`

## Global Constraints

- Complete Parts 1 and 2 and satisfy their exit gates before executing this plan.
- Preserve `EndToEndPedagogicalRAGPipeline.ask()` as a paired legacy baseline until the final promotion task.
- Never use vector similarity as an event count, prevalence measure, frequency rank, or cohort definition.
- Macro execution order is fixed: SQL facts first, authorized Session/Turn set second, micro evidence materialization third, optional LLM wording fourth, audit fifth, rendering last.
- A model may select only backend-issued evidence IDs; it cannot emit Session IDs, Turn IDs, quotes, ranks, counts, percentages, or status values as authoritative data.
- The existing student and tutor collections remain physically isolated evidence perspectives.
- Student-only plans must not retrieve tutor strategy candidates; tutor-only plans must not retrieve student misconception candidates; dual-perspective plans preserve `[STUDENT, TUTOR]` order.
- Representative evidence for a macro item must belong to the exact event cluster and Session set that produced that aggregate.
- Parent-5 means at most five representative aggregate/event anchors enter narrative context; W7/S3 means seven-Turn windows with step three around authorized anchor Turns. A Top-10 report may render all ten SQL rows, but the LLM narrative context remains bounded to five representative anchors.
- The 4000-token budget applies to the final narrative generator context, including evidence catalog; no selected logical window is silently truncated.
- `UNSUPPORTED` and `NO_EVIDENCE` paths do not call the generator.
- User-visible output is rendered only after contract validation and evidence audit. Internal streaming may measure TTFT but cannot expose unvalidated text.
- Root data paths must resolve from the repository root, not from the current working directory.
- Production profile names must match the actual embedding, reranker, route, Parent count, W7 selection count, and artifact versions used at runtime.

## Delivery Boundary

```text
BusinessQueryRequest
  -> route + plan
  -> [macro] AnalyticsRepository -> Aggregate Facts
     [micro] Existing dual-lane RAG
  -> EvidenceBridge -> authorized exact Turns + bounded W7 context
  -> intent-specific narrative generator
  -> deterministic response builder
  -> four-field evidence audit
  -> intent-specific renderer
```

This part implements the six currently supportable internal-business responses: FAQ, difficulty, misconception, exam-error, tutor-strategy, and content-improvement, plus the existing specific-question path. Cohort queries remain conditionally supported and return `UNSUPPORTED` on the current Eedi artifact.

## File Structure

```text
src/
├── evidence_search.py        # perspective-aware wrapper over current retriever
├── evidence.py               # analytics pointer authorization and W7 materialization
├── business_generation.py    # bounded intent-specific narrative generation
├── business_response.py      # deterministic response construction and rendering
├── business_pipeline.py      # route/plan/execute/audit orchestration
├── retriever.py              # add optional perspective/filter inputs, defaults unchanged
├── reranker.py               # add authorized-window assembly entrypoint
├── rag_pipeline.py           # keep existing ask(); expose reusable citation audit
└── cli.py                    # root paths, business mode, truthful profile, audited rendering

tests/
├── test_evidence_search.py
├── test_business_evidence.py
├── test_business_generation.py
├── test_business_response.py
├── test_business_pipeline.py
└── test_business_cli.py

scripts/
└── evaluate_business_pipeline.py

evals/datasets/business-response-v1/
├── queries.jsonl
├── expected_facts.jsonl
├── expected_evidence.jsonl
├── split_manifest.json
└── README.md
```

---

### Task 1: Add Perspective-Aware, Session-Bounded Evidence Search

**Files:**
- Create: `src/evidence_search.py`
- Create: `tests/test_evidence_search.py`
- Modify: `src/retriever.py:676-859`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `EvidenceSearchRequest(query, perspectives, allowed_session_ids, top_k_each, fetch_evidence)`.
- Produces: `EvidenceSearchResult` and backward-compatible optional parameters on `DualMetricRetriever.retrieve_multi_perspective_rrf()`.

- [ ] **Step 1: Write tests that prove perspective isolation and Session filters**

```python
# tests/test_evidence_search.py
from src.business_models import EvidencePerspective
from src.evidence_search import EvidenceSearchEngine, EvidenceSearchRequest


def test_student_plan_does_not_query_tutor_lane(spy_retriever) -> None:
    engine = EvidenceSearchEngine(spy_retriever)
    result = engine.search(
        EvidenceSearchRequest(
            query="学生最常问什么？",
            perspectives=[EvidencePerspective.STUDENT],
            allowed_session_ids=[10, 20],
            top_k_each=3,
            fetch_evidence=True,
        )
    )
    assert result.strategy_candidates == []
    assert spy_retriever.strategy_call_count == 0
    assert {item["metadata"]["session_id"] for item in result.misconception_candidates} <= {10, 20}


def test_dual_perspective_order_is_stable(spy_retriever) -> None:
    engine = EvidenceSearchEngine(spy_retriever)
    result = engine.search(
        EvidenceSearchRequest(
            query="学生错在哪里，导师如何引导？",
            perspectives=[EvidencePerspective.STUDENT, EvidencePerspective.TUTOR],
            allowed_session_ids=None,
            top_k_each=3,
            fetch_evidence=True,
        )
    )
    assert result.executed_perspectives == [
        EvidencePerspective.STUDENT,
        EvidencePerspective.TUTOR,
    ]
```

- [ ] **Step 2: Run the new tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evidence_search.py -v
```

Expected: collection fails because `src.evidence_search` does not exist.

- [ ] **Step 3: Add strict search request/result models and wrapper**

```python
# src/evidence_search.py
from __future__ import annotations

from pydantic import Field, model_validator

from src.business_models import EvidencePerspective, StrictBusinessModel


class EvidenceSearchRequest(StrictBusinessModel):
    query: str = Field(min_length=1)
    perspectives: list[EvidencePerspective] = Field(min_length=1, max_length=2)
    allowed_session_ids: list[int] | None = None
    top_k_each: int = Field(default=3, ge=1, le=20)
    fetch_evidence: bool = True

    @model_validator(mode="after")
    def perspectives_are_unique_and_ordered(self) -> "EvidenceSearchRequest":
        if len(set(self.perspectives)) != len(self.perspectives):
            raise ValueError("perspectives must be unique")
        if self.perspectives == [EvidencePerspective.TUTOR, EvidencePerspective.STUDENT]:
            raise ValueError("dual perspective order must be STUDENT then TUTOR")
        if self.allowed_session_ids is not None and not self.allowed_session_ids:
            raise ValueError("allowed_session_ids cannot be an empty list")
        return self


class EvidenceSearchResult(StrictBusinessModel):
    misconception_candidates: list[dict]
    strategy_candidates: list[dict]
    executed_perspectives: list[EvidencePerspective]
    allowed_session_ids: list[int] | None
    trace: dict
```

The wrapper calls only requested lanes and records candidate IDs/counts. An empty allowed Session set is rejected before retrieval so callers return `NO_EVIDENCE` instead of accidentally searching the whole corpus.

- [ ] **Step 4: Extend the existing RRF API with backward-compatible filters**

Add keyword-only parameters with defaults that preserve all current callers:

```python
def retrieve_multi_perspective_rrf(
    self,
    raw_query: str,
    top_k_each: int = 3,
    fetch_evidence: bool = True,
    k_constant: int = 60,
    *,
    perspectives: tuple[str, ...] = ("student", "tutor"),
    where_filter: dict[str, object] | None = None,
) -> dict[str, object]:
```

Propagate `where_filter` into all `_retrieve_misconceptions_by_vector()` and `_retrieve_strategies_by_vector()` calls. For a bounded Session set use:

```python
where_filter = {"session_id": {"$in": sorted(set(allowed_session_ids))}}
```

Skip construction, embedding, retrieval, and RRF fusion for an unrequested perspective. Keep the existing trace keys but set skipped lanes to empty arrays and add `executed_perspectives` and `where_filter_applied`.

- [ ] **Step 5: Run new and existing retrieval tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_evidence_search.py tests/test_retriever.py -v
```

Expected: all tests pass; calls without the new parameters still execute both lanes and retain prior ranking behavior.

- [ ] **Step 6: Commit perspective-aware search**

```powershell
git add src/evidence_search.py src/retriever.py tests/test_evidence_search.py tests/test_retriever.py
git commit -m "feat(eedi-rag): constrain evidence search by perspective and session"
```

---

### Task 2: Materialize Authorized Macro Evidence and Bounded W7 Context

**Files:**
- Create: `src/evidence.py`
- Create: `tests/test_business_evidence.py`
- Modify: `src/retriever.py:472-647`
- Modify: `src/reranker.py:657-1209`
- Reference: `src/generator_contract.py:254-344`
- Reference: `src/rag_pipeline.py:1075-1135`

**Interfaces:**
- Consumes: aggregate `RepresentativeEvidenceRef` values from Part 2, authoritative storage, `max_parent_anchors=5`, `window_size=7`, `step=3`, and `max_prompt_tokens=4000`.
- Produces: `AuthorizedEvidenceBundle(pointers, selected_parent_anchors, selected_windows, assembled_context, evidence_catalog, audit_status)` and `AnalyticsEvidenceBridge.materialize_for_rows(plan, rows)`.

- [ ] **Step 1: Write tests that bind aggregate rows to exact authoritative Turns**

```python
# tests/test_business_evidence.py
import pytest

from src.analytics_models import RepresentativeEvidenceRef
from src.business_models import EvidencePerspective
from src.evidence import AnalyticsEvidenceBridge


def test_bridge_rejects_representative_turn_outside_source_event(storage, analytics_store) -> None:
    ref = RepresentativeEvidenceRef(
        source_event_id="sqe-v1-10-4",
        session_id=10,
        turn_ids=[99],
        perspective=EvidencePerspective.STUDENT,
        subject_path="Number",
    )
    with pytest.raises(ValueError, match="authorized event"):
        AnalyticsEvidenceBridge(storage, analytics_store).materialize(
            query="学生最常问什么？",
            ranked_representatives=[ref],
        )


def test_bridge_uses_at_most_parent5_and_w7_s3(storage, analytics_store) -> None:
    refs = make_verified_representative_refs(count=8)
    bundle = AnalyticsEvidenceBridge(storage, analytics_store).materialize(
        query="学生最常问什么？",
        ranked_representatives=refs,
        max_parent_anchors=5,
        window_size=7,
        step=3,
        max_prompt_tokens=4000,
    )
    assert len(bundle.selected_parent_anchors) == 5
    assert all(len(window.evidence_turns) <= 7 for window in bundle.selected_windows)
    assert bundle.generator_context_token_count <= 4000
    assert bundle.audit_status == "AUDITED_100_VERIFIED"
```

- [ ] **Step 2: Run the bridge tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_evidence.py -v
```

Expected: collection fails because `src.evidence` does not exist.

- [ ] **Step 3: Add a generic authorized-anchor expansion API**

Define the context models before implementing expansion:

```python
class AuthorizedWindow(StrictBusinessModel):
    window_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    start_turn_id: int = Field(ge=1)
    end_turn_id: int = Field(ge=1)
    evidence_turns: list[dict] = Field(min_length=1)
    source_event_ids: list[str] = Field(min_length=1)
    perspective: EvidencePerspective


class AuthorizedEvidenceBundle(StrictBusinessModel):
    pointers: list[EvidencePointer] = Field(min_length=1)
    selected_parent_anchors: list[RepresentativeEvidenceRef] = Field(min_length=1, max_length=5)
    selected_windows: list[AuthorizedWindow] = Field(min_length=1)
    assembled_context: str = Field(min_length=1)
    evidence_catalog: dict[str, EvidenceCatalogItem]
    generator_context_token_count: int = Field(ge=1, le=4000)
    audit_status: Literal["AUDITED_100_VERIFIED"]
```

In `src/retriever.py`, add:

```python
def expand_authorized_turn_anchors(
    self,
    anchors: list[dict[str, object]],
    *,
    window_size: int = 7,
    step: int = 3,
) -> list[dict[str, object]]:
    """Window complete authoritative Sessions and retain only windows intersecting allowed Turn anchors."""
```

Each anchor must contain `source_event_id`, `session_id`, `source_turn_ids`, `subject_path`, and `perspective`. Fetch the complete Session through `storage.get_dialogue_turns(session_id)`, build W7/S3 windows, and keep only windows intersecting the declared source Turns. Do not infer missing Turn IDs. Attach source event IDs as metadata, never as dialogue content.

- [ ] **Step 4: Add a reusable authorized-window assembler entrypoint**

In `PedagogicalGoldAssembler`, add:

```python
def assemble_authorized_windows(
    self,
    raw_query: str,
    parent_anchors: list[dict[str, object]],
    ranked_windows: list[dict[str, object]],
) -> GoldAssembledContext:
```

Reuse `_ordered_unit_evidence`, `_render_compact_evidence_catalog`, `_render_unique_window_nodes`, and `estimate_text_tokens`. Pack complete windows in deterministic rank order, skip wholly duplicate windows, stop before 4000 tokens, and fail if no complete window fits. Do not call `_select_mmr_best()` or consult retrieval qrels on this already-authorized analytics path.

- [ ] **Step 5: Implement the analytics evidence bridge**

`AnalyticsEvidenceBridge.materialize()` must:

```text
1. look up each source_event_id in the analytics sidecar;
2. require exact session_id and turn_ids equality with the aggregate ref;
3. fetch authoritative source Turns;
4. require perspective/role agreement and exact text;
5. issue deterministic E001... evidence IDs in ranked-item order;
6. retain evidence pointers for every rendered aggregate item;
7. choose at most five ranked refs as parent narrative anchors;
8. expand W7/S3 only around those anchors;
9. assemble complete windows within 4000 tokens;
10. run the existing four-field citation audit before returning.
```

The bundle model contains:

- [ ] **Step 6: Run bridge, assembler, generator-contract, and citation-audit tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_business_evidence.py `
  tests/test_reranker.py `
  tests/test_generator_contract.py `
  tests/test_rag_pipeline.py::test_citation_auditor `
  tests/test_rag_pipeline.py::test_citation_auditor_never_trusts_preverified_flag -v
```

Expected: all tests pass; a pointer that exists in DuckDB but was not authorized by the aggregate event remains rejected.

- [ ] **Step 7: Commit evidence materialization**

```powershell
git add src/evidence.py src/retriever.py src/reranker.py tests/test_business_evidence.py tests/test_reranker.py
git commit -m "feat(eedi-rag): bridge macro facts to audited dialogue evidence"
```

---

### Task 3: Add Intent-Specific Business Response Models and Builders

**Files:**
- Modify: `src/business_models.py`
- Create: `src/business_response.py`
- Create: `tests/test_business_response.py`
- Reference: `docs/Query_Intent_Response_Executable_Design.md:405-626`

**Interfaces:**
- Consumes: strict aggregate models, `AuthorizedEvidenceBundle`, `QueryUnderstanding`, and `DataScope`.
- Produces: discriminated report result models, `BusinessResponseBuilder.build_success(...)`, `.build_no_evidence(...)`, `.build_unsupported(...)`, and deterministic Markdown rendering.

- [ ] **Step 1: Write response tests proving that facts are backend-owned**

```python
# tests/test_business_response.py
from src.business_models import BusinessResponseStatus
from src.business_response import BusinessResponseBuilder


def test_faq_builder_preserves_sql_rank_count_and_share(faq_aggregate, evidence_bundle, query_plan) -> None:
    response = BusinessResponseBuilder().build_faq(
        plan=query_plan,
        rows=[faq_aggregate],
        evidence=evidence_bundle,
        narrative="学生最常寻求下一步提示。",
    )
    item = response.result.items[0]
    assert item.rank == faq_aggregate.rank
    assert item.event_count == faq_aggregate.event_count
    assert item.distinct_session_count == faq_aggregate.distinct_session_count
    assert item.session_share == faq_aggregate.session_share
    assert response.status == BusinessResponseStatus.SUCCESS
    assert response.audit_status == "AUDITED_100_VERIFIED"


def test_empty_analytics_returns_no_evidence_without_result(query_plan, empty_scope) -> None:
    response = BusinessResponseBuilder().build_no_evidence(
        plan=query_plan,
        data_scope=empty_scope,
        reason="当前过滤范围没有经过验证的事件",
    )
    assert response.status == BusinessResponseStatus.NO_EVIDENCE
    assert response.result is None
    assert response.evidence == []
    assert response.audit_status == "NOT_APPLICABLE"


def test_unsupported_cohort_has_missing_fields_and_no_generated_result(cohort_reject_plan) -> None:
    response = BusinessResponseBuilder().build_unsupported(cohort_reject_plan)
    assert response.status == BusinessResponseStatus.UNSUPPORTED
    assert "attempt_number" in response.limitations[0]
    assert response.result is None
```

- [ ] **Step 2: Run response tests and observe the missing builder failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_response.py -v
```

Expected: collection fails because `src.business_response` does not exist and result models are incomplete.

- [ ] **Step 3: Implement all report result types as a discriminated union**

Add these exact result types to `src/business_models.py`:

```text
StudentFAQReport / FrequentQuestionItem
DifficultyReport / DifficultConceptItem / DifficultySignals
MisconceptionReport / MisconceptionItem
ExamErrorPatternReport / ExamErrorPatternItem
StudentCohortPatternReport / StudentCohortPatternItem
TutorStrategyReport / TutorStrategyItem
ContentImprovementBrief / ContentOpportunityItem
SocraticTurn
```

Use the fields and bounds from design sections 9.1-9.8. Add `StudentCohortPatternReport` with:

```python
class StudentCohortPatternItem(StrictBusinessModel):
    rank: int = Field(ge=1)
    pattern_id: str = Field(min_length=1)
    pattern_name: str = Field(min_length=1)
    distinct_student_count: int = Field(ge=1)
    distinct_session_count: int = Field(ge=1)
    session_share_within_cohort: float = Field(ge=0.0, le=1.0)
    representative_evidence_ids: list[str] = Field(min_length=1, max_length=3)


class StudentCohortPatternReport(StrictBusinessModel):
    response_type: Literal["STUDENT_COHORT_PATTERN_REPORT"] = "STUDENT_COHORT_PATTERN_REPORT"
    cohort_id: str = Field(min_length=1)
    cohort_definition_version: str = Field(min_length=1)
    metadata_coverage: float = Field(ge=0.0, le=1.0)
    items: list[StudentCohortPatternItem] = Field(min_length=1, max_length=10)
```

- [ ] **Step 4: Implement deterministic builders and rendering**

Each builder copies all rank/count/share/category fields from aggregate models, maps only bridge-authorized evidence IDs, and accepts generated text only for narrative fields such as `cross_pattern_summary`, `deep_mechanism_summary`, or content outline. Rendering includes:

```text
query interpretation
data scope and versions
ranked deterministic facts
representative exact dialogue evidence
limitations/unmeasured signals
audit status and trace ID
```

Do not render raw debug context unless `BusinessQueryRequest.debug=True`.

- [ ] **Step 5: Run response and contract tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_models.py tests/test_business_response.py -v
```

Expected: all report types validate, false-success combinations fail, and SQL facts survive builder construction unchanged.

- [ ] **Step 6: Commit response models and builders**

```powershell
git add src/business_models.py src/business_response.py tests/test_business_models.py tests/test_business_response.py
git commit -m "feat(eedi-rag): build intent-specific audited business responses"
```

---

### Task 4: Add Bounded Intent-Specific Narrative Generation

**Files:**
- Create: `src/business_generation.py`
- Create: `tests/test_business_generation.py`
- Reference: `src/rag_pipeline.py:705-1020`
- Reference: `src/generator_contract.py:16-126`

**Interfaces:**
- Consumes: `QueryPlan`, immutable aggregate facts, and an `AuthorizedEvidenceBundle`.
- Produces: strict intent-specific narrative payloads that contain no authoritative numeric/citation fields, plus generation telemetry.

- [ ] **Step 1: Write tests for bounded output, no unknown evidence, and no calls on rejected plans**

```python
# tests/test_business_generation.py
def test_faq_generation_receives_read_only_facts_and_bounded_context(fake_client, faq_prompt_input) -> None:
    generator = BusinessNarrativeGenerator(client=fake_client, model="mimo-v2.5")
    payload, telemetry = generator.generate_faq_summary(faq_prompt_input)
    request = fake_client.chat.completions.requests[0]
    assert request["max_completion_tokens"] == 160
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "distinct_session_count" in request["messages"][1]["content"]
    assert telemetry.thinking_mode == "disabled"


def test_generator_payload_cannot_override_counts_or_quotes() -> None:
    with pytest.raises(ValidationError):
        FAQNarrativePayload.model_validate(
            {
                "cross_pattern_summary": "summary",
                "event_count": 999,
                "quote_text": "fabricated",
            }
        )


def test_reject_and_no_evidence_paths_do_not_call_generator(fake_client, business_pipeline) -> None:
    response = business_pipeline.ask_business(
        BusinessQueryRequest(query="重考者有什么共同模式？")
    )
    assert response.status == BusinessResponseStatus.UNSUPPORTED
    assert fake_client.chat.completions.requests == []
```

- [ ] **Step 2: Run generation tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_generation.py -v
```

Expected: collection fails because `src.business_generation` does not exist.

- [ ] **Step 3: Implement one strict payload per narrative task**

Examples:

```python
class FAQNarrativePayload(StrictBusinessModel):
    cross_pattern_summary: str = Field(min_length=1, max_length=600)


class DifficultyNarrativePayload(StrictBusinessModel):
    overall_interpretation: str = Field(min_length=1, max_length=800)
    measurement_limitation: str = Field(min_length=1, max_length=500)


class ContentBriefNarrativePayload(StrictBusinessModel):
    learning_objective: str = Field(min_length=1, max_length=500)
    suggested_content_outline: list[str] = Field(min_length=1, max_length=6)
    validation_method: str = Field(min_length=1, max_length=500)
```

No payload includes rank, event count, Session ID, Turn ID, quote text, status, or audit flags.

- [ ] **Step 4: Implement one prompt template and output budget per intent**

Use:

```python
OUTPUT_BUDGETS = {
    QueryIntent.STUDENT_FREQUENT_QUESTIONS: 160,
    QueryIntent.DIFFICULT_CONCEPTS: 240,
    QueryIntent.RECURRING_MISCONCEPTIONS: 220,
    QueryIntent.EXAM_ERROR_PATTERNS: 220,
    QueryIntent.STUDENT_COHORT_PATTERNS: 220,
    QueryIntent.TUTOR_STRATEGY_PATTERNS: 300,
    QueryIntent.CONTENT_IMPROVEMENT: 420,
    QueryIntent.QUESTION_TUTORING: 96,
}
```

Every prompt separates `User Query`, `Read-only SQL Facts`, `Authorized Evidence Catalog`, and `Task`. Set `temperature=0.1`, `extra_body={"thinking": {"type": "disabled"}}`, and the intent budget. Reuse one initialized client. Retry only once for JSON/contract failure; provider/network failure raises a typed `BusinessGenerationError` unless the caller explicitly selected deterministic narrative omission.

- [ ] **Step 5: Collect internal streaming without exposing pre-audit text**

Record TTFT and chunks internally, concatenate the full JSON, validate it, and return it to the response builder. The CLI receives stage events only. Remove no existing streaming code in this task; Part 3 Task 6 changes the business CLI path after tests exist.

- [ ] **Step 6: Run generation and existing generator-contract tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_business_generation.py `
  tests/test_generator_contract.py `
  tests/test_rag_pipeline.py::test_auto_mode_without_llm_credentials_does_not_silently_fallback -v
```

Expected: all tests pass; missing credentials and provider failures remain visible failures.

- [ ] **Step 7: Commit narrative generation**

```powershell
git add src/business_generation.py tests/test_business_generation.py
git commit -m "feat(eedi-rag): add bounded business narrative generation"
```

---

### Task 5: Implement the Business RAG Orchestrator

**Files:**
- Create: `src/business_pipeline.py`
- Create: `tests/test_business_pipeline.py`
- Modify: `src/rag_pipeline.py:1075-1135`
- Reference: `src/rag_pipeline.py:1215-1318`

**Interfaces:**
- Consumes: Part 1 router/planner/capabilities, Part 2 analytics repository, Tasks 1-4 evidence/generation/response components, and the legacy micro RAG pipeline.
- Produces: `BusinessRAGPipeline.ask_business(request: BusinessQueryRequest) -> BusinessResponseEnvelope`.

- [ ] **Step 1: Write orchestration tests with call-order spies**

```python
# tests/test_business_pipeline.py
def test_macro_query_runs_sql_before_bounded_evidence_and_generation(pipeline_spies) -> None:
    response = pipeline_spies.pipeline.ask_business(
        BusinessQueryRequest(query="学生最常提出的问题是什么？", top_k=10)
    )
    assert pipeline_spies.calls == [
        "route",
        "capabilities",
        "plan",
        "analytics:student_frequent_questions",
        "evidence:student",
        "generate:student_faq",
        "build",
        "audit",
    ]
    assert response.status == BusinessResponseStatus.SUCCESS


def test_specific_case_query_skips_macro_analytics(pipeline_spies) -> None:
    pipeline_spies.pipeline.ask_business(
        BusinessQueryRequest(query="学生为什么会把 5.4598 选成 5.45？")
    )
    assert not any(call.startswith("analytics:") for call in pipeline_spies.calls)
    assert "legacy_micro_rag" in pipeline_spies.calls


def test_empty_aggregate_returns_no_evidence_before_rag_or_llm(pipeline_spies) -> None:
    pipeline_spies.analytics.rows = []
    response = pipeline_spies.pipeline.ask_business(
        BusinessQueryRequest(query="不存在主题中学生最常问什么？")
    )
    assert response.status == BusinessResponseStatus.NO_EVIDENCE
    assert not any(call.startswith("evidence:") for call in pipeline_spies.calls)
    assert not any(call.startswith("generate:") for call in pipeline_spies.calls)


def test_retaker_query_without_capability_is_rejected_before_execution(pipeline_spies) -> None:
    response = pipeline_spies.pipeline.ask_business(
        BusinessQueryRequest(query="重考者经常问什么？")
    )
    assert response.status == BusinessResponseStatus.UNSUPPORTED
    assert pipeline_spies.calls == ["route", "capabilities", "plan", "build:unsupported"]
```

- [ ] **Step 2: Run orchestrator tests and observe the missing-module failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_pipeline.py -v
```

Expected: collection fails because `src.business_pipeline` does not exist.

- [ ] **Step 3: Implement fail-fast orchestration with explicit operation dispatch**

```python
# src/business_pipeline.py
class BusinessRAGPipeline:
    def ask_business(self, request: BusinessQueryRequest) -> BusinessResponseEnvelope:
        trace = BusinessTrace.start(request)
        understanding = self.router.route(request)
        capabilities = self.capability_registry.inspect()
        plan = self.planner.plan(understanding, capabilities)
        trace.record_route(understanding, plan)

        if plan.plan_type == QueryPlanType.REJECT:
            return self.response_builder.build_unsupported(plan, trace_id=trace.trace_id)

        if plan.analytics_operations:
            rows, data_scope = self.analytics.execute(plan, request)
            if not rows:
                return self.response_builder.build_no_evidence(
                    plan=plan,
                    data_scope=data_scope,
                    reason="当前查询范围没有经过验证的分析事件",
                    trace_id=trace.trace_id,
                )
            evidence = self.evidence_bridge.materialize_for_rows(plan, rows)
            narrative = self.generator.generate(plan, rows, evidence)
            response = self.response_builder.build(plan, rows, evidence, narrative, trace)
        else:
            response = self.micro_executor.execute(plan, request, trace)

        self.auditor.audit_business_response(response)
        return response
```

Use an explicit operation-to-method map; never use `getattr()` on untrusted operation text. Wrap known component errors with stage context and preserve the original exception. A provider failure returns `FAILED`, not `SUCCESS` with a blank narrative, unless the request explicitly selects a deterministic report mode that requires no LLM.

- [ ] **Step 4: Expose the existing citation audit as a reusable service**

Move no behavior yet. Add a thin public adapter in `rag_pipeline.py` or `evidence.py` that invokes the existing exact `(session_id, turn_id, speaker, quote)` authorization logic. Tests must prove legacy citations and business `EvidencePointer` values use the same authoritative comparison.

- [ ] **Step 5: Add stage timings and non-secret trace fields**

Record:

```text
trace_id
intent/scope/computation/perspectives
plan and capability snapshots
analytics operation and row counts
aggregate cluster IDs
authorized event/Session/Turn IDs
retrieval candidates and final windows
context token count
generator model/thinking/budget/TTFT/visible tokens
audit result
final status and failure stage
source/analytics/Chroma/config versions
```

Do not log API keys, cookies, raw authorized cohort rows, or unrestricted full dialogue text.

- [ ] **Step 6: Run orchestrator and legacy pipeline tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_business_pipeline.py `
  tests/test_business_response.py `
  tests/test_rag_pipeline.py -v
```

Expected: new orchestration tests pass and the legacy baseline remains measurable. Existing tests that intentionally assert pre-audit streaming must remain as legacy behavior until Task 6 switches only the business CLI path.

- [ ] **Step 7: Commit business orchestration**

```powershell
git add src/business_pipeline.py src/rag_pipeline.py tests/test_business_pipeline.py tests/test_rag_pipeline.py
git commit -m "feat(eedi-rag): orchestrate analytics and micro evidence"
```

---

### Task 6: Harden CLI Paths, Profiles, and Audited Rendering

**Files:**
- Modify: `src/cli.py:79-350`
- Create: `tests/test_business_cli.py`
- Reference: `scripts/interactive_cli.py`

**Interfaces:**
- Consumes: `BusinessRAGPipeline`, explicit artifact paths/profile configuration, stage callbacks.
- Produces: an interactive business CLI that cannot accidentally open `src/data`, accurately reports the active RAG route, and displays final content only after audit.

- [ ] **Step 1: Write CLI tests for path resolution and truthful active configuration**

```python
# tests/test_business_cli.py
from pathlib import Path

from src.cli import CLIConfig, resolve_cli_config


def test_default_data_paths_resolve_from_repository_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config = resolve_cli_config([])
    assert config.db_path == PROJECT_ROOT / "data/db/tutoring_knowledge.duckdb"
    assert config.chroma_dir == PROJECT_ROOT / "data/chroma"


def test_local_profile_does_not_claim_parent5_w7() -> None:
    config = resolve_cli_config(["--profile", "local"])
    assert config.rerank_unit == "card"
    assert config.active_route_label == "card baseline"


def test_production_profile_requires_matching_artifact_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embedding index contract"):
        resolve_cli_config(
            ["--profile", "production", "--chroma-dir", str(tmp_path / "empty-chroma")]
        )


def test_business_cli_emits_only_stage_events_before_audit(fake_business_pipeline, capsys) -> None:
    run_one_query(fake_business_pipeline, "学生最常问什么？")
    output = capsys.readouterr().out
    assert output.index("审计完成") < output.index("高频问题报告")
    assert "unvalidated model text" not in output
```

- [ ] **Step 2: Run CLI tests and observe failures against current relative defaults/streaming**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_cli.py -v
```

Expected: tests fail because `CLIConfig`, repository-root resolution, profile validation, and business-stage rendering do not exist.

- [ ] **Step 3: Add strict CLI configuration and repository-root defaults**

Resolve defaults from `Path(__file__).resolve().parent.parent`:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
DEFAULT_ANALYTICS_PATH = PROJECT_ROOT / "data" / "analytics" / "business-analytics.duckdb"
```

Add:

```text
--pipeline business|legacy        default business only after Task 8 promotion
--profile local|production        default local
--analytics-path PATH
--detail brief|standard|full
--debug
```

`local` accurately exposes deterministic embedding/no reranker/card route. `production` requires explicit matching embedding index metadata and a configured real reranker before it can label the route `Anchored Parent-5/W7/S3`; otherwise initialization fails.

- [ ] **Step 4: Print startup provenance and storage counts**

Before accepting a query, print:

```text
resolved DuckDB path
resolved Chroma path
resolved analytics path
DuckDB session/turn counts
Chroma collection counts
source and analytics version/hash prefix
embedding backend
query-rewrite backend
retrieval mode and BM25 weight
reranker backend/unit/pool
Parent/window/selection/token budget
business or legacy pipeline
cohort capabilities and coverage
```

Do not print keys or full credential-bearing environment values.

- [ ] **Step 5: Replace business-path answer streaming with stage events plus post-audit rendering**

Allowed pre-audit output:

```text
理解问题
统计全库
检索代表案例
装配原文证据
生成摘要
审计引用
```

Collect model JSON internally. Render `BusinessResponseEnvelope` only after status/contract/evidence validation. Keep the legacy pipeline flag temporarily for paired comparison; label it visibly as legacy.

- [ ] **Step 6: Run CLI, pipeline, and anchored-production tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_business_cli.py `
  tests/test_business_pipeline.py `
  tests/test_anchored_production.py `
  tests/test_rag_pipeline.py -v
```

Expected: business CLI never opens `src/data`, never mislabels the local card route, and never renders unaudited model text.

- [ ] **Step 7: Commit CLI hardening**

```powershell
git add src/cli.py tests/test_business_cli.py
git commit -m "feat(eedi-rag): expose audited business query CLI"
```

---

### Task 7: Freeze End-to-End Business Gold Facts and Evidence

**Files:**
- Create: `evals/datasets/business-response-v1/queries.jsonl`
- Create: `evals/datasets/business-response-v1/expected_facts.jsonl`
- Create: `evals/datasets/business-response-v1/expected_evidence.jsonl`
- Create: `evals/datasets/business-response-v1/split_manifest.json`
- Create: `evals/datasets/business-response-v1/README.md`
- Create: `scripts/evaluate_business_pipeline.py`
- Create: `tests/test_business_pipeline_eval.py`

**Interfaces:**
- Consumes: current immutable source/analytics artifacts and human-reviewed expected facts/evidence kept outside the system under test.
- Produces: case traces, exact deterministic metrics, optional semantic metrics, immutable report artifacts, and a release decision.

- [ ] **Step 1: Write evaluation contract tests before creating the runner**

```python
# tests/test_business_pipeline_eval.py
def test_eval_rejects_gold_that_leaks_into_request() -> None:
    with pytest.raises(ValueError, match="Gold"):
        BusinessEvalQuery.model_validate(
            {
                "case_id": "business-001",
                "query": "学生最常问什么？",
                "expected_facts": {"rank": 1},
                "split": "holdout",
            }
        )


def test_eval_marks_failed_pipeline_case_as_failed_not_zero_score(fake_failed_pipeline) -> None:
    report = evaluate_cases([faq_case()], fake_failed_pipeline)
    assert report.cases[0].status == "FAILED"
    assert report.cases[0].metrics == {}
    assert report.failure_count == 1
```

- [ ] **Step 2: Run evaluation tests and observe the missing-runner failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_pipeline_eval.py -v
```

Expected: collection fails because `scripts.evaluate_business_pipeline` does not exist.

- [ ] **Step 3: Freeze at least 60 human-reviewed cases across the supported boundary**

Required distribution:

```text
10 Student FAQ queries
10 Difficulty queries
10 Recurring misconception queries
8 Exam error queries
8 Tutor strategy queries
6 Content improvement queries
5 specific micro/case queries
3 cohort/unsupported queries
```

For every case, store only query/filters in `queries.jsonl`. Store SQL facts separately in `expected_facts.jsonl` and exact authoritative Turn keys/text separately in `expected_evidence.jsonl`. Freeze dev/holdout membership and file hashes in `split_manifest.json`. Expected facts must be computed independently, then human-reviewed; they cannot be created by calling `BusinessRAGPipeline`.

- [ ] **Step 4: Implement exact metrics and release gates**

Measure:

```text
route intent/scope/perspective exact match
plan type exact match
ranked item exact count match
top-k cluster precision/recall
distinct-session-count exact match
session-share numeric equality within 1e-9
DataScope exact match
citation precision/recall
role accuracy
quote exact match
candidate authorization rate
status accuracy for NO_EVIDENCE/UNSUPPORTED/FAILED
per-stage latency and token telemetry completeness
```

Hard gates:

```python
HARD_GATES = {
    "count_exact_match": 1.0,
    "data_scope_exact_match": 1.0,
    "citation_precision": 1.0,
    "citation_recall": 1.0,
    "role_accuracy": 1.0,
    "quote_exact_match": 1.0,
    "candidate_authorization_rate": 1.0,
    "unsupported_status_accuracy": 1.0,
}
```

Semantic answer quality may be evaluated separately with DeepEval when the evaluation model is configured. A missing judge yields `UNMEASURED`, not a substitute heuristic score.

- [ ] **Step 5: Run end-to-end evaluation into a new output directory**

Run:

```powershell
.venv\Scripts\python.exe scripts/evaluate_business_pipeline.py `
  --dataset evals/datasets/business-response-v1 `
  --source-db data/db/tutoring_knowledge.duckdb `
  --chroma-dir data/chroma `
  --analytics-db reports/staging/business-analytics-v1/business-analytics.duckdb `
  --split holdout `
  --output reports/eval/business-pipeline-v1-20260907
```

Expected: `report.json`, `report.md`, `metrics.jsonl`, `traces.jsonl`, and `failures.jsonl`. Exit non-zero on any hard-gate failure.

- [ ] **Step 6: Commit evaluation fixtures and runner**

```powershell
git add evals/datasets/business-response-v1 scripts/evaluate_business_pipeline.py tests/test_business_pipeline_eval.py
git commit -m "test(eedi-rag): freeze business analytics evidence evaluation"
```

---

### Task 8: Run Regression, Paired Comparison, and Controlled Promotion

**Files:**
- Modify: `src/cli.py`
- Modify: `docs/Query_Intent_Response_Executable_Design.md:1288-1318`
- Create: `docs/Business_RAG_Capability_Boundary_Runbook.md`
- Create at runtime: `reports/eval/business-rag-release-20260907/`

**Interfaces:**
- Consumes: all Part 1-3 tests, current legacy baseline, new business evaluation reports, artifact hashes.
- Produces: evidence-backed release decision, runbook, and only then a CLI-default promotion.

- [ ] **Step 1: Run focused business tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_business_models.py `
  tests/test_query_intent.py `
  tests/test_query_planner.py `
  tests/test_data_capabilities.py `
  tests/test_analytics_models.py `
  tests/test_analytics_store.py `
  tests/test_event_extraction.py `
  tests/test_event_derivation.py `
  tests/test_analytics.py `
  tests/test_cohort_data.py `
  tests/test_business_evidence.py `
  tests/test_business_generation.py `
  tests/test_business_response.py `
  tests/test_business_pipeline.py `
  tests/test_business_cli.py `
  tests/test_business_pipeline_eval.py -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the entire repository test suite and preserve the log**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -v 2>&1 | Tee-Object reports/eval/business-rag-release-20260907/pytest-full.log
```

Expected: all tests pass. If an unrelated pre-existing failure remains, record it explicitly and do not claim a green release.

- [ ] **Step 3: Run paired legacy-vs-business cases on identical inputs**

Run the 60-case set through:

```text
A: existing EndToEndPedagogicalRAGPipeline.ask()/ask_stream baseline
B: BusinessRAGPipeline.ask_business()
```

Use identical source DuckDB/Chroma hashes and queries. Compare route correctness, count exactness, evidence correctness, answer relevance, and latency. Gold stays in the evaluation control plane.

- [ ] **Step 4: Audit artifact immutability and capability truthfulness**

Require:

```text
source DuckDB hash before == after
source Chroma hash before == after
analytics source hash == source DuckDB hash
analytics manifest == READY
invalid event evidence == 0
current Eedi RETAKE_COHORT == unavailable
current Eedi ABILITY_COHORT == unavailable
no src/data artifact selected by CLI
```

- [ ] **Step 5: Write the capability-boundary runbook from measured behavior**

The runbook states:

```text
DuckDB/analytics answers what patterns exist and how often.
The student/tutor RAG retrieves how those patterns appear in real dialogue.
Parent-5/W7 supplies bounded narrative context, not population statistics.
The LLM explains authorized facts; it does not create facts.
Cohort claims require authorized longitudinal data and explicit definitions.
```

Include exact startup commands for local and production profiles, expected startup provenance, data promotion procedure, failure/status meanings, and rollback to `--pipeline legacy`.

- [ ] **Step 6: Promote business pipeline only if every hard gate passes**

Change `--pipeline` default from `legacy` to `business` in one isolated commit. If any hard gate fails, leave the default unchanged and record `BLOCKED` with the failing metrics.

Run after the default change:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_business_cli.py tests/test_business_pipeline.py -v
```

Expected: business is the default and legacy remains explicitly selectable.

- [ ] **Step 7: Commit documentation and gated promotion**

```powershell
git add src/cli.py docs/Query_Intent_Response_Executable_Design.md docs/Business_RAG_Capability_Boundary_Runbook.md
git commit -m "feat(eedi-rag): promote evidence-backed business query pipeline"
```

## Part 3 Exit Gate

Part 3 is complete only when:

- Macro questions obtain every number, rank, and scope field from the analytics sidecar.
- Micro questions use the existing student/tutor RAG path without running full-corpus SQL unnecessarily.
- Student/tutor evidence perspective controls which physical retrieval lanes execute.
- Every macro representative citation is authorized by the exact aggregate event and authoritative Turn.
- Parent-5/W7/4000-token configuration is used only when runtime provenance proves that route is active.
- Top-10 reports can render ten deterministic rows while narrative context remains bounded to at most five representative anchors.
- Current Eedi retaker/weak-foundation questions return `UNSUPPORTED` without an LLM call.
- User-visible content appears only after response and evidence audit.
- Full regression, business holdout hard gates, artifact immutability checks, and paired baseline comparison have passed.

## Execution Order Across the Plan Suite

Execute and review in this order:

```text
Part 1: Contract and Routing
  -> Part 2: Analytics Events and Data Capability
     -> Part 3 Tasks 1-5: Evidence Integration
        -> Part 3 Task 6: CLI Business Mode
           -> Part 3 Task 7: Full Evaluation
              -> Part 3 Task 8: Controlled Promotion
```

Do not parallelize tasks that edit `src/business_models.py`, `src/retriever.py`, `src/reranker.py`, `src/rag_pipeline.py`, or `src/cli.py`. Data-fixture authoring and independent evaluator implementation may run in parallel only after their consumed contracts are committed.
