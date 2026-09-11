# Knowledge Graph Gold Validation and Task 6+ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Within the next five hours, connect the approved business Gold set to the implemented Session/Card knowledge-graph route, execute real Router/Planner/SQL/Graph/Card/W7/Prompt/Citation validation, diagnose failures, and finish the remaining approved tasks from the Session-Card Knowledge Graph plan where time allows.

**Architecture:** The source DuckDB and READY graph sidecar remain authoritative. Each Gold case is run through the graph business route: deterministic Router, capability-aware Planner, SQL/Graph analytics or graph evidence traversal, Card evidence selection, W7/S3 window expansion and Window-level reranking/assembly, then structured response and citation audit. The evaluator records stage traces and applies hard gates; diagnostic metrics explain failures without converting failures into passes.

**Tech Stack:** Python 3.14, DuckDB, ChromaDB, Pydantic 2, JSONL, pytest, SHA-256, existing Qwen reranker/generator providers where configured.

**Spec:** `docs/superpowers/plans/2026-09-08-session-card-knowledge-graph.md`; approved Gold contract in `data/business/business-graph-gold-v1.jsonl`.

## Current State and Evidence

- Active branch: `codex/eedi-rag-session-card-knowledge-graph`.
- Git root: `E:\PIAgent`; project: `E:\PIAgent\10-projects\AgentLearn\Eedi-RAG`.
- Original graph plan is implemented through Task 5: graph contracts/store, Session/Card projection, taxonomy, analytics, evidence bridge, and component evaluators.
- Graph topology hardening was committed in `a315db8`.
- Business Gold was built from real artifacts and expanded to 83 cases; the latest macro/query correction is `17d4c02`.
- Gold was manually approved for this evaluation in `data/business/business-graph-gold-v1.manifest.json`; this approval is scoped to this evaluation run.
- Gold coverage is 17 categories: 35 macro, 24 micro, 15 mixed, 3 `NO_EVIDENCE`, 3 `UNSUPPORTED`, and 3 `FAILED`.
- Current authoritative source DB: `data/db/tutoring_knowledge.duckdb`.
- Current source SHA-256: `f089c56f162666ab71b3beecef33a871f3e140608721b3f61f96f9ed384b6157`.
- Matching READY graph sidecar used to author Gold: `reports/staging/session-card-graph-v1/graph-taxonomy-verify-2-20260908.duckdb`.
- Current graph runtime policy frozen in Gold: W7/S3, Window is the smallest reranking and prompt unit, Turn is provenance and citation evidence only.
- Source database contains 1,576 sessions, 1,576 misconception Cards, 1,576 tutor strategy Cards, 37,186 Turns, and 11,337 stored legacy sliding windows.
- Existing source tables are `tutoring_sessions`, `session_dialogue_turns`, `misconception_chunks`, `tutor_strategy_chunks`, and `sliding_window_chunks`.
- Current business models are in `src/business_models.py`; Router is `src/query_intent.py`; Planner is `src/query_planner.py`; Graph analytics is `src/graph_analytics.py`; Bridge is `src/graph_evidence_bridge.py`; RAG route is `src/rag_pipeline.py`.
- Current known configuration split: anchored runtime uses W7/S3 in `src/rag_pipeline.py`; legacy source/index builders still contain W6/S3 references. Do not silently mix these policies in a single evaluation.
- Current worktree is dirty with unrelated user changes. Preserve them and stage only files belonging to this plan.

## Non-negotiable Truth Rules

- Gold cases are never regenerated from actual SUT output.
- Positive Card, Graph, Window, Turn, question, and quote evidence must resolve against the frozen source artifacts.
- `COUNT(DISTINCT source_session_id)` is the primary macro prevalence measure; event counts remain separate.
- A macro Case must validate aggregate facts and representative evidence authorization; a single unrelated Session cannot stand in for a corpus ranking.
- W7 Window is evaluated as one ranking unit. Individual Turn recall must not be interpreted as Window recall.
- `NO_EVIDENCE`, `UNSUPPORTED`, and `FAILED` are valid expected outcomes and must remain explicit.
- Missing model credentials or missing artifacts produce `UNMEASURED`/`FAILED`, never a fabricated success.
- The final release decision is for the graph business route only; legacy Card-only and unrelated L1/L2 metrics are comparison diagnostics.

## Five-Hour Execution Schedule

The schedule is a priority order, not permission to weaken hard gates. If a stage consumes more time, skip optional expansion before skipping runtime validation.

| Time | Objective | Required output | Exit condition |
|---|---|---|---|
| 00:00–00:30 | Freeze runtime and inspect interfaces | runtime manifest, artifact checks, route matrix | no unresolved path/config ambiguity |
| 00:30–01:20 | Build graph-business execution adapter | one traceable Case execution | one positive and one negative Case run end to end |
| 01:20–02:20 | Implement actual-vs-Gold evaluator | per-stage trace and hard-gate report | evaluator runs all 83 cases without synthetic evidence |
| 02:20–03:10 | Run evaluation and classify failures | holdout/dev metrics and failure buckets | every failure has a stage and reason |
| 03:10–04:20 | Fix graph-route blockers | focused tests and rerun delta | only evidence-backed fixes are applied |
| 04:20–05:00 | Full verification and handoff | final report, diagnosis, remaining Task 6+ plan | truthful PASS/FAILED/UNMEASURED decision |

## Task 0: Freeze Runtime Inputs and Gold Preconditions

**Files:**
- Read: `data/business/business-graph-gold-v1.manifest.json`
- Read: `data/business/business-graph-gold-v1.jsonl`
- Read: `src/query_intent.py`
- Read: `src/query_planner.py`
- Read: `src/data_capabilities.py`
- Read: `src/rag_pipeline.py`
- Read: `src/retriever.py`
- Read: `src/reranker.py`
- Read: `src/graph_analytics.py`
- Read: `src/graph_evidence_bridge.py`
- Create: `reports/eval/business-graph-v1-runtime-freeze.json`

**Interfaces:**
- Consumes: approved Gold manifest and real source/graph artifacts.
- Produces: a runtime freeze containing source hashes, graph hash, taxonomy hash, W7/S3, parent limit, reranker unit, prompt budget, generator contract, and capability snapshot.

- [ ] Verify Gold manifest `review_status` is `HUMAN_APPROVED_FOR_THIS_EVALUATION`.
- [ ] Verify source and graph SHA-256 values still match the manifest.
- [ ] Verify exactly one READY graph manifest is used.
- [ ] Resolve actual Planner matrix names and map each approved Gold lever to an existing or explicitly missing implementation.
- [ ] Record that `CONTENT_IMPROVEMENT` currently carries the cross-session co-occurrence route because there is no separate `CO_OCCURRENCE` intent enum.
- [ ] Record whether Qwen reranker and generator credentials are available; mark missing external services as `UNMEASURED`, not pass.
- [ ] Run `python -m pytest tests/test_business_graph_gold.py -q` before changing runtime code.

## Task 1: Define a Trace Contract for the Graph Business Route

**Files:**
- Create: `src/business_graph_trace.py`
- Test: `tests/test_business_graph_trace.py`

**Interfaces:**
- `BusinessGraphRuntimeTrace`
- `run_business_case(case, runtime) -> BusinessGraphRuntimeTrace`
- Trace fields: `actual_router`, `actual_plan`, `capabilities`, `analytics_result`, `graph_paths`, `card_candidates`, `parent_cards`, `candidate_windows`, `ranked_windows`, `selected_windows`, `assembled_prompt`, `citations`, `answer`, `stage_status`, `errors`, and timings.

- [ ] Write tests for one micro Case, one macro Case, one `NO_EVIDENCE` Case, one `UNSUPPORTED` Case, and one `FAILED` precondition.
- [ ] Ensure trace serialization never includes Gold expected answers inside the runtime request.
- [ ] Ensure each stage records `SUCCESS`, `NO_EVIDENCE`, `UNSUPPORTED`, `FAILED`, or `UNMEASURED` explicitly.
- [ ] Ensure Window records include `window_id`, `session_id`, `source_turn_ids`, `anchor_turn_ids`, and rerank rank.
- [ ] Ensure Turn records are stored as authorized evidence, never as ranking candidates.

## Task 2: Implement the Graph Business Execution Adapter

**Files:**
- Create or modify: `src/business_graph_pipeline.py`
- Modify: `src/rag_pipeline.py` only where a narrow reusable seam is required
- Test: `tests/test_business_graph_pipeline.py`

**Interfaces:**
- `BusinessGraphPipeline.__init__(source_db, graph_db, chroma_dir, runtime_config)`
- `BusinessGraphPipeline.prepare_case(query) -> BusinessGraphRuntimeTrace`
- `BusinessGraphPipeline.ask_case(query, mode=...) -> BusinessGraphRuntimeTrace`

**Route rules:**

```text
QUESTION_TUTORING
  -> MICRO
  -> question/session-scoped Card retrieval
  -> Parent Card anchoring
  -> W7/S3 Window expansion
  -> Window rerank
  -> Prompt assembly

RECURRING_MISCONCEPTIONS / DIFFICULT_CONCEPTS / EXAM_ERROR_PATTERNS /
STUDENT_FREQUENT_QUESTIONS / TUTOR_STRATEGY_PATTERNS
  -> HYBRID
  -> DuckDB/Graph aggregate first
  -> authorized representative evidence second

CONTENT_IMPROVEMENT with cross-session language
  -> misconception_strategy_pairs when the query asks co-occurrence

STUDENT_COHORT_PATTERNS without cohort fields
  -> REJECT / UNSUPPORTED
```

- [ ] Write a failing integration test proving a micro query reaches Router, Planner, Card, W7, and Prompt stages.
- [ ] Write a failing integration test proving a macro query uses analytics output rather than vector score as its ranking fact.
- [ ] Write a failing integration test proving a cross-session query uses `CO_OCCURS_WITH`/`misconception_strategy_pairs`.
- [ ] Write a failing integration test proving an unsupported cohort query stops before evidence generation.
- [ ] Use GraphEvidenceBridge to authorize representative references and verify all materialized Turns against source DuckDB.
- [ ] Use the existing production assembler with `rerank_unit="anchored_logical_window"`, Window size 7, step 3, and Parent-5 where the model reranker is available.
- [ ] If the reranker is unavailable, return `UNMEASURED` for model-ranking metrics and run only deterministic structural checks.
- [ ] Do not force macro queries through a single-session Card/Window path when the Gold has no required Window evidence.

## Task 3: Implement Actual-vs-Gold Evaluation

**Files:**
- Create: `scripts/evaluate_business_graph.py`
- Create: `evals/business/business_graph_metrics.py`
- Test: `tests/test_business_graph_eval.py`

**Interfaces:**
- `evaluate_case(case, trace) -> CaseEvaluation`
- `evaluate_dataset(gold_path, runtime) -> BusinessGraphEvaluationReport`
- CLI options: `--gold`, `--source-db`, `--graph-db`, `--chroma-dir`, `--split`, `--output`, `--mode`.

**Per-case checks:**

- Router exact match: intent, scope, computation scope, perspectives.
- Planner exact match: plan type, selected lever, analytics operations, retrieval perspectives.
- SQL scope: filters, required tables/operations/keywords, aggregate semantics, and scope confinement.
- Graph path: required node/edge path authorization, source session consistency, and no unsupported causal edge.
- Card evidence: expected Card existence, candidate recall, Parent Card inclusion, and source pointer validity.
- Window evidence: expected W7/S3 Window candidate recall, selected Window coverage, anchor containment, and rank diagnostics.
- Prompt: Window-level coverage, overlap-aware deduplication, required sections, source session confinement, and recomputed token count.
- Citation: authorized Window membership, source Turn membership, speaker correctness, exact quote equality, citation precision, and citation recall.
- Status: exact `SUCCESS`, `NO_EVIDENCE`, `UNSUPPORTED`, or `FAILED` behavior.
- Answer: deterministic required/forbidden claim checks; LLM judge scores remain diagnostic and cannot override hard gates.

**Hard gates:**

```python
HARD_GATES = {
    "router_exact_match": 1.0,
    "planner_exact_match": 1.0,
    "sql_graph_scope_confinement": 1.0,
    "card_evidence_validity": 1.0,
    "window_coverage": 1.0,
    "prompt_window_coverage": 1.0,
    "citation_precision": 1.0,
    "citation_recall": 1.0,
    "role_accuracy": 1.0,
    "exact_quote_match": 1.0,
    "status_accuracy": 1.0,
    "source_artifact_unchanged": 1.0,
}
```

- [ ] Write failing metric tests using deliberately wrong Router, wrong Window, wrong quote, cross-session leakage, and fabricated Card IDs.
- [ ] Implement per-stage comparison with stable reason codes such as `ROUTER_INTENT_MISMATCH`, `PLAN_LEVER_MISMATCH`, `SQL_SCOPE_MISMATCH`, `GRAPH_PATH_MISSING`, `CARD_NOT_RETRIEVED`, `WINDOW_NOT_RETRIEVED`, `PROMPT_WINDOW_MISSING`, `CITATION_UNAUTHORIZED`, `QUOTE_MISMATCH`, and `STATUS_MISMATCH`.
- [ ] Separate `candidate_window_recall`, `selected_window_coverage`, and `prompt_window_coverage`; do not collapse them into Turn recall.
- [ ] Recompute all token counts from emitted Prompt text.
- [ ] Hash source artifacts before and after evaluation and fail if either changes.
- [ ] Write JSON and Markdown reports with dev/holdout breakdown and first-failure stage per Case.

## Task 4: Run the Graph-Only Gold Evaluation

**Files:**
- Create at runtime: `reports/eval/business-graph-v1-<run-id>/report.json`
- Create at runtime: `reports/eval/business-graph-v1-<run-id>/report.md`
- Create at runtime: `reports/eval/business-graph-v1-<run-id>/cases.jsonl`

- [ ] Run a five-case smoke set first: one micro student case, one micro tutor case, one macro recurrence case, one mixed case, and one unsupported case.
- [ ] Inspect traces manually for Router → Planner → Graph/SQL → Card → Window → Prompt ordering.
- [ ] Run all 83 cases on `dev` and `holdout` separately.
- [ ] Run with real model reranker/generator only when configured; otherwise record exact unmeasured dependencies.
- [ ] Preserve the existing component reports as diagnostics, but use only the new graph-business report for the graph-line release decision.
- [ ] Confirm all macro Cases contain real aggregate rows and do not inherit Session 10 micro evidence.
- [ ] Confirm all micro Cases have non-noise Window source Turns.
- [ ] Confirm all negative Cases produce their intended explicit status.

## Task 5: Diagnose Failures Without Masking Them

Classify every failed Case by first failing layer:

```text
DATA_ARTIFACT
  source/graph hash, READY manifest, missing table, stale taxonomy

ROUTER
  intent/scope/perspective mismatch, ambiguity, protected-token loss

PLANNER
  wrong plan type, missing operation, wrong lever, unsupported capability handling

ANALYTICS
  wrong filter, wrong aggregate, wrong distinct-session denominator, hidden UNCLASSIFIED

GRAPH
  missing node/edge/path, wrong relation direction, unauthorized causal inference

CARD
  wrong lane, missing Parent Card, source pointer mismatch

WINDOW
  wrong W6/W7 policy, wrong step, anchor outside Window, candidate loss

ASSEMBLY
  Window omitted, overlap deduplication loss, token budget loss, section omission

CITATION
  unauthorized Window/Turn, role mismatch, altered quote, missing required role

GENERATION
  claim unsupported, forbidden claim, invalid response contract, audit failure
```

- [ ] Produce failure counts by layer, category, split, intent, and subject path.
- [ ] For each hard-gate failure, save the minimal reproducer query and Case ID.
- [ ] Distinguish data quality issues from route implementation issues and from external model unavailability.
- [ ] Never lower a threshold or remove a Gold field to make a failing report pass.

## Task 6: Fix Only Evidence-Backed Graph-Line Problems

**Files:**
- Modify only the files implicated by Task 5 reason codes.
- Add focused tests beside each fix.

- [ ] Fix Router only when an approved Gold label is consistent with the business intent matrix and real query wording.
- [ ] Fix Planner/lever mappings only when the implementation lacks a route required by an approved Case.
- [ ] Fix GraphAnalytics only when an independent source SQL check demonstrates a factual mismatch.
- [ ] Fix EvidenceBridge only when an authorized pointer can be verified in source data and the bridge rejects/loses it incorrectly.
- [ ] Fix W7 expansion/assembly only when the expected Window is generated from real source Turns and the production route cannot reach it.
- [ ] Keep every unrelated dirty worktree change untouched.
- [ ] After each fix, rerun the focused reproducer before rerunning the full suite.

## Task 7: Complete Remaining Original Plan Items If Time Remains

The original plan after Task 5 requires business generation integration, pipeline execution, CLI integration, full Gold evaluation, and promotion controls. Implement in this order:

- [ ] Add `BusinessResponseEnvelope` production integration carrying understanding, plan, data scope, result, evidence, limitations, and audit status.
- [ ] Add macro execution adapters for all `PLAN_MATRIX` analytics operations, including `content_opportunities` and `misconception_strategy_pairs`.
- [ ] Add Graph path materialization to the final generator context while keeping raw Turn evidence authorized by the bridge.
- [ ] Add a graph-business CLI mode with explicit `--route business-graph` and no silent default switch.
- [ ] Add `business-graph-v1` dataset selection, dev/holdout split checks, and artifact hash preconditions to the CLI.
- [ ] Add regression comparison against the legacy Card-only route using identical query lists and source/index hashes.
- [ ] Keep default production route unchanged unless every graph hard gate and legacy micro regression gate pass.
- [ ] If the graph route cannot pass, publish `BLOCKED` with measured first-failure reasons and leave the legacy route unchanged.

## Task 8: Final Verification and Handoff

- [ ] Run Gold structural tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_business_graph_gold.py -q
```

- [ ] Run focused graph tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_models.py tests/test_graph_store.py tests/test_graph_projection.py tests/test_graph_analytics.py tests/test_graph_evidence_bridge.py -q
```

- [ ] Run the new end-to-end evaluator on smoke, dev, and holdout.
- [ ] Run the full relevant pytest suite; report pre-existing failures separately from changes in this plan.
- [ ] Verify source DB, Graph DB, Card/Chroma artifacts, and Gold hashes in the final report.
- [ ] Report hard-gate result, diagnostic metrics, failure buckets, model availability, and whether graph-line promotion is allowed.
- [ ] Commit only evaluation and graph-line files created or modified by this plan.

## Deliverables for the Next Conversation

The next agent must start by reading this plan and then inspect:

```text
data/business/business-graph-gold-v1.manifest.json
data/business/business-graph-gold-v1.coverage.json
data/business/business-graph-gold-v1.jsonl
reports/eval/business-graph-v1-runtime-freeze.json
docs/superpowers/plans/2026-09-08-session-card-knowledge-graph.md
```

The first message should state the current stage, the latest evaluator result, and the next unchecked Todo. It must not regenerate Gold, select a different graph sidecar, or treat the existing component 100% reports as end-to-end business success.

