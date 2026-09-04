# Chunk Selection E2E Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Build a fast, production-shaped 10-session chunk-strategy benchmark that uses an independent test database, exercises ingestion, retrieval, reranking/assembly, evidence integrity, and produces an auditable selection report.

**Architecture:** The benchmark owns an isolated fixture under `tests/db/chunk_selection_10_v1/`; it never reads or hashes `data/db` or `data/chroma`. A canonical test seed is loaded into a fresh DuckDB/Chroma pair per candidate strategy. Each candidate then runs through the production storage manager, retriever, assembler/MMR, and citation audit adapters. LLM generation and provider reranking are disabled and reported as `UNMEASURED`; deterministic grounding and business/MMR assembly remain measurable.

**Tech Stack:** Python 3.14, pytest, Pydantic, DuckDB, ChromaDB, production `DualEngineStorageManager`, `DualMetricRetriever`, `PedagogicalGoldAssembler`, `src.rag_pipeline`, existing `evals` contracts/runners, JSONL fixtures.

**Spec:** Brainstorming design approved in chat on 2026-09-03; Luna read-only review `luna_e2e_eval_design`.

## Global Constraints

- The test database is independent test data under `tests/db`; production `data/db` and `data/chroma` are out of scope.
- No real LLM/API calls, provider embeddings, or provider reranker calls are allowed in this smoke test.
- Every candidate uses its own temporary DuckDB file and Chroma directory; no candidate mutates another candidate's artifact.
- Every reported quality number must identify whether it is measured, derived, or `UNMEASURED`.
- The existing 10 sample sessions are the input boundary; quality qrels must cover all 10 sessions, including Session 80.
- No single weighted score may override a failed integrity or citation hard gate.
- Test output must include the fixture version, candidate configuration, and test-artifact hashes generated inside `tests/db` only.

---

### Task 1: Freeze the independent 10-session test table

**Files:**
- Create: `tests/db/chunk_selection_10_v1/run_manifest.json`
- Create: `tests/db/chunk_selection_10_v1/sessions.jsonl`
- Create: `tests/db/chunk_selection_10_v1/extracted_pius.jsonl`
- Create: `tests/db/chunk_selection_10_v1/queries.jsonl`
- Create: `tests/db/chunk_selection_10_v1/qrels.jsonl`
- Create: `tests/db/chunk_selection_10_v1/candidate_config.json`
- Test: `tests/evals/test_slect_chunking.py`

**Interfaces:**
- Consumes: the existing 10-session sample and 10 existing golden queries as source material during fixture creation only.
- Produces: a self-contained fixture with 10 sessions, 10 audited query cases (one per session), 10 extracted card payloads, graded Turn qrels, and candidate metadata consumed by later tasks.

- [ ] **Step 1: Write the failing fixture validation test**

Add a test that loads only `tests/db/chunk_selection_10_v1/` and asserts:

```python
assert len(sessions) == 10
assert len(queries) == 10
assert {query["target_intervention_id"] for query in queries} == {session.intervention_id for session in sessions}
assert all(case["label_source"] == "human_reviewed" for case in qrels)
```

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\evals\\test_slect_chunking.py -k fixture`

Expected: FAIL because the independent fixture and Session 80 qrel do not exist yet.

- [ ] **Step 2: Create the fixture files**

Copy the 10 JSON objects from `data/sample/cleaned_sessions_sample.jsonl` into `sessions.jsonl`; copy the 10 corresponding objects from `data/sample/extracted_pius_sample.jsonl` into `extracted_pius.jsonl`; copy the existing query cases and replace the duplicate Session 10 case with a human-reviewed Session 80 query. Store each qrel with:

```json
{
  "query_id": "q80",
  "target_session_ids": [80],
  "required_evidence": [{"turn_id": 5, "role": "tutor", "grade": 2}, {"turn_id": 7, "role": "tutor", "grade": 2}, {"turn_id": 11, "role": "tutor", "grade": 3}],
  "acceptable_supporting_turn_ids": [4, 6, 8, 9, 10],
  "expected_collection": "strategy",
  "label_source": "human_reviewed"
}
```

`run_manifest.json` must record fixture version, source file paths (not production paths), Python version, dependency lock hash, and `llm_calls_allowed=false`. `candidate_config.json` must enumerate the candidate strategies and slot policies without embedding query-specific decisions.

- [ ] **Step 3: Run the fixture validation test**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\evals\\test_slect_chunking.py -k fixture`

Expected: PASS with exactly 10 sessions and 10 unique target sessions.

- [ ] **Step 4: Commit the frozen test table**

```bash
git add tests/db/chunk_selection_10_v1 tests/evals/test_slect_chunking.py
git commit -m "test: freeze independent chunk selection fixture"
```

### Task 2: Add independent test-database builders and ingestion audits

**Files:**
- Modify: `tests/evals/test_slect_chunking.py`
- Test: `tests/evals/test_slect_chunking.py`

**Interfaces:**
- Consumes: fixture files from Task 1; candidate Chunk factories from the existing test module.
- Produces: `build_candidate_artifacts(candidate_name, sessions, chunks, root) -> CandidateArtifacts` and `audit_candidate_ingestion(artifacts, sessions, chunks) -> dict`.

- [ ] **Step 1: Write failing tests for isolation and round-trip integrity**

Assert that each candidate creates `tests/db`-scoped or pytest-temporary `tutoring_knowledge.duckdb` and `chroma/` paths, that production paths are never opened, and that the audit reports:

```text
session_ingestion_coverage = 1.0
turn_ingestion_coverage = 1.0
source_turn_id_valid_rate = 1.0
source_turn_role_valid_rate = 1.0
exact_text_roundtrip_rate = 1.0
duplicate_chunk_rate = 0.0
empty_chunk_rate = 0.0
```

Run the focused test and expect failure because the builder/audit functions are absent.

- [ ] **Step 2: Implement candidate artifact construction**

For each candidate, create a fresh temporary directory under `tests/db/chunk_selection_10_v1/runs/<run_id>/<candidate>/` (or a pytest `tmp_path` child when cleanup is enabled). Instantiate `DualEngineStorageManager(db_path=..., chroma_dir=..., embedding_backend="deterministic")`; call `ingest_sessions()` for all 10 sessions. Map experimental Action/Exchange objects into the production fallback-window contract with metadata fields `experimental_strategy`, `role_scope`, and `boundary_reason`; call `ingest_sliding_window_chunks()`. For the card candidate, call `ingest_extracted_pius()` with the frozen test `ExtractedPIU` objects.

Do not call `ingest_all()` for mixed experimental candidates because its strict card requirement would make zero-LLM candidates impossible. Keep the test adapter explicit about which production methods are exercised.

- [ ] **Step 3: Implement and run the ingestion audit**

Read the candidate DB through DuckDB read-only queries and Chroma through the existing immutable snapshot adapter. Verify session/Turn counts, all candidate chunk pointers, role constraints, exact Turn text presence, parent IDs where applicable, duplicate IDs, empty documents, and collection counts. Persist the audit dictionary in the run report.

- [ ] **Step 4: Run tests and commit**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\evals\\test_slect_chunking.py -k ingestion`

Expected: PASS with all integrity hard gates at 1.0/0.0. Commit with `git add tests/evals/test_slect_chunking.py && git commit -m "test: build isolated candidate storage artifacts"`.

### Task 3: Execute the production retrieval and assembly path per candidate

**Files:**
- Modify: `tests/evals/test_slect_chunking.py`
- Test: `tests/evals/test_slect_chunking.py`

**Interfaces:**
- Consumes: `CandidateArtifacts` and qrels from Tasks 1–2.
- Produces: per-query traces with `retrieval_before_rerank`, `rerank_after`, `selected_chunk_ids`, `selected_evidence_turn_ids`, prompt token counts, and explicit reranker status.

- [ ] **Step 1: Write failing production-path tests**

For every candidate/query pair, assert that the trace contains:

```text
query_rewrite_backend
embedding_backend
retrieved_candidate_ids
retrieval_scores
assembler_selected_ids
assembler_evidence_turns
reranking_applied
reranker_status
audit_status
```

Assert `reranker_status == "UNMEASURED_PROVIDER_RERANKER"` when no provider reranker is configured, and never label MMR as a model reranker.

- [ ] **Step 2: Implement candidate-specific production adapters**

Instantiate `DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic", alpha=0.5)` and `PedagogicalGoldAssembler(lambda_diversity=0.7, max_prompt_tokens=1500, model_reranker=None, chunk_strategy=...)`. Use `retrieve_multi_perspective_rrf()` for the card candidate and `retrieve_fallback_windows()` for window/Action/Exchange candidates. Feed the resulting trace to `assemble_with_trace()` from `evals.adapters.assembler`. Record ranking before assembly, MMR/business selection after assembly, and actual evidence Turn IDs.

For fixed-slot experimental policies, execute each lane independently and merge only by the declared slot policy (`card:1 + window:2`, `action:2 + window:1`, and `action:1 + window:2`). Do not inspect qrels while choosing slots.

- [ ] **Step 3: Run retrieval/assembly tests**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\evals\\test_slect_chunking.py -k production_path`

Expected: PASS with complete traces for all candidates and `UNMEASURED` provider-reranker status.

- [ ] **Step 4: Commit**

```bash
git add tests/evals/test_slect_chunking.py
git commit -m "test: exercise production retrieval and assembler per candidate"
```

### Task 4: Add evidence, context, rerank-delta, and performance metrics

**Files:**
- Modify: `tests/evals/test_slect_chunking.py`
- Test: `tests/evals/test_slect_chunking.py`

**Interfaces:**
- Consumes: production traces and ingestion audits from Tasks 2–3.
- Produces: per-query and aggregate candidate rows with hard-gate status, measured metrics, and `UNMEASURED` fields.

- [ ] **Step 1: Write failing metric tests**

Assert the report includes:

```text
required_evidence_recall_at_3/5
supporting_evidence_recall_at_3/5
context_precision_at_3/5
context_required_turn_recall_after_assembly
role_correctness
turn_continuity
wrong_session_turn_rate
token_count
budget_violation
retrieval_mrr/nDCG
rerank_mrr_delta/nDCG_delta
assembler_drop_rate
retrieval_p50/p95
assembly_p50/p95
total_p50/p95
llm_request_count
llm_generation_cost
provider_reranker_quality
```

The last three must be explicitly `0`, `UNMEASURED`, and `UNMEASURED` respectively for this no-LLM run; do not infer cost from card loading time.

- [ ] **Step 2: Implement graded evidence and context metrics**

Use qrels grades to separate required and supporting evidence. Compute candidate-level retrieval metrics before assembly, then recompute after assembler selection. Deduplicate `(session_id, turn_id)` pairs before precision/recall calculations. Count role errors, gaps in Turn continuity, wrong-session evidence, token budget violations, and truncation loss. Invoke production `_audit_citations()` only on citations constructed from assembled evidence; record its actual status and failure count.

- [ ] **Step 3: Implement rerank delta and performance capture**

Record the exact candidate order before MMR/business selection and after it. Compute MRR/nDCG/evidence recall deltas. Measure ingestion, isolated index build, retrieval, assembly, and audit stages with `time.perf_counter()`. Report p50 and p95; do not combine latency and quality into an unbounded opaque score.

- [ ] **Step 4: Run metric tests**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests\\evals\\test_slect_chunking.py -k metrics`

Expected: PASS; all quality values remain in `[0, 1]`, hard-gate failures are visible, and unmeasured fields are not silently converted to zero-quality claims.

### Task 5: Generate the auditable short-validation report and selector

**Files:**
- Modify: `tests/evals/test_slect_chunking.py`
- Create: `tests/db/chunk_selection_10_v1/last_run_report.json` (generated, ignored if policy requires)
- Create: `tests/db/chunk_selection_10_v1/last_run_report.md` (generated, ignored if policy requires)
- Test: `tests/evals/test_slect_chunking.py`

**Interfaces:**
- Consumes: all candidate traces and metrics from Tasks 1–4.
- Produces: JSON/Markdown report with per-query rows, aggregate rows, candidate status, failure reasons, and a gated recommendation.

- [ ] **Step 1: Write failing selector tests**

Assert selection order:

```text
1. reject candidates failing ingestion integrity;
2. reject candidates failing required evidence or citation hard gates;
3. compare post-assembly nDCG/MRR/context precision;
4. use latency, token inflation, and cost only as tie-breakers;
5. report multiple candidates as “no winner” when no candidate passes the hard gates.
```

- [ ] **Step 2: Implement report generation**

Write `last_run_report.json` and `.md` only under `tests/db/chunk_selection_10_v1/`. Include fixture hash, test DB/Chroma hashes, candidate config, per-query traces, aggregate metrics, failure categories, and the exact list of `UNMEASURED` metrics. The recommendation must be one of `PASS_RECOMMENDED`, `DIAGNOSTIC_ONLY`, or `NO_WINNER_HARD_GATE_FAILED`.

- [ ] **Step 3: Run the complete short validation**

Run:

```bash
.venv\\Scripts\\python.exe -m pytest -q -s tests\\evals\\test_slect_chunking.py
.venv\\Scripts\\python.exe -m pytest -q tests\\test_chunker.py tests\\evals\\test_chunking_and_rewrite.py tests\\evals\\test_l1_v2_contracts.py
.venv\\Scripts\\python.exe -m compileall -q tests\\evals\\test_slect_chunking.py
git diff --check -- tests\\evals\\test_slect_chunking.py
```

Expected: the dedicated test suite passes; related production/evaluation tests remain green; report states whether a candidate passed hard gates or remains diagnostic-only.

- [ ] **Step 4: Commit the completed benchmark**

```bash
git add tests/evals/test_slect_chunking.py tests/db/chunk_selection_10_v1 docs/superpowers/plans/2026-09-03-chunk-selection-e2e-test-plan.md
git commit -m "test: add production-shaped chunk strategy selection benchmark"
```

## Self-Review Checklist

- The test DB is independent and never reads production artifacts.
- Session 80 receives a quality qrel; the report is genuinely 10-session, not 9-session plus speed-only.
- Every candidate passes through isolated persistence and production retrieval/assembly adapters.
- MMR is labeled assembly reranking; provider Cross-Encoder reranking remains `UNMEASURED`.
- Required and supporting evidence are graded separately.
- Citation audit uses assembled evidence, never copied gold citations.
- No opaque composite score can override integrity or evidence hard gates.
- LLM generation cost/latency is never inferred from loading frozen cards.
