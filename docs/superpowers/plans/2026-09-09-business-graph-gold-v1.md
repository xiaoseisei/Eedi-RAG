# Business Graph Gold v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a database-grounded end-to-end Gold dataset and evaluator covering business routing, planning, SQL/Graph scope, Card retrieval, W7 window retrieval/reranking, prompt assembly, citation audit, and final answer constraints.

**Architecture:** Gold cases are authored independently from system outputs, then validated against immutable source DuckDB, graph sidecar, Card records, and authoritative turns. The evaluator runs the real pipeline and compares stage traces at the Window level; Turns are used only for provenance, exact-quote, role, and authorization checks.

**Tech Stack:** Python, DuckDB, Pydantic 2, JSONL, JSON Schema, pytest, SHA-256.

**Spec:** Approved in chat on 2026-09-09; W7/S3 is frozen from the current anchored production path.

## Global Constraints

- Positive evidence must exist in the current database and graph artifacts.
- Gold must never be generated from Retriever, Reranker, Assembler, or evaluator outputs.
- W7 Window is the smallest ranking and prompt-assembly unit; Turn is not a ranking unit.
- Missing evidence produces `NO_EVIDENCE`; unsupported business claims produce `UNSUPPORTED`; execution failures produce `FAILED`.
- No source artifact mutation is allowed during authoring or evaluation.
- Existing dirty worktree changes must be preserved; only target files may be added/modified.

---

### Task 1: Freeze runtime configuration and build an independent authoring catalog

**Files:**
- Create: `scripts/build_business_graph_gold_catalog.py`
- Create: `data/business/business-graph-gold-v1.runtime.json`
- Test: `tests/test_business_graph_gold_catalog.py`

**Interfaces:**
- Consumes: source DuckDB, cleaned sessions, ready graph sidecar, taxonomy.
- Produces: deterministic catalog of real sessions/cards/turns/windows and immutable artifact hashes.

- [ ] Validate source and graph artifacts exist, are ready, and have matching provenance.
- [ ] Generate W7/S3 windows directly from authoritative turns, not from retrieval results.
- [ ] Export card IDs, source turn IDs, roles, exact quotes, graph nodes/edges, taxonomy labels, and question IDs.
- [ ] Fail closed if a pointer, role, quote, or window is invalid.
- [ ] Add tests proving catalog generation rejects fabricated Card IDs, wrong roles, and altered quotes.

### Task 2: Define and validate the end-to-end Gold schema

**Files:**
- Create: `data/business/business-graph-gold-v1.schema.json`
- Create: `src/business_graph_gold.py`
- Test: `tests/test_business_graph_gold_schema.py`

**Interfaces:**
- `load_gold_cases(path: Path) -> list[BusinessGoldCase]`
- `validate_gold_case(case: BusinessGoldCase, catalog: AuthoringCatalog) -> None`

- [ ] Model the approved fields: router, plan/lever, SQL scope, graph paths, Card evidence, W7 window evidence, prompt contract, citation contract, status, and ideal answer.
- [ ] Enforce status-specific invariants for `SUCCESS`, `NO_EVIDENCE`, `UNSUPPORTED`, and `FAILED`.
- [ ] Enforce Window-level evidence semantics and forbid Turn-level ranking fields.
- [ ] Enforce exact quote and source pointer validation against the independent catalog.
- [ ] Enforce unique case IDs and category coverage metadata.

### Task 3: Author real-data Gold cases across macro, micro, and mixed coverage

**Files:**
- Create: `data/business/business-graph-gold-v1.jsonl`
- Create: `data/business/business-graph-gold-v1.coverage.json`
- Test: `tests/test_business_graph_gold_coverage.py`

**Interfaces:**
- Every positive case contains the full approved end-to-end structure.
- Negative cases contain explicit status and rejection/limitation expectations.

- [ ] Select only cases whose facts and evidence can be verified from the current artifacts.
- [ ] Cover the 17 requested categories, including macro, micro, mixed, no-evidence, unsupported, and failure behavior.
- [ ] Use `QUESTION_TUTORING`/`MICRO` for a specific student error and `RECURRING_MISCONCEPTIONS`/`HYBRID` for cross-session prevalence.
- [ ] Bind positive cases to W7/S3 windows and exact source turns without using Turn as a ranking target.
- [ ] Split by session isolation and record counts per category/split; do not invent a target count when real data cannot support it.

### Task 4: Implement the real end-to-end evaluator

**Files:**
- Create: `evals/business/business_graph_e2e_eval.py`
- Create: `tests/test_business_graph_e2e_eval.py`

**Interfaces:**
- `evaluate_case(case, runtime) -> CaseEvaluation`
- `evaluate_dataset(gold_path, runtime) -> BusinessGraphGoldReport`
- CLI emits JSON and Markdown reports.

- [ ] Run Router and compare exact expected fields.
- [ ] Run Planner with real capability snapshots and compare plan type, operations, perspectives, and lever.
- [ ] Verify SQL/Graph scope without matching implementation-specific SQL text.
- [ ] Compare Card candidates/parents, candidate W7 windows, ranked windows, selected windows, and prompt window coverage.
- [ ] Recompute prompt token counts and verify overlap-aware deduplication from authoritative turns.
- [ ] Verify citations are authorized by selected Windows and exact source quotes.
- [ ] Check required and forbidden answer claims with deterministic token/phrase rules; judge-based scoring remains diagnostic.
- [ ] Apply hard gates: router, plan, scope, Card validity, Window coverage, prompt coverage, citation fidelity, and status correctness.

### Task 5: Add manifest, reproducible CLI, and documentation

**Files:**
- Create: `data/business/business-graph-gold-v1.manifest.json`
- Modify: `evals/cli.py`
- Create: `docs/business-graph-gold-v1.md`
- Test: `tests/test_business_graph_gold_manifest.py`

- [ ] Bind the Gold manifest to source DB, graph DB, taxonomy, runtime configuration, schema, case file, and split hashes.
- [ ] Add a CLI command that refuses stale artifacts and reports `PASSED`/`FAILED`/`UNMEASURED` truthfully.
- [ ] Document category coverage, status semantics, Window-level metrics, and known unavailable business outcomes.
- [ ] Run focused tests and the full relevant test suite.

