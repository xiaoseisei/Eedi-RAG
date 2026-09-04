# Generator-First L2 Evaluation and Contract v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the Generator prompt, evidence citation contract, claim-level provenance, Gold Context, and DeepEval rubric so L2 can distinguish generator failures from upstream context failures.

**Architecture:** Keep the current Anchored Parent-3/W5 retrieval and overlap-aware Assembler frozen. Add an explicit Generator contract version: legacy v1 remains available for rollback, while v2 presents a deterministic evidence catalog, requires evidence IDs instead of free-form quotes, validates every factual claim against cited evidence, and materializes canonical citation quadruples server-side. Add a structured Gold Context v2 that contains real card facts plus complete logical windows, and evaluate v1/v2 Generator behavior on the same 30-case Gold/Real tracks with the same Judge.

**Tech Stack:** Python 3.14+, Pydantic v2, OpenAI-compatible Generator/Judge APIs, DeepEval 4.2.0, existing DuckDB/Chroma artifacts, pytest.

**Spec:** User-approved Generator-first requirements in the active conversation; L2 baseline evidence in `docs/L2_DeepEval_Anchored_Full_20260904.md`.

## Global Constraints

- Freeze retrieval and Assembler configuration at Anchored Parent-3/W5, Top-20, BM25 weight `0.35`, W6/S3, 1500-token context budget.
- Preserve legacy Generator contract v1 and make v2 explicitly selectable; do not silently change unrelated callers.
- Generator citations may only be materialized from the final `GoldAssembledContext.evidence_turns`; no model-supplied `quote_text` is trusted.
- Every factual claim in v2 must carry at least one valid evidence ID; unsupported inference must be explicitly typed and qualified.
- Gold Context is evaluation-only and must not enter production retrieval.
- DeepEval failures, missing Judge configuration, API errors, and invalid Generator output remain explicit `ERROR`/`UNMEASURED`; never fill scores with deterministic guesses.
- No API key, cookie, or secret may be written to code, plans, traces, or reports.
- Production DB/Chroma remain read-only during evaluation; artifact hashes are checked before and after each run.

---

### Task 1: Evidence catalog and Generator contract v2

**Files:**
- Create: `src/generator_contract.py`
- Modify: `src/rag_pipeline.py`
- Test: `tests/test_generator_contract.py`

**Interfaces:**
- `build_evidence_catalog(context: GoldAssembledContext) -> dict[str, EvidenceCatalogItem]`
- `render_evidence_catalog(catalog: dict[str, EvidenceCatalogItem]) -> str`
- `materialize_generator_citations(citation_refs: list[GeneratorCitationRef], catalog: dict[str, EvidenceCatalogItem]) -> list[DialogueCitation]`
- `GeneratorClaim(claim_id, claim_text, claim_type, evidence_ids, qualification)` where `claim_type` is `fact|inference|recommendation`.
- `GeneratorCitationRef(evidence_id)` with strict ID validation.
- `GeneratorGuidancePayloadV2(subject_path, answer, claims, dialogue_citations, transfer_question)`.
- `EndToEndPedagogicalRAGPipeline(generator_contract_version="v1"|"v2")`.

- [ ] **Step 1: Write failing tests** for deterministic evidence IDs, exact catalog provenance, rejection of unknown IDs, server-side quote materialization, factual claims without evidence, and v1 compatibility.
- [ ] **Step 2: Run focused tests** with `pytest tests/test_generator_contract.py -q`; expect missing-module/attribute failures.
- [ ] **Step 3: Implement the minimal catalog and strict Pydantic v2 models**. Generate IDs from final evidence order (`E001`, `E002`, …), keyed by `(session_id, turn_id)`, and reject duplicate or missing provenance.
- [ ] **Step 4: Implement server-side materialization** so the model supplies only evidence IDs; lookup fills exact `session_id`, `turn_id`, `speaker`, and `quote_text` from the catalog. Unknown IDs raise a checked `RAGGenerationError`.
- [ ] **Step 5: Run focused tests** and existing citation-audit tests; keep v1 payload parsing unchanged.

### Task 2: Prompt v2 and Generator integration

**Files:**
- Modify: `src/rag_pipeline.py`
- Modify: `src/cli.py`
- Test: `tests/test_rag_pipeline.py`, `tests/test_generator_contract.py`

**Interfaces:**
- `SYSTEM_PEDAGOGICAL_PROMPT_V2` requires structured diagnosis, evidence explanation, actionable teaching intervention, qualified inference, and evidence IDs only.
- `EndToEndPedagogicalRAGPipeline._generate_with_llm(..., contract_version="v1"|"v2") -> PedagogicalGuidanceResponse`.
- `EndToEndPedagogicalRAGPipeline._generator_context_text(context) -> str` adds the deterministic evidence catalog to the exact Generator input.
- CLI flag `--generator-contract {v1,v2}`; default remains `v1` until the L2 A/B decision.

- [ ] **Step 1: Write failing tests** proving v2 prompt has no 80-character/no-steps contradiction, includes fact/inference/recommendation rules, exposes evidence IDs, and sends the exact same catalog text seen by the Generator.
- [ ] **Step 2: Run focused tests** and verify they fail against the v1 prompt/contract.
- [ ] **Step 3: Implement v2 prompt routing**. Permit a structured answer up to 600 characters, require claim-level evidence IDs, require at least one citation per factual claim, and require both student/tutor roles when the query explicitly asks for both.
- [ ] **Step 4: Parse v2 payload, materialize citations, and pass the materialized response through the existing field-by-field citation audit**. Do not silently strip bracket markers from free text.
- [ ] **Step 5: Run the focused and existing RAG tests**; verify `--generator-contract v1` remains behavior-compatible and v2 is explicit.

### Task 3: Structured Gold Context v2

**Files:**
- Modify: `scripts/run_l2_deepeval.py`
- Test: `tests/test_l2_deepeval_runner.py`

**Interfaces:**
- `_build_gold_context_v2(case, sessions, storage) -> GoldAssembledContext`.
- Gold Context v2 contains: source session question/subject, structured card facts read from DuckDB, complete deterministic W6/S3 logical windows, and explicit labels `[AUTHORITATIVE_TURN]`, `[CARD_FACT]`, `[ALLOWED_INFERENCE]`.
- Gold track remains evaluation-only; expected answer text is never inserted into the Generator context.

- [ ] **Step 1: Write failing tests** proving Gold Context v2 contains card facts and complete real windows, labels fact/inference boundaries, and excludes `ground_truth`/qrels as prompt text.
- [ ] **Step 2: Run focused tests** and observe missing helper/labels.
- [ ] **Step 3: Implement read-only DuckDB card loading** for source sessions and deterministic evidence-index window rendering; use only real stored card fields and real Turns.
- [ ] **Step 4: Add `--gold-context-version {v1,v2}`** and include the selected version, source sessions, artifact hash, and context policy in dataset/run provenance.
- [ ] **Step 5: Run focused tests and a local no-Judge fixture** to verify Gold Context v2 serialization is deterministic.

### Task 4: DeepEval rubric v2 and L2 A/B runner

**Files:**
- Modify: `scripts/run_l2_deepeval.py`
- Modify: `evals/deepeval_adapter.py`
- Test: `tests/test_l2_deepeval_runner.py`, `tests/evals/test_reporting_and_deepeval.py`

**Interfaces:**
- `_metrics(judge_model, rubric_version="v1"|"v2") -> list[tuple[str, Any, float]]`.
- v2 Faithfulness rubric: direct facts must be supported by final context; inferences are allowed only when explicitly qualified and grounded by cited evidence; unsupported claims fail.
- v2 Pedagogical GEval rubric: must separate student misconception, tutor strategy, evidence explanation, and actionable intervention; concise answers are acceptable when all required elements are present.
- Runner flags `--generator-contract`, `--gold-context-version`, and `--rubric-version`.

- [ ] **Step 1: Write failing tests** for rubric selection/provenance, exact final Generator context passed to DeepEval, and report fields that distinguish v1/v2.
- [ ] **Step 2: Run focused tests** and verify the runner still defaults to v1 for backward compatibility.
- [ ] **Step 3: Implement v2 rubric text and provenance fields**. Keep thresholds unchanged initially (`0.90/0.85/0.90/0.85/0.80/0.95`) so score movement is attributable to contract/rubric/context changes.
- [ ] **Step 4: Run a 1-case v2 Gold/Real smoke** with timeout and artifact hash checks; require explicit `MEASURED`, `ERROR`, or `UNMEASURED` statuses.
- [ ] **Step 5: Run the fixed 30-case v1/v2 A/B using four resumable shards**, then merge with `scripts/merge_l2_deepeval_reports.py`; no production promotion.

### Task 5: Analysis, documentation, and verification

**Files:**
- Create: `docs/L2_Generator_First_AB_20260904.md`
- Modify: `docs/L2_DeepEval_Anchored_Full_20260904.md`
- Modify: `HANDOFF.md`
- Test: full suite

**Interfaces:**
- Report must contain per-track/per-metric mean, pass count, case-level failures, contract version, Gold Context version, rubric version, Generator/Judge model names, source artifact hash, and reproduction commands.
- Decision table must distinguish: Gold v2 failure (Generator/contract/rubric issue), Gold v2 pass + Real failure (upstream context gap), and Real citation audit failure (claim/citation contract issue).

- [ ] **Step 1: Add report tests** for complete provenance and no secret leakage.
- [ ] **Step 2: Write the v1/v2 comparison and root-cause analysis**. Do not call a score improvement causal unless the fixed-context A/B supports it.
- [ ] **Step 3: Run `pytest -q`, `python -m compileall -q src scripts evals tests`, and `git diff --check`**.
- [ ] **Step 4: Review Git diff and preserve unrelated worktree changes; commit Generator v2 separately from the frozen baseline.**
