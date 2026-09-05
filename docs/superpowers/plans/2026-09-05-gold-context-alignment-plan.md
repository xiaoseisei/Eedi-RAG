# Gold Context Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Gold track a clean, auditable semantic alignment baseline before comparing Real retrieval or Generator variants.

**Architecture:** Keep production Anchored Parent-3/W5 retrieval unchanged. Refactor evaluation-only Gold Context v2 into deduplicated, separately addressable card/window/inference nodes selected around required evidence; keep compact text for the Generator while passing node-level context to DeepEval. Evaluate the narrow Generator core answer for relevancy and the structured response for grounding/pedagogy, and persist claim-level provenance plus data-conflict warnings without injecting expected answers or qrels.

**Tech Stack:** Python 3.14+, Pydantic v2, DeepEval 4.2, DuckDB/Chroma read-only artifacts, pytest.

**Spec:** User-approved Gold alignment design in the active conversation and `docs/L2_Generator_First_AB_20260904.md`.

## Global Constraints

- Keep Anchored Parent-3/W5, Top-20, BM25 weight `0.35`, W6/S3, and production 1,500-token budget unchanged.
- Gold Context remains evaluation-only and must not contain `ground_truth`, qrels, or expected-answer text.
- Gold evidence must come from real stored card fields and real `DialogueTurn` text; no synthetic quotes or fake defaults.
- DeepEval thresholds remain `Answer Relevancy 0.85`, `Contextual Recall 0.90`, `Faithfulness 0.90`, `Pedagogical GEval 0.80`, and `Citation Audit 0.95`.
- API, Judge, embedding, and retrieval failures remain explicit `ERROR` or `UNMEASURED` records.
- Production DB/Chroma are read-only during all evaluation runs and artifact hashes must remain unchanged.

---

### Task 1: Add auditable Gold Context node contracts

**Files:**
- Modify: `src/reranker.py:42-64`
- Modify: `scripts/run_l2_deepeval.py:284-400`
- Test: `tests/test_l2_deepeval_runner.py`

**Interfaces:**
- `GoldAssembledContext.deepeval_context_nodes: list[str]` stores separately addressable, non-empty DeepEval nodes.
- `_build_gold_context_v2(case, sessions, storage) -> GoldAssembledContext` selects complete W6/S3 windows intersecting required Turn IDs, merges overlapping ranges, and records card facts, raw evidence windows, and inference boundary as separate nodes.
- Every node is deterministic and contains only real artifact content or explicit boundary labels.

- [ ] **Step 1: Write failing tests** for required evidence retention, merged overlapping windows without duplicate Turn lines, irrelevant session-prefix exclusion, separate card/inference nodes, and absence of `ground_truth`/qrels.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest -q tests/test_l2_deepeval_runner.py -k gold_context`; expect failure because the node field and compact union do not exist.
- [ ] **Step 3: Implement** the node field and deterministic window union. Use required `(session_id, turn_id)` pairs, retain W6/S3 windows intersecting required Turns, merge overlapping intervals, and render each Turn once in chronological order. Store card nodes, merged authoritative window nodes, and the inference boundary separately.
- [ ] **Step 4: Re-run** the focused tests and inspect case 573; Turns 10–18 must remain while unrelated Turns 1–6 are excluded.
- [ ] **Step 5: Commit** with `git add src/reranker.py scripts/run_l2_deepeval.py tests/test_l2_deepeval_runner.py` and `git commit -m "feat(eedi-rag): split gold context into deduplicated nodes"`.

### Task 2: Make DeepEval metric inputs intent-appropriate

**Files:**
- Modify: `src/rag_pipeline.py:625-652`
- Modify: `scripts/run_l2_deepeval.py:563-580,900-914`
- Test: `tests/test_rag_pipeline.py`, `tests/test_l2_deepeval_runner.py`

**Interfaces:**
- v2 responses expose `generator_core_answer` and `generator_claims` in runtime trace metadata.
- `_run_deepeval_case(..., actual_output_by_metric: dict[str, str] | None = None, retrieval_context: list[str] | None = None)` builds one test case per metric using the selected output/context without changing metric names or thresholds.

- [ ] **Step 1: Write failing tests** proving Answer Relevancy receives only core `answer`, other semantic metrics receive structured output, and Gold uses multiple context nodes.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest -q tests/test_rag_pipeline.py tests/test_l2_deepeval_runner.py -k "generator_v2 or deepeval_case or gold_context"`; expect failure because one output and one context string are used for every metric.
- [ ] **Step 3: Persist** `response.__dict__["generator_core_answer"] = payload.answer` while preserving structured `answer_content` for production rendering and validated `generator_claims`.
- [ ] **Step 4: Implement** metric-specific payloads: core answer for `answer_relevancy`; structured output for faithfulness, contextual recall/precision, and pedagogy; `context.deepeval_context_nodes` for DeepEval with a legacy single-node fallback. Keep citation audit deterministic and separate.
- [ ] **Step 5: Re-run** the focused tests and commit with `git commit -m "fix(eedi-rag): align DeepEval inputs with metric intent"`.

### Task 3: Add claim coverage and Gold data-conflict diagnostics

**Files:**
- Modify: `src/generator_contract.py:80-101`
- Modify: `scripts/run_l2_deepeval.py:800-915,950-1010`
- Test: `tests/test_generator_contract.py`, `tests/test_l2_deepeval_runner.py`

**Interfaces:**
- `summarize_claim_coverage(payload_or_response) -> dict[str, int | float]` reports total claims, factual claims, factual claims with evidence, and cited evidence IDs.
- `audit_gold_alignment(case, card_rows) -> list[dict[str, str]]` reports deterministic card-vs-Golden conflicts without changing scores or context.

- [ ] **Step 1: Write failing tests** for claim coverage, missing fact evidence, and an explicit card option conflict warning.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest -q tests/test_generator_contract.py tests/test_l2_deepeval_runner.py -k "claim or alignment"`; expect failure because the summaries do not exist.
- [ ] **Step 3: Implement** summaries using only validated v2 claims. Facts are covered when their evidence IDs intersect citations; inference qualification is preserved; recommendations may have no IDs. Compare only explicit option tokens and stored card values for alignment warnings.
- [ ] **Step 4: Add** `claim_coverage`, `gold_alignment_warnings`, contract/context/rubric versions to traces and report provenance; warnings must not alter metric values or gates.
- [ ] **Step 5: Run focused tests and commit** with `git commit -m "feat(eedi-rag): audit claim coverage and gold conflicts"`.

### Task 4: Gold smoke and threshold gate

**Files:**
- Modify: `docs/L2_Generator_First_AB_20260904.md`
- Modify: `docs/L2_DeepEval_Anchored_Full_20260904.md`
- Modify: `HANDOFF.md`
- Test: full suite and one-case live Gold smoke

**Interfaces:**
- Gold smoke uses the existing artifact hash, `generator-contract=v2`, `gold-context-version=v2`, `rubric-version=v2`, and the configured Judge.
- No production promotion occurs until Gold meets unchanged thresholds or an explicit data/evaluator blocker is documented.

- [ ] **Step 1: Run** `.venv\Scripts\python.exe -m pytest -q`, `.venv\Scripts\python.exe -m compileall -q src scripts evals tests`, and `git diff --check`.
- [ ] **Step 2: Run** one Gold v2 smoke with node-level context and metric-specific outputs; require explicit `SUCCESS`, `ERROR`, or `UNMEASURED` statuses.
- [ ] **Step 3: Read** every Gold metric, context-node count, rough context size, citation components, claim coverage, and alignment warnings. Never use expected answer text as Generator input.
- [ ] **Step 4: Update** the two L2 reports and HANDOFF with threshold status and root-cause classification. Do not proceed to Real A/B as if Gold were aligned when a Gold threshold fails.
- [ ] **Step 5: Commit** documentation with `git add docs/L2_Generator_First_AB_20260904.md docs/L2_DeepEval_Anchored_Full_20260904.md HANDOFF.md` and `git commit -m "docs(eedi-rag): record gold alignment smoke"`.

