# Card Evidence Binding and Incremental Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate semantic card extraction from deterministic logical-chain evidence binding and add resumable, contract-keyed incremental ingestion.

**Architecture:** Build a deterministic raw evidence index from cleaned dialogue; bind every card to role-correct logical-chain entries without trusting LLM-selected pointers; cache successful per-session semantic artifacts using a complete contract key; make staging reuse cache records and process only changed or explicitly selected sessions.

**Tech Stack:** Python 3.11+, Pydantic v2, DuckDB, ChromaDB, pytest, existing OpenAI-compatible LLM client.

**Spec:** `docs/superpowers/specs/2026-09-04-card-evidence-incremental-design.md`

## Global Constraints

- Never write production DB/Chroma during extraction or evaluation.
- Cache only successful outputs whose full contract key matches; failures remain explicit.
- Raw `session_dialogue_turns` is authoritative; no generated text or Turn IDs.
- Preserve backward compatibility for existing `source_turn_ids` consumers.
- Keep the production default `chunk_strategy=card` unchanged.
- Every new manifest records source/index hashes, contract fields, reuse counts, failures, and reproduction parameters.

---

### Task 1: Deterministic logical evidence index

**Files:**
- Create: `src/evidence_index.py`
- Modify: `src/models.py` to add typed evidence-index provenance fields
- Test: `tests/test_evidence_index.py`

**Interfaces:**
- `build_logical_evidence_index(session: CleanedSession, *, window_size: int = 6, step: int = 3, version: str = "logical-chain-v1") -> LogicalEvidenceIndex`
- `bind_card_to_evidence_index(card: ExtractedPIU, index: LogicalEvidenceIndex, *, policy: str = "session_role_chain_v1") -> ExtractedPIU`
- `LogicalEvidenceIndex.entries`, `.session_id`, `.version`, `.artifact_hash`, `.union_turn_ids`

- [ ] **Step 1: Write failing tests** for overlapping windows, full union coverage, role separation, deterministic IDs, noise marking, sorted/unique bound pointers, and no fabricated Turn IDs.
- [ ] **Step 2: Run tests** with `pytest tests/test_evidence_index.py -q`; expect import or implementation failures.
- [ ] **Step 3: Implement** immutable Pydantic/dataclass records. Reuse real `CleanedSession.turns`; generate `index_id` from session/version/sequence; compute content hashes from actual text and Turn IDs.
- [ ] **Step 4: Implement binding** so misconception cards use non-noise student IDs, strategy cards use non-noise tutor IDs, and `source_index_ids`/policy/version/hash are stored in card metadata without using LLM pointer IDs.
- [ ] **Step 5: Run focused tests** and verify all output IDs are in the session Turn set.

### Task 2: Semantic-only extraction and cache contract

**Files:**
- Create: `src/extraction_cache.py`
- Modify: `src/extract_knowledge.py`
- Test: `tests/test_extraction_cache.py`, `tests/test_extract_knowledge.py`

**Interfaces:**
- `build_session_content_hash(session: CleanedSession) -> str`
- `ExtractionCacheKey.from_session(session, *, model, prompt_version, schema_version, temperature, binding_policy, index_version) -> ExtractionCacheKey`
- `ExtractionCache.get(key) -> CacheHit | None`
- `ExtractionCache.put_success(key, extracted: ExtractedPIU, evidence_metadata: dict) -> None`
- `ExtractionCache.put_failure(key, error_type: str, error_message: str) -> None`
- `extract_semantic_cards_from_session(...) -> ExtractedPIU`

- [ ] **Step 1: Add tests** proving equivalent sessions produce the same hash, any dialogue/question change changes the hash, contract changes miss the cache, and failed records are never returned as hits.
- [ ] **Step 2: Add tests** proving semantic extraction validates quotes against authoritative role turns but does not trust returned pointer IDs for final binding.
- [ ] **Step 3: Implement** JSON cache records with atomic temp-file replacement and strict key comparison; serialize no API key or prompt secrets.
- [ ] **Step 4: Implement** an explicit semantic extraction mode that accepts the existing strict payload, validates model facts, and leaves evidence binding to Task 1. Keep the legacy function behavior unchanged unless the new mode is explicitly selected.
- [ ] **Step 5: Run focused tests** and the existing extraction tests.

### Task 3: Incremental staging/re-extraction pipeline

**Files:**
- Modify: `scripts/reextract_cards_staging.py`
- Create: `scripts/incremental_card_ingest.py`
- Test: `tests/test_incremental_card_ingest.py`

**Interfaces:**
- `run_incremental_ingestion(..., mode: Literal["changed-only", "audit-only", "full"], cache_dir: Path, session_ids: list[int] | None, workers: int, ...) -> dict`
- `IncrementalRunManifest` fields: `status`, `reused_count`, `llm_processed_count`, `audit_count`, `failed_count`, `session_statuses`, `source_hashes`, `contract`

- [ ] **Step 1: Write tests** for unchanged rerun (`llm_processed_count=0`), prompt invalidation, targeted `--session-ids`, audit-only zero LLM calls, and failure resume behavior.
- [ ] **Step 2: Implement** session selection from production DB read-only; construct the logical evidence index for every selected session before cache lookup.
- [ ] **Step 3: Implement** changed-only cache lookup and bounded worker submission for misses; bind reused and newly extracted cards through Task 1.
- [ ] **Step 4: Persist** per-session cache and staging JSONL atomically; emit a manifest with exact counts and errors. Do not create/promote final DB/Chroma when failures remain.
- [ ] **Step 5: Add CLI flags** (`--mode`, `--cache-dir`, `--session-ids`, `--workers`) and a reproduction command to the manifest.
- [ ] **Step 6: Run focused tests** and a local fixture run with a fake LLM that records call count.

### Task 4: Storage/Chroma metadata and compatibility integration

**Files:**
- Modify: `src/storage_manager.py`
- Test: `tests/test_storage_manager.py`, `tests/test_evidence_index.py`

**Interfaces:**
- Extend card metadata with `source_index_ids`, `evidence_binding_policy`, `evidence_index_version`, and `evidence_index_hash`.
- Preserve existing DuckDB columns and `source_turn_ids` semantics as the deterministic bound union.

- [ ] **Step 1: Add tests** that ingested Chroma metadata contains deterministic binding provenance and that old cards without the optional fields still load.
- [ ] **Step 2: Implement** metadata propagation through card chunk creation and `ingest_extracted_pius`; do not alter authoritative dialogue rows.
- [ ] **Step 3: Run storage tests** and verify DuckDB/Chroma ID parity remains unchanged.

### Task 5: Evaluation, documentation, and full verification

**Files:**
- Modify: `scripts/run_l1_qwen_full_eval.py` if the new provenance must be reported
- Modify: `HANDOFF.md`
- Create: `docs/Incremental_Card_Ingestion_Runbook.md`
- Test: `tests/test_incremental_card_ingest.py`, full suite

**Interfaces:**
- Reports must expose cache reuse counts, deterministic index version/hash, binding policy, and per-stage measured/failed/unmeasured status.

- [ ] **Step 1: Add tests** for evaluation manifest provenance and unchanged artifact hashes.
- [ ] **Step 2: Document** operational commands for first build, changed-only rerun, audit-only repair, explicit full rebuild, failure resume, and blue/green promotion.
- [ ] **Step 3: Run** `pytest -q`, `python -m compileall -q src scripts evals tests`, and `git diff --check`.
- [ ] **Step 4: Run** a no-op incremental smoke test and record `llm_processed_count=0` evidence; do not call production promotion.
- [ ] **Step 5: Review** changed files and ensure unrelated user modifications remain untouched.
