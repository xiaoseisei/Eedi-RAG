# Full Knowledge Base Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import all 1,576 cleaned Eedi sessions into DuckDB and Chroma with concurrent semantic extraction, resumable checkpoints, incremental updates, and verification after every 100-session batch.

**Architecture:** Keep semantic extraction in the existing cache-keyed staging runner and add a production import orchestrator around it. Workers only perform LLM extraction; the coordinator alone writes DuckDB/Chroma for an all-or-nothing batch. A durable manifest records source hashes, per-session status, batch commit status, and post-commit audit counts so interrupted runs resume without reprocessing successful sessions.

**Tech Stack:** Python 3.14, Pydantic models, DuckDB, ChromaDB, OpenAI-compatible extraction client, `ThreadPoolExecutor`, atomic JSON/JSONL artifacts.

**Spec:** `docs/需求.md`

## Global Constraints

- Use `data/cleaned_sessions.jsonl` as the authoritative full-session input.
- Never fabricate cards or silently downgrade an unavailable LLM; failures remain explicit and retryable.
- Production writes must contain only validated `ExtractedPIU` records with deterministic evidence binding.
- Batch verification must run after each 100-session commit and must fail closed on missing sessions, cards, windows, or evidence pointers.
- Re-running with the same source and contract must be idempotent and must reuse successful extraction cache entries.

---

### Task 1: Add authoritative JSONL loading and batch/checkpoint contracts

**Files:**
- Create: `src/knowledge_import.py`
- Test: `tests/test_knowledge_import.py`

**Interfaces:**
- `load_cleaned_sessions(path: Path) -> list[CleanedSession]`
- `partition_sessions(sessions: Sequence[CleanedSession], batch_size: int) -> list[list[CleanedSession]]`
- `KnowledgeImportManifest` and `KnowledgeImportBatch` dataclasses with atomic JSON serialization.

- [x] **Step 1: Write tests for strict source loading, deterministic ordering, duplicate rejection, and 100-session partitioning.**
- [x] **Step 2: Run focused tests and confirm the new tests pass after implementation.**
- [x] **Step 3: Implement JSONL validation, stable `intervention_id` ordering, duplicate checks, and atomic manifest helpers.**
- [x] **Step 4: Run the focused tests and confirm they pass.**

### Task 2: Add production batch promotion and post-commit checks

**Files:**
- Modify: `src/knowledge_import.py`
- Test: `tests/test_knowledge_import.py`

**Interfaces:**
- `promote_batch(storage, sessions, extracted, chunker) -> BatchAudit`
- `audit_batch(storage, session_ids, expected_cards=True, expected_windows=True) -> BatchAudit`

- [x] **Step 1: Write tests using a fake storage adapter that verifies each batch writes sessions, two cards per session, and windows exactly once.**
- [x] **Step 2: Run the focused tests and confirm they pass after implementation.**
- [x] **Step 3: Implement promotion through `DualEngineStorageManager.ingest_all`, generating hybrid chunks with `SessionChunker`, then query authoritative counts and IDs.**
- [x] **Step 4: Make audit failures raise a typed error and never mark the batch committed.**
- [x] **Step 5: Run the focused tests and confirm they pass.**

### Task 3: Build resumable full import CLI with concurrent extraction

**Files:**
- Create: `scripts/import_full_knowledge_base.py`
- Modify: `src/extraction_cache.py` only if the integration test proves a cache contract gap
- Test: `tests/test_import_full_knowledge_base.py`

**Interfaces:**
- `run_full_import(source_path, db_path, chroma_dir, state_dir, batch_size=100, workers=8, resume=True, ...) -> dict`
- CLI flags: `--source`, `--db`, `--chroma-dir`, `--state-dir`, `--batch-size`, `--workers`, `--model`, `--resume`, `--start-batch`, `--max-batches`.

- [x] **Step 1: Write tests for concurrent extraction, cache reuse after interruption, failed-session retry, batch stop-on-audit-failure, and idempotent rerun.**
- [x] **Step 2: Run the focused tests and confirm they pass after implementation.**
- [x] **Step 3: Implement the coordinator: load source, partition batches, call `run_incremental_ingestion` for each batch, read validated cards, promote only complete batches, and atomically update manifest after every session and batch.**
- [x] **Step 4: Ensure workers never write DuckDB, Chroma, or the shared manifest; only the coordinator performs those writes.**
- [x] **Step 5: On restart, skip `committed` batches and rely on session/contract hashes to reuse successful extraction results.**
- [x] **Step 6: Run focused tests and confirm they pass.**

### Task 4: Add 100-session verification and operational documentation

**Files:**
- Modify: `scripts/import_full_knowledge_base.py`
- Modify: `README.md` or `docs/knowledge_import.md`
- Test: `tests/test_import_full_knowledge_base.py`

- [x] **Step 1: Add a batch report containing expected/actual session, card, window, and evidence counts plus source and contract hashes.**
- [x] **Step 2: Add a documented command for a 100-session smoke run and a resume command.**
- [x] **Step 3: Run the 100-session smoke import against an isolated DB/Chroma directory; do not overwrite production artifacts during the test.**
- [x] **Step 4: Verify the smoke manifest shows one committed batch, zero missing IDs, and explicit failure records if any LLM call fails.**

### Task 5: Full import execution and final audit

**Files:**
- Runtime artifacts: configured `state_dir`, production DuckDB, and production Chroma directory.

- [x] **Step 1: Run the importer over all source sessions with a bounded worker count and the approved LLM configuration.**
- [x] **Step 2: After every 100 sessions, inspect the generated batch audit before continuing.**
- [x] **Step 3: Resume failed batches using their checkpoints without reprocessing successful sessions.**
- [x] **Step 4: Run the final storage audit and compare all imported session IDs with the authoritative JSONL source.**
- [x] **Step 5: Run the complete relevant test suite and record command output, final counts, failed sessions, and artifact hashes.**
