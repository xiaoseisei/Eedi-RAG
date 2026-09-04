# Card Evidence Binding and Incremental Ingestion Design

## Goal

Improve card evidence completeness without asking the LLM to select authoritative `source_turn_ids`, and make historical ingestion incremental so changes do not reprocess all 100 sessions.

## Scope and boundaries

- Production supports two retrieval strategies: `card` and `fallback`.
- This design changes how cards are bound to evidence; it does not remove the existing `source_turn_ids` compatibility field.
- Raw dialogue remains authoritative in `session_dialogue_turns`; cards remain semantic summaries.
- The deterministic evidence index is built from cleaned dialogue only. It must never invent text or Turn IDs.
- No production DB/Chroma overwrite occurs during extraction or evaluation. Promotion remains an explicit blue/green operation.
- Existing successful cards are reusable only when the session fingerprint, extraction contract, schema version, and index contract match.
- Failed or partially validated outputs are never cacheable.

## Current problem

The LLM currently emits both card semantics and evidence pointers. Strict validation proves that emitted IDs are real and role-correct, but not that the set is complete. The 30-case root-cause evaluation found role-correct card pointer recall of `0.6833` before the latest re-extraction and `0.8722` after the completeness prompt; remaining misses are concentrated in misconception cards. The first extraction prompt encourages representative summaries, while evidence completeness is not independently represented in the data model.

The current re-extraction script calls the LLM for every selected session and only writes staging storage after the corpus traversal. Consequently, a prompt, evidence rule, or unrelated chunking change can cause an unnecessary full run.

## Proposed architecture

### A. Deterministic raw evidence index

Create a versioned logical evidence index from every cleaned session before card generation. The index is a deterministic derivative of `session_dialogue_turns`:

```text
CleanedSession
  -> role/noise-aware Turn records
  -> overlapping logical-chain windows (window=6, step=3)
  -> complete session union and index entries
```

Each entry contains:

```json
{
  "index_id": "session_14_chain_0002",
  "session_id": 14,
  "sequence": 2,
  "chain_id": "session_14_logic_v1",
  "turn_ids": [4, 5, 6, 7, 8, 9],
  "student_turn_ids": [4, 6, 8],
  "tutor_turn_ids": [5, 7, 9],
  "content_hash": "...",
  "source": "session_dialogue_turns"
}
```

Windows overlap, and the index manifest records union coverage. The index intentionally favors complete causal chains over minimal snippets. Greeting/noise turns may be marked but are never silently substituted for missing evidence.

### B. Card semantics and evidence binding are separate

The LLM still produces semantic fields (`misconception_name`, `deep_mechanism`, `key_aha_question`, etc.) and may provide optional quotes for readability. The LLM-provided `source_turn_ids` are not authoritative in the new path.

After semantic validation, a deterministic binder maps each card to the existing logical evidence index by:

1. `session_id` equality;
2. card role (`misconception -> student`, `tutor_strategy -> tutor`);
3. index entry membership in the session logical chain;
4. exclusion of turns marked greeting/noise where a non-noise alternative exists.

The binder writes:

- `source_index_ids`: all logical-chain entries supporting the card role;
- `source_turn_ids`: sorted union of role-correct Turn IDs from those entries;
- `evidence_binding_policy`: `session_role_chain_v1`;
- `evidence_index_version` and `evidence_index_hash`.

This gives deterministic complete coverage at the session-role level. It does not claim that every retained role Turn is semantically minimal; precision/noise remains a measured property and can be reduced later with human labels.

The old extraction path remains available for backward compatibility. The new re-extraction path explicitly requests semantic-only extraction plus deterministic binding.

### C. Versioned per-session cache and invalidation

Store one cache record per session under an isolated cache root. The semantic cache key is deliberately independent of evidence binding/index versions:

```text
session_content_hash
extraction_prompt_version
llm_provider
llm_model
extraction_schema_version
temperature
semantic_extraction_mode
```

Only records with `status=SUCCESS` and matching all semantic fields are reusable. Binding policy and index version are recorded separately in the run manifest and are applied deterministically on every cache hit. Cache records include the serialized `ExtractedPIU`, timestamps, latency, and error history. Cache writes are atomic per session.

Invalidation rules:

| Change | Reuse semantic card | Rebuild evidence binding | Rebuild windows | Rebuild embeddings |
| --- | --- | --- | --- | --- |
| BM25/RRF/reranker/Generator | yes | no | no | no |
| Window size/step | yes | yes | yes | yes for window collection |
| Evidence binding policy | yes | yes | no | yes for changed card metadata |
| Extraction prompt/schema/model | no | after new card | no | yes for changed cards |
| Session dialogue/question | no for that session | yes for that session | yes for that session | yes for that session |

### D. Incremental staging command

Add an incremental command with explicit modes:

```text
--mode changed-only       # default: reuse valid cache, process only misses
--mode audit-only         # bind/re-audit selected cached cards, no semantic LLM call
--mode full               # explicit rebuild of all selected sessions
--session-ids 14,206      # optional targeted repair
--cache-dir <isolated-dir>
```

The command emits a manifest containing `reused_count`, `llm_processed_count`, `audit_count`, `failed_count`, per-session statuses, cache key fields, and source/index hashes. A failed session is recorded and does not discard successful cache records; production promotion is blocked until the caller explicitly resolves failures.

### E. Provider concurrency and resumability

Keep the current strict provider behavior and success-only cache. Use bounded worker concurrency, with an explicit `--workers` value and provider timeout. Resume reads cache records first and submits only missing sessions. This reduces repeated runs without hiding provider failures. A future provider batch endpoint is optional and out of scope for this change.

## Error handling

- Invalid cache contract: cache miss, not a successful default.
- Invalid or stale evidence index: fail the session and record `FAILED`.
- LLM unavailable or malformed semantic output: record `FAILED`, never create a card.
- Binder cannot find a session/role Turn: record `FAILED` unless the card itself is explicitly `partial` under the existing contract.
- Any staging artifact hash change during evaluation: abort promotion.

## Verification criteria

1. A second run with unchanged inputs performs zero LLM calls and reuses all successful sessions.
2. A prompt-version change reprocesses only sessions without a matching semantic cache record.
3. `audit-only` changes `source_turn_ids` without invoking the semantic extraction LLM.
4. Binder output is role-correct, sorted, duplicate-free, in-bounds, and traceable to index entries.
5. The union of logical index entries covers every source Turn in every indexed session.
6. A targeted failure leaves successful cache records intact and reports the failure explicitly.
7. Existing 179-test suite remains green; new tests cover cache key invalidation, atomic cache writes, deterministic role binding, and changed-only counts.

## Non-goals

- Proving semantic minimality of every evidence Turn without human graded labels.
- Changing the production default from `card` to `fallback`.
- Replacing the Qwen embedding or reranker model.
- Automatically promoting staging artifacts into production.
