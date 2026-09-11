# Session-Card Knowledge Graph for Business Insight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Session + student misconception Card + tutor strategy Card RAG into a rebuildable knowledge graph that answers all business questions in `docs/需求.md` through deterministic macro analysis, graph relationships, micro raw-dialogue evidence, and an LLM final response.

**Architecture:** Keep the current Card documents, BM25/Dense/RRF retrieval, Parent-5/W7 assembler, and exact DuckDB Turn audit unchanged as the micro baseline. Build a versioned graph projection in a separate analytics artifact: one Session node owns one Question, one Subject path, one student Card event, and one tutor Card event; controlled labels connect equivalent events across Sessions. Query execution becomes `intent/router -> plan -> graph/SQL scope -> existing Card RAG evidence -> original query + structured facts + authorized raw dialogue -> generator`.

**Tech Stack:** Python 3.14, DuckDB, existing ChromaDB/BM25/Dense retriever, Pydantic 2, pytest 9, JSONL evaluation fixtures, SHA-256 provenance. No standalone Neo4j/GraphRAG service in this phase.

**Spec:** `docs/需求.md`; `docs/Query_Intent_Response_Executable_Design.md`; `docs/superpowers/plans/2026-09-07-business-rag-contract-routing.md`; `docs/superpowers/plans/2026-09-07-business-rag-analytics-assets.md`

## Global Constraints

- Preserve the current Card embedding documents, Chroma collections, BM25 index, RRF ranking, Parent-5/W7 settings, and legacy `EndToEndPedagogicalRAGPipeline.ask()` as regression baselines.
- Graph data is a derived, versioned projection; authoritative facts remain the source DuckDB Session/Turn tables and existing Card tables.
- Building or rebuilding the graph must never call an LLM on the full raw transcript and must never mutate the production Card/Chroma artifacts.
- Every graph node and edge carries `source_session_id`, source Card ID where applicable, source Turn IDs where applicable, derivation version, and evidence verification status.
- Use controlled labels for cross-Session grouping. Preserve original Card text separately; never replace it with a generated label.
- Deterministic labels may be auto-assigned; automatic assignments must be marked `AUTO_ASSIGNED` and must not be described as human-reviewed truth.
- Graph edges describe observed relationships (`HAS_CARD`, `ABOUT_SUBJECT`, `USES_STRATEGY`, `EVIDENCED_BY`, `CO_OCCURS_WITH`, `FOLLOWED_BY`). Do not create `CAUSES`, `IMPROVES_SCORE`, or `EFFECTIVE_FOR` edges without approved longitudinal/experimental data.
- DuckDB computes counts, rankings, percentages, and data scope. Graph traversal finds connected Sessions and relation paths. Existing Card RAG finds representative micro evidence. LLM only generates bounded narrative fields.
- Macro outputs must use `COUNT(DISTINCT session_id)` as the primary prevalence measure and must expose event count separately.
- Current Eedi data has no verified student longitudinal identity, retake count, or ability field. Retaker/weak-foundation questions remain `UNSUPPORTED` until an authorized cohort sidecar is provided.
- User-visible text is rendered only after response contract validation and exact evidence audit. Internal streaming may record telemetry but cannot expose unaudited claims.
- All new code and tests use the project’s Chinese module banner, section dividers, field comments, invariant comments, and business-facing docstrings demonstrated by `src/business_models.py`.

## Business Coverage Matrix

本计划逐项覆盖 `需求.md` 的业务问题原文：

```text
维度 A：关于学生端（学情与错题洞察）
  1. 高频困惑：学生在辅导中最常提出的问题是什么？
  2. 核心难点：哪些概念、主题或题型是学生觉得最难理解、卡壳时间最长的？
  3. 共性误区：有哪些认知错误（Misconceptions）在不同学生身上反复出现？
  4. 错题模式：学生在做题/考试时最容易犯的错误类型是什么？
  5. 困难学生画像：多次重考或基础薄弱的学生是否存在共性思维卡点？

维度 B：关于导师端（名师教学方法论与策略沉淀）
  6. 核心解题套路：经验丰富的导师反复传授哪些通用做题和思考策略？
  7. 启发式教学技巧：导师用了哪些比喻、分步引导（Scaffolding）帮助学生理解？
  8. 排除法与决策技巧：导师如何帮助学生在相似选项中做最终抉择？
  9. 教研内容改进：哪些信息可以直接优化教材、微课、题库解析？
 10. 未来 AI 应用机会：这些真实对话还能孵化哪些有商业价值的 AI 产品？
```

第 1-4 项由 Task 4 的学生事件/误区/错误聚合覆盖；第 5 项由 Task 7 的 cohort 能力门禁覆盖；第 6-8 项由 Task 4 的导师策略与关系查询覆盖；第 9-10 项由 Task 4 的内容机会查询和 Task 6 的受限生成覆盖。`系统功能范围与交付边界` 的证据审计、隐私、失败状态和不伪造原则贯穿所有任务，而不是单独交给 LLM。

| Requirement from `需求.md` | Graph/SQL source | Evidence perspective | Current Eedi status |
|---|---|---|---|
| 学生最常提出的问题 | `StudentQuestionEvent -> QuestionCluster` + SQL counts | STUDENT | Supported after event extraction |
| 最难概念/卡壳最长 | `DifficultyEvent -> Subject` + SQL signals | STUDENT, optionally TUTOR | Proxy signals only; no subjective difficulty claim |
| 反复认知误区 | `MisconceptionEvent -> MisconceptionConcept` + SQL counts | STUDENT | Supported after controlled taxonomy |
| 错题模式 | `ExamErrorEvent -> ErrorPattern` + SQL counts | STUDENT | Supported with `UNCLASSIFIED` retained |
| 重考/基础薄弱画像 | `StudentSessionFeature -> Cohort` + graph links | STUDENT | `UNSUPPORTED` on current Eedi |
| 导师通用策略 | `TutorStrategyEvent -> StrategyConcept` + SQL counts | TUTOR | Supported after strategy normalization |
| 启发式技巧/脚手架 | StrategyEvent fields + Session/Turn links | TUTOR | Supported; analogy coverage disclosed |
| 相似选项排除/决策 | Question/Option + student/tutor events | STUDENT, TUTOR | Supported where source fields exist |
| 教材/微课/题库改进 | joins across question/ misconception/strategy/difficulty | STUDENT, TUTOR | Recommendation, not approved fact |
| 未来 AI 应用机会 | aggregated gaps + content opportunities | BOTH | Exploratory recommendation only |

## Target Graph Shape

```text
(Session)
  ├─[:ANCHORS]─>(Question)
  ├─[:ABOUT_SUBJECT]─>(Subject)
  ├─[:HAS_MISCONCEPTION]─>(MisconceptionEvent)─[:TYPE_OF]─>(MisconceptionConcept)
  ├─[:HAS_STRATEGY]─>(TutorStrategyEvent)─[:TYPE_OF]─>(StrategyConcept)
  ├─[:HAS_DIFFICULTY_SIGNAL]─>(DifficultyEvent)
  ├─[:HAS_STUDENT_QUESTION]─>(StudentQuestionEvent)─[:TYPE_OF]─>(QuestionCluster)
  └─[:HAS_TURN]─>(Turn)

(MisconceptionEvent) ─[:EVIDENCED_BY]─>(Student Turn)
(TutorStrategyEvent) ─[:EVIDENCED_BY]─>(Tutor Turn)
(TutorStrategyEvent) ─[:USES_TALK_MOVE]─>(TalkMove)
(MisconceptionEvent) ─[:CO_OCCURS_WITH]─>(TutorStrategyEvent)
```

Session is the traceability center. Card events are not flattened into anonymous edges: they retain their original Card identity, fields, and exact evidence pointers. Shared concept nodes are the only cross-Session grouping layer.

---

### Task 1: Add Graph Projection Contracts and Sidecar Manifest

**Files:**
- Create: `src/graph_models.py`
- Create: `src/graph_store.py`
- Create: `tests/test_graph_models.py`
- Create: `tests/test_graph_store.py`
- Reference: `src/business_models.py`
- Reference: `src/storage_manager.py:246-325`

**Interfaces:**
- Consumes: `CleanedSession`, Card rows, authoritative Turn rows, and Part 1 business contracts.
- Produces: `GraphNode`, `GraphEdge`, `GraphProjectionManifest`, `GraphStore.create_staging()`, `GraphStore.open_ready()`, and read-only node/edge queries.

- [ ] **Step 1: Write failing tests for node/edge provenance and manifest hash binding**

```python
def test_graph_edge_requires_source_session_and_derivation_version():
    with pytest.raises(ValidationError):
        GraphEdge(
            edge_id="edge-1",
            source_node_id="session:10",
            target_node_id="subject:rounding",
            edge_type="ABOUT_SUBJECT",
            source_session_id=None,
            source_turn_ids=[],
            derivation_version="graph-v1",
            review_status="AUTO_ASSIGNED",
        )


def test_graph_store_rejects_stale_source_hash(tmp_path):
    store = GraphStore.create_staging(tmp_path / "graph.duckdb", manifest("a" * 64))
    store.mark_ready()
    store.close()
    with pytest.raises(ValueError, match="source hash mismatch"):
        GraphStore.open_ready(tmp_path / "graph.duckdb", expected_source_hash="b" * 64)
```

- [ ] **Step 2: Run tests and verify missing-module red state**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_models.py tests/test_graph_store.py -v`

Expected: collection fails because `src.graph_models` and `src.graph_store` do not exist.

- [ ] **Step 3: Implement strict graph models with Chinese business comments**

Define:

```python
class GraphNode(StrictBusinessModel):
    node_id: str = Field(min_length=1)
    node_type: Literal[
        "SESSION", "QUESTION", "SUBJECT", "MISCONCEPTION_EVENT",
        "MISCONCEPTION_CONCEPT", "TUTOR_STRATEGY_EVENT", "STRATEGY_CONCEPT",
        "STUDENT_QUESTION_EVENT", "QUESTION_CLUSTER", "DIFFICULTY_EVENT", "TURN", "TALK_MOVE"
    ]
    label: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    source_session_id: int | None = Field(default=None, ge=1)
    source_card_id: str | None = None
    source_turn_ids: list[int] = Field(default_factory=list)
    derivation_version: str = Field(min_length=1)
    review_status: Literal["SOURCE", "AUTO_ASSIGNED", "HUMAN_APPROVED"]


class GraphEdge(StrictBusinessModel):
    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    edge_type: Literal[
        "ANCHORS", "ABOUT_SUBJECT", "HAS_MISCONCEPTION", "HAS_STRATEGY",
        "HAS_STUDENT_QUESTION", "HAS_DIFFICULTY_SIGNAL", "HAS_TURN",
        "TYPE_OF", "EVIDENCED_BY", "USES_TALK_MOVE", "CO_OCCURS_WITH", "FOLLOWED_BY"
    ]
    source_session_id: int = Field(ge=1)
    source_turn_ids: list[int] = Field(min_length=1)
    derivation_version: str = Field(min_length=1)
    review_status: Literal["SOURCE", "AUTO_ASSIGNED", "HUMAN_APPROVED"]
```

Add validators requiring source node/edge provenance and forbidding `CAUSES`, `IMPROVES_SCORE`, `EFFECTIVE_FOR`, and arbitrary edge types. Include `GraphProjectionManifest` with `graph_version`, `source_artifact_hash`, `node_count`, `edge_count`, `build_status`, `built_at_utc`, and `failure_reason` consistency.

- [ ] **Step 4: Implement a separate DuckDB graph sidecar**

Create tables:

```sql
CREATE TABLE graph_manifest (...);
CREATE TABLE graph_nodes (... properties_json VARCHAR, source_turn_ids INTEGER[], ...);
CREATE TABLE graph_edges (... source_turn_ids INTEGER[], ...);
CREATE TABLE graph_label_assignments (
    assignment_id VARCHAR PRIMARY KEY,
    source_node_id VARCHAR NOT NULL,
    label_type VARCHAR NOT NULL,
    canonical_label VARCHAR NOT NULL,
    assignment_source VARCHAR NOT NULL,
    review_status VARCHAR NOT NULL,
    taxonomy_version VARCHAR NOT NULL
);
```

`GraphStore.open_ready()` uses `read_only=True`, requires exactly one READY manifest row, verifies source SHA-256, and never creates missing tables. `replace_projection()` is transactional and refuses duplicate node/edge IDs.

- [ ] **Step 5: Run graph contract/store tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_models.py tests/test_graph_store.py -v`

Expected: all tests pass and no production artifact changes.

Commit: `git add src/graph_models.py src/graph_store.py tests/test_graph_models.py tests/test_graph_store.py; git commit -m "feat(eedi-rag): add rebuildable graph projection contracts"`

---

### Task 2: Project Each Session and Its Two Cards into the Graph

**Files:**
- Create: `src/graph_projection.py`
- Create: `tests/test_graph_projection.py`
- Create: `scripts/build_session_card_graph.py`
- Reference: `src/models.py:107-144`
- Reference: `src/storage_manager.py:280-308,461-601,734-758`

**Interfaces:**
- Consumes: read-only source DuckDB rows for one Session, its misconception Card, tutor strategy Card, and authoritative Turns.
- Produces: `project_session_graph(session_row, misconception_row, strategy_row, turns) -> SessionGraphProjection` and full-build CLI.

- [ ] **Step 1: Write failing projection tests**

```python
def test_projection_creates_one_session_two_card_events_and_source_edges():
    projection = project_session_graph(session_row(), misconception_row(), strategy_row(), turns())
    assert [node.node_type for node in projection.nodes].count("SESSION") == 1
    assert [node.node_type for node in projection.nodes].count("MISCONCEPTION_EVENT") == 1
    assert [node.node_type for node in projection.nodes].count("TUTOR_STRATEGY_EVENT") == 1
    assert edge_types(projection) == {
        "ANCHORS", "ABOUT_SUBJECT", "HAS_MISCONCEPTION", "HAS_STRATEGY",
        "EVIDENCED_BY", "USES_TALK_MOVE", "HAS_TURN"
    }


def test_projection_rejects_wrong_role_card_pointer():
    bad_card = {**misconception_row(), "source_turn_ids": [2]}  # authoritative Turn 2 is tutor
    with pytest.raises(ValueError, match="student"):
        project_session_graph(session_row(), bad_card, strategy_row(), turns())
```

- [ ] **Step 2: Run tests and observe missing projection module**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_projection.py -v`

Expected: collection fails because `src.graph_projection` does not exist.

- [ ] **Step 3: Implement deterministic Session/Card projection**

For every source Session create:

```text
session:{session_id}
question:{question_id}
subject:{normalized_subject_path}
misconception-event:{card_id}
tutor-strategy-event:{card_id}
turn:{session_id}:{turn_id}
talk-move:{normalized_talk_move}
```

Create only edges whose endpoints and source pointers exist. Use the existing Card `source_turn_ids` exactly; validate misconception pointers are student Turns and strategy pointers are tutor Turns. Store full Card fields in event-node properties, including original names, mechanisms, key question, scaffolding, talk moves, and resolution outcome. Do not put generated canonical labels into original Card text.

- [ ] **Step 4: Add stable normalization helpers for graph keys**

Implement:

```python
normalize_subject_path(value: str) -> str
normalize_strategy_category(value: str) -> str
normalize_talk_move(value: str) -> str
stable_node_id(node_type: str, *parts: str | int) -> str
stable_edge_id(edge_type: str, source_id: str, target_id: str) -> str
```

Normalization may trim whitespace, Unicode-normalize, case-fold English, and map known Talk Move spelling variants. It must not rewrite the original Card field stored in properties.

- [ ] **Step 5: Implement full-build CLI with source immutability checks**

Run:

```powershell
.venv\Scripts\python.exe scripts/build_session_card_graph.py `
  --source-db data/db/tutoring_knowledge.duckdb `
  --output reports/staging/session-card-graph-v1/graph.duckdb `
  --graph-version session-card-graph-v1
```

The builder hashes source before/after, opens source read-only, projects all paired Sessions, validates every edge pointer, writes a READY manifest only after all checks, and exits non-zero with FAILED manifest on error.

- [ ] **Step 6: Run projection/storage regression tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_projection.py tests/test_graph_store.py tests/test_evidence_index.py -v`

Commit: `git add src/graph_projection.py scripts/build_session_card_graph.py tests/test_graph_projection.py; git commit -m "feat(eedi-rag): project sessions and cards into rebuildable graph"`

---

### Task 3: Add Controlled Cross-Session Labels Without Changing Card Retrieval

**Files:**
- Create: `src/graph_taxonomy.py`
- Create: `tests/test_graph_taxonomy.py`
- Create: `data/business/graph-taxonomy-v1.json`
- Modify: `src/graph_projection.py`
- Modify: `src/graph_store.py`

**Interfaces:**
- Consumes: Card event-node properties and controlled taxonomy JSON.
- Produces: deterministic label assignments for question function, misconception, exam error, tutor strategy, difficulty signal, and content opportunity.

Define these concrete types in `src/graph_taxonomy.py` before implementing assignment rules:

```python
class Taxonomy(StrictBusinessModel):
    taxonomy_version: str = Field(min_length=1)
    label_type: Literal[
        "QUESTION_FUNCTION", "MISCONCEPTION", "EXAM_ERROR",
        "TUTOR_STRATEGY", "DIFFICULTY_SIGNAL", "CONTENT_OPPORTUNITY"
    ]
    labels: dict[str, str] = Field(min_length=1)  # canonical_id -> human-readable name
    signals: dict[str, list[str]] = Field(default_factory=dict)


class LabelAssignment(StrictBusinessModel):
    source_node_id: str = Field(min_length=1)
    label_type: str = Field(min_length=1)
    canonical_label: str = Field(min_length=1)
    assignment_source: Literal["deterministic_rules", "llm_suggestion"]
    review_status: Literal["AUTO_ASSIGNED", "HUMAN_APPROVED", "UNCLASSIFIED"]
    taxonomy_version: str = Field(min_length=1)
    source_session_id: int = Field(ge=1)
    source_turn_ids: list[int] = Field(min_length=1)
```

`UNCLASSIFIED` is a valid canonical label and must never be silently dropped. `llm_suggestion` is allowed only as an explicitly labelled suggestion; it cannot change the Card embedding document or be promoted to `HUMAN_APPROVED` by agent code.

- [ ] **Step 1: Write taxonomy tests proving original Card documents remain unchanged**

```python
def test_label_assignment_is_separate_from_embedding_document(card, original_document):
    assignment = assign_misconception_label(card, TAXONOMY)
    assert assignment.review_status == "AUTO_ASSIGNED"
    assert card.embedding_document == original_document


def test_unmatched_card_stays_unclassified():
    assignment = assign_misconception_label(ambiguous_card(), TAXONOMY)
    assert assignment.canonical_label == "UNCLASSIFIED"
    assert assignment.assignment_source == "deterministic_rules"
```

- [ ] **Step 2: Run taxonomy tests and observe missing-module red state**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_taxonomy.py -v`

Expected: collection fails because `src.graph_taxonomy` and the taxonomy artifact do not exist.

- [ ] **Step 3: Define a small, reviewable taxonomy**

Create `graph-taxonomy-v1.json` with controlled labels:

```json
{
  "misconception": [
    "PLACE_VALUE_CONFUSION",
    "ROUNDING_RULE_CONFUSION",
    "ORDER_OF_OPERATIONS_CONFUSION",
    "FRACTION_DENOMINATOR_CONFUSION",
    "NEGATIVE_SIGN_SCOPE_CONFUSION",
    "FORMULA_SELECTION_ERROR",
    "LANGUAGE_INTERPRETATION_ERROR",
    "OPTION_COMPARISON_ERROR",
    "UNCLASSIFIED"
  ],
  "exam_error": [
    "CONCEPTUAL_ERROR", "PROCEDURAL_ERROR", "REPRESENTATION_ERROR",
    "LANGUAGE_INTERPRETATION_ERROR", "CONDITION_OMISSION",
    "OVERGENERALIZATION_ERROR", "OPTION_COMPARISON_ERROR", "UNCLASSIFIED"
  ],
  "tutor_strategy": [
    "ASK_FOR_REASONING", "PRESS_FOR_ACCURACY", "USE_COUNTER_EXAMPLE",
    "RESTATE_STUDENT_IDEA", "BREAK_INTO_STEPS", "CONNECT_TO_PRIOR_KNOWLEDGE",
    "COMPARE_OPTIONS", "CHECK_UNDERSTANDING", "UNCLASSIFIED"
  ],
  "question_function": [
    "REQUEST_EXPLANATION", "REQUEST_CONFIRMATION", "ASK_NEXT_STEP",
    "COMPARE_OPTIONS", "RECALL_RULE", "ASK_DEFINITION",
    "EXPRESS_UNCERTAINTY", "OTHER"
  ]
}
```

Every label has a description, matching signals, taxonomy version, and review status. The JSON artifact is a draft taxonomy until a human approves its names; assignments remain AUTO_ASSIGNED.

- [ ] **Step 4: Implement deterministic label assignment**

Implement:

```python
assign_question_function(student_turn_text: str) -> LabelAssignment
assign_misconception_label(card_row: dict[str, Any], taxonomy: Taxonomy) -> LabelAssignment
assign_exam_error_label(card_row: dict[str, Any], taxonomy: Taxonomy) -> LabelAssignment
assign_strategy_labels(strategy_row: dict[str, Any], taxonomy: Taxonomy) -> list[LabelAssignment]
derive_difficulty_signals(session_row: dict[str, Any], turns: list[dict[str, Any]]) -> list[DifficultySignal]
```

Use first-match declared rules only. Preserve `UNCLASSIFIED` rather than forcing a label. `assign_strategy_labels()` may emit multiple labels because one Tutor Card can contain several real Talk Moves. Do not use `resolution_outcome` to claim strategy effectiveness.

- [ ] **Step 5: Verify Card/Chroma immutability and run tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_graph_taxonomy.py tests/test_graph_projection.py tests/test_retriever.py tests/test_reranker.py -v
```

Expected: taxonomy tests pass and existing retrieval/reranker tests remain green with unchanged Card documents.

- [ ] **Step 6: Commit taxonomy projection**

Commit: `git add src/graph_taxonomy.py src/graph_projection.py src/graph_store.py data/business/graph-taxonomy-v1.json tests/test_graph_taxonomy.py; git commit -m "feat(eedi-rag): add controlled cross-session graph labels"`

---

### Task 4: Implement Macro Graph/SQL Analytics for Dimension A and B

**Files:**
- Create: `src/graph_analytics.py`
- Create: `tests/test_graph_analytics.py`
- Reference: `src/storage_manager.py:761-842`
- Reference: `src/query_planner.py`

**Interfaces:**
- Consumes: READY graph sidecar, source DuckDB read-only connection, taxonomy assignments, subject filters, and `top_k`.
- Produces: strict aggregate reports with rank/count/share, connected Session IDs, and representative event references; no LLM calls.

Use this shared result shell for every graph aggregate method:

```python
class RepresentativeEvidenceRef(StrictBusinessModel):
    source_event_id: str = Field(min_length=1)
    session_id: int = Field(ge=1)
    turn_ids: list[int] = Field(min_length=1)
    perspective: Literal["STUDENT", "TUTOR"]
    subject_path: str = Field(min_length=1)


class DataScope(StrictBusinessModel):
    dataset_version: str = Field(min_length=1)
    graph_version: str = Field(min_length=1)
    total_sessions_considered: int = Field(ge=0)
    matched_sessions: int = Field(ge=0)
    subject_filters: list[str] = Field(default_factory=list)
    unmeasured_metrics: list[str] = Field(default_factory=list)


class AnalyticsResult(StrictBusinessModel):
    operation: str = Field(min_length=1)
    rows: list[dict[str, Any]]
    data_scope: DataScope
    representative_refs: list[RepresentativeEvidenceRef] = Field(default_factory=list)
    derivation_version: str = Field(min_length=1)
```

Part 3 response builders may convert `rows` into the typed report models from the existing business response plan; this graph analytics layer itself never emits generated narrative text.

- [ ] **Step 1: Write aggregate truth tests**

```python
def test_frequent_questions_count_distinct_sessions(graph_repo):
    rows, scope = GraphAnalytics(graph_repo).frequent_student_questions(top_k=10)
    assert rows[0].distinct_session_count == 2
    assert rows[0].event_count == 3
    assert rows[0].session_share == 2 / 3


def test_graph_analytics_keeps_unclassified_and_unmeasured_visible(graph_repo):
    result = GraphAnalytics(graph_repo).recurring_misconceptions(top_k=10)
    assert any(row.canonical_name == "UNCLASSIFIED" for row in result.rows)
    assert "UNRESOLVED_ENDING" in result.data_scope.unmeasured_metrics


def test_no_vector_score_is_used_as_count(graph_repo, spy_retriever):
    GraphAnalytics(graph_repo).frequent_student_questions(top_k=10)
    assert spy_retriever.calls == []
```

- [ ] **Step 2: Run analytics tests and observe missing-module red state**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_analytics.py -v`

Expected: collection fails because `src.graph_analytics` does not exist.

- [ ] **Step 3: Implement the five student/teacher aggregate queries**

Implement these methods:

```python
frequent_student_questions(top_k, subject_filters) -> AnalyticsResult
difficult_concepts(top_k, subject_filters) -> AnalyticsResult
recurring_misconceptions(top_k, subject_filters) -> AnalyticsResult
exam_error_patterns(top_k, subject_filters) -> AnalyticsResult
tutor_strategy_patterns(top_k, subject_filters) -> AnalyticsResult
```

Each method must:

1. filter graph/source data with bound parameters;
2. group by controlled label ID, not free-form Card prose;
3. compute `event_count`, `distinct_session_count`, and `session_share` deterministically;
4. return top rows with deterministic tie-breaking;
5. select representatives from distinct Sessions;
6. return DataScope with source/graph versions, total Sessions, matched Sessions, filters, and unmeasured signals;
7. preserve `UNCLASSIFIED` rows and never silently drop them.

- [ ] **Step 4: Implement relation analytics for combined student/tutor questions**

Add:

```python
misconception_strategy_pairs(top_k, subject_filters) -> AnalyticsResult
content_opportunities(top_k, subject_filters) -> AnalyticsResult
```

`misconception_strategy_pairs()` joins only events from the same Session and labels the relation `CO_OCCURS_WITH`; it must not report causal effectiveness. `content_opportunities()` ranks gaps by explicit formula using coverage, repeated error count, and missing content flag; the LLM may later write the outline, but cannot change priority counts.

- [ ] **Step 5: Run analytics tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_analytics.py tests/test_storage_manager.py::test_sql_analytics_and_group_by -v`

Commit: `git add src/graph_analytics.py tests/test_graph_analytics.py; git commit -m "feat(eedi-rag): add graph-backed macro analytics"`

---

### Task 5: Add Authorized Graph-to-Card-RAG Evidence Bridge

**Files:**
- Create: `src/graph_evidence_bridge.py`
- Create: `tests/test_graph_evidence_bridge.py`
- Modify: `src/retriever.py`
- Modify: `src/reranker.py`
- Reference: `src/generator_contract.py:254-344`

**Interfaces:**
- Consumes: aggregate representative refs, graph connected Sessions, existing retriever/assembler, max Parent anchors 5, W7/S3, and 4000-token budget.
- Produces: `AuthorizedGraphEvidenceBundle` with exact Evidence IDs, representative original Card context, W7 windows, and audit status.

Define the bridge output before implementing it:

```python
class AuthorizedGraphEvidenceBundle(StrictBusinessModel):
    pointers: list[EvidencePointer] = Field(min_length=1)
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    selected_parent_session_ids: list[int] = Field(min_length=1, max_length=5)
    selected_windows: list[dict[str, Any]] = Field(min_length=1)
    assembled_context: str = Field(min_length=1)
    evidence_catalog: dict[str, EvidenceCatalogItem]
    generator_context_token_count: int = Field(ge=1, le=4000)
    audit_status: Literal["AUDITED_100_VERIFIED"]
```

`EvidencePointer` is the Part 1 business contract with `source_event_id`, Session/Turn, speaker, exact quote, and verified status. `EvidenceCatalogItem` is the existing `src.generator_contract` catalog entry; Task 5 must import it rather than define a second citation format.

- [ ] **Step 1: Write evidence authorization tests**

```python
def test_bridge_rejects_turn_not_declared_by_aggregate_event(graph_store, source_storage):
    ref = RepresentativeEvidenceRef(
        source_event_id="misconception-event:10",
        session_id=10,
        turn_ids=[99],
        perspective="STUDENT",
        subject_path="Number",
    )
    with pytest.raises(ValueError, match="authorized"):
        GraphEvidenceBridge(graph_store, source_storage).materialize([ref])


def test_bridge_uses_existing_card_rag_inside_graph_session_scope(spy_retriever):
    bundle = make_bundle(spy_retriever)
    assert spy_retriever.received_allowed_session_ids
    assert bundle.generator_context_token_count <= 4000
    assert bundle.audit_status == "AUDITED_100_VERIFIED"
```

- [ ] **Step 2: Run evidence tests and observe missing-module red state**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_evidence_bridge.py -v`

Expected: collection fails because `src.graph_evidence_bridge` does not exist.

- [ ] **Step 3: Implement graph-to-RAG authorization**

Execution order:

```text
1. receive aggregate row representative event refs
2. verify event ID, Session ID, controlled label, and source Turn IDs in graph
3. fetch exact source Turns from authoritative DuckDB
4. verify speaker role and exact text
5. derive allowed Session IDs from the aggregate row only
6. invoke existing Card retriever with where/session restriction
7. expand only authorized Card pointers to W7/S3 windows
8. keep at most 5 Parent anchors for narrative context
9. build Evidence catalog E001... in deterministic order
10. run shared four-field citation audit
```

The bridge must never search the full corpus after an aggregate row has supplied a bounded Session set. If no authorized representative exists, return `NO_EVIDENCE` rather than selecting a nearby Session.

- [ ] **Step 4: Extend current retriever/assembler through backward-compatible APIs**

Add keyword-only optional parameters:

```python
DualMetricRetriever.retrieve_multi_perspective_rrf(
    ..., *, allowed_session_ids: list[int] | None = None,
    perspectives: tuple[str, ...] = ("student", "tutor"),
)

PedagogicalGoldAssembler.assemble_authorized_graph_context(
    raw_query: str,
    parent_anchors: list[dict[str, Any]],
    ranked_units: list[dict[str, Any]],
) -> GoldAssembledContext
```

Defaults preserve existing callers and ranking. New code must not add labels to embedding documents or modify Chroma metadata used by baseline retrieval.

- [ ] **Step 5: Run evidence/grounding/reranker tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_graph_evidence_bridge.py tests/test_generator_contract.py tests/test_reranker.py tests/test_retriever.py -v`

Commit: `git add src/graph_evidence_bridge.py src/retriever.py src/reranker.py tests/test_graph_evidence_bridge.py; git commit -m "feat(eedi-rag): bridge graph aggregates to audited Card RAG evidence"`

---

### Task 6: Add Final Business Generator Context and Response Contracts

**Files:**
- Create: `src/business_graph_models.py`
- Create: `src/business_graph_generation.py`
- Create: `tests/test_business_graph_generation.py`
- Modify: `src/business_models.py`
- Reference: `src/rag_pipeline.py:705-1020`

**Interfaces:**
- Consumes: original user query, `QueryPlan`, deterministic aggregate rows, graph paths, `AuthorizedGraphEvidenceBundle`.
- Produces: strict final prompt context and intent-specific business response narratives.

- [ ] **Step 1: Write generator tests for context ordering and backend-owned facts**

```python
def test_prompt_contains_original_query_facts_graph_paths_then_raw_evidence():
    text = build_business_generator_context(request, plan, facts, graph_paths, evidence)
    assert text.index("用户原始提问") < text.index("宏观统计事实")
    assert text.index("宏观统计事实") < text.index("图关系")
    assert text.index("图关系") < text.index("授权原始对白")


def test_generator_payload_cannot_return_authoritative_count_or_quote():
    with pytest.raises(ValidationError):
        BusinessNarrativePayload.model_validate(
            {"summary": "x", "distinct_session_count": 999, "quote_text": "fake"}
        )
```

- [ ] **Step 2: Run generation tests and observe missing-module red state**

Run: `.venv\Scripts\python.exe -m pytest tests/test_business_graph_generation.py -v`

Expected: collection fails because new business graph generation modules do not exist.

- [ ] **Step 3: Implement the four-section generator context**

Final context must be assembled in this order:

```text
【用户原始提问】
raw query exactly as entered

【宏观统计事实】
rank/count/share/DataScope from AnalyticsRepository

【图关系】
authorized graph paths and relation labels, no causal wording

【授权原始对白】
Evidence ID, Session, Turn, speaker, exact quote, Parent/W7 context
```

The generator receives all four sections, but only narrative fields are model-generated. Rank, counts, percentages, Session/Turn IDs, Evidence IDs, quotes, status, and audit state are filled by backend code.

- [ ] **Step 4: Implement intent-specific narrative payloads and budgets**

Use these API output limits:

```text
FAQ 160 tokens
Difficulty 240 tokens
Misconception 220 tokens
Exam error 220 tokens
Tutor strategy 300 tokens
Content improvement 420 tokens
Specific question 96 tokens
```

Set `thinking.type=disabled` by default, `temperature=0.1`, reuse one client, and record TTFT, visible tokens, reasoning tokens, and retries. `UNSUPPORTED`/`NO_EVIDENCE` bypass the generator.

- [ ] **Step 5: Run generator/contract tests and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_business_graph_generation.py tests/test_generator_contract.py -v`

Commit: `git add src/business_graph_models.py src/business_graph_generation.py src/business_models.py tests/test_business_graph_generation.py; git commit -m "feat(eedi-rag): generate answers from facts graph paths and raw evidence"`

---

### Task 7: Implement the Business Graph Pipeline and CLI Integration

**Files:**
- Create: `src/business_graph_pipeline.py`
- Create: `tests/test_business_graph_pipeline.py`
- Modify: `src/cli.py`
- Modify: `src/query_planner.py`
- Reference: `scripts/interactive_cli.py`

**Interfaces:**
- Consumes: Part 1 Router/Planner, graph store, analytics repository, evidence bridge, generator, existing micro pipeline.
- Produces: `BusinessGraphPipeline.ask_business(request) -> BusinessResponseEnvelope` and explicit CLI `--pipeline business-graph|legacy`.

- [ ] **Step 1: Write orchestration call-order tests**

```python
def test_macro_query_executes_graph_sql_then_card_rag_then_generator(spies):
    response = spies.pipeline.ask_business(
        BusinessQueryRequest(query="学生最常提出的问题是什么？")
    )
    assert spies.calls == [
        "route", "capabilities", "plan", "graph_analytics",
        "graph_scope", "card_rag_student", "evidence_audit",
        "generator", "response_build",
    ]


def test_micro_query_reuses_existing_rag_without_graph_analytics(spies):
    spies.pipeline.ask_business(
        BusinessQueryRequest(query="为什么 (-q)^2 和 -q^2 不同？")
    )
    assert "legacy_micro_rag" in spies.calls
    assert not any(call == "graph_analytics" for call in spies.calls)


def test_current_eedi_cohort_query_returns_unsupported_without_llm(spies):
    response = spies.pipeline.ask_business(
        BusinessQueryRequest(query="重考者有哪些共性思维卡点？")
    )
    assert response.status == BusinessResponseStatus.UNSUPPORTED
    assert "generator" not in spies.calls
```

- [ ] **Step 2: Run pipeline tests and observe missing-module red state**

Run: `.venv\Scripts\python.exe -m pytest tests/test_business_graph_pipeline.py -v`

Expected: collection fails because `src.business_graph_pipeline` does not exist.

- [ ] **Step 3: Add GraphRAG plan types and explicit dispatch**

Extend `QueryPlanType` with:

```text
GRAPH_ANALYTICS_WITH_EVIDENCE
GRAPH_RELATION_REPORT
```

Planner rules:

```text
FAQ/Difficulty/Misconception/ExamError/TutorStrategy:
  SQL aggregate first; graph scope when relation paths are requested; Card RAG for representatives.

Combined misconception + tutor strategy:
  GRAPH_RELATION_REPORT with CO_OCCURS_WITH paths.

Content improvement:
  graph/SQL gap aggregation + dual-perspective evidence.

Specific question:
  existing micro RAG; no graph aggregate.

Retaker/ability query without sidecar:
  REJECT/UNSUPPORTED before graph, RAG, or LLM.
```

- [ ] **Step 4: Implement fail-fast business pipeline**

The pipeline must enforce this sequence:

```text
route -> capability check -> plan
  -> REJECT: deterministic unsupported response
  -> MICRO: existing Card RAG -> exact audit -> response
  -> MACRO/HYBRID: SQL facts -> graph Session scope -> Card RAG -> evidence audit -> generator -> response
```

No generic fallback from analytics to vector-only RAG is allowed. Any stage failure returns `FAILED` with the stage and preserved exception context.

- [ ] **Step 5: Integrate CLI without breaking legacy mode**

Add:

```text
--pipeline legacy|business-graph
--graph-db PATH
--analytics-db PATH
--debug-graph
```

Resolve all defaults from repository root. Startup prints source/graph/analytics counts, hashes, active pipeline, actual reranker/Parent/W7 profile, and cohort capability status. Business output is rendered only after audit; legacy mode remains selectable for paired comparison.

- [ ] **Step 6: Run pipeline, CLI, legacy RAG, and anchored tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_business_graph_pipeline.py tests/test_business_cli.py tests/test_rag_pipeline.py tests/test_anchored_production.py -v`

Commit: `git add src/business_graph_pipeline.py src/query_planner.py src/cli.py tests/test_business_graph_pipeline.py; git commit -m "feat(eedi-rag): add graph-backed business pipeline"`

---

### Task 8: Freeze Business Graph Evaluation and Controlled Promotion

**Files:**
- Create: `evals/datasets/business-graph-v1/queries.jsonl`
- Create: `evals/datasets/business-graph-v1/expected_facts.jsonl`
- Create: `evals/datasets/business-graph-v1/expected_graph_paths.jsonl`
- Create: `evals/datasets/business-graph-v1/expected_evidence.jsonl`
- Create: `evals/datasets/business-graph-v1/split_manifest.json`
- Create: `evals/datasets/business-graph-v1/README.md`
- Create: `scripts/evaluate_business_graph.py`
- Create: `tests/test_business_graph_eval.py`
- Create: `docs/Business_Graph_RAG_Runbook.md`

**Interfaces:**
- Consumes: immutable source/Card/graph/analytics artifacts and independently human-reviewed facts, paths, and Turn evidence.
- Produces: exact macro metrics, graph path metrics, micro citation metrics, status accuracy, latency telemetry, and a release decision.

- [ ] **Step 1: Write evaluation contract tests**

```python
def test_gold_facts_are_not_passed_to_sut():
    with pytest.raises(ValueError, match="Gold"):
        BusinessGraphEvalCase.from_payload({"query": "...", "expected_facts": {"count": 1}})


def test_failed_graph_case_is_failed_not_zero():
    report = evaluate_cases([case()], failing_pipeline)
    assert report.cases[0].status == "FAILED"
    assert report.cases[0].metrics == {}
```

- [ ] **Step 2: Freeze at least 80 independently reviewed cases**

Required distribution:

```text
10 Student FAQ
10 Difficulty
10 Recurring misconception
10 Exam error
10 Tutor strategy
8 combined misconception-strategy relation
8 Content improvement
8 specific micro cases
6 Unsupported/cohort cases
```

Store queries, deterministic facts, graph paths, and exact evidence in separate files. Gold is never loaded by the SUT.

- [ ] **Step 3: Implement exact evaluation metrics**

Measure:

```text
route intent/scope/perspective exact match
plan type exact match
rank/count/session-share exact match
graph node/edge/path precision and recall
representative Session authorization rate
citation precision/recall
role accuracy and exact quote match
UNSUPPORTED/NO_EVIDENCE/FAILED status accuracy
source/Card/graph immutability
Router/SQL/graph traversal/RAG/assembly/TTFT/generation/audit latency
```

Hard gates:

```python
HARD_GATES = {
    "count_exact_match": 1.0,
    "graph_path_authorization": 1.0,
    "citation_precision": 1.0,
    "citation_recall": 1.0,
    "role_accuracy": 1.0,
    "exact_quote_match": 1.0,
    "unsupported_status_accuracy": 1.0,
    "card_embedding_artifact_unchanged": 1.0,
}
```

- [ ] **Step 4: Run paired legacy vs graph-business comparison**

Use identical queries, source hashes, Card/Chroma hashes, and model configuration. Compare:

```text
legacy Card-only retrieval
graph scope + Card representative retrieval
```

Do not promote based only on narrative quality. Promotion requires graph path correctness, macro fact exactness, evidence authorization, and no regression on existing micro RAG.

- [ ] **Step 5: Publish runbook and controlled promotion**

The runbook must state plainly:

```text
Session is the case center.
Student and tutor Cards are case-level evidence events.
Controlled labels connect Sessions.
DuckDB computes macro truth.
Graph finds related Sessions and paths.
Existing Card RAG finds representative raw dialogue.
LLM receives the original query + facts + graph paths + authorized raw evidence.
Current Eedi cannot answer retaker/weak-foundation claims.
```

Only if all hard gates pass, change CLI default from `legacy` to `business-graph`; otherwise keep legacy default and record `BLOCKED` with measured failures.

- [ ] **Step 6: Run full verification and commit evaluation assets**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/evaluate_business_graph.py `
  --dataset evals\datasets\business-graph-v1 `
  --source-db data\db\tutoring_knowledge.duckdb `
  --chroma-dir data\chroma `
  --graph-db reports\staging\session-card-graph-v1\graph.duckdb `
  --analytics-db reports\staging\business-analytics-v1\business-analytics.duckdb `
  --split holdout `
  --output reports\eval\business-graph-v1-YYYYMMDD
```

Commit: `git add evals/datasets/business-graph-v1 scripts/evaluate_business_graph.py tests/test_business_graph_eval.py docs/Business_Graph_RAG_Runbook.md; git commit -m "test(eedi-rag): evaluate session-card graph business pipeline"`

## Part Exit Criteria

The implementation is complete only when:

- Graph projection can be deleted and rebuilt from source Session/Card data with identical node/edge hashes.
- Every Session has one traceable student Card event and one tutor Card event where source data is complete.
- Controlled labels connect equivalent events without changing existing Card retrieval documents or Chroma embeddings.
- Dimension A and B macro questions obtain counts/ranks from SQL/graph analytics, not vector similarity.
- Macro reports include representative micro evidence from the exact connected Session set.
- Specific micro questions continue to use the existing Card RAG and Parent-5/W7 path.
- Final generator context contains original user query, deterministic facts, authorized graph paths, and exact raw dialogue in that order.
- Unsupported cohort claims stop before graph/RAG/LLM when required fields are absent.
- Full evaluation hard gates and legacy paired regression pass.
