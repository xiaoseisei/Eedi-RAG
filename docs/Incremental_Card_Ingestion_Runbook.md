# Incremental Card Ingestion Runbook

## 目的

卡片语义抽取和原文逻辑链索引分开运行。卡片的 `source_turn_ids` 由真实 `session_dialogue_turns` 的确定性 evidence index 绑定，不再采用 LLM 返回的 Turn ID 作为最终指针。成功结果按会话和完整契约缓存，后续运行只处理变化或显式指定的会话。

## 安全边界

- 所有抽取命令默认写入隔离 staging/cache 目录，不写 `data/db/tutoring_knowledge.duckdb` 或 `data/chroma`。
- 只有 staging 全部成功、hash 和评测通过后，才能人工执行蓝绿 promotion。
- LLM/API 失败会记录 `FAILED`，不会生成默认卡片，也不会覆盖已有成功缓存。
- `--mode full` 是唯一允许显式重新处理全部会话的模式；默认使用 `changed-only`。

## 1. 首次构建

```powershell
python scripts/incremental_card_ingest.py `
  --db data/db/tutoring_knowledge.duckdb `
  --cache-dir reports/cache/card-extraction `
  --mode full `
  --workers 4
```

输出 manifest 必须记录：

- `llm_processed_count`；
- `reused_count`；
- `audit_count`；
- `failed_count`；
- 每个 session 的状态、content hash、契约 key；
- evidence index version/hash；
- 生产 DB 只读源 hash。

## 2. 无变化增量运行

```powershell
python scripts/incremental_card_ingest.py `
  --db data/db/tutoring_knowledge.duckdb `
  --cache-dir reports/cache/card-extraction `
  --mode changed-only `
  --workers 4
```

当会话、Prompt、模型、Schema、绑定策略和 evidence index 版本都没有变化时，预期 `llm_processed_count=0`，所有成功会话从缓存复用。若不是 0，必须检查 manifest 中的契约差异，不能把结果当作正常复用。

## 3. 只修复指定会话

```powershell
python scripts/incremental_card_ingest.py `
  --db data/db/tutoring_knowledge.duckdb `
  --cache-dir reports/cache/card-extraction `
  --mode changed-only `
  --session-ids 14,206,469 `
  --workers 3
```

未指定的会话不会被提交到 LLM。成功的旧缓存仍保持不变。

## 4. 只重新绑定原文证据

```powershell
python scripts/incremental_card_ingest.py `
  --db data/db/tutoring_knowledge.duckdb `
  --cache-dir reports/cache/card-extraction `
  --mode audit-only `
  --session-ids 14,206,469
```

该模式不得调用语义抽取 LLM；它只读取成功缓存、重建确定性 evidence index 并更新绑定元数据。若 index 构建失败，该会话为 `FAILED`。

## 5. Prompt/模型更新

修改 Prompt 或模型后仍运行 `changed-only`。因为契约 key 改变，只有受影响的缓存记录会 miss；未变化的会话仍可复用旧版本，除非调用方明确要求全量重建。

## 6. 失败恢复

失败信息写入 staging manifest 和 cache failure record。修复 provider、输入或契约后，重新使用相同 `--cache-dir` 和新的 staging 目录运行 `changed-only`；成功会话会复用，失败会话重新提交。只有 `failed_count=0` 才允许创建可 promotion 的 staging DB/Chroma。

## 7. 显式全量重建

只有在模型迁移、Schema 不兼容或人工批准的完整重算时使用：

```powershell
python scripts/incremental_card_ingest.py `
  --db data/db/tutoring_knowledge.duckdb `
  --cache-dir reports/cache/card-extraction-v2 `
  --mode full `
  --workers 4
```

## 8. Promotion 前检查

1. manifest `status=SUCCESS` 且 `failed_count=0`；
2. 所有卡片 source Turn 角色正确、无重复、无越界；
3. index union 覆盖每场会话全部有效 Turn；
4. DuckDB/Chroma ID parity 和 evidence provenance 通过；
5. 在隔离副本运行当前 L1 和 chunk A/B 评测；
6. 记录 artifact hash before/after；
7. 人工确认后才执行 promotion，保留旧 DB/Chroma 备份。

## 9. 现有卡片证据回填与评测

对已经存在的 100 场历史卡片，不需要重新调用 LLM。使用以下命令重建逻辑链索引并写入隔离 staging：

```powershell
python scripts/rebind_card_evidence_staging.py `
  --db data/db/tutoring_knowledge.duckdb `
  --chroma data/chroma `
  --output-dir reports/eval/card-evidence-rebind-staging-<timestamp>
```

然后使用 card-specific evaluator 比较生产快照和 staging 快照：

```powershell
python scripts/eval_card_evidence_rebind.py `
  --before-db data/db/tutoring_knowledge.duckdb `
  --before-chroma data/chroma `
  --after-db reports/eval/card-evidence-rebind-staging-<timestamp>/tutoring_knowledge.duckdb `
  --after-chroma reports/eval/card-evidence-rebind-staging-<timestamp>/chroma `
  --output-dir reports/eval/card-evidence-rebind-card-metrics-<timestamp> `
  --top-k 20 `
  --bm25-weight 0.35
```

本次实际运行：

- staging：`reports/eval/card-evidence-rebind-staging-20260904-001500/`；100 sessions、200 cards、`llm_calls=0`。
- 评测：`reports/eval/card-evidence-rebind-card-metrics-20260904-003000/`；30 cases、同一 BM25+Dense 配置。
- card evidence Recall@20：`0.8722 → 1.0000`；Assembler final Context evidence recall：`0.6556 → 0.7667`。
- 这是 recall-first 回填；指针总量均值 `128.2 → 242.1`，因此 evidence precision 下降，不能直接宣称业务准确率。

## 10. Anchored Logical-Window Reranker 运行

生产环境使用真实 SiliconFlow Reranker 时，未显式指定 `--rerank-unit` 会默认启用 Anchored Parent-3/W5：

```powershell
python -m src.cli --embedding-backend siliconflow `
  --retrieval-mode bm25_dense --bm25-weight 0.35 `
  --reranker-backend siliconflow --chunk-strategy card --reranker-pool-size 20 `
  --rerank-unit anchored_logical_window --parent-card-count 3 `
  --evidence-selection-count 5
```

执行链路为：Top-20 卡片 → Card Rerank → 总 Parent Top-3（两个非空语义 lane 各保留至少 1 张，再按分数补足）→ 仅保留覆盖 `source_turn_ids` 的 W6/S3 窗口 → 注入父卡元数据进行 Window Rerank → W5 → overlap-aware Assembler。Assembler 按 rank 装配并对重叠真实 Turn 去重，父卡元数据只用于排序，最终引用仍只能来自 DuckDB 真实 Turn。

该模式不会重新抽取卡片，也不会改动缓存；Assembler 最多选择 5 个 evidence units，按 rank 和 1500-token budget 逐个加入，预算不足时停止而不是截断已选逻辑链。若需要旧的全量窗口路线，可显式使用 `--rerank-unit logical_evidence --reranker-pool-size 20`；无外部 Reranker 的本地 deterministic 运行默认仍为 `--rerank-unit card`。
