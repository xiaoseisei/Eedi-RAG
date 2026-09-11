# Business Graph Gold v1

这是基于当前真实 Eedi 数据库建立的端到端业务 Gold 基线集。

本评测集已通过本次人工查验，允许用于本次评测。该放行不等同于生产业务效果已验证，也不替代后续数据版本变更后的重新审核。

## 数据边界

- 源库：`data/db/tutoring_knowledge.duckdb`
- 图库：READY 的 `session-card-graph-v1` sidecar
- 当前窗口：W7/S3
- Window 是最小重排和 Prompt 装配单元；Turn 只用于原文、角色和引用授权校验。
- Gold 编写不读取 Retriever、Reranker、Assembler 或评测结果。

## 覆盖

当前版本覆盖 17 个请求类别，共 83 条可审计样本：微观类别每类 6 条，宏观类别每类 5 条，混合类别每类 5 条，`NO_EVIDENCE`、`UNSUPPORTED`、`FAILED` 各 3 条。具体数量以 coverage 文件为准。

所有正向样本均来自真实库并经过指针、角色、Window 和原文校验；后续新增 Case 必须通过同一构建器和人工业务审核。

## 状态语义

- `SUCCESS`：真实 Card/Graph/Window/Turn 可以支持问题。
- `NO_EVIDENCE`：问题可理解，但当前真实库没有目标证据。
- `UNSUPPORTED`：当前数据能力不支持该业务结论，例如重考后成绩变化。
- `FAILED`：执行前提失败，例如 source 与 graph hash 不一致；必须阻断。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\build_business_graph_gold.py
.\.venv\Scripts\python.exe -m pytest tests/test_business_graph_gold.py -q
.\.venv\Scripts\python.exe evals/business/business_graph_e2e_eval.py
```
