# Query Router v1 人工审核说明

推荐审核文件是 `query-router-v1.review-v2.jsonl`。它有 135 条候选样本，覆盖当前 9 个业务意图各 15 条；所有行都显式标为：

```json
"label_source": "draft"
```

这表示标签由 AI 起草、尚未经过人工审核。它不是正式 Gold，不能直接传给 `scripts/evaluate_business_router.py`，也不能作为 Part 1 的发布依据。

`query-router-v1.review.jsonl` 是上一版不可覆盖草案，保留用于版本追溯；请审核 v2，不要修改或覆盖旧版。

审核时请逐条确认或修改：

- `primary_intent`：业务意图；
- `scope`：`CORPUS`、`SUBJECT`、`CASE`、`QUESTION`；
- `computation_scope`：`MACRO`、`MICRO`、`HYBRID`、`REJECT`；
- `evidence_perspectives`：学生、导师或双视角；
- `subject_filters`、`protected_tokens`、`requested_top_k`；
- `split`：保持 `dev` / `holdout` 分割稳定。

人工审核完成后，由审核人执行：

1. 把确认后的内容复制到 `evals/datasets/query-router-v1.jsonl`；
2. 将每一行的 `label_source` 改为 `human`；
3. 保留原有 `case_id`，不要重排、删改或复用 ID；
4. 执行正式评测：

```powershell
.venv\Scripts\python.exe scripts\evaluate_business_router.py `
  --dataset evals\datasets\query-router-v1.jsonl `
  --split holdout `
  --output reports\eval\business-router-v1-YYYYMMDD
```

正式评测器会拒绝非 `human` 标签、重复 ID、非法枚举、空数据集和已有输出目录。若任一门禁失败，报告会保留失败样本并以非零退出；不得手工改写报告为通过。
