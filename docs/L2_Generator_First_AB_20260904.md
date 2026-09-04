# L2 Generator-first A/B 记录（2026-09-04/05）

## 目的与固定边界

本轮只比较 Generator contract/prompt；检索、Anchored Parent-3/W5、BM25+Dense、Top-20 card pool、artifact 和 30-case golden split 固定不变。Gold Context v2 仅用于评测，不进入生产数据。

固定运行参数：

```text
artifact = b45860c80d02b999c9c9bf6d960c50779b84aad1b76fa74bef02da6ac9a8aeb9
retrieval = bm25_dense, bm25_weight=0.35
route = anchored_logical_window, pool=20, parent=3, evidence=5
gold_context = v2
generator = LLM_MODEL from .env
```

## v2 已完成全量

报告目录：`reports/eval/l2-generator-v2-full-20260905-000000/`。

四个分片共 30 个 case，Gold/Real 各 30 条 trace。源 artifact 前后 hash 一致。4 条 trace 为远端 connection error，未使用默认值或伪造结果。

| Track | Answer Relevancy | Contextual Precision | Contextual Recall | Faithfulness | Pedagogical GEval | Citation Audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gold | 0.7932 (29) | 1.0000 (28) | 0.9557 (29) | 0.9241 (29) | 0.9286 (28) | 0.5690 (29) |
| Real | 0.8625 (26) | 0.8979 (26) | 0.7404 (26) | 0.7704 (27) | 0.9231 (26) | 0.4877 (27) |

括号内为实际测量 case 数；指标 threshold 仍见 `report.json`。v2 的结构化教学输出改善明显，但 claim-level contract 仍不能保证覆盖 provisional required qrels，因此 Citation Audit 未达 0.95。

## v1 对照状态

首轮 v1（`l2-generator-v1-full-20260905-0100-partial`）在 API 余额不足时运行：30 个 case attempted，39 条 ERROR、21 条 SUCCESS，不能作为完整 A/B 结论。错误包含真实的 `402 Insufficient account balance`，另有少量旧 v1 quote 标记污染。

在 API 恢复后，已重新启动四个新分片（run id 后缀 `20260905-0200`），参数为 `generator-contract=v1`、`gold-context-version=v2`、`rubric-version=v1`，其余参数与 v2 相同。必须等四个分片均完成并合并后，才可报告 v1 全量均值。

## 解释规则

- `Gold` 高、`Real` 低：仍有检索/装配上下文损失。
- 两条 track 都低：优先查 Generator、输出契约或 Judge rubric。
- v1 使用 rubric v1、v2 使用 rubric v2 时，Faithfulness/Pedagogical 的绝对差异存在 rubric confound；若需要严格隔离 Generator，应在同一 rubric v2 下重放 v1 contract。
- Citation Audit 是 deterministic backend audit，不能用 Judge 分数替代，也不能因模型输出更长而视为通过。

## 当前阻断

生产 release 仍受历史 `BLOCKED_L1_PRECONDITION` 阻断。L2 v2 结果不构成自动 promotion；v1 全量完成前不做 Generator 选型结论。

