# L1 评测数据集目录

`l1-component-suite.example.json` 是一个明确标注为 `fixture` 的最小契约样例，只用于验证 Suite CLI 和报告写入；它不代表当前 RAG 业务质量，也不能用于签署基线。

正式数据集应按组件分别版本化：

- `grounding-vN.json`：输出引用、权威 Turn、required evidence 四元组。
- `chunking-vN.json`：Query、目标 Session/Turn、真实 Chunk 排名和 token 统计。
- `rewrite-vN.json`：原始/改写 Query、受保护 token、关键实体、意图标注和配对检索结果。
- `retrieval-vN.json`：真实 Query、多个 graded qrels（0/1/2/3）、slice 和 latency。
- `assembler-vN.json`：固定候选集、最终 Context、required evidence、预算和压缩统计。

数据集不得写入 API key、token、cookie 或其他凭据；同一 `intervention_id` 的记录必须处于同一 split，避免 Session 泄漏。
