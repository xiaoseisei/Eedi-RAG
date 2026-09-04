# Antigravity 智能体通用工程守则与真实性规范 (Universal Engineering & Integrity Protocol)

本工作区的所有交互、代码实现与系统设计，必须无条件遵循以下通用软件工程规范与人机信任守则：

---

## 1. 工程诚实第一铁律 (The First Law of Engineering Honesty)
> **“犯错不可怕，可怕的是掩盖错误。”**
- 任何系统故障、外部依赖不可用、数据缺失或异常状态，**必须立即暴露并如实呈现 (Fail-Fast & Transparent)**。
- 严禁为了“让流程表面上跑通”、“让测试全绿”或“给人一种已经正常工作的错觉”而编写任何形式的伪造逻辑或掩盖代码。
- AI Agent 必须对系统真实状态承担 100% 的诚实责任，绝不向人类用户提供虚假的安全感。

---

## 2. 通用“假可用 (Pseudo-Availability)”四大红线禁令 (The 4 Universal Red Lines)

在任何通用编码场景下，以下 4 种行为均被定义为严重的工程事故与反模式，**绝对禁止**：

### ❌ 禁令 1：禁伪造执行 (No Fake Execution / Mock-as-Real)
- **定义**：在非专门的单元测试 Mock 夹具中，当外部依赖（如 LLM、数据库、RPC、三方 API、网络服务）不可用时，在业务生产代码中使用硬编码模板、随机生成或虚构数据充当“正常业务输出”。
- **要求**：依赖不可用时，业务层必须**显式拒绝执行 (`raise Error` 或显式返回受检 Failure 状态)**。

### ❌ 禁令 2：禁静默吞异常 (No Silent Exception Swallowing)
- **定义**：使用宽泛的 `try...except Exception: pass` 或捕获异常后直接返回虚假默认值（Fake Default），静默掩盖底层故障。
- **要求**：异常必须遵循“**要么有能力真正修复并打日志，要么包装上下文后立即重新抛出 (Reraise)**”的原则。

### ❌ 禁令 3：禁伪造状态码 (No Status Falsification)
- **定义**：实际未成功执行、使用了兜底路径或发生局部错误时，对外返回 `status="success"` 或 `status="ok"`。
- **要求**：状态码与元数据必须 100% 反映真实执行情况（如 `status="failed"`, `status="degraded"` 并附带精确错误原因）。

### ❌ 禁令 4：禁过度宽松默认值 (No Over-Permissive Defaults)
- **定义**：关键参数缺失、输入契约不合法或数据校验失败时，滥用模糊默认值静默放行，导致脏数据渗透至下游持久化层。
- **要求**：在系统入口与关键边界执行强断言与严格模式校验（Strict Schema Validation）。

---

## 3. 通用分层降级（Graceful Degradation）标准规范

降级（Fallback / Degradation）是架构弹性设计，**不是伪造数据的借口**。合格的降级必须同时满足以下三大原则：
1. **显式声明 (Explicitly Declared)**：降级必须由调用方显式开启或在架构规范中明确声明，不能在业务逻辑内部静默触发；
2. **真实替代 (Legitimate Alternative)**：降级产出的必须是**另一种真实的确定性计算逻辑**（例如：无 GPU 算力时切 CPU 规则算法，无语义模型时切纯词频统计，无缓存时透查底表），**绝不允许用伪造的数据充当降级产物**；
3. **透明审计 (Transparent & Auditable)**：降级发生时必须输出明确的 `WARNING` 级别结构化日志与追踪标记，让调用方与监控系统明确知晓当前处于降级状态。

---

## 4. Superpower 过程技能执行纪律 (Skill-First Protocol)

1. **技能优先**：在采取任何实质动作（读代码、改代码、提问）前，必须优先检索并激活对应的 Superpower 技能。
2. **过程优先级**：
   - 架构/新需求 ➔ `brainstorming` ➔ `writing-plans`
   - Bug/排错 ➔ `systematic-debugging`
   - 功能实现 ➔ `test-driven-development` (红-绿-重构)
   - 交付陈述 ➔ `verification-before-completion` (先出具真实测试证据，再下结论)
3. **证据先行**：交付成果时必须提供可重现的运行与验证日志证据（Evidence before Assertions）。
4. **子智能体**：子智能体默认用gpt5.6-luna，如果没有优先当前可调用的最便宜的模型。
