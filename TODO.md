# Xcode TODO

按优先级排序，仅保留未完成计划。

## 边界

- 默认路径：REPL/CLI → StructuredAgent → core tools → permission/risk/audit
- 默认工具组：`tools.enabled_groups=["core"]`
- 新能力以 opt-in group 或 `experimental.*` 进入
- 不做默认 MCP 全量注入，不做不可观测 swarms，不做绕过权限的外部工具直连，不做企业级 RBAC/Grafana/Phoenix/RAGAS

---

## 待实现

### P1 评估基础设施

评估系统"骨架完整、肌肉不足"。当前仅 7 个自定义合成任务，需接入外部标准 benchmark 并激活已有 LLM-as-judge 能力。

- **外部 coding benchmark 接入**（source-review §10）：至少接入 HumanEval（代码补全）+ SWE-bench Lite（任务级修复），与当前 fixture-based eval 架构天然匹配。同时将 Pass@k 从朴素 `any(pass in k)` 改为无偏估计量 `1 - C(n-c,k)/C(n,k)`。
- **内置 LLM-as-judge 任务启用**（source-review §10）：`src/xcode/evals/graders.py` 中 `llm_judge` 接口完整，但内置 `tasks.py` 套件均未设置 `llm_judge_criteria`。为内置套件补充 LLM 评判标准，使其在每次评测中自动触发。

### P2 Session 并发安全

`SessionStore` 引入轻量 `filelock` 上下文管理器保护写操作（fork、compact、rewind）。

### P3 分支摘要自动压缩

`LayeredCompactor` 增加分支摘要层：上下文紧张时调用 LLM 压缩非活跃分支，以 `BranchSummaryMessage` 替换原始内容。需要 P2 的 session 安全写入作为前置。

### P4 Turn Snapshot 隔离

`StructuredAgent.execute_turn()` 在 turn 开始时冻结 config/tools/skills 快照，turn 内使用快照而非全局引用。

### P5 会话持久化协议

存储协议（JSONL vs SQLite）、恢复边界、分支联动设计。

### P6 分支导航

依赖 P5。分支 fork/摘要/切换，需要会话状态树存储。

### P7 类型化事件流（含 P9 订阅事件）

定义统一 `HarnessEvent` union，提供强类型 `subscribe()`。保留 `HookManager` 做内部桥接。

### P7 Tasks+Progress 编排

从"有场景再动"升级。现有 `tasks` + `progress` 实验组支持轻量任务图和 checklist，但缺少可重入长任务的完整编排能力：中断恢复、超时/重试、子任务分发。实现后使长任务在真实场景中可靠运行。无硬性前置依赖。

### P8 类型化事件流（含 P9 订阅事件）

定义统一 `HarnessEvent` union，提供强类型 `subscribe()`。保留 `HookManager` 做内部桥接。

### P9 Daemon 生命周期

从"有场景再动"升级。现有 `daemon` 实验组支持 `HeartbeatDaemon` 周期性检查，但缺少健康检查、自动重启和回调注册。完善后使后台守护过程可观测、可恢复。

### P10 维护契约补齐（source-review §10）

当前 `tool_catalog.py` 已覆盖所有产出 `ToolSpec` 的模块（包括 mailbox/progress/mcp），但缺少"新增 `build_*_tools()` 须同步注册 builder 条目"的显式维护契约。在 `tool_catalog.py` docstring 中注明此规则。

### P11 Provider 代码清理（source-review §10）

`src/xcode/ai/providers/metrics.py` 的 `ProviderMetricsMixin` 已完成三个模式的提取，子类覆写 `_record_usage` 属合理多态。仅 `OpenAIResponsesProvider.intercept_events`（`openai.py:126-141`）因 Responses API 事件模型差异存在一个同构闭包，可提取为 mixin 中的 `_intercept_responses_stream` 方法。

### P12 模型模式解析

从"有场景再动"升级。`model:thinking_level` 三段式解析（provider / model ID / thinking level）。独立、范围小、可实现为纯函数，适合作为低优先级 quick win。

### P13 队列模式

从"有场景再动"升级。REPL 主循环改为 asyncio 双路架构后支持流式期间输入。UX 改善显著但前置重构较大，放在低优先级等待 REPL 架构自然演进。

### P14 ExecutionEnv 抽象

从"有场景再动"升级。`ExecutionEnv` protocol，默认调 subprocess，web sandbox 可注入 mock。纯架构改善，无直接用户影响，放在最后。
