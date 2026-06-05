# Xcode TODO

按优先级排序，仅保留未完成计划。

## 边界

- 默认路径：REPL/CLI → StructuredAgent → core tools → permission/risk/audit
- 默认工具组：`tools.enabled_groups=["core"]`
- 新能力以 opt-in group 或 `experimental.*` 进入
- 不做默认 MCP 全量注入，不做不可观测 swarms，不做绕过权限的外部工具直连，不做企业级 RBAC/Grafana/Phoenix/RAGAS

---

## 已完成

### P1 评估基础设施

- 已接入本地 HumanEval 与 SWE-bench Lite JSON/JSONL benchmark loader。
- 已将 Pass@k 改为无偏估计量 `1 - C(n-c,k)/C(n,k)`。
- 已为内置 eval suites 补充 `llm_judge_criteria`，触发现有 LLM-as-judge 路径。

### P2 Session 并发安全

- 已为 `SessionStore` 写操作增加 `filelock` 保护。
- 已覆盖 append、fork、clean fork、rewind、compact 与 metadata 写入路径。

### P3 分支摘要自动压缩

- 已为 `LayeredCompactor` 增加非活跃分支摘要层。
- 已在 token 压力触发时将带分支 metadata 的非活跃消息替换为分支摘要消息。

### P4 Turn Snapshot 隔离

- 已增加 `TurnSnapshot` 冻结 dataclass，包含 config/registry/tool_map/provider 等 turn 依赖。
- 已增加 `_turn_snapshot()` 在 `arun_stream()` 开头冻结快照（`deepcopy(config)` + `tuple(registry)`）。
- turn 内所有操作均使用快照引用而非 `self` 全局引用。

### P5 会话持久化协议

- 已显式记录 `jsonl-v1` 会话索引协议。
- 已在 index 中写入恢复边界 `current_transcript_and_session_tree`。
- 已保持旧 index 读取兼容。

### P6 分支导航

- 已增加 `SessionStore.switch_branch()`。
- 已增加 `/branch list|tree|<id|title>` 命令复用会话树切换分支。

### P7 类型化事件流（含 P9 订阅事件）

- 已定义 `HarnessEvent` union 与类型化事件 dataclass。
- 已增加 `HookManager.subscribe()` / `unsubscribe()`。
- 已保留 `HookManager.register()` 作为内部桥接。

### P7 Tasks+Progress 编排

- 已增加长任务运行状态 `TaskRunState`。
- 已支持中断恢复、租约超时释放、重试预算和子任务分发。
- 已通过 `progress` 实验组暴露对应编排工具。

### P8 类型化事件流（含 P9 订阅事件）

- 与 P7 类型化事件流重复，已随 P7 统一完成。

### P9 Daemon 生命周期

- 已增加 `DaemonHealth` 健康快照。
- 已增加 `register_callback()` 事件回调。
- 已增加任务失败计数、错误事件和 `ensure_healthy()` 自愈重启。

### P10 维护契约补齐（source-review §10）

- 已在 `tool_catalog.py` docstring 中注明新增 `build_*_tools()` 须同步注册 `_builders()`。

### P11 Provider 代码清理（source-review §10）

- 已将 Responses API stream usage 拦截提取到 `ProviderMetricsMixin._intercept_responses_stream()`。
- 已保留 `OpenAIResponsesProvider` 内的 stateful response id 更新逻辑。

### P12 模型模式解析

- 已增加纯函数 `parse_model_mode()`。
- 已支持 `/model [profile/]name[:thinking]`。
- 已保留 `/model <name> --thinking <level>` 用法。

### P13 队列模式

- 已增加 opt-in `/queue on|off`。
- 已支持流式 turn 期间读取 queued follow-up，并在当前 turn 结束后注入 agent follow-up 队列。

### P14 ExecutionEnv 抽象

- 已定义 `ExecutionEnv` protocol（`run(argv, cwd, timeout, cancel_event) → ExecutionResult`）。
- 已增加 `SubprocessExecutionEnv` 默认实现（提取自 `bash.py` 的子进程管理逻辑）。
- 已增加 `SandboxExecutionEnv` 用于测试/mock 注入（记录调用，支持预设返回）。
- 已接入 `bash` 工具（替换原有的未使用 `BashOperations` Protocol，新增 `env` 参数）。
- 已通过 `registry.py` / `assembly.py` 将 `env` 参数透传至顶层注入点。
