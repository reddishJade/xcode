# Xcode 源码级审查报告

本文基于当前独立 checkout 的 `src/xcode/` 源码整理。目标不是宣传能力，而是记录当前实现边界、默认路径和实验性功能的真实接入方式。

---

## 1. 系统定位

Xcode 是一个轻量级 Python coding agent harness。核心设计是：

- 模型负责推理和工具选择。
- Harness 负责工具协议、安全边界、上下文装配、会话状态、审计和验证。
- 默认路径保持小而稳定，只暴露 `core` 工具组。
- 实验性能力必须通过 `tools.enabled_groups` 显式启用。

运行路径：

```text
src/xcode/main.py
  -> discover_runtime_config()
  -> build_app()
  -> StructuredAgent
  -> provider stream
  -> tool execution
  -> final answer / REPL transcript
```

---

## 2. 装配中心

`src/xcode/harness/app.py::build_app()` 是应用装配入口。装配委托给 `assembly.py`，后者负责：

- 解析 config。
- 初始化 shared infra（`ContextualRetrievalState`、`CancellationToken`、`CompactController`）。
- 根据 `tools.enabled_groups` 构造可见工具 registry。
- 构造 provider bundle。
- 构造 `StructuredAgent`。
- 按 opt-in group 连接 experimental 组件。

`EXPERIMENTAL_FEATURE_GROUPS` 当前展开为：

```text
worktree, mcp, tasks, memory, plugins, daemon, mailbox, progress
```

`experimental` 是总开关。启用它等价于启用全部上述 group。`bm25` 是 `memory` 的内部实现，不是独立 group。

---

## 3. Context Surface

### Prompt 构造

`src/xcode/harness/agent_runtime/prompting.py` 负责 system prompt 拼装。默认模块包括：

- identity
- tool discipline
- tools
- environment
- git preflight
- cwd
- instructions
- notices

可选模块包括 `search_strategy`、`contextual_retrieval`、`skills`。

### Git Preflight

`git_preflight.py` 在 prompt 中注入当前 Git 状态、最近 commit 和 dirty diff stat。它只提供环境上下文，不自动 stage、commit 或回滚文件。

### Contextual Retrieval

`ContextualRetrievalState` 记录最近访问文件和工具结果摘要。只有 prompt modules 包含 `contextual_retrieval` 时才注入 prompt。

---

## 4. Action Surface

### Core tools

默认 `core` 工具组提供：

- `read_file`
- `write_file`
- `edit_file`
- `glob_files`
- `grep_search`
- `ls`
- `bash`

`edit_file` 依赖 read-before-edit 指纹校验。`read_file` 会记录文件 hash、mtime、size；后续编辑前必须匹配该版本。

`bash` 通过 `ShellAdapter` 选择宿主 shell，并使用 `Popen` 生命周期控制、超时和 cancellation token。命令风险由 `CommandRiskEvaluator` 判定。

REPL 中的 `!COMMAND` 是 `bash` 工具的快捷入口，输出按原始终端文本展示；它不绕过工具适配器的权限判定和脱敏路径。

### Tool execution

生产工具调用路径是：

```text
StructuredAgent
  -> Agent
  -> agent/tool_execution.py
  -> ToolSpecAdapter
  -> ToolSpec.handler
```

执行路径会统一处理：

- 未知工具错误
- 统一权限决策（`check_tool_permission`）
- execution mode policy（`PermissionDecision`: allow/deny/ask）
- HITL approval
- secret redaction
- structured audit record 和 hook

### Tool partitioning

`agent/tool_execution.py` 会把连续的并发安全工具分区并行执行。`ToolSpecAdapter.execution_mode` 根据工具声明决定调度方式：只读、并发安全且非 high risk 的工具默认并行；其他工具默认串行。这样写工具、高风险工具和不可并发工具按模型原始顺序串行执行。

---

## 5. Control Surface

### Permission policy

`PermissionDecision` 统一了 execution mode 和 permission policy 的三态决策：`allow`、`deny`、`ask`。`check_tool_permission()` 合并 `PermissionPolicy.decide()` 和 `risk_evaluator` 两层检查，返回 `PermissionCheckResult(blocked, reason)`。

`SettingsSandboxPermissionPolicy` 可从 `.local/settings.json` 或 `settings.json` 读取规则，与调用方传入 policy 组合。

### HITL

高风险工具或 policy 判定为 `ask` 时，会要求 approval callback。没有 callback 时，工具返回 `approval_required`，不会执行 handler。

### Redaction

`redact_text()` 对工具输出和审计日志进行脱敏，覆盖 API key 和常见 key/secret/token 键值格式。

---

## 6. Isolation Surface

### Plan / Review / Act

REPL 支持 `/plan`、`/review`、`/act`。执行模式由 `execution_modes.py` 提供说明和权限边界。

### Plan Exit

`/act --clear` 会把 plan artifact 写入 `.local/session_artifacts/`，再创建干净会话并把 approved plan 注入下一轮运行上下文。

### Subagent

`subagent` group 提供：

- `submit_subagent`
- `check_subagent`
- `cancel_subagent`

`ManagedSubagentRunner` 控制并发、超时和递归深度。子 agent 使用过滤后的 tool registry，避免看到父 agent 未启用的工具。

### Worktree

`worktree` group 提供 Git worktree 沙箱：

- `create_worktree_task`
- `remove_worktree_task`

移除 worktree 时会检查 dirty status 和未合并/未推送提交，防止误删工作产物。

---

## 7. State And Verification Surface

### Layered Compaction

`LayeredCompactor` 包含：

1. stale `read_file` 裁剪
2. 大工具输出头尾截断
3. 旧 `tool_result` 微压缩
4. transcript 落盘
5. older messages summary compact

只有启用 `memory` group 时，压缩摘要才会交给 `MemoryManager.consolidate()` 写入 `MEMORY.md` 候选块。

### Read version restoration

`restore_read_versions` 在压缩或恢复后，从历史消息和磁盘文件重建 read-before-edit 指纹。若文件已变化，则要求重新读取。

### Watchdog

`StructuredAgent` 会检测重复工具调用签名，超过 `watchdog_repeated_tool_limit` 时停止，避免完全重复的工具循环。

---

## 8. Experimental Feature Review

所有 experimental 能力默认关闭。启用方式是把对应 group 加入 `tools.enabled_groups`。

### `mcp`

文件：`src/xcode/experimental/mcp.py`

能力：

- stdio JSON-RPC MCP client
- `Content-Length` framing
- `.local/mcp_cache.json` schema cache
- `defer_loading` bootstrap/search flow
- dynamic `mcp__server__tool` registration
- explicit tool risk overrides

边界：

- 只在启用 `mcp` 或 `experimental` 后读取 MCP 配置。
- MCP 工具风险必须通过 server `overrides` 显式声明；未声明工具默认 high risk。
- 只支持当前实现里的 stdio 传输；SSE/WebSocket 仍属于后续方向。

### `tasks`

文件：`src/xcode/experimental/tasks.py`

能力：

- `.local/tasks.json.d/{id}.json` task storage
- filelock 保护 ID 分配和更新
- dependency topological sort
- Kanban rendering
- tools：`create_task`、`update_task`、`list_tasks`、`get_task`

边界：

- 这是轻量任务图，不是完整长任务编排系统。

### `worktree`

文件：`src/xcode/experimental/worktree.py`

能力：

- `WorktreeTaskRunner`
- tools：`create_worktree_task`、`remove_worktree_task`
- Git worktree physical isolation
- dirty/unmerged commit removal guard

边界：

- 工具风险为 high，需要按现有 HITL 规则执行。

### `mailbox`

文件：`src/xcode/experimental/mailbox.py`

能力：

- `.team/inbox/{agent_id}.jsonl` append-only mailbox
- filelock 写入和读取
- ACK 事件保证幂等
- tools：`send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message`

边界：

- 当前是本地文件邮箱，不含跨机器 transport。

### `progress`

文件：`src/xcode/experimental/progress.py`

能力：

- `TaskProgress.save_progress()`
- `TaskProgress.resume_task()`
- `claude-progress.txt` 派生只读视图
- tools：`save_task_progress`、`resume_task_progress`

边界：

- 依赖 `TaskStore` 作为真值源。
- 不会自动决定任务计划，只保存调用方传入的 checklist。

### `memory`

文件：`src/xcode/experimental/memory.py`、`src/xcode/experimental/bm25.py`

能力：

- `MEMORY.md` H2 block parsing
- required fields validation
- corrupt candidate archive
- BM25 recall
- metadata reranking
- search eval helpers
- compaction summary consolidation hook

边界：

- 默认不启用。
- 不自动注入主 prompt。
- `bm25` 是内部算法实现，不单独启用。

### `plugins`

文件：`src/xcode/experimental/plugins.py`

能力：

- 扫描 `.local/plugins/*.py`
- 收集 `exposed_tools`
- 收集 `exposed_hooks`
- 收集 `exposed_skills`

边界：

- 使用 in-process `exec()` 动态加载，插件等同宿主代码，必须保持 opt-in 且只加载已审核可信插件。

### `daemon`

文件：`src/xcode/experimental/daemon.py`

能力：

- `HeartbeatDaemon`
- 周期检查 mailbox、git dirty status、background tasks
- 把事件写入 mailbox

边界：

- `build_app()` 只在启用 `daemon` 或 `experimental` 后构造 daemon。

---

## 9. Evals

`src/xcode/evals/` 包含两条验证线：

- `EvalRunner`：消费 `XcodeApp.aask_stream()` 事件流，生成 trace、JSON report 和 HTML report。

当前 grader 是确定性规则，覆盖最终事件、答案片段、工具调用约束、工具错误数和文件证据。

---

## 10. Current Gaps

- `cli/tool_catalog.py` 仍只扫描部分工具 builder，若 REPL 目录要展示 mailbox/progress/mcp 等所有 opt-in 工具，需要继续同步。
- `memory` 仍缺少 consolidation 质量门、冲突合并和长期遗忘策略。
- `plugins` 使用动态加载，应继续保持显式 opt-in，并补更细的安全策略。
- `daemon` 当前由 app 构造，但生命周期启动仍需要调用方控制。
- `tasks` + `progress` 能表达任务和 checklist，但还不是完整可重入长任务编排器。
- eval 仍以确定性 grader 为主，没有 LLM-as-judge、Pass@k 和外部 benchmark 接入。
- `intercept_usage` closure 和 `_record_usage`/`_ensure_metrics` 在 4-5 个 provider 中有重复，可抽取到基类或工具函数。
- Provider 层已完成 OpenAI-compatible 基类 `openai_compat.py` 抽取，`deepseek`、`chatglm`、`mimo` 均已继承自该基类。

---

## 11. 维护规则

1. 不要绕过 `ToolSpecAdapter` 和 `agent/tool_execution.py` 直接执行工具 handler。
2. 不要默认启用 `src/xcode/experimental/` 中的能力。
3. 新 experimental 能力必须有独立 group；`experimental` 总开关应同步展开它。
4. 新工具必须声明 group、risk、schema 和 read-only/concurrency 属性。
5. 修改工具或 group 后，应更新 `CONFIG.md`、`README.md`、`docs/code-organization.md` 和相关测试。
