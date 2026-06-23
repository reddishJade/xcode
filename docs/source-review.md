# Xcode 源码级审查报告

基于 `src/xcode/` 源码整理。

---

## 1. 系统定位

四层模型：`ai/` (Provider) → `agent/` (Loop Core) → `harness/` (Runtime Infra) → `coding_agent/` (Coding Product) → `cli/` (UI)

运行路径：
```
main.py → discover_runtime_config() → build_app() [harness/app.py]
  → assembly.py → StructuredAgent [harness/agent_runtime/structured.py]
  → Agent loop [agent/agent_loop.py] → provider stream [ai/providers/]
  → tool execution [agent/tool_execution.py → coding_agent/tools/]
  → final answer / REPL transcript
```

---

## 2. 装配中心

`build_app()` → `assembly.py` 负责：
- 配置解析（config.py）
- 共享基础设施：ContextualRetrievalState、CancellationToken、CompactController、LayeredCompactor
- Provider bundle 构造
- 工具 registry（coding_agent/registry.py + assembly 扩展）
- StructuredAgent 构造
- 服务：daemon、mailbox、progress


---

## 3. Context Surface

`prompting/builder.py` + `prompting/identity.py` 负责 system prompt 构造。

模块顺序（`STABLE_PROMPT_MODULE_ORDER`）：identity → tool_discipline → citations → tools → search_strategy
动态模块（`DYNAMIC_PROMPT_MODULE_ORDER`）：environment → cwd
易变模块（`VOLATILE_PROMPT_MODULE_ORDER`）：git_preflight → contextual_retrieval → notices

分三个缓存区域：
- **STABLE**: identity, tool_discipline, citations, tools, search_strategy
- **DYNAMIC**: environment, cwd（按项目缓存）
- **VOLATILE**: git_preflight, contextual_retrieval, notices（每轮重建）

`ContextualRetrievalState` 记录最近访问文件和工具结果摘要。

---

## 4. Action Surface

### Core tools

实现归 `coding_agent/tools/` 层所有，group 均为 `core`：
- `read_file` / `write_file` / `edit_file`（file.py）
- `glob_files` / `find_files` / `grep_search` / `ls`（code_search.py）
- `bash`（bash.py，通过 ShellAdapter 选择宿主 shell）
- `search_tools`（assembly.py，按关键字搜索已注册工具）

`edit_file` 依赖 read-before-edit 指纹校验。`bash` 通过 `ShellAdapter` 选择宿主 shell，`Popen` 生命周期控制、超时和 cancellation token。

### 工具注册

`build_project_scoped_registry()` 构建 core 工具（file/code_search/bash）和 skills 工具。`_extend_registry_with_features()` 添加 worktree/mcp/tasks/mailbox/progress/memory 组工具。`build_search_tools_tool()` 注册 `search_tools` 后追加。最后 `_build_subagent_integration()` 注册 subagent 工具。

### Tool execution

```
StructuredAgent → Agent → agent/tool_execution.py → ToolSpecAdapter → ToolSpec.handler
```

`PermissionEngine` 统一处理：权限决策、HITL 审批、脱敏、审计记录。`tool_execution.py` 把并发安全工具分区并行执行。

---

## 5. Control Surface

### Permission engine

`PermissionDecision`: `allow` / `deny` / `ask`。`PermissionEngine` 以 `SecurityRuntimeConfig` 派生静态策略。

`PermissionEngineConfig` 参数：`static_policy`、`restricted_dirs`、`defer_static_ask`、`shadow_model_enabled`、`project_root`、`session_grant_store`、`permanent_grant_store`、`hook_constraint_providers`。

`SecurityRuntimeConfig`（config.py）：
- `permission_mode`: strict / normal / permissive
- `sandbox_mode`: bool
- `approval_policy`: always / never
- `writable_roots`、`restricted_dirs`
- `rules`: 静态权限规则列表（替换 `deny_tools`/`ask_tools`/`allow_tools`）
- `global_default`: 无规则匹配时的默认决策

规则格式（`StaticPermission`）：
```json
{
  "tool": "bash",
  "decision": "deny",
  "target": null,
  "target_type": null,
  "input_contains": null,
  "input_prefix": null,
  "input_regex": null
}
```

`StaticPolicyEvaluator` 按声明顺序遍历规则，最后一个匹配的规则生效（last-match-wins）。
全局 `PermissionResolver` 优先级不变：`non_bypassable_deny > deny > ask > allow`。

配置示例：
```json
{
  "security": {
    "rules": [
      {"tool": "bash", "decision": "deny"},
      {"tool": "write_file", "decision": "ask"},
      {"tool": "read_file", "decision": "allow"}
    ],
    "global_default": "ask"
  }
}
```

权限配置仅支持 `security.rules` + `security.global_default`。

### HITL

高风险工具或 policy 判定为 `ask` 时触发 approval callback。REPL 中 `ReplHITLHandler` 提供交互式选择（once/session/permanent）。

### Redaction

`redact_text()` 覆盖 API key 和常见 key/secret/token 键值格式。

---

## 6. Isolation Surface

### Plan / Build / Act

REPL 支持三种执行模式。`/act --clear` 把 plan artifact 写入 `.local/session_artifacts/`，创建干净会话并注入 approved plan。

### Subagent

`ManagedSubagentRunner` 控制并发和超时。支持 `worktree` 隔离。子 agent 使用
过滤后的 tool registry，且不包含 subagent 工具，因此当前只允许一层委托，
不存在递归深度控制。

权限边界：子 agent 继承父级的静态策略（rules + global_default）、restricted_dirs、hook_constraint_providers。不继承 `approval_callback`、`session_grant_store`、`permanent_grant_store`，因此 `ask` 决策在子 agent 中退化为硬阻断。`project_root` 正确传递给子 agent 的 `PermissionEngine.boundary_context()`。

### Worktree

Git worktree 沙箱。移除时检查 dirty status 和未合并/未推送提交。

---

## 7. State And Verification Surface

### Layered Compaction

- stale read_file 裁剪
- 大工具输出头尾截断
- 旧 tool_result 微压缩
- transcript 落盘
- older messages summary compact

`memory` group 启用时将压缩摘要交给 `MemoryManager.consolidate()`。

### Read version restoration

压缩/恢复后从历史消息和磁盘文件重建 read-before-edit 指纹。

### Watchdog

重复工具调用签名检测，超过 `watchdog_repeated_tool_limit` 时停止。文件变更后自动清除只读调用历史。

---

## 8. Feature Review

### MCP (`harness/mcp/tools.py` + `client.py`)
- 官方 Python SDK `ClientSession` + stdio transport（仅本地子进程）
- 单一 async owner task 管理 transport/session 完整生命周期，并向同步
  `ToolSpec` handler 提供受控适配
- initialize 协议版本与 server capability 协商
- `.local/mcp_cache.json` schema cache
- `defer_loading` bootstrap/search flow
- 动态 `mcp__{server}__{tool}` 注册
- **有意未启用**: remote transports、OAuth、resources、prompts、sampling、
  elicitation 等非 coding-agent 必需能力

### `tasks` (`experimental/task_store.py`)
- `.local/tasks.json.d/{id}.json` 文件存储
- filelock 保护
- 依赖拓扑排序
- Kanban 渲染
- 工具：`create_task`、`update_task`、`advance_task`、`list_tasks`、`get_task`、`resolve_blocked`

### `worktree` (`experimental/worktree.py`)
- `WorktreeTaskRunner`
- 工具：`create_worktree_task`、`remove_worktree_task`

### `mailbox` (`experimental/mailbox.py`)
- `.local/team/inbox/{agent_id}.jsonl` append-only mailbox
- filelock 写入和读取
- ACK 幂等
- 工具：`send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message`

### `progress` (`experimental/task_progress.py`)
- TaskProgress save/resume
- 租约机制（expire/retry）
- 工具：`save_task_progress`、`resume_task_progress`、`start_task_run`、`resume_task_run`、`retry_task_run`、`expire_task_runs`

### `memory` (`harness/memory/manager.py`)
- `MEMORY.md` H2 block parsing
- required fields validation
- BM25 recall
- metadata reranking
- compaction summary consolidation hook

### `daemon` (`harness/daemon.py`)
- `HeartbeatDaemon`
- 周期检查 mailbox、git dirty、background tasks
- `DaemonHealth` 健康快照
- `ensure_healthy()` 自愈重启

### `execution_env` (`harness/execution_env.py`)
- `ExecutionEnv` protocol
- `SubprocessExecutionEnv`（Popen、线程 drain、进程树清理）
- `SandboxExecutionEnv`（测试 mock）

---

## 9. Evals

四条验证线：
- `pipeline`：离线 eval pipeline 回归（1 任务）
- `tool-policy`：离线工具策略回归（3 任务）
- `coding-fixture`：真实 provider sandbox 编码任务（4 任务）
- `smoke`/`tool`/`context`/`multi`/`plan`：单一维度离线任务

`EvalRunner` 消费 `XcodeApp.aask_stream()` 事件流，生成 trace、JSON/HTML/CSV report。

Grader 分四类：确定性（runtime_error/final_event/answer_contains/expected_tool/disallowed_tool）、文件证据（file_exists/contains/not_contains/changed）、validation command、基于 `StreamProvider` 的 LLM-as-judge。judge 未执行或输出不可解析时显式记录 skipped。

`pass@k` 使用无偏估计量 `1 - C(n-c,k)/C(n,k)`；`pass^k` 为全部 trial 成功。

内置 HumanEval/EvalPlus/MBPP loader：`evals/benchmarks.py`。

CLI `--benchmark` 支持：humaneval、swebench-lite、evalplus-humaneval、evalplus-mbpp。

---

## 10. 已知约束

- `cli/tool_catalog.py` 通过 `_builders()` 和 `CATALOG_COVERED_BUILDERS` 维护 builder 与目录的一致性契约。
- `memory` 实现了 BM25 检索、质量门、冲突合并和 LRU 遗忘策略，但 `consolidate()` 质量门宽松（仅检查 `##` 标题和必需字段关键字），可能接受低质量块。
- `daemon` 由 `build_app()` 构造，生命周期由调用方控制。
- `tasks` + `progress` 支持任务和 checklist，不提供完整可重入长任务编排。
- eval 已接入 HumanEval/MBPP loader；LLM-as-judge 的稳定性和成本仍依赖外部
  provider 治理。
- `ProviderMetricsMixin` 位于 `ai/providers/metrics.py`。

---

## 11. 维护规则

1. 不要绕过 `ToolSpecAdapter` 和 `agent/tool_execution.py` 直接执行工具 handler。
2. 新工具按 group 归类，通过 ToolSpec 声明注册。
3. 不新增 `experimental` package 或 aggregate switch。
4. 新工具必须声明 group、risk、schema、read-only、concurrency 属性。
5. 修改工具或 group 后，应更新 `CONFIG.md`、`README.md`、`docs/code-organization.md` 和相关测试。
