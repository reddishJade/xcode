# Xcode 代码组织说明

本文描述独立 checkout 的代码布局和模块边界。使用 `src/` 包布局。

---

## 四层模型

```
cli/ ──→ coding_agent/ ──→ harness/ ──→ agent/ ──→ ai/
```

| 层 | 目录 | 职责 |
|---|---|---|
| Provider | `ai/` | 统一多 provider LLM API |
| Loop Core | `agent/` | 通用 agent 循环合约、消息/事件类型 |
| Runtime Infra | `harness/` | 应用装配、config、session、安全、观测、ToolSpec 协议 |
| Coding Product | `coding_agent/` | coding 产品工具（file/search/bash/shell）及 registry |

---

## 顶层结构

```
.
├── pyproject.toml
├── AGENTS.md / CONFIG.md / README.md / TODO.md
├── docs/ examples/ skills/
└── src/xcode/
    ├── main.py → 入口：解析参数 → 配置发现 → build_app() → REPL/--prompt
    ├── __main__.py → `python -m xcode` 入口
    ├── cli/
    ├── coding_agent/
    ├── experimental/
    ├── harness/
    ├── agent/
    ├── ai/
    ├── evals/
    └── tests/
```

---

## 模块详细职责

### `src/xcode/ai/` — Provider 层

| 模块 | 职责 |
|---|---|
| `types.py` | LLM 共享类型：`Model`、`Usage`、`StreamOptions`、`ToolDefinition`、`ThinkingBudgets` |
| `events.py` | provider stream 事件：`TextDelta`、`ReasoningDelta`、`ToolCallEvent`、`FinalMessage` |
| `registry.py` | 模型注册中心：`get_model`、`get_models`、`get_providers`、`resolve_model` |
| `cache.py` | 缓存统计与工具稳定化（规范化、排序、指纹） |
| `model_modes.py` | 模型模式支持 |
| `providers/` | Provider 适配器 |
| `providers/protocol.py` | `ModelProvider` 协议 |
| `providers/factory.py` | `build_provider_bundle`、`ProviderSettings` |
| `providers/_registry.py` | `PROVIDER_REGISTRY` |
| `providers/router.py` | `RouterProvider` |
| `providers/runtime.py` | 重试+限流：`ProviderRuntime` |
| `providers/metrics.py` | `ProviderMetricsMixin` |
| `providers/codec.py` | schema/delta 编解码、跨 provider 消息归一化 |
| `providers/stream_codec.py` | stream delta → event 编解码 |
| `providers/openai_compat.py` | OpenAI Chat 基类 |
| `providers/openai.py` | `OpenAIChatProvider` |
| `providers/deepseek.py` | `DeepSeekProvider` |
| `providers/chatglm.py` | `ChatGLMProvider` |
| `providers/mimo.py` | `MiMoProvider` |
| `providers/faux.py` | 测试用 `FauxProvider` |

### `src/xcode/agent/` — Loop Core 层

| 模块 | 职责 |
|---|---|
| `protocols.py` | `AgentTool`、`CancellationSignal`、`ContentBlock`、`ToolExecutionMode` |
| `messages.py` | 消息类型 union：`SystemMessage`、`UserMessage`、`AssistantMessage`、`ToolResultMessage` |
| `events.py` | `AgentEvent` union（TurnStart/MessageUpdate/ToolExecutionEnd 等） |
| `config.py` | `AgentContext`、`AgentLoopConfig`、`AgentLoopResult`、hook 类型 |
| `types.py` | 基础类型：`ShellCallOutputContent` |
| `agent.py` | `Agent` 封装，包装 `run_agent_loop`，管理 steer/followup 队列 |
| `agent_loop.py` | 无状态 Agent loop |
| `tool_execution.py` | 工具执行调度、串行/并行分区和参数校验 |
| `compaction.py` | Agent 层上下文压缩 |
| `history.py` | 历史记录管理、request hygiene |
| `hooks.py` | Agent 层 hook 点 |
| `message_converter.py` | 消息格式转换 |
| `watchdog.py` | 重复工具调用检测 |
| `_provider.py` | 内部 provider 适配器 |
| `context_collector.py` | `ContextCollectorRegistry`、6 个上下文收集器 |
| `context_assembly.py` | `DefaultContextAssembler`、context block 排序/裁剪 |
| `results.py` | `AgentToolResult`、`ToolResultMessage` |

### `src/xcode/harness/` — Runtime Infra 层

| 模块 | 职责 |
|---|---|
| `app.py` | `XcodeApp` dataclass、`build_app()` 入口 |
| `assembly.py` | 装配：config 解析、shared infra、provider bundle、tool registry、agent 构建、services |
| `config.py` | 9 个 dataclass、配置发现/序列化/合并/环境变量覆盖 |
| `skills.py` | `ToolSpec`、`ToolOutput` dataclass |
| `session.py` | JSONL 会话存储、索引、resume、fork、rewind |
| `execution_env.py` | `ExecutionEnv` protocol、`SubprocessExecutionEnv`、`SandboxExecutionEnv` |
| `daemon.py` | `HeartbeatDaemon` |
| `snapshot.py` | Git tree 文件快照、每轮 pre/post snapshot、undo |
| `worktree.py` | Git worktree 隔离 backend；subagent worktree isolation 依赖 |
| `skill_activation.py` | Skill activation 内容解析 |
| `skills_registry.py` | Skill 发现、索引、懒加载 |
| `session_todo.py` | 主 agent 会话级 `update_todo` 工具与内存状态 |
| `mcp/` | 官方 MCP SDK stdio adapter、schema cache 和动态工具集成 |
| `mcp/client.py` | 官方 Python SDK 的同步生命周期适配层 |
| `mcp/tools.py` | MCP 配置、schema cache 和动态 ToolSpec 构建 |
| `mcp/results.py` | MCP structuredContent 校验和 typed content 宿主映射 |
| `memory/` | Memory 管理与解析 |
| `memory/manager.py` | 项目/用户 `MEMORY.md`、BM25 召回、consolidation 和 LRU |
| `memory/parsing.py` | 记忆块数据类型、解析和评分辅助 |
| `memory/tools.py` | `memory` group 的只读 `search_memory` ToolSpec |
| `agent_runtime/` | StructuredAgent 运行时 |
| `agent_runtime/structured.py` | `StructuredAgent`（harness 对 agent loop 的适配器） |
| `agent_runtime/subagent.py` | `DelegatedTaskRunner`、`delegate_task` 工具 |
| `agent_runtime/compaction.py` | `CompactController`、`LayeredCompactor` |
| `agent_runtime/prompting/` | 系统提示词构造 |
| `agent_runtime/prompting/builder.py` | `SystemPromptBuilder`、`build_runtime_context_provider` |
| `agent_runtime/prompting/identity.py` | `PROMPT_VERSION`、缓存区域定义 |
| `agent_runtime/prompting/token_budget.py` | token 预算管理 |
| `agent_runtime/config.py` | `AgentRuntimeConfig`、`GateConfig` |
| `agent_runtime/events.py` | Agent Runtime 事件类型 |
| `agent_runtime/result.py` | `RunState`、`StructuredAgentResult` |
| `agent_runtime/cancellation.py` | `CancellationToken` |
| `agent_runtime/contextual.py` | `ContextualRetrievalState` |
| `agent_runtime/git_preflight.py` | Git 状态预检 |
| `agent_runtime/message_codec.py` | 消息编解码 |
| `agent_runtime/tool_adapter.py` | 工具适配器 |
| `agent_runtime/tool_audit.py` | 工具审计 |
| `agent_runtime/tool_gate.py` | 工具门禁 |
| `agent_runtime/tool_hooks.py` | 工具 hook 集成 |
| `agent_runtime/agent_helpers.py` | Agent 辅助函数 |
| `agent_runtime/async_worker.py` | 异步工作线程 |
| `agent_runtime/fallback.py` | 模型 fallback |
| `agent_runtime/execution_modes.py` | 执行模式（plan/build/act） |
| `observability/` | 可观测性 |
| `observability/audit.py` | `AuditRecord`、`JsonlAuditLogger`、`redact_text` |
| `observability/correlation.py` | hook、结构化事件与 audit 共享的 session/turn/request/tool-call 关联状态 |
| `observability/hooks.py` | `HookManager`、6 个事件类型 |
| `observability/external_hooks.py` | 外部命令 hook 的 JSON 进程边界、失败策略和诊断状态 |
| `observability/permissions.py` | `PermissionEngine`、`PermissionEngineConfig`、`PermissionPolicy` |
| `observability/permission_model.py` | `Action`、`PermissionResolver`、`GrantStore`、权限模型轴 |
| `observability/_safety_backstop.py` | `SafetyBackstopPolicyEvaluator`、shell 命令三桶分类 |

### `src/xcode/coding_agent/` — Coding Product 层

| 模块 | 职责 |
|---|---|
| `__init__.py` | 导出 `build_project_scoped_registry` |
| `registry.py` | `build_project_scoped_registry()` — 构建项目级工具注册表 |
| `tools/` | 内置工具实现 |
| `tools/file.py` | `build_file_tools` → `read_file`、`write_file`、`edit_file` |
| `tools/code_search.py` | `build_code_tools` → `glob_files`、`find_files`、`grep_search`、`ls` |
| `tools/bash.py` | `build_bash_tool` → `bash` |
| `tools/shell_adapter.py` | `ShellSpec`、`detect_shell`、`build_shell_argv` |
| `tools/path_utils.py` | 路径解析、`is_path_blocked` |
| `tools/file_index.py` | 有时间和数量预算的 `.gitignore` 感知项目文件索引 |
| `tools/truncate.py` | 输出截断 |
| `tools/output_accumulator.py` | 命令输出累积 |
| `tools/tools_manager.py` | 外部工具检测（ripgrep 等） |
| `tools/file_handlers.py` | 文件操作处理 |
| `tools/file_image.py` | 图片文件处理 |
| `tools/file_mutation_queue.py` | 文件变更队列 |
| `tools/edit_diff.py` | diff 编辑 |
| `tools/_constants.py` | 工具常量、风险评估 |

### `src/xcode/experimental/` — 实验能力

该目录下能力默认关闭，由 `experimental.*` 配置显式启用。

| 模块 | 职责 |
|---|---|
| `task_store.py` | `tasks` group：任务存储、依赖排序、Kanban |
| `task_progress.py` | `progress` group：长任务 checklist 和 lease |
| `orchestration_store.py` | progress 运行编排状态 |
| `mailbox.py` | 基于共享本地文件系统的跨进程 mailbox |

### `src/xcode/cli/` — UI 层

| 模块 | 职责 |
|---|---|
| `repl.py` | REPL 主循环和事件流展示编排 |
| `repl_commands.py` | slash command 注册与分发，包括会话、模式、模型、权限、hook、工具和退出命令 |
| `repl_hitl.py` | HITL 审批（once/session/permanent） |
| `repl_rendering.py` | 终端渲染、`LiveMarkdownStream`、`LiveReasoningPreview` |
| `repl_sessions.py` | 会话恢复、历史记录转换 |
| `repl_settings.py` | `/model`、`/effort`、`/thinking`、`/permissions` 命令处理 |
| `repl_tools.py` | `/tool` 解析、`!COMMAND` 快捷入口、事件序列化 |
| `repl_turn_handler.py` | 单轮处理 |
| `commands.py` | `CommandRegistry`、`ReplState`、`CommandContext` |
| `completion.py` | `ReplCompleter` fuzzy 命令/工具补全、项目级 `@file` 索引和 `!shell` 路径补全 |
| `repl_skills.py` | `$skill-name`、`/skill` 激活解析和 transcript 记录 |
| `file_refs.py` | `@relative/path` 解析和文件内容注入 |
| `markdown.py` | `TerminalMarkdownRenderer` |
| `tool_catalog.py` | 工具目录自省 |
| `setup_wizard.py` | 配置向导 |
| `reasoning_effort.py` | reasoning effort 处理 |
| `app_contract.py` | CLI-应用合约类型 |

### `src/xcode/evals/` — 评估框架

| 模块 | 职责 |
|---|---|
| `schema.py` | `EvalTask`、`EvalReport`、`GraderResult`、`TrialResult` |
| `runner.py` | `EvalRunner` — 事件流 eval runner |
| `tasks.py` | 预定义套件：`pipeline`、`tool-policy`、`coding-fixture`、`smoke`、`tool`、`context`、`multi`、`plan`、`all` |
| `cli.py` | CLI 入口 |
| `graders.py` | 确定性 grader |
| `reporting.py` | JSON/HTML 报告 |
| `tracing.py` | JSONL trace 记录 |
| `validation.py` | Validation command grader |
| `benchmarks.py` | 外部 benchmark loader（HumanEval、EvalPlus、MBPP） |
| `sandbox.py` | Sandbox 项目根选择 |
| `adapters/registry.py` | 外部 benchmark adapter registry |
| `adapters/swebench.py` | SWE-bench predictions helper |

## 工具组

`core`（`read_file`、`write_file`、`edit_file`、`glob_files`、`find_files`、`grep_search`、`ls`、`bash`、`search_tools`）
→ `session`（`update_todo`）
→ `skills`（`load_skill`）
→ `subagent`（`delegate_task`）
→ `worktree`（`create_worktree_task`、`remove_worktree_task`、`list_worktrees`、`prune_stale_worktrees`）
→ `tasks`（实验，`experimental.tasks`）
→ `mailbox`（实验，`experimental.mailbox`）
→ `progress`（实验，`experimental.progress`，依赖 tasks）
→ `memory`（`search_memory`、主动召回、consolidation）
→ `daemon`（`HeartbeatDaemon`）
→ `mcp`（`mcp__{server}__{tool}`、`mcp_tool_search`，由 `.local/mcp_config.json` 自动发现）

---

## 本地状态路径

`.local/sessions/`、`.local/session_index.json`、`.local/session_artifacts/`、`.local/mcp_cache.json`、`.local/mcp_config.json`、`.local/tasks.json.d/`、`.local/team/inbox/`（mailbox）、项目根 `MEMORY.md`、用户级 `~/.xcode/memory/MEMORY.md`。

---

## 测试目录

`src/xcode/tests/` 覆盖核心装配、provider、runtime、coding tools、observability、REPL、evals 和全部工具组（71 个测试文件 + fixtures.py + conftest.py）。
