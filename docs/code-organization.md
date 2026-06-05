# Xcode 代码组织说明

本文描述独立 checkout `xcode` 的代码布局和模块边界。当前项目使用 `src/` 包布局，Python 包、测试和 eval 代码位于 `src/xcode/`。

---

## 分层概览

Xcode 按四层模型组织，依赖方向从左到右：

```text
cli/ ──→ coding_agent/ ──→ harness/ ──→ agent/ ──→ ai/
                                    └────────────────────→ ai/ (layer skip)
```

| 层 | 目录 | 职责 |
|---|------|------|
| Provider | `ai/` | 统一多 provider LLM API |
| Loop Core | `agent/` | 通用 agent 循环合约，中性消息/事件类型 |
| Runtime Infra | `harness/` | 应用装配、config、session、安全、观测、ToolSpec 协议 |
| Coding Product | `coding_agent/` | coding 产品工具（file/search/bash/edit）及产品层逻辑 |

## 顶层结构

```text
.
├── pyproject.toml
├── AGENTS.md
├── CONFIG.md
├── README.md
├── TODO.md
├── docs/
├── examples/
├── skills/
└── src/xcode/
    ├── main.py
    ├── cli/
    ├── coding_agent/
    ├── harness/
    ├── ai/
    ├── evals/
    ├── experimental/
    ├── tests/
    └── agent/
```

`pyproject.toml` 使用 `where = ["src"]` 和 `include = ["xcode*"]`，因此包内引用应指向 `src/xcode/...`。

---

## 运行入口

```text
src/xcode/main.py
  -> parse_args()
  -> discover_runtime_config()
  -> src/xcode/harness/app.py::build_app()
       -> provider bundle
       -> tool registry
       -> StructuredAgent
  -> src/xcode/cli/repl.py::run_repl() 或单次 --prompt
```

`build_app()` 是应用装配入口。装配逻辑委托给 `assembly.py`，后者通过 `coding_agent/registry.py` 构造工具注册表：

---

## 主要目录职责

各层按依赖方向排序：Provider → Loop Core → Runtime Infra → Coding Product → UI。

### `src/xcode/ai/`

**Provider 层**：统一多 provider LLM API，提供 OpenAI、Anthropic、Google 等模型接入。

| 模块 | 职责 |
| --- | --- |
| `types.py` | LLM 可见的共享接口类型，例如 `ToolDefinition` |
| `events.py` | provider stream 输出事件协议 |
| `providers/factory.py` | 根据 profile 构造 provider bundle |
| `providers/protocol.py` | `ModelProvider` 协议 |
| `providers/codec.py` | OpenAI-compatible schema 和 delta 编解码 |
| `providers/openai_compat.py` | OpenAI Chat Completions 适配基类 |
| `providers/openai.py` | OpenAI Chat Completions 和 stateful Responses |
| `providers/anthropic.py` | Anthropic Messages 适配 |
| `providers/deepseek.py` | DeepSeek thinking mode 适配 |
| `providers/chatglm.py` | ChatGLM 适配 |
| `providers/mimo.py` | MiMo 适配 |
| `providers/faux.py` | 测试用 provider |
| `providers/runtime.py` | retry/rate limit 运行期控制 |
| `providers/metrics.py` | provider metrics 追踪 |
| `providers/stream_codec.py` | provider stream delta 到 event 的编解码 |

### `src/xcode/agent/`

**Loop Core 层**：通用 agent 循环合约。定义中性消息类型、事件协议和可注入的 agent 循环。依赖 `ai/` provider 协议。

| 文件 | 职责 |
| --- | --- |
| `protocols.py` | `AgentTool`、`CancellationSignal` 协议；`ContentBlock`、`ToolExecutionMode` 等基础类型 |
| `messages.py` | 消息类型（`SystemMessage`、`AssistantMessage` 等）、`AgentMessage` union、LLM 格式转换 |
| `events.py` | Agent 事件类型（`TurnStartEvent`、`MessageUpdateEvent`、`ToolExecutionEndEvent` 等）和 `AgentEvent` union |
| `config.py` | `AgentLoopConfig`、`AgentLoopResult`、hook 上下文类型、压缩指令、指标 |
| `agent.py` | `Agent` 薄封装，包装 `run_agent_loop` 并管理 steer/followup 队列 |
| `agent_loop.py` | 无状态 Agent loop，provider、工具执行、turn hooks 均通过合约注入 |
| `tool_execution.py` | 工具执行逻辑（串行/并行调度、before/after hook），从 `agent_loop.py` 提取 |

### `src/xcode/harness/`

**Runtime Infra 层**：跨产品应用基础设施。负责配置解析、会话管理、安全策略、观测和 agent 适配器。

| 模块 | 职责 |
| --- | --- |
| `app.py` | 应用装配入口，委托 `assembly.py` |
| `assembly.py` | 装配核心：config 解析、shared infra、provider bundle、experimental services |
| `config.py` | runtime config dataclass、配置读取、相对路径解析 |
| `session.py` | JSONL 会话存储、索引、resume、fork、plan artifact |
| `skills.py` | `ToolSpec`、工具输入解析、HITL 执行和脱敏入口 |
| `execution_env.py` | `ExecutionEnv` protocol、`SubprocessExecutionEnv`（本地子进程）、`SandboxExecutionEnv`（测试 mock） |
| `skill_loader.py` | `SKILL.md` catalog 扫描和 `load_skill` 工具 |
| `agent_runtime/` | `StructuredAgent`（harness 对 agent loop 的适配器）、subagent、prompt、compaction、cancellation |
| `observability/` | audit、permission policy、hook manager |

### `src/xcode/coding_agent/`

**Coding Product 层**：coding 产品工具（file/search/bash/edit）和产品级 registry 构造。

| 模块 | 职责 |
| --- | --- |
| `tools/` | coding 内置工具实现（`read_file`、`write_file`、`edit_file`、`bash`、`grep`、`glob`、`ls`） |
| `registry.py` | coding 产品工具注册表构造 |
| `__init__.py` | 对外门面，re-export 工具 builder（`build_file_tools`、`build_code_tools`、`build_bash_tool`） |

### `src/xcode/cli/`

**UI 层**：终端入口和交互体验。负责 REPL、命令、渲染和会话命令。通过 `harness/` 和 `coding_agent/` 访问下层能力。

| 文件 | 职责 |
| --- | --- |
| `repl.py` | REPL 主循环和事件流展示编排 |
| `repl_commands.py` | slash command 注册表和命令处理 |
| `repl_hitl.py` | REPL 人工授权提示 |
| `repl_rendering.py` | 终端渲染、prompt session、推理预览 |
| `repl_sessions.py` | REPL 会话恢复和历史展示 |
| `repl_settings.py` | 模型、thinking、effort、权限命令处理 |
| `repl_tools.py` | `/tool` 输入解析、`!COMMAND` shell 快捷入口、工具意图摘要、事件序列化 |
| `completion.py` | 命令、工具名、`@file` 和 `!COMMAND` 路径补全 |
| `file_refs.py` | `@relative/path` 解析和文件内容注入 |
| `markdown.py` | 终端 markdown/diff 渲染 |
| `tool_catalog.py` | 构造 REPL 可见工具目录 |
| `setup_wizard.py` | 配置向导 |

### `src/xcode/evals/`

评估框架。

| 文件 | 职责 |
| --- | --- |
| `schema.py` | `EvalTask` 和配置 schema |
| `runner.py` | 事件流 eval runner |
| `sandbox.py` | 真实 provider eval 的 sandbox 项目根选择 |
| `validation.py` | validation command grader |
| `tracing.py` | JSONL trace 记录 |
| `graders.py` | 确定性 grader |
| `reporting.py` | JSON/HTML 报告 |
| `cli.py` | `xcode-eval` / `python -m xcode.evals.cli` |
| `adapters/` | 外部 benchmark adapter registry 与 SWE-bench predictions helper |

### `src/xcode/experimental/`

实验性能力层，默认不加载。每项能力都有独立启用 group，`experimental` group 会展开为全部实验性能力。

| 文件 | group | 职责 |
| --- | --- | --- |
| `worktree.py` | `worktree` | Git worktree 任务隔离；工具：`create_worktree_task`、`remove_worktree_task` |
| `mcp.py` | `mcp` | stdio MCP client、schema cache、动态 MCP tool proxy |
| `mcp_client.py` | internal | MCP stdio JSON-RPC 客户端，被 `mcp.py` 引用 |
| `tasks.py` | `tasks` | JSON/filelock 任务存储、依赖排序、Kanban 视图；工具：`create_task`、`update_task`、`list_tasks`、`get_task` |
| `mailbox.py` | `mailbox` | append-only JSONL mailbox；工具：`send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message` |
| `progress.py` | `progress` | 长任务 checklist 保存/恢复；工具：`save_task_progress`、`resume_task_progress` |
| `memory.py` | `memory` | `MEMORY.md` 记忆块校验、BM25 召回、压缩摘要 consolidation |
| `memory_parsing.py` | internal | 记忆块解析数据类型，被 `memory.py` 引用 |
| `bm25.py` | internal | `memory` 使用的纯 Python BM25Okapi，随 `memory` group 启用 |
| `plugins.py` | `plugins` | `.local/plugins/*.py` 动态加载，收集 tools/hooks/skills |
| `daemon.py` | `daemon` | `HeartbeatDaemon`，轮询 mailbox/git/tasks；`DaemonHealth` 健康快照、`ensure_healthy()` 自愈重启 |

---

## 工具组与默认可见工具

工具实现归 `coding_agent/` 层所有。默认 `enabled_groups=("core",)`，可见工具为：

- `read_file`
- `write_file`
- `edit_file`
- `glob_files`
- `grep_search`
- `ls`
- `bash`

可选非 experimental group：

- `skills`：`load_skill`
- `subagent`：`submit_subagent`、`check_subagent`、`cancel_subagent`

experimental group：

- `worktree`
- `mcp`
- `tasks`
- `mailbox`
- `progress`
- `memory`
- `plugins`
- `daemon`
- `experimental`：展开为全部 experimental group

---

## 本地状态路径

- `.local/sessions/`：REPL transcript JSONL
- `.local/session_index.json`：会话索引
- `.local/session_artifacts/`：Plan artifact
- `.local/mcp_cache.json`：MCP schema cache
- `.local/mcp_config.json`：本地 MCP server 配置
- `.local/tasks.json.d/`：实验性 task store
- `.team/inbox/`：实验性 mailbox
- `.local/plugins/`：实验性插件目录

这些路径由不同模块按需创建；默认 core 路径不会创建 experimental 状态。

---

## 测试目录

`src/xcode/tests/` 覆盖核心装配、provider、runtime、coding tools、observability、REPL、evals 和 experimental 组件。常用命令：

```powershell
uv run python -m unittest discover src\xcode\tests
uv run python -m compileall src
```
