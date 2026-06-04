# Xcode 代码组织说明

本文描述独立 checkout `xcode` 的代码布局和模块边界。当前项目使用 `src/` 包布局，Python 包、测试和 eval 代码位于 `src/xcode/`。

---

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

`build_app()` 是应用装配入口。装配逻辑主要委托给 `assembly.py`，后者负责：

- 读取 runtime config
- 构造 shared infra（`ContextualRetrievalState`、`CancellationToken`、`CompactController`）
- 构造 provider bundle
- 构造 core tools 和 opt-in tools
- 构造 tool registry
- 构造 `StructuredAgent`
- 按配置连接 compactor、audit logger、hooks、subagent runner、experimental 组件

---

## 主要目录职责

### `src/xcode/cli/`

交互层，负责终端体验和会话命令。

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

### `src/xcode/harness/`

核心运行层，负责 Agent 生命周期、工具协议、安全和观测。

| 模块 | 职责 |
| --- | --- |
| `app.py` | 应用装配入口，委托 `assembly.py` |
| `assembly.py` | 装配核心：config 解析、shared infra、tool registry、provider bundle、experimental services |
| `config.py` | runtime config dataclass、配置读取、相对路径解析 |
| `session.py` | JSONL 会话存储、索引、resume、fork、plan artifact |
| `skills.py` | `ToolSpec`、工具输入解析、HITL 执行和脱敏入口 |
| `skill_loader.py` | `SKILL.md` catalog 扫描和 `load_skill` 工具 |
| `adapters/` | harness 类型到下层协议的适配，例如 `ToolSpec` -> provider tool schema |
| `agent_runtime/` | `StructuredAgent`、工具执行结果、subagent、prompt、compaction、cancellation |
| `tools/` | core file/search/bash tools |
| `observability/` | audit、permission policy、hook manager |

### `src/xcode/ai/`

模型传输和 provider 适配层。

| 模块 | 职责 |
| --- | --- |
| `types.py` | LLM 可见的共享接口类型，例如 `ToolDefinition` |
| `events.py` | provider stream 输出事件协议 |
| `providers/factory.py` | 根据 profile 构造 provider bundle |
| `providers/protocol.py` | `ModelProvider` 协议 |
| `providers/codec.py` | OpenAI-compatible schema 和 delta 编解码 |
| `providers/openai_compat.py` | OpenAI Chat Completions 兼容基类 |
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

纯 Agent core 层，负责中性消息类型、消息转换和可注入的 Agent 循环合约。该层可以依赖 `ai/` provider 协议，但不依赖 `harness/`、`cli/` 或 Xcode 运行时配置。

| 文件 | 职责 |
| --- | --- |
| `types.py` | Agent 消息、事件、工具运行时协议、loop callback 合约 |
| `messages.py` | Agent message 到 LLM message 的转换 |
| `agent.py` | Agent 薄封装，包装 `run_agent_loop` |
| `agent_loop.py` | 无状态 Agent loop，provider、工具执行、turn hooks 均通过合约注入 |
| `provider_response.py` | Provider 响应类型 |
| `tool_execution.py` | 工具执行逻辑，从 `agent_loop.py` 提取 |

### `src/xcode/evals/`

评估框架。

| 文件 | 职责 |
| --- | --- |
| `schema.py` | `EvalTask` 和配置 schema |
| `runner.py` | 事件流 eval runner |
| `tracing.py` | JSONL trace 记录 |
| `graders.py` | 确定性 grader |
| `reporting.py` | JSON/HTML 报告 |
| `cli.py` | `xcode-eval` / `python -m xcode.evals.cli` |

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
| `bm25.py` | internal | `memory` 使用的纯 Python BM25Okapi，不单独作为启用入口 |
| `plugins.py` | `plugins` | `.local/plugins/*.py` 动态加载，收集 tools/hooks/skills |
| `daemon.py` | `daemon` | `HeartbeatDaemon`，轮询 mailbox/git/tasks |
| `speculation.py` | `speculation` | 无副作用 UI 预热事件规划 |

---

## 工具组与默认可见工具

默认 `enabled_groups=("core",)`，可见工具为：

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
- `speculation`
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

`src/xcode/tests/` 覆盖核心装配、provider、runtime、tools、observability、REPL、evals 和 experimental 组件。常用命令：

```powershell
uv run python -m unittest discover src\xcode\tests
uv run python -m compileall src
```
