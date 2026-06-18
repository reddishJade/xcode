# Xcode Coding Agent Harness

Xcode 是一个轻量级 Python coding agent harness。围绕结构化事件流、路径安全、工具审批、审计脱敏、上下文压缩和 REPL 会话管理提供一个可测试的 Agent 运行骨架。

默认配置只启用 `core` 工具组。扩展能力必须显式 opt-in。

---

## 核心能力

- **结构化 Agent 循环**：`StructuredAgent` 消费 provider 流式事件，统一处理 text、reasoning、tool_use、tool_result 和 final answer。
- **核心工具闭环**：默认提供文件读写编辑、词法搜索和受控 bash。`edit_file` 依赖 read-before-edit 指纹校验。
- **工具并发分区**：只读且并发安全的工具并行执行；写操作、高风险命令保持串行。
- **权限与审计**：`PermissionEngine` 统一执行工具权限判定、HITL 审批和输出脱敏；`JsonlAuditLogger` 记录审计日志。
- **上下文压缩与恢复**：`LayeredCompactor` 裁剪过期读取、大输出和旧工具结果，支持压缩后重建文件指纹。
- **REPL 会话管理**：支持 `/plan`、`/build`、`/act`、`/compact`、`/sessions`、`/resume`、`/branch`、`/queue`、`/model`、`/tool`、`!COMMAND` shell 快捷入口、`@file` 引用和 session transcript 落盘。
- **可选扩展**：subagent、worktree、tasks、mailbox、progress、daemon；MCP、memory、plugins 位于 `experimental/`。

---

## 目录结构

```text
.
├── pyproject.toml
├── AGENTS.md
├── CONFIG.md
├── README.md
├── docs/
├── examples/
├── skills/
└── src/xcode/
    ├── main.py
    ├── cli/            → REPL、命令、渲染、补全
    ├── coding_agent/   → 产品工具（file/search/bash/shell）
    ├── harness/        → 应用装配、config、安全、观测、session
    │   ├── agent_runtime/ → StructuredAgent、subagent、prompting
    │   └── observability/ → audit、hooks、permissions
    ├── agent/          → Agent 循环、消息/事件类型、工具执行调度
    ├── ai/             → LLM provider 抽象、流事件协议
    │   └── providers/  → DeepSeek、ChatGLM、MiMo、OpenAI 等适配
    ├── evals/
    ├── experimental/   → MCP、memory、plugins
    └── tests/
```

---

## 快速开始

```powershell
uv pip install -e .
uv run python -m unittest discover src\xcode\tests
uv run xcode
uv run xcode --prompt "说明项目的核心工具。"
uv run xcode --resume
```

---

## 配置

运行时配置来自项目根 `xcode.config.json`，没有配置文件时使用默认配置：

```json
{"tools": {"enabled_groups": ["core"]}}
```

配置发现栈：全局 `~/.xcode/settings.json` → 项目 `xcode.config.json` → 本地 `.local/settings.json` → 环境变量覆盖。完整配置说明见 [CONFIG.md](CONFIG.md)。

---

## 工具组

默认 `enabled_groups=["core"]`。可用 group：

| group | 状态 | 工具 |
|---|---|---|
| `core` | 默认 | `read_file`、`write_file`、`edit_file`、`glob_files`、`find_files`、`grep_search`、`ls`、`bash`、`shell`、`search_tools` |
| `skills` | 可选 | `load_skill` |
| `subagent` | 可选 | `submit_subagent`、`check_subagent`、`cancel_subagent` |
| `worktree` | 可选 | `create_worktree_task`、`remove_worktree_task` |
| `mcp` | experimental | 动态 `mcp__{server}__{tool}`、`mcp_tool_search` |
| `tasks` | 可选 | `create_task`、`update_task`、`advance_task`、`list_tasks`、`get_task`、`resolve_blocked` |
| `mailbox` | 可选 | `send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message` |
| `progress` | 可选 | `save_task_progress`、`resume_task_progress`、`start_task_run`、`resume_task_run`、`retry_task_run`、`expire_task_runs` |
| `memory` | experimental | 启用 `MemoryManager` 压缩摘要 consolidation |
| `plugins` | experimental | 扫描 `.local/plugins/*.py` 动态加载 |
| `daemon` | 可选 | 构造 `HeartbeatDaemon` |
| `experimental` | 总开关 | 展开为 `mcp`、`memory`、`plugins` |

---

## MCP

MCP server 配置放在 `.local/mcp_config.json`：

```json
{"mcpServers": {"demo": {"command": "python", "args": ["path/to/server.py"], "defer_loading": true}}}
```

只有启用 `mcp` 或 `experimental` 后才会读取。schema 缓存在 `.local/mcp_cache.json`。

---

## 编程式使用

```python
from pathlib import Path
from xcode.harness.app import build_app

app = build_app(project_root=Path.cwd())
answer = app.ask("Find all Python test files.")
print(answer)

answer = await app.aask("Find all Python test files.")
app.close()
```

---

## 评估与验证

```powershell
uv run python -m unittest discover src\xcode\tests
uv run python -m compileall src
uv run python -m xcode.evals.cli --list-suites
uv run python -m xcode.evals.cli --suite pipeline
uv run python -m xcode.evals.cli --suite tool-policy
uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3
uv run python -m xcode.evals.cli --list-benchmarks
uv run python -m xcode.evals.cli --real --benchmark evalplus-humaneval --benchmark-path <url> --trials 1
```

更多说明见 [docs/evaluation-guide.md](docs/evaluation-guide.md)。

---

## 文档导航

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | 本仓库 Agent 开发入口和约束 |
| [CONFIG.md](CONFIG.md) | 运行时配置参考 |
| [docs/code-organization.md](docs/code-organization.md) | 模块职责与工具组映射 |
| [docs/source-review.md](docs/source-review.md) | 源码级架构审查 |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | 测试和 eval 工作流 |
| [docs/api-reference.md](docs/api-reference.md) | 公开 API 参考 |
