# Xcode Coding Agent Harness

Xcode 是一个轻量级 Python coding agent harness。默认运行模式只启用少量核心工具，围绕结构化事件流、路径安全、工具审批、审计脱敏、上下文压缩和 REPL 会话管理提供一个可测试的 Agent 运行骨架。

默认配置只启用 `core` 工具组。扩展能力必须显式 opt-in：转正能力通过各自 group 启用，`experimental` 总开关只打开仍在实验状态的能力。

---

## 核心能力

- **结构化 Agent 循环**：`StructuredAgent` 消费 provider 流式事件，统一处理 `text_delta`、`reasoning_delta`、`tool_use`、`tool_result` 和 final answer。
- **核心工具闭环**：默认提供文件读写编辑、词法搜索和受控 bash。`edit_file` 依赖 read-before-edit 指纹校验，避免基于过期文件内容写入。
- **工具并发分区**：只读且并发安全的工具可以并行执行；写操作、高风险命令和需要审批的工具保持串行顺序。
- **权限与审计**：`run_tool_result()` 统一执行工具权限判定、HITL 审批和输出脱敏；`JsonlAuditLogger` 可把工具调用写入本地审计日志。
- **上下文压缩与恢复**：`LayeredCompactor` 裁剪过期读取、大输出和旧工具结果；`restore_read_versions` 在压缩或恢复后重建文件指纹状态。
- **REPL 会话管理**：支持 `/plan`、`/review`、`/act`、`/compact`、`/sessions`、`/resume`、`/branch`、`/queue`、`/model`、`/tool`、`!COMMAND` shell 快捷入口、`@file` 引用和 session transcript 落盘。
- **可选扩展**：tasks、mailbox、progress、daemon 位于 `harness/`，worktree 位于 `coding_agent/tools/`，MCP、memory 和 plugins 仍位于 `experimental/`；这些能力默认不加载。

---

## 目录结构

```text
.
├── pyproject.toml
├── AGENTS.md
├── CLAUDE.md
├── CONFIG.md
├── README.md
├── docs/
├── examples/
├── skills/
└── src/xcode/
    ├── main.py
    ├── cli/
    ├── harness/      → 应用装配、工具协议、安全、观测
    ├── agent/        → Agent 协议、消息/事件类型、循环配置、无状态循环
    ├── ai/           → LLM provider 抽象、流事件协议
    ├── evals/
    ├── experimental/  → 仍在审核的动态插件等研究性能力
    └── tests/
```

三个核心模块的详细说明见各目录下的 `README.md`。其余模块说明见 [docs/code-organization.md](docs/code-organization.md)。

---

## 快速开始

在独立 checkout 根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

运行离线测试：

```powershell
uv run python -m unittest discover src\xcode\tests
```

启动 REPL：

```powershell
.\.venv\Scripts\python.exe -m xcode.main
```

单次 prompt：

```powershell
.\.venv\Scripts\python.exe -m xcode.main --prompt "说明当前项目的核心工具。"
```

恢复最近会话：

```powershell
.\.venv\Scripts\python.exe -m xcode.main --resume
```

---

## 配置

运行时配置来自项目根目录的 `xcode.config.json`。没有配置文件时使用默认配置：

```json
{
  "tools": {
    "enabled_groups": ["core"]
  }
}
```

常用配置示例：

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "deepseek_chat",
        "chat_model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": ""
      },
      "subagent": {
        "chat_model": "deepseek-v4-flash"
      },
      "fallback": {
        "chat_model": "deepseek-v4-flash"
      }
    }
  },
  "agent": {
    "max_steps": 20,
    "compact_threshold": 8,
    "compact_token_threshold": 12000,
    "max_recent_messages": 10,
    "tool_workers": 4,
    "watchdog_repeated_tool_limit": 3
  },
  "tools": {
    "enabled_groups": ["core", "mcp"],
    "bash": {
      "network_commands": "ask",
      "shell": "auto"
    }
  },
  "paths": {
    "sessions_dir": ".local/sessions",
    "skills_dir": "skills"
  },
  "observability": {
    "audit_path": ".local/audit.jsonl"
  }
}
```

完整配置说明见 [CONFIG.md](CONFIG.md)。

---

## 工具组

默认 `enabled_groups=["core"]`。可选 group：

- `core`：`read_file`、`write_file`、`edit_file`、`glob_files`、`grep_search`、`ls`、`bash`
- `skills`：`load_skill`
- `subagent`：`submit_subagent`、`check_subagent`、`cancel_subagent`
- `worktree`：`create_worktree_task`、`remove_worktree_task`
- `mcp`：根据 `.local/mcp_config.json` 或 `mcp_config.json` 动态生成 `mcp__server__tool`
- `tasks`：`create_task`、`update_task`、`list_tasks`、`get_task`
- `mailbox`：`send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message`
- `progress`：`save_task_progress`、`resume_task_progress`
- `memory`：启用压缩摘要到 `MEMORY.md` 的 consolidation hook
- `plugins`：扫描 `.local/plugins/*.py` 并注册 `exposed_tools` / `exposed_hooks` / `exposed_skills`
- `daemon`：构造 `HeartbeatDaemon`
- `experimental`：展开为 `mcp`、`memory` 和 `plugins`

`skills` 由 `skills` group 启用；`skills.auto_trigger=true` 随该 group 生效。

---

## MCP

MCP server 配置不放在 `xcode.config.json` 中，而是放在 `.local/mcp_config.json` 或项目根 `mcp_config.json`：

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {},
      "defer_loading": true
    }
  }
}
```

只有启用 `mcp` 或 `experimental` 后，Xcode 才会读取 MCP 配置。MCP schema 缓存在 `.local/mcp_cache.json`，缓存随 command、args、env hash 失效。

---

## 编程式使用

```python
from pathlib import Path
from xcode.harness.app import build_app

app = build_app(project_root=Path.cwd())
answer = app.ask("Find all Python test files and summarize their checks.")
print(answer)
```

异步入口：

```python
answer = await app.aask("Find all Python test files and summarize their checks.")
```

调用方完成后可以显式关闭后台 runner：

```python
app.close()
```

---

## 评估与验证

完整单元测试：

```powershell
uv run python -m unittest discover src\xcode\tests
```

编译检查：

```powershell
uv run python -m compileall src
```

离线 eval：

```powershell
uv run python -m xcode.evals.cli --list-suites
uv run python -m xcode.evals.cli --suite pipeline
uv run python -m xcode.evals.cli --suite tool-policy
```

真实 provider coding fixture eval：

```powershell
uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3
```

外部 benchmark adapter 目标：

```powershell
uv run python -m xcode.evals.cli --list-benchmarks
uv run python -m xcode.evals.cli --real --benchmark evalplus-humaneval --benchmark-path https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-Mini.jsonl.gz --trials 1
```

更多说明见 [docs/evaluation-guide.md](docs/evaluation-guide.md)。

---

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 本仓库 Agent 开发入口和约束 |
| [CONFIG.md](CONFIG.md) | 运行时配置参考 |
| [docs/code-standards.md](docs/code-standards.md) | 详细代码质量、依赖、验证规则 |
| [docs/git-workflow.md](docs/git-workflow.md) | 多会话 Git 工作流和提交边界 |
| [docs/code-organization.md](docs/code-organization.md) | 模块职责与工具组映射 |
| [docs/source-review.md](docs/source-review.md) | 源码级架构审查 |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | 测试和 eval 工作流 |
