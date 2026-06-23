# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Xcode 是轻量级 Python Agent 运行骨架，四层架构：
`ai/` (Provider) → `agent/` (Loop Core) → `harness/` (Runtime Infra) → `coding_agent/` (Coding Product)

顶层 `cli/` 提供 REPL UI 和 slash command 系统。包名 `xcode*`，位于 `src/`。

运行路径：`main.py` → `build_app()` → `StructuredAgent` → `Agent` loop → provider stream → tool execution。

**入口**：`xcode` CLI / `python -m xcode` / `build_app()` 编程式 API。
**Eval 入口**：`xcode-eval` / `python -m xcode.evals.cli`。

## 常用命令

```powershell
# 安装
uv pip install -e .                          # 运行时
uv pip install -e ".[dev]"                   # 开发依赖（ruff, pyright, pytest）

# 静态检查（lint → format → typecheck）
uv run ruff check src/ --fix
uv run ruff format src/
uv run pyright src/

# 编译检查
uv run python -m compileall src

# 测试
uv run pytest src/xcode/tests/ -q --tb=short                         # 全部测试
uv run pytest src/xcode/tests/test_xcode_file_tools.py -q --tb=short # 单个测试文件

# Eval 回归
uv run python -m xcode.evals.cli --suite all               # 离线回归集合
uv run python -m xcode.evals.cli --suite pipeline           # 单个套件
uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3  # 真实 provider
uv run python -m xcode.evals.cli --list-suites              # 列出可用套件
uv run python -m xcode.evals.cli --list-benchmarks          # 列出 benchmark
```

## 架构要点

### 关键入口文件

| 文件 | 职责 |
|---|---|
| `src/xcode/main.py` | CLI 入口：参数解析 → 配置发现 → `build_app()` → REPL 或单次 prompt |
| `src/xcode/harness/app.py` | `XcodeApp` dataclass 和 `build_app()` 装配编排 |
| `src/xcode/harness/assembly.py` | 实际装配逻辑：config 解析、provider bundle、tool registry、agent 构建 |
| `src/xcode/harness/config.py` | 所有配置 dataclass（9 个）、配置发现栈、合并与序列化 |
| `src/xcode/agent/agent_loop.py` | 无状态 agent loop 核心 |
| `src/xcode/agent/tool_execution.py` | 工具执行调度（读写分区、串行/并行） |
| `src/xcode/coding_agent/registry.py` | `build_project_scoped_registry()` — 构建项目级工具注册表 |
| `src/xcode/harness/agent_runtime/structured.py` | `StructuredAgent` — harness 对 agent loop 的适配器 |

### 四层职责

- **`ai/`** — 多 provider LLM API（OpenAI-compatible 基类 + DeepSeek/ChatGLM/MiMo 适配器）。Provider 注册、缓存统计、工具 schema 稳定化。
- **`agent/`** — 通用 agent loop 合约。消息/事件类型、工具执行分区、context 收集与组装、watchdog 重复检测、request hygiene。
- **`harness/`** — 应用装配、运行时配置、session 存储（JSONL）、权限引擎、审计日志、hook 管理、MCP 集成、Memory、tasks、progress、mailbox、daemon。
- **`coding_agent/`** — coding 产品工具实现：file（read/write/edit）、code_search（glob/find/grep/ls）、bash、worktree。`edit_file` 依赖 read-before-edit 指纹校验。

### 配置发现栈（优先级从低到高）

全局 `~/.xcode/settings.json` → 项目 `xcode.config.json` → 本地 `.local/settings.json` → 环境变量覆盖。

没有配置文件时使用默认配置。配置结构详见 [CONFIG.md](CONFIG.md)。

### 工具组

| group | 工具 |
|---|---|
| `core` | `read_file`, `write_file`, `edit_file`, `glob_files`, `find_files`, `grep_search`, `ls`, `bash`, `search_tools` |
| `skills` | `load_skill`（发现 project/user skill 时自动注册） |
| `subagent` | `submit_subagent`, `check_subagent`, `cancel_subagent` |
| `worktree` | `create_worktree_task`, `remove_worktree_task` |
| `tasks` | `create_task`, `update_task`, `advance_task`, `list_tasks`, `get_task`, `resolve_blocked` |
| `mailbox` | `send_mailbox_message`, `read_mailbox_messages`, `acknowledge_mailbox_message` |
| `progress` | `save_task_progress`, `resume_task_progress`, `start_task_run`, `resume_task_run`, `retry_task_run`, `expire_task_runs` |
| `memory` | `search_memory` + 主动召回 + 压缩摘要 consolidation |
| `daemon` | `HeartbeatDaemon` 后台服务（由 `daemon.enabled` 控制） |
| `mcp` | `mcp__{server}__{tool}`、`mcp_tool_search`（存在 `.local/mcp_config.json` 时自动注册） |

### 权限模型

`PermissionEngine` 统一处理权限决策、HITL 审批和输出脱敏。权限规则通过 `security.rules` 配置，按声明顺序遍历，最后匹配的规则生效（last-match-wins）。全局 resolver 优先级：`non_bypassable_deny > deny > ask > allow`。

## 代码规范

- Python 3.12+，完整类型注解，ruff 格式化（行宽 88），零 `# noqa` / `# type: ignore`
- 注释和 docstring 使用简体中文
- 函数职责单一，纯函数优先，异常捕获具体类型（禁止 bare `except:`）
- `*args`/`**kwargs` 只在不可避免的边界使用；禁止动态 `importlib`/`getattr`/`setattr`
- 依赖变更视为代码变更，需要 review
- 不维护向后兼容，除非用户要求

## Git 规则

- 只允许 `git add <exact-path>`，不允许 `git add -A` / `git add .`
- 禁止历史重写操作（`reset --hard`、`checkout .`、`clean -fd`、`stash`）
- 禁止 `git commit --no-verify` 和 force push
- 每个 commit 只含一个逻辑变更
- 提交前检查：`git status --short && git diff --cached --stat`
- commit message 格式：`type: one-line title` + body（英文）
- 详细 git 工作流见 [docs/git-workflow.md](docs/git-workflow.md)

## 测试

- `pytest-asyncio`，`asyncio_mode = "auto"`，测试位于 `src/xcode/tests/`
- 71 个测试文件覆盖核心装配、provider、runtime、coding tools、observability、REPL 和 evals
- 回归套件（`--suite all`）：`pipeline` + `tool-policy` + `context` + `multi`
- 仅修改文档时：`git diff --check -- <modified-docs>`
- 不运行需要外部环境变量的端到端套件（除非明确要求）

## 引用文档

需要深入了解特定主题时阅读：

| 文档 | 内容 |
|---|---|
| [CONFIG.md](CONFIG.md) | 运行时配置完整参考（provider、agent、security、hooks、tools、prompt 等） |
| [docs/code-organization.md](docs/code-organization.md) | 每个模块的详细职责和文件清单 |
| [docs/source-review.md](docs/source-review.md) | 源码级架构审查：数据流、权限边界、隔离面 |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | Eval 框架、套件、grader、benchmark adapter 使用指南 |
| [docs/git-workflow.md](docs/git-workflow.md) | Git 操作详细规则 |
