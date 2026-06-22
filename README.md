<div align="center">
  <br/>
  <h1>Xcode Coding Agent</h1>
  <p><strong>轻量级 Python Agent 运行骨架</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-3.12-%23141413?style=flat-square" alt="Python 3.12"/>&nbsp;
    <img src="https://img.shields.io/badge/version-0.1.1-%23d97757?style=flat-square" alt="Version 0.1.1"/>&nbsp;
    <img src="https://img.shields.io/badge/license-MIT-%23788c5d?style=flat-square" alt="License MIT"/>&nbsp;
    <img src="https://img.shields.io/badge/status-active-%236a9bcc?style=flat-square" alt="Status Active"/>
  </p>
  <br/>
</div>

围绕结构化事件流、路径安全、工具审批、审计脱敏、上下文压缩和 REPL 会话管理构建的可测试 Agent 运行骨架。默认配置只启用 `core` 工具组；扩展能力必须显式 opt-in。

---

## 安装

### 前置条件

- Python **3.12** 或更高
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 从源码安装

```powershell
git clone https://github.com/your-org/xcode.git
cd xcode
uv pip install -e .
```

仅安装运行时依赖，不含开发工具。

### 安装开发环境

```powershell
uv pip install -e ".[dev]"
```

开发依赖包括：ruff（格式化/lint）、pyright（类型检查）、pytest（测试框架）。

---

## 快速开始

### 编程式调用

```python
from pathlib import Path
from xcode.harness.app import build_app

app = build_app(project_root=Path.cwd())

answer = app.ask("列出当前目录所有 Python 文件。")
print(answer)

answer = await app.aask("列出当前目录所有 Python 文件。")

app.close()
```

### REPL 交互

```powershell
uv run python -m xcode
```

或通过安装后的命令入口：

```powershell
xcode
```

REPL 支持以下会话命令：

| 命令 | 功能 |
|---|---|
| `/plan` | 查看/编辑当前计划 |
| `/build` | 执行当前计划 |
| `/act` | 切换至执行模式 |
| `/compact` | 手动触发上下文压缩 |
| `/sessions` | 列出所有历史会话 |
| `/resume` | 恢复历史会话 |
| `/branch` | 基于已有会话创建分支 |
| `/queue` | 查看消息队列 |
| `/model` | 切换当前模型 |
| `/tool` | 查看/切换工具组 |
| `/memory` | 检索、列出或添加项目级与用户级记忆 |
| `!COMMAND` | 执行 shell 命令 |
| `@file` | 引用并读取文件内容 |

---

## 核心能力

- **结构化 Agent 循环** — `StructuredAgent` 消费 provider 流式事件，统一处理
  text、reasoning、tool_use、tool_result 和 final answer。
- **核心工具闭环** — 默认提供文件读写编辑、词法搜索和受控 bash。`edit_file`
  依赖 read-before-edit 指纹校验。
- **工具并发分区** — 只读且并发安全的工具并行执行；写操作、高风险命令保持串行。
- **权限与审计** — `PermissionEngine` 统一执行工具权限判定、HITL 审批和输出
  脱敏；`JsonlAuditLogger` 记录审计日志。
- **上下文压缩与恢复** — `LayeredCompactor` 裁剪过期读取、大输出和旧工具结果，
  支持压缩后重建文件指纹。
- **REPL 会话管理** — `/plan`、`/build`、`/act`、`/compact`、`/sessions`、
  `/resume`、`/branch`、`/queue`、`/model`、`/tool`、`!COMMAND` shell 快捷入口、
  `@file` 引用和 session transcript 落盘。
- **MCP 工具集成** — 核心运行时自动读取 `.local/mcp_config.json`；无配置时
  不注册额外工具。
- **可选扩展** — subagent、worktree、tasks、mailbox、progress、memory、daemon。

---

## 配置

运行时配置来自项目根 `xcode.config.json`，没有配置文件时使用默认配置：

```json
{"tools": {"enabled_groups": ["core"]}}
```

配置发现栈：全局 `~/.xcode/settings.json` → 项目 `xcode.config.json` → 本地
`.local/settings.json` → 环境变量覆盖。完整配置说明见 [CONFIG.md](CONFIG.md)。

---

## 工具组

默认 `enabled_groups=["core"]`。可用 group：

| group | 状态 | 工具 |
|---|---|---|
| `core` | 默认 | `read_file`、`write_file`、`edit_file`、`glob_files`、`find_files`、`grep_search`、`ls`、`bash`、`shell`、`search_tools` |
| `skills` | 可选 | `load_skill` |
| `subagent` | 可选 | `submit_subagent`、`check_subagent`、`cancel_subagent` |
| `worktree` | 可选 | `create_worktree_task`、`remove_worktree_task` |
| `tasks` | 可选 | `create_task`、`update_task`、`advance_task`、`list_tasks`、`get_task`、`resolve_blocked` |
| `mailbox` | 可选 | `send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message` |
| `progress` | 可选 | `save_task_progress`、`resume_task_progress`、`start_task_run`、`resume_task_run`、`retry_task_run`、`expire_task_runs` |
| `memory` | 可选 | `search_memory`；启用主动召回、压缩摘要 consolidation |
| `daemon` | 可选 | 构造 `HeartbeatDaemon` |

启用 `memory` 后，每轮会按用户问题检索项目根 `MEMORY.md` 与用户级
`~/.xcode/memory/MEMORY.md`，将最多 3 条匹配记录注入 `<memory>` 上下文。
`search_memory` 是只读、低风险工具；schema 支持 `query`、`limit`、`scope`
和 `layer`。显式 `/memory list|search|add` 命令不依赖工具组启用状态。

---

## 评估与验证

### 单元测试

```powershell
# 运行全部测试
uv run python -m unittest discover src\xcode\tests

# 或使用 pytest
uv run pytest src/xcode/tests -q --tb=short
```

### Eval 工作流

```powershell
# 列出可用 eval 套件
uv run python -m xcode.evals.cli --list-suites

# 运行 pipeline 套件
uv run python -m xcode.evals.cli --suite pipeline

# 运行工具策略套件
uv run python -m xcode.evals.cli --suite tool-policy

# 运行真实模型 eval（需要配置 API key）
uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3

# 基准测试
uv run python -m xcode.evals.cli --list-benchmarks
uv run python -m xcode.evals.cli --real --benchmark evalplus-humaneval --benchmark-path <url> --trials 1
```

更多说明见 [docs/evaluation-guide.md](docs/evaluation-guide.md)。

---

## 开发指南

### 静态检查

```powershell
uv run ruff check src/ --fix
uv run ruff format src/
uv run pyright src/
```

### 代码规范

- Python 3.12+，完整类型注解
- ruff 格式化（行宽 88），零 `# noqa`
- 纯函数优先，职责分离（IO / 计算 / 展示）
- 异常捕获明确具体类型，禁止 bare `except:`

详细规范见 [docs/code-standards.md](docs/code-standards.md)。

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
| [docs/code-standards.md](docs/code-standards.md) | Python 编码规范 |

---

## 许可

[MIT](LICENSE) © 2026 Xcode Contributors
