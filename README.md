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

围绕结构化事件流、路径安全、工具审批、审计脱敏、上下文压缩和 REPL 会话管理构建的可测试 Agent 运行骨架。零配置即可运行。

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
| `/plan` | 进入 Plan 模式（只读检查，禁止写入） |
| `/build` | 进入 Build 模式（允许写入，高风险需审批） |
| `/act` | 进入 Act 模式（完整权限） |
| `/compact` | 手动触发上下文压缩 |
| `/clear` | 开始新会话 |
| `/fork [type]` | 分支（explore/verify/isolate） |
| `/rewind [n]` | 回退 n 轮 |
| `/resume [last\|id]` | 恢复历史会话 |
| `/sessions` | 列出所有历史会话 |
| `/branch [list\|tree\|id]` | 切换分支 |
| `/tree` | 查看会话树 |
| `/model [name]` | 切换当前模型 |
| `/effort <level>` | 设置推理 effort |
| `/thinking on/off` | 切换 thinking 显示 |
| `/tool [list\|NAME INPUT]` | 查看/调用工具 |
| `/skill NAME` | 显式激活技能 |
| `/memory` | 检索、列出或添加项目级与用户级记忆 |
| `/permissions [revoke\|clear]` | 权限管理 |
| `/hooks` | 查看 hook 状态 |
| `/context` | 查看上下文 token 占用 |
| `/btw` | 侧问题快速问答 |
| `/undo` | 文件级撤销 |
| `/exit\|/quit` | 退出 |
| `$skill-name ...` | 行首 `$` 激活技能并传递任务 |
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
- **REPL 会话管理** — `/plan`、`/build`、`/act`、`/compact`、`/clear`、`/fork`、
  `/rewind`、`/sessions`、`/resume`、`/branch`、`/tree`、`/model`、`/effort`、
  `/thinking`、`/tool`、`/skill`、`/memory`、`/permissions`、`/hooks`、`/undo`、
  `$skill-name` 快捷入口、`!COMMAND` shell 快捷入口、`@file` 引用、
  `/context`、`/btw` 和 session transcript 落盘。
- **Subagent 委托** — `ManagedSubagentRunner` 并行子任务调度，支持 worktree 沙箱隔离和模型 profile 切换。
- **MCP 协议** — 自动发现 `.local/mcp_config.json`，注册 `mcp__{server}__{tool}` 动态工具。
- **任务与协作** — tasks 任务图、mailbox 跨 agent 消息、progress 断点续传、daemon 后台心跳。

---

## 配置

配置发现栈：全局 `~/.xcode/settings.json` → 项目 `xcode.config.json` → 本地
`.local/settings.json` → 环境变量覆盖。所有字段默认值见 [CONFIG.md](CONFIG.md)。

---

## 工具组

| group | 工具 |
|---|---|
| `core` | `read_file`、`write_file`、`edit_file`、`glob_files`、`find_files`、`grep_search`、`ls`、`bash`、`search_tools` |
| `skills` | `load_skill`（发现 skill 时自动注册） |
| `subagent` | `submit_subagent`、`check_subagent`、`cancel_subagent` |
| `worktree` | `create_worktree_task`、`remove_worktree_task` |
| `tasks` | `create_task`、`update_task`、`advance_task`、`list_tasks`、`get_task`、`resolve_blocked` |
| `mailbox` | `send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message` |
| `progress` | `save_task_progress`、`resume_task_progress`、`start_task_run`、`resume_task_run`、`retry_task_run`、`expire_task_runs` |
| `memory` | `search_memory`；主动召回、压缩摘要 consolidation |
| `daemon` | `HeartbeatDaemon`（由 `daemon.enabled` 配置项控制） |
| `mcp` | `mcp__{server}__{tool}`、`mcp_tool_search`（存在 `.local/mcp_config.json` 时自动注册） |

每轮会按用户问题检索项目根 `MEMORY.md` 与用户级
`~/.xcode/memory/MEMORY.md`，将最多 3 条匹配记录注入 `<memory>` 上下文。
`search_memory` 是只读、低风险工具；schema 支持 `query`、`limit`、`scope`
和 `layer`。

---

## 评估与验证

### 单元测试

```powershell
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

详细规范见 [AGENTS.md](AGENTS.md)。

---

## 文档导航

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | Agent 开发入口和 Python 编码规范 |
| [CONFIG.md](CONFIG.md) | 运行时配置参考 |
| [docs/code-organization.md](docs/code-organization.md) | 模块职责与工具组映射 |
| [docs/source-review.md](docs/source-review.md) | 源码级架构审查 |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | 测试和 eval 工作流 |

---

## 许可

[MIT](LICENSE) © 2026 Xcode Contributors
