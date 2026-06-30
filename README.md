<div align="center">
  <br/>
  <h1>
    <code style="color:#141413; background:#e8e6dc; padding:0 12px; border-radius:4px;">xcode</code>
  </h1>
  <p style="font-size:1.2em; color:#141413;">
    <strong>轻量级 Python Agent 运行骨架</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/python-3.12-%23141413?style=flat-square" alt="Python 3.12"/>&nbsp;
    <img src="https://img.shields.io/badge/version-0.1.1-%23d97757?style=flat-square" alt="Version 0.1.1"/>&nbsp;
    <img src="https://img.shields.io/badge/license-MIT-%23788c5d?style=flat-square" alt="License MIT"/>&nbsp;
    <img src="https://img.shields.io/badge/status-active-%236a9bcc?style=flat-square" alt="Status Active"/>
  </p>
  <br/>
</div>

围绕结构化事件流、路径安全、工具审批、审计脱敏、上下文压缩和 REPL 会话管理构建的可测试 Agent 运行骨架。**零配置即可运行。**

---

## 安装

### 前置条件

- Python **3.12** 或更高
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 从源码安装（开发模式）

```powershell
git clone https://github.com/your-org/xcode.git
cd xcode
uv pip install -e .
```

以 editable 模式安装到当前项目虚拟环境，源码修改即时生效。

### 全局安装（uv tool）

```powershell
uv tool install --python 3.12 <path-to-xcode>
```

安装后 `xcode` 成为系统级 CLI 命令，任意目录下均可调用。升级：

```powershell
uv tool upgrade xcode --no-cache
```

### 安装开发环境

```powershell
uv pip install -e ".[dev]"
```

开发依赖包括：ruff（格式化/lint）、pyright（类型检查）、pytest（测试框架）。

---

## 打包为独立二进制

使用 PyInstaller 将 xcode 打包为免 Python 环境的可执行文件：

```powershell
# 安装打包依赖
uv pip install -e ".[pack]"

# 打包（onedir 模式，启动快、更新方便）
uv run pyinstaller --onedir --name xcode --paths src src/xcode/__main__.py

# 产出在 dist/xcode/
# 直接运行（Windows）：
.\dist\xcode\xcode.exe --help
# 或（Linux/macOS）：
./dist/xcode/xcode --help
```

| 模式 | 产出 | 启动速度 | 适用场景 |
|---|---|---|---|
| `--onedir`（默认） | `dist\xcode\` 目录（exe + 依赖） | 无延迟 | 开发/调试、频繁更新 |
| `--onefile` | 单个 `dist\xcode.exe` | 慢 1-3 秒（需解压） | 分发给终端用户 |

`onedir` 模式下，依赖层不变时只需重新打包主 exe，`_internal\` 目录可复用。

---

## 快速开始

### 编程式调用

```python
from pathlib import Path
from xcode.harness.app import build_app

app = build_app(project_root=Path.cwd())

answer = app.ask("列出当前目录所有 Python 文件。")
print(answer)
```

应用配置通过 `build_app()` 参数、配置文件或环境变量注入。详细配置见 [CONFIG.md](CONFIG.md)。

### CLI 交互

```powershell
# 直接提问（单轮，自动退出）
xcode "列出当前目录所有 Python 文件。"

# REPL 模式（多轮对话，支持 /slash 命令）
xcode

# 自定义配置
xcode --config .local/settings.json
```

### REPL 命令概览

| 命令 | 功能 |
|---|---|
| `/plan [目标]` | 制定执行计划 |
| `/build` | 执行当前 `plan.md` |
| `/act [需求]` | plan + build 二合一 |
| `/compact` | 手动触发上下文压缩 |
| `/clear` | 清屏 |
| `/fork [消息序号]` | 从指定消息分支新会话 |
| `/rewind [N]` | 撤销最近 N 轮交互 |
| `/sessions` | 列出所有历史会话 |
| `/branch [list\|tree\|id]` | 切换分支 |
| `/tree` | 查看会话树 |
| `/model [provider/model[:thinking_level]]` | 切换模型 |
| `/effort <level>` | 设置推理 effort |
| `/thinking on/off` | 切换 thinking 显示 |
| `/tool [list\|NAME INPUT]` | 查看/调用工具 |
| `/skill NAME` | 显式激活技能 |
| `/memory` | 检索、列出或添加记忆 |
| `/permissions [list\|clear]` | 查看或清除权限授权 |
| `/hooks` | 查看 hook 状态 |
| `/context` | 查看上下文 token 占用 |
| `/btw` | 侧问题快速问答 |
| `/undo` | 文件级撤销 |
| `/exit` | 退出（`/quit` 为隐藏别名） |
| `$skill-name ...` | 行首 `$` 激活技能并传递任务 |
| `!COMMAND` | 执行 shell 命令 |
| `@file` | 引用并读取文件内容 |

---

## 核心能力

- **结构化 Agent 循环** — `StructuredAgent` 消费 provider 流式事件，统一处理 text、reasoning、tool_use、tool_result 和 final answer。
- **核心工具闭环** — 默认提供文件读写编辑、词法搜索和受控 bash。`edit_file` 依赖 read-before-edit 指纹校验。
- **工具并发分区** — 只读且并发安全的工具并行执行；写操作、高风险命令保持串行。
- **权限与审计** — `PermissionEngine` 统一执行工具权限判定、HITL 审批和输出脱敏；`JsonlAuditLogger` 记录审计日志。
- **上下文压缩与恢复** — `LayeredCompactor` 裁剪过期读取、大输出和旧工具结果，支持压缩后重建文件指纹。
- **REPL 会话管理** — 丰富的 `/slash` 命令体系，支持 plan/build/act、会话分支、回退、模型切换、session transcript 落盘。
- **Subagent 委托** — `delegate_task` 单入口委派子任务，实时流式展示子 agent 进度；支持正式的 worktree 隔离。
- **MCP 协议** — 基于官方 Python SDK 连接本地 stdio server，自动发现
  `.local/mcp_config.json` 并注册 `mcp__{server}__{tool}` 动态工具。
- **实验能力** — 可显式启用 tasks、mailbox 和 progress 断点续传；默认全部关闭。

---

## 工具组

| Group | Tools |
|---|---|
| `core` | `read_file`, `write_file`, `edit_file`, `glob_files`, `find_files`, `grep_search`, `ls`, `bash`, `search_tools` |
| `skills` | `load_skill`（发现 skill 时自动注册） |
| `subagent` | `delegate_task` |
| `worktree` | `create_worktree_task`, `remove_worktree_task`, `list_worktrees`, `prune_stale_worktrees` |
| `tasks` | 实验：`experimental.tasks=true` |
| `mailbox` | 实验：`experimental.mailbox=true` |
| `progress` | 实验：`experimental.progress=true`，且要求 `tasks=true` |
| `memory` | `search_memory`；主动召回、压缩摘要 consolidation |
| `daemon` | `HeartbeatDaemon`（由 `daemon.enabled` 配置项控制） |
| `mcp` | `mcp__{server}__{tool}`, `mcp_tool_search`（存在 `.local/mcp_config.json` 时自动注册） |

每轮会按用户问题检索项目根 `MEMORY.md` 与用户级 `~/.xcode/memory/MEMORY.md`，将最多 3 条匹配记录注入 `<memory>` 上下文。`search_memory` 是只读、低风险工具。

---

## 配置

配置发现栈（优先级从低到高）：

```
~/.xcode/settings.json          ← 全局默认
     ↓
xcode.config.json               ← 项目级
     ↓
.local/settings.json            ← 本地覆盖
     ↓
环境变量                          ← 最高优先级
```

所有字段默认值及完整参考见 [CONFIG.md](CONFIG.md)。

---

## 架构

四层架构，自底向上：

| Layer | 职责 |
|---|---|
| `ai/` | 多 provider LLM API（OpenAI-compatible 基类 + DeepSeek/ChatGLM/MiMo 适配器） |
| `agent/` | 通用 agent loop 合约：消息/事件类型、工具执行分区、上下文收集、watchdog |
| `harness/` | 应用装配、运行时配置、session 存储、权限引擎、审计日志、MCP 集成 |
| `coding_agent/` | Coding 产品工具实现：file、code_search、bash、worktree |
| `cli/` | REPL UI 和 slash command 系统 |

运行路径：`main.py` → `build_app()` → `StructuredAgent` → `Agent` loop → provider stream → tool execution。

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

# 运行 tool-policy 套件
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
