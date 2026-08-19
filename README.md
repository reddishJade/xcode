<div align="center">
  <br/>
  <h1>
    <code style="color:#141413; background:#e8e6dc; padding:0 12px; border-radius:4px;">xcode</code>
  </h1>
  <p style="font-size:1.2em; color:#141413;">
    <strong>轻量级 Python Coding Agent 运行骨架</strong>
  </p>
  <p>
    <img src="https://img.shields.io/badge/python-3.12-%23141413?style=flat-square" alt="Python 3.12"/>&nbsp;
    <img src="https://img.shields.io/badge/version-0.1.2-%23d97757?style=flat-square" alt="Version 0.1.2"/>&nbsp;
    <img src="https://img.shields.io/badge/license-MIT-%23788c5d?style=flat-square" alt="License MIT"/>&nbsp;
    <img src="https://img.shields.io/badge/status-active-%236a9bcc?style=flat-square" alt="Status Active"/>
  </p>
  <br/>
</div>

围绕结构化事件流、执行模式、路径安全、工具审批、审计脱敏、上下文压缩、REPL/TUI 会话和记忆系统构建的可测试 Coding Agent 运行骨架。**零配置即可运行。**

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

## 快速开始

### 编程式调用

```python
from pathlib import Path
from xcode.coding_agent.app import build_app

app = build_app(project_root=Path.cwd())

answer = app.ask("列出当前目录所有 Python 文件。")
print(answer)
```

应用配置通过 `build_app()` 参数、配置文件或环境变量注入。详细配置见 [CONFIG.md](CONFIG.md)。

### CLI 子命令

```powershell
# 直接提问（单轮，自动退出）
xcode "列出当前目录所有 Python 文件。"

# TUI 全屏终端界面（默认启动方式）
xcode

# 显式启动 TUI
xcode tui

# CLI / REPL 模式（多轮对话，支持 /slash 命令）
xcode cli

# 管理 provider API 配置
xcode config list
xcode config add main
xcode config set main chat_model deepseek-v4-flash

# 首次使用引导
xcode setup

# 自定义配置
xcode --config .local/settings.json

# 恢复最近会话
xcode --resume
```

### REPL 命令概览

| 命令 | 组 | 功能 |
|---|---|---|
| `/plan [目标]` | 模式控制 | 进入 Plan 模式（只读） |
| `/build` | 模式控制 | 进入 Build 模式 |
| `/act [需求]` | 模式控制 | plan + build 二合一 |
| `/verbose` | 模式控制 | 设置日志详细度 |
| `/debug` | 模式控制 | 切换 debug 模式 |
| `/steer` | 模式控制 | 注入实时引导 |
| `/queue` | 模式控制 | 设置忙时消息策略，或在当前 run 后排队新 run |
| `/help` | 信息工具 | 显示帮助 |
| `/compact` | 会话回滚 | 手动触发上下文压缩 |
| `/rewind [N]` | 会话回滚 | 撤销最近 N 轮交互 |
| `/undo [N\|--list]` | 会话回滚 | 文件级撤销（快照恢复） |
| `/clear` | 会话生命周期 | 开始新会话 |
| `/continue` | 会话生命周期 | 恢复本项目最近会话 |
| `/new` | 会话生命周期 | `/clear` 别名 |
| `/resume` | 会话生命周期 | 选择历史会话恢复 |
| `/sessions` | 会话生命周期 | 列出历史会话 |
| `/rename` | 会话生命周期 | 重命名当前会话 |
| `/fork [消息序号]` | 会话分支 | 从指定消息分支新会话 |
| `/clone` | 会话分支 | 克隆当前会话为新文件 |
| `/tree` | 会话分支 | 查看会话分叉树 |
| `/model` | 模型配置 | 显示/切换当前模型 |
| `/effort <level>` | 模型配置 | 设置推理 effort |
| `/thinking on/off` | 模型配置 | 切换 thinking 显示 |
| `/config` | 模型配置 | 管理 provider profile |
| `/tool [list\|NAME INPUT]` | 信息工具 | 查看/调用工具 |
| `/skill NAME` | 信息工具 | 显式激活技能 |
| `/memory` | 信息工具 | 列出、检索或显式添加长期记忆 |
| `/permissions [list\|clear]` | 信息工具 | 查看或清除权限授权 |
| `/hooks` | 信息工具 | 查看外部 hook 状态 |
| `/mcp status\|reload` | 信息工具 | 查看 MCP 状态或重载 |
| `/context` | 信息工具 | 查看上下文 token 占用 |
| `/btw` | 信息工具 | 侧问题快速问答 |
| `/exit` | 退出 | 退出 REPL |
| `$skill-name ...` | — | 行首 `$` 激活技能并传递任务 |
| `!COMMAND` | — | 执行 shell 命令 |
| `@file` | — | 引用并读取文件内容 |

---

## 核心能力

- **结构化 Agent 循环** — `CodingAgentHarness` 消费 provider 流式事件，统一处理 text、reasoning、tool_use、tool_result 和 final answer。
- **可回放事实账本** — session 以 append-only 事件记录用户输入、provider 实际请求、工具语义、compaction epoch、子代理生命周期和最终回答。
- **三执行模式** — `plan`（只读）、`build`（自动执行并由独立 reviewer 审批边界动作）、`act`（边界动作询问用户），规则引擎按 findLast 覆盖权限。
- **核心工具闭环** — 内置文件读写编辑、glob/grep/bash/subagent/webfetch/websearch/question/todowrite 等工具。`edit_file` 依赖 read-before-edit SHA256 指纹校验。
- **工具并发分区** — 只读且并发安全的工具并行执行；写操作、高风险命令保持串行。
- **权限与审计** — `PermissionEngine` 统一执行工具权限判定、自动/人工审批和输出脱敏；`JsonlAuditLogger` 记录审计日志；Build 中需要 review 的 shell 动作不会暂停询问用户。
- **上下文压缩与恢复** — `LayeredCompactor` 裁剪过期读取、大输出和旧工具结果；compact 后按 session 写入 checkpoint，resume 使用 checkpoint + 原文 tail 重建上下文。
- **REPL 会话管理** — `/slash` 命令支持 plan/build/act、会话分支、回退、undo（快照恢复）、模型切换、config 管理、session transcript 落盘。
- **TUI 全屏终端** — 基于 `prompt-toolkit` 的类 VSCode 全屏交互界面。
- **Subagent 委托** — `subagent` 单入口委派子任务，持久化 batch/run 谱系与终态；子 agent 共享项目目录，并继承父 agent 的权限门控。
- **类型化工具呈现** — terminal、diff、location 和 subagent 由工具产生结构化 intent，REPL/TUI 共享投影逻辑。
- **MCP 协议** — 基于官方 Python SDK 连接本地 stdio server，自动发现 `.local/mcp_config.json` 并注册 `mcp__{server}__{tool}` 动态工具。
- **记忆系统** — 项目根 `MEMORY.md` + 用户级 `~/.xcode/memory/` 是可审查的长期事实源；Agent 通过 BM25 工具按需检索。
- **外部 Hook** — 可配置事件驱动的外部命令 hooks（git 前置检查、自定义通知等）。

---

## 工具能力

稳定工具默认注册：`read_file`、`write_file`、`edit_file`、`apply_patch`、
`glob_files`、`find_files`、`list_dir`、`grep_search`、`websearch`、
`webfetch`、`question`、`bash`、`search_tools`、`subagent`、`todowrite`、
`history`、`search_memory`。发现 skill 时注册 `load_skill`；存在 MCP 配置时
注册 `mcp__{server}__{tool}` 动态工具。

`search_memory` 是只读、低风险的 BM25 检索工具。运行时不会在每轮自动
注入检索结果；resume/rebuild 才会在独立预算内注入项目与用户记忆。长期
记忆只保存用户规则、架构决定和经过验证的跨 session 事实，当前进度与
下一步动作由 `.xcode/checkpoints/<session-id>/checkpoint.md` 负责。

`history` 只读取当前 session 的当前分支：`search` 按关键词定位旧消息，
`around` 按 message id 读取原文邻域。compact 后的 checkpoint 滚动更新，
旧 checkpoint 是下一轮摘要的权威基线；退化摘要不会覆盖已有可用状态。

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

**零配置可用**：无配置文件时启用核心工具、subagent 和 memory。

Xcode 只执行本地 filesystem 和本地 shell，不提供容器、远程环境或远程执行
provider。权限提示和 shell 效果分析用于帮助用户了解并确认操作，不构成 OS
级安全隔离；需要隔离时必须由运行 Xcode 的外部环境提供。

所有字段默认值及完整参考见 [CONFIG.md](CONFIG.md)。

---

## 架构

详细不变量与运行路径见 [docs/architecture.md](docs/architecture.md)，测试边界见
[docs/testing.md](docs/testing.md)。工程决策记录在
[.agents/notes/README.md](.agents/notes/README.md)，事故复盘规范见
[docs/postmortem/README.md](docs/postmortem/README.md)。

五层架构，自底向上：

| Layer | 路径 | 职责 |
|---|---|---|
| `ai/` | `src/xcode/ai/` | 多 provider LLM API：OpenAI-compatible 基类 + DeepSeek/ChatGLM/MiMo 适配器，流式传输、缓存、thinking |
| `agent/` | `src/xcode/agent/` | Agent loop 合约：消息/事件类型、上下文压缩、工具执行分区、watchdog、provider 抽象 |
| `harness/` | `src/xcode/harness/` | 运行时配置、session 事实账本、权限/审计、MCP、skill、记忆、hooks 和本地执行协议 |
| `coding_agent/` | `src/xcode/coding_agent/` | 产品工具装配：文件读写编辑、glob/grep/bash/subagent/webfetch/websearch 等 |
| `cli/` | `src/xcode/cli/` | REPL UI、TUI、slash command 系统、setup wizard、配置管理 |

运行路径：`main.py` → `build_app()` → `CodingAgentHarness` → `Agent` loop → provider stream → tool execution。

---

## 评估与验证

### 长程任务 benchmark

`benchmarks/` 提供上下文压缩消融实验：对同一模型、温度和任务，配对运行
完整历史 baseline 与启用 `LayeredCompactor`、checkpoint、resume 的 Xcode
配置。任务成功由测试进程判定，状态保持由文件哈希、禁止路径和验证命令判定。

```powershell
uv run python -m benchmarks.runners.run_ablation benchmarks/tasks/long_horizon `
  --repeat 3 --temperature 0 --max-pair-attempts 2 --require-complete-usage
```

实验设计、任务格式和报告口径见 [benchmarks/README.md](benchmarks/README.md)。

### 工具调度 benchmark

确定性消融实验通过生产 `execute_tool_calls()` 重放相同的 5、10、20 文件
读取批次，对比强制串行与副作用感知并发调度，并用混合读写 workload 验证
写操作不与其他工具重叠。该命令不调用模型 API：

```powershell
uv run python -m benchmarks.runners.run_tool_scheduling `
  benchmarks/tasks/parallel_reads --repeat 10 --warmup 1
```

报告按 workload 给出工具阶段 P50/P95 延迟、配对加速比、最大并发度、输出
等价率和写隔离率；这些结果不等同于端到端 Agent 延迟。

### 单元测试

```powershell
uv run pytest src/xcode/tests -q --tb=short
```

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
| [AGENTS.md](AGENTS.md) | Agent 开发入口、编码规范 |
| [CONFIG.md](CONFIG.md) | 运行时配置参考 |
| `src/xcode/main.py` | CLI 入口点与子命令 |
| `src/xcode/coding_agent/assembly/` | Coding 产品装配与工具注册 |
| [docs/source-review.md](docs/source-review.md) | 源码级架构审查 |


---

## 许可

[MIT](LICENSE) © 2026 Xcode Contributors
