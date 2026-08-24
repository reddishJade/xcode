# Xcode 实战场景与使用示例

本文档提供 Xcode 在日常软件开发中的核心工作流与常见实战示例。

---

## 1. 快速提问与单轮执行

对于简单的代码咨询、目录分析或单步操作，可以使用单轮提问模式：

```bash
# 查询代码实现
xcode -p "解释 src/xcode/agent/agent.py 中的 AgentLoop 运行机制"

# 执行特定定位
xcode -p "检查项目中所有未被使用的 import 语句"
```

---

## 2. 交互式全屏终端 (TUI) 与 REPL

Xcode 默认启动类 VSCode 的全屏终端界面（TUI）：

```bash
# 启动 TUI 全屏交互
xcode

# 或启动标准 CLI / REPL 模式
xcode cli
```

在交互界面中，支持丰富的输入与控制能力：
* **`@` 引用文件**：输入 `@src/xcode/main.py` 可直接将文件内容加入提问上下文。
* **`!` 执行 Shell**：输入 `!git status` 可在不通过模型的情况下直接运行本地命令。
* **`$` 调用技能**：输入 `$refactor 提取通用基类` 可显式激活已加载的 Skill。

---

## 3. 典型工作流：从 Plan 规划到 Build 实现

对于中大型功能开发或复杂重构，推荐采用 **Plan 模式规划 → 人工确认 → Build 模式实现** 的两阶段工作流：

### 第一步：进入 Plan 模式进行架构调研与规划
在 REPL 或 TUI 中输入：
```text
/plan 分析现有权限判定逻辑，并规划基于角色（RBAC）的扩展方案
```
* **运行机制**：Agent 处于只读模式，通过 `read_file`、`grep_search` 等工具勘察代码，分析现有实现并在 `.xcode/plans/rbac_design.md` 中生成详细的实施计划，包括修改文件清单、数据结构设计与测试用例规划。
* **安全保证**：此阶段 Agent 无法修改任何业务代码，无法执行任意 Shell 命令。

### 第二步：审查计划并一键切换到 Build 模式
审阅生成的 Markdown 计划后，输入：
```text
/build 按照 .xcode/plans/rbac_design.md 中的方案实现代码并跑通测试
```
* **运行机制**：Agent 自动切换为构建模式，依次使用 `edit_file` / `write_file` 应用修改，并通过 `bash` 运行单元测试。
* **自动审查**：执行测试等低风险命令由独立的 Reviewer 模型自动审查放行；如果涉及跨工作区写入或敏感命令，则自动暂停向用户请求确认。

---

## 4. 会话分支探索与文件级快照撤销 (Undo)

在探索不确定的重构思路时，Xcode 提供了完整的版本回退与分支能力：

### 4.1 会话分支 (/fork)
如果你想在某一轮对话的基础上尝试另一种技术方案：
```text
/fork 3
```
系统会从第 3 条消息分叉出一个全新会话，原有会话的完整历史账本保持不变。

### 4.2 文件级快照撤销 (/undo)
如果 Agent 修改了多个文件后测试未通过，或者你不满意当前修改：
```text
# 查看最近可撤销的文件快照
/undo --list

# 撤销最近一次工具执行所做的所有文件修改
/undo
```
Xcode 会利用内置的快照引擎瞬间将涉及的文件原子恢复至修改前的状态。

---

## 5. 扩展能力：连接 MCP Server (Model Context Protocol)

Xcode 原生支持 Model Context Protocol (MCP) 标准，可通过配置本地 stdio server 接入丰富的第三方工具（如数据库、GitLab、Sentry 等）。

在项目根目录创建 `.xcode/mcp_config.json`：
```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./data.db"]
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
```

启动 Xcode 后，Agent 会自动发现并注册对应的动态工具：
```text
● mcp__sqlite__read_query
● mcp__sqlite__describe_table
● mcp__fetch__fetch
```
你可以直接向 Agent 提问：“查询 SQLite 数据库中最新的 10 条用户记录”，Agent 会自动调用该 MCP 工具并汇总结果。

---

## 6. 长期记忆管理 (Memory)

Xcode 内置了分层的长期记忆系统：
* **项目记忆**：位于项目根目录的 `MEMORY.md`，用于记录跨会话的技术选型、业务约定与关键约束；
* **用户记忆**：位于 `~/.xcode/memory/MEMORY.md`，用于记录个人代码风格偏好。

### 查看与检索记忆
在 REPL 中：
```text
# 检索关于认证方式的记忆
/memory search auth

# 列出当前已持久化的所有记忆
/memory list
```

Agent 在执行任务时，会根据需要通过 `search_memory` 工具按需检索，无需每轮浪费 Token 注入完整背景。

---

## 7. 外部事件 Hooks

你可以通过配置外部命令 Hook，在 Agent 执行关键动作前后进行自动化校验或通知。

在 `xcode.config.json` 中配置：
```json
{
  "hooks": {
    "entries": [
      {
        "event": "pre_tool",
        "matcher": "bash",
        "command": ["python", "scripts/pre_bash_check.py"],
        "timeout": 5,
        "failure_policy": "fail"
      },
      {
        "event": "post_tool",
        "matcher": "edit_file",
        "command": ["uv", "run", "ruff", "check", "--fix"],
        "timeout": 10,
        "failure_policy": "warn"
      }
    ]
  }
}
```

* `pre_tool`：在 Agent 执行 `bash` 命令前运行自定义校验脚本，可动态决定 `allow`、`deny` 或 `ask`；
* `post_tool`：在 Agent 修改文件后自动运行 linter 修复格式。
