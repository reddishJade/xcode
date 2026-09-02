# 快速上手与交互模式

## 1. 第一次运行

在项目根目录完成 provider 配置后：

```bash
xcode
```

默认进入终端 TUI。输入任务并按 Enter 提交，Shift+Enter 插入换行；部分终端可以使用 Esc、Enter 作为换行组合键。

常见的第一条任务：

```text
先浏览项目结构，说明入口、核心模块和当前 Git 状态。
```

Agent 通常先使用读取与搜索工具建立上下文，再根据任务需要编辑文件、运行命令和返回验证结果。

## 2. 三种常用运行方式

### 单次 prompt

```bash
xcode -p "解释 src/xcode/harness/agent_runtime 的运行流程"
xcode -p "检查当前修改并给出风险摘要"
```

单次模式消费事件流，打印文本结果后退出。它适合快速分析和脚本化调用。

### 终端 CLI / REPL

```bash
xcode cli
```

CLI 保留多轮 session，并提供命令补全、Markdown 输出、推理摘要和工具摘要。详细命令位于 [slash-commands.md](slash-commands.md)。

### TUI

```bash
xcode
xcode tui
```

TUI 在当前终端中显示 inline transcript，包含输入、步骤、推理、工具卡片、授权面板、滚动历史和状态栏。

## 3. 输入语法

| 语法 | 行为 | 示例 |
| --- | --- | --- |
| `@path` | 读取项目内文件并附加为 `<file-reference>` | `@src/xcode/main.py 解释参数解析` |
| `!command` | 直接调用注册的 `bash` 工具 | `!git status --short` |
| `$skill task` | 激活指定技能，再提交剩余任务 | `$code-review 检查这个补丁` |
| `/command` | 执行 session、模式、模型等控制命令 | `/plan 分析实现路径` |

`@file` 使用项目文件读取路径，`!command` 仍然经过工具门控和当前执行模式。

## 4. 会话恢复

```bash
# 当前项目最近会话
xcode --continue

# 打开会话选择器
xcode --resume

# 恢复指定 session
xcode --session SESSION_ID
```

进入 CLI 后也可以使用 `/continue`、`/resume`、`/sessions`。恢复过程从 session branch 重建模型历史、运行模式、Goal、todo、技能激活和相关上下文。

## 5. 一个稳妥的编码流程

1. `/plan`：读取结构、搜索相关符号、确认修改范围。
2. `/build`：让项目内结构化写入自动进行，边界动作交给自动 reviewer。
3. 先使用 `edit_file` 做局部修改；多文件关联变更使用 `apply_patch`。
4. 使用 `bash` 运行聚焦验证。
5. `/context` 查看上下文与用量，需要主动清理工作集时使用 `/new-context`。
6. `/undo` 回退文件快照，或 `/rewind` 回退当前 session branch。
7. `/act` 回到每项写入和 Shell 的人工审批模式。

## 6. Python 调用

```python
from pathlib import Path
from xcode.coding_agent.app import build_app

app = build_app(project_root=Path.cwd())
try:
    answer = app.ask("检查当前项目的配置和工具注册，输出结构化摘要。")
    print(answer)
finally:
    app.close()
```

异步环境使用 `await app.aask(...)`。持续消费事件时使用 `ask_stream` 或 `aask_stream`，前端可以直接消费 `AgentHarnessEvent`。
