# 快速上手与交互模式

Xcode 提供了单轮问答、全屏终端交互（TUI）、交互式命令行（CLI/REPL）以及编程式调用等多种交互形态。

---

## 1. 单轮提问模式 (-p)

对于不需要多轮上下文的快速任务或脚本化调用，直接使用 `-p`（`--prompt`）选项：

```bash
# 查询代码实现
xcode -p "解释当前项目中的执行模式是如何定义的"

# 查找特定文件
xcode -p "列出 src/xcode/ai/ 目录下所有的 provider 适配器"
```
Agent 会完成分析与回答后自动退出。

---

## 2. 全屏终端界面 (TUI)

直接运行 `xcode`（或 `xcode tui`）会启动基于 `prompt-toolkit` 的全屏交互界面：

```bash
xcode
```

### TUI 界面核心特性
* **顶部状态栏**：实时展示当前执行模式（`plan`/`build`/`act`）、活跃模型 profile 与上下文占用；
* **流式 Markdown 渲染**：支持代码高亮、表格与实时 Reasoning 思考链折叠；
* **结构化工具面板**：终端命令输出（Terminal Intent）、代码 Diff（Diff Intent）与子代理运行状态均以独立视窗投影展示；
* **快捷交互键**：
  * `Esc`：快速关闭当前弹出菜单或表单；
  * `Tab`：快速切换输入焦点或补全 Slash 命令。

---

## 3. 标准 REPL 模式 (CLI)

如果你更习惯标准流式终端输出，可以使用 CLI 模式：

```bash
xcode cli
```

在 REPL 模式中，支持多轮对话与以下增强输入语法：

| 语法 | 说明 | 示例 |
|---|---|---|
| `@path/to/file` | 将本地文件内容读取并注入到提问中 | `@src/xcode/main.py 请分析该文件的参数定义` |
| `!command` | 直接在本地终端执行命令，不经过模型 | `!git status` |
| `$skill-name ...` | 显式激活特定 Skill 并委派任务 | `$refactor 重构工具注册逻辑` |
| `/command` | 执行 Xcode 内置 Slash 控制命令 | `/plan`, `/undo`, `/config` |

---

## 4. 会话恢复与历史管理

Xcode 自动落盘所有会话：

```bash
# 恢复当前项目最近一次的会话
xcode --resume

# 列出所有历史会话并选择恢复
xcode
# 在交互菜单中选择 "Resume Session" 或使用 /resume 命令
```

---

## 5. Python 编程式调用

你也可以在 Python 脚本中将 Xcode 作为组件引入：

```python
from pathlib import Path
from xcode.coding_agent.app import build_app

# 构造 Coding Agent 实例
app = build_app(project_root=Path.cwd())

# 发起问答
response = app.ask("审查当前仓库的配置体系，并列出所有支持的环境变量。")
print(response)

# 关闭并释放资源
app.close()
```

---

← **上一篇**：[模型与 Provider 配置 (providers.md)](providers.md) | **下一篇**：[执行模式：Plan / Build / Act (modes.md)](modes.md) →

