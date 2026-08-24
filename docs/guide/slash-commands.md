# Slash 命令完整参考手册

Xcode 在 REPL / TUI 会话中提供了超过 30 条实用的 `/slash` 控制命令。所有命令均以斜杠 `/` 开头，支持 `Tab` 键自动补全。

---

## 1. 模式控制 (Mode Control)

| 命令 | 语法 | 功能说明 |
|---|---|---|
| `/plan` | `/plan [目标]` | 切换至 Plan 只读规划模式，专注于架构分析与设计文档生成 |
| `/build` | `/build` | 切换至 Build 自动构建模式，允许文件修改，Shell 命令由 Reviewer 自动放行 |
| `/act` | `/act [需求]` | 切换至 Act 人机协作模式，文件写入与 Shell 执行需用户确认 |
| `/verbose` | `/verbose [0\|1\|2]` | 设置运行日志与思考链的详细输出级别 |
| `/debug` | `/debug` | 切换 Debug 调试模式开关 |
| `/steer` | `/steer <指令>` | 在 Agent 执行循环中动态注入实时引导建议 |
| `/queue` | `/queue <消息>` | 设置忙时消息排队策略，或在当前 Run 结束后立即排队执行新任务 |

---

## 2. 会话生命周期与分支 (Session Lifecycle & Branching)

| 命令 | 语法 | 功能说明 |
|---|---|---|
| `/new` 或 `/clear` | `/new` | 清空当前上下文，开启全新会话 |
| `/continue` | `/continue` | 自动恢复当前项目最近一次的活跃会话 |
| `/resume` | `/resume [ID]` | 交互式选择或按 Session ID 恢复历史会话 |
| `/sessions` | `/sessions` | 列出当前项目的所有持久化历史会话 |
| `/rename` | `/rename <新名称>` | 为当前会话设置易于识别的可读名称 |
| `/fork` | `/fork [消息序号]` | 从指定的对话轮次创建新的会话分叉分支 |
| `/clone` | `/clone` | 将当前完整会话克隆为一个独立的 JSONL 文件 |
| `/tree` | `/tree` | 以 ASCII 树状图展示当前项目的会话派生谱系 |

---

## 3. 回滚与快照恢复 (Rollback & Undo)

| 命令 | 语法 | 功能说明 |
|---|---|---|
| `/undo` | `/undo [N\|--list]` | **文件级原子撤销**：根据快照瞬时将受影响的文件还原至修改前 |
| `/rewind` | `/rewind [N]` | **上下文撤销**：撤销最近 N 轮问答交互历史 |
| `/compact` | `/compact` | 手动触发上下文分层压缩并写入滚动 Checkpoint |

---

## 4. 模型与配置 (Model & Config)

| 命令 | 语法 | 功能说明 |
|---|---|---|
| `/model` | `/model [Profile/Model]` | 查看或动态切换当前使用的 LLM 模型及 Thinking 参数 |
| `/effort` | `/effort <level>` | 调整推理思考力度（`off`/`low`/`medium`/`high`/`max`） |
| `/thinking` | `/thinking on\|off` | 开启或关闭模型思考链（Reasoning）的实时展示 |
| `/config` | `/config [setting]` | 打开交互式配置浏览器，或直接调整特定运行参数 |

---

## 5. 工具与外部扩展 (Tools & Extensions)

| 命令 | 语法 | 功能说明 |
|---|---|---|
| `/tool` | `/tool [list\|NAME]` | 查看已注册工具列表、Schema 定义或手动测试调用 |
| `/skill` | `/skill <NAME>` | 显式激活特定 Skill 并将指令加载至上下文 |
| `/memory` | `/memory [list\|search\|add]` | 列出、检索或显式追加长期项目记忆 |
| `/permissions` | `/permissions [list\|clear]` | 查看当前会话已授予的临时权限或一键清除授权 |
| `/mcp` | `/mcp status\|reload` | 检查 MCP 服务器连接状态或热重载 MCP 配置 |
| `/hooks` | `/hooks` | 检查外部事件 Hook 的配置与执行统计 |
| `/context` | `/context` | 查看当前上下文窗口的 Token 消耗分布与压缩水位线 |
| `/btw` | `/btw <问题>` | 发起侧问题快速问答，不污染主任务会话历史 |
| `/exit` | `/exit` 或 `/quit` | 安全保存状态并退出会话 |

---

← **上一篇**：[外部事件 Hooks (hooks.md)](hooks.md) | **下一篇**：[CLI 命令行参数速查 (cli.md)](cli.md) →

