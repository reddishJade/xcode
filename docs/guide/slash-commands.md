# Slash 命令参考

REPL 和 TUI 使用统一命令注册表。输入 `/` 后按 Tab 可以补全命令、说明和参数提示。

## 1. 模式与运行控制

| 命令 | 用法 | 作用 |
| --- | --- | --- |
| `/plan` | `/plan [prompt]` | 进入 Plan；带 prompt 时排入下一次运行 |
| `/build` | `/build` | 进入 Build |
| `/act` | `/act` | 进入 Act |
| `/steer` | `/steer MESSAGE` | 将消息送入当前 run 的下一次模型边界 |
| `/queue` | `/queue steer\|followup\|interrupt\|MESSAGE` | 设置 busy policy，或排入 follow-up |
| `/verbose` | `/verbose normal\|verbose\|debug` | 设置输出详细程度 |
| `/debug` | `/debug on\|off` | 切换 debug 输出 |
| `/compact` | `/compact` | 立即执行完整上下文压缩并保存 replacement |
| `/goal` | `/goal CONDITION\|pause\|resume\|clear` | 设置、暂停、恢复或清除独立验收目标 |

`/steer` 适合当前 run 的即时纠偏；`/queue` 的 follow-up 在当前 run 完成后启动新的 run。忙时普通输入默认按 `busy_mode` 处理。

## 2. Session 生命周期

| 命令 | 用法 | 作用 |
| --- | --- | --- |
| `/new` | `/new` | 建立空 session |
| `/clear` | `/clear` | 清空当前 session 并恢复空状态 |
| `/continue` | `/continue` | 恢复当前项目最近的 session |
| `/resume` | `/resume [ID]` | 选择或恢复指定 session |
| `/sessions` | `/sessions` | 打开历史 session 选择器 |
| `/rename` | `/rename TITLE` | 修改当前 session 标题 |
| `/fork` | `/fork` | 从当前 branch 的用户消息选择 fork 起点 |
| `/clone` | `/clone` | 复制当前 session |
| `/tree` | `/tree` | 浏览 session entry tree 并移动 head |
| `/rewind` | `/rewind [N]` | 回退最近 N 个用户 turn，默认 1 |

切换、fork、clone、rewind 后，Agent 从新的 branch 重新恢复 history、运行状态和上下文。

## 3. 文件回滚

| 命令 | 用法 | 作用 |
| --- | --- | --- |
| `/undo` | `/undo [N\|--list]` | 查看或回退 Git snapshot turn |

`/undo` 校验 post snapshot 冲突和权限。冲突或无法恢复的文件进入 skipped；session transcript 保持可追踪。

## 4. 模型与配置

| 命令 | 用法 | 作用 |
| --- | --- | --- |
| `/model` | `/model` | 显示当前模型和 base URL |
| `/model` | `/model MODEL` | 切换 main profile 模型 |
| `/model` | `/model PROFILE/MODEL:LEVEL` | 切换 main/subagent 与 thinking level |
| `/effort` | `/effort LEVEL` | 设置当前 provider 的 reasoning effort |
| `/thinking` | `/thinking on\|off` | 切换 thinking |
| `/config` | `/config [setting]` | 打开或定位交互式设置浏览器 |

`PROFILE` 当前使用 `main` 或 `subagent`。具体 effort 选项由 active transport 决定。

## 5. 工具与扩展

| 命令 | 用法 | 作用 |
| --- | --- | --- |
| `/tool` | `/tool list` | 列出注册工具 |
| `/tool` | `/tool NAME INPUT` | 通过当前权限 gate 直接执行工具 |
| `/skill` | `/skill NAME [prompt]` | 显式激活技能并可追加任务 |
| `/memory` | `/memory list\|search\|add\|update\|delete` | 检索和维护项目/用户记忆 |
| `/permissions` | `/permissions [list\|clear]` | 查看权限状态或清除 session grant |
| `/hooks` | `/hooks` | 查看外部 hook 诊断 |
| `/mcp` | `/mcp status\|reload` | 查看或重载 MCP runtime |
| `/context` | `/context` | 查看 token、工具、记忆和技能占用 |
| `/btw` | `/btw QUESTION` | 发起侧问题并恢复主 session |

工具输入可以使用 JSON object；只有一个 required 参数的工具支持对应的文本简写。

## 6. 退出

| 命令 | 用法 | 作用 |
| --- | --- | --- |
| `/exit` | `/exit` | 保存当前摘要并退出 |
| `/quit` | `/quit` | `/exit` 的隐藏 alias |
| `/revert` | `/revert [N\|--list]` | `/undo` 的隐藏 alias |

终端 Ctrl+C：输入栏有内容时先清空；活动 run 中请求取消；空闲状态连续触发后退出。
