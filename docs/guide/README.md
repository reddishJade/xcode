# Xcode 使用指南

Xcode 是运行在本地工作区中的编码 Agent。它把模型推理、工具执行、权限审批、上下文管理、会话恢复和交互界面组合成一条可持续的工作流。

## 推荐阅读路径

| 目标 | 文档 |
| --- | --- |
| 安装并完成第一次运行 | [install.md](install.md) → [quickstart.md](quickstart.md) |
| 理解运行流程 | [architecture.md](architecture.md) |
| 配置模型和运行参数 | [providers.md](providers.md) → [configuration.md](configuration.md) |
| 选择自主性和安全边界 | [modes.md](modes.md) → [security.md](security.md) |
| 使用本地工具 | [tools.md](tools.md) |
| 管理长任务 | [sessions.md](sessions.md) → [memory.md](memory.md) |
| 扩展 Agent | [skills.md](skills.md) → [mcp.md](mcp.md) → [hooks.md](hooks.md) |
| 使用子代理 | [subagents.md](subagents.md) |
| 查阅命令和界面 | [slash-commands.md](slash-commands.md) → [cli.md](cli.md) → [web.md](web.md) |

## 三个核心概念

### Turn

用户输入进入一个 turn。Agent 可以在同一个 turn 内多次调用模型和工具，直到模型完成回答、达到步骤限制、触发看门狗、被取消或遇到 provider error。

### Session

Session 以 JSONL entry 保存事实。输入、工具调用、工具结果、压缩替换、provider request 和最终结果都可以从账本恢复。当前模型上下文是账本的 branch projection。

### Tool gate

模型产生工具调用后，Xcode 先解析工具意图、目标路径和未决效果，再执行权限规则、执行模式、已有授权和审批流程。工具 handler 在 gate 放行后运行。

## 最小工作流

```bash
# 在项目根目录完成配置
xcode setup

# 默认启动终端 TUI
xcode

# 或使用传统 REPL
xcode cli

# 单次任务
xcode -p "检查当前项目的 provider 配置，并给出改进建议"
```

常用输入入口：

- `@path/to/file`：读取项目内文件并附加到当前用户消息。
- `!command`：通过注册的 `bash` 工具直接执行 Shell 命令。
- `$skill-name task`：显式激活技能并继续任务。
- `/command`：控制模式、会话、模型、权限和运行状态。

## 配置与数据位置

- 项目配置：`xcode.config.json`
- 项目本地配置：`.xcode/settings.json`
- 会话账本：`.xcode/sessions/`
- 永久授权：`.xcode/approval_grants.json`
- MCP 配置与缓存：`.xcode/mcp_config.json`、`.xcode/mcp_cache.json`
- 项目记忆：`MEMORY.md`
- 用户记忆：`~/.xcode/memory/MEMORY.md`
- 审计日志：由 `observability.audit_path` 指定

工具以当前 `--project-root` 作为默认工作区边界；项目外访问需要 `external_directories` 与对应 access 授权。启动时明确指定项目根目录，可以让会话、工具、配置和 Git 状态保持同一工作区语义。
