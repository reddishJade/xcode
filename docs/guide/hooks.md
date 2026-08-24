# 外部事件 Hooks

Xcode 提供了事件驱动的外部命令 Hook 机制，允许开发者在 Agent 生命周期和工具调用的各个阶段注入自定义脚本，实现自动化校验、拦截、格式化与审计通知。

---

## 1. 支持的事件类型

| 事件名称 | 触发时机 | 典型应用场景 |
|---|---|---|
| `pre_tool` | 工具执行之前 | 动态安全检查、危险命令拦截、参数变换 |
| `post_tool` | 工具执行完成之后 | 自动代码格式化（Ruff/Black/Prettier）、Lint 检查 |
| `on_error` | 工具或 Provider 发生错误时 | 错误通知、报警、异常上下文记录 |
| `on_compact` | 上下文压缩触发时 | 记录 Token 消耗统计、备份 Checkpoint |
| `before_agent_start` | Agent 启动执行前 | 环境变量预热、Git 状态检查 |
| `before_provider_request`| 请求大模型之前 | 发送前最终请求脱敏与统计 |

---

## 2. 配置示例

在 `xcode.config.json` 中配置 `hooks.entries`：

```json
{
  "hooks": {
    "entries": [
      {
        "event": "pre_tool",
        "matcher": "bash",
        "command": ["python", "scripts/check_shell_command.py"],
        "timeout": 5.0,
        "failure_policy": "fail"
      },
      {
        "event": "post_tool",
        "matcher": "edit_file",
        "command": ["uv", "run", "ruff", "format", "--check"],
        "timeout": 10.0,
        "failure_policy": "warn"
      }
    ]
  }
}
```

---

## 3. Hook 交互协议 (Stdin / Stdout)

Xcode 通过标准输入以 JSON 格式向 Hook 进程传递事件上下文（`HookRecord`）：

```json
{
  "event": "pre_tool",
  "tool_name": "bash",
  "arguments": {
    "command": "git push origin main --force"
  },
  "session_id": "sess-xyz",
  "timestamp": "2026-08-24T15:35:00Z"
}
```

对于 `pre_tool` 事件，Hook 脚本可通过 Stdout 返回判定决策：
```json
{
  "decision": "deny",
  "reason": "禁止在自动化会话中执行强制推送 (force push)"
}
```

* `decision` 支持 `allow`（放行）、`ask`（交由用户或 Reviewer 审批）、`deny`（直接拒绝）。
* 如果 Hook 脚本因超时或崩溃报错，Xcode 会根据配置的 `failure_policy`（`ignore`/`warn`/`fail`）进行容错处理。

---

## 4. 查看 Hook 状态

在 REPL / TUI 会话中，输入 `/hooks` 即可实时查看当前加载的所有 Hook 状态、调用次数及最近错误：

```text
/hooks
```

---

← **上一篇**：[长期记忆系统 (memory.md)](memory.md) | **下一篇**：[Slash 命令完全参考手册 (slash-commands.md)](slash-commands.md) →

