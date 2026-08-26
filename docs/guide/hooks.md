# 外部事件 Hooks

Hooks 把 Agent 运行事件转换为同步回调、结构化订阅或受信任外部命令。它们用于观测、校验、参数变换和外部自动化。

## 1. 事件

| event | 时机 | 可用输入/作用 |
| --- | --- | --- |
| `before_agent_start` | Agent turn 开始前 | 记录问题与 session 关联 |
| `before_provider_request` | provider 请求组装后 | 查看最终 messages、tools、options、预算和 trace |
| `pre_tool` | 工具执行前 | 参数变换、收紧 allow/ask/deny |
| `post_tool` | 工具成功完成后 | 接收工具输入和输出 |
| `on_error` | 工具或运行错误后 | 错误通知与诊断 |
| `on_compact` | 压缩启动时 | 记录压缩触发和消息数量 |

## 2. 配置

```json
{
  "hooks": {
    "entries": [
      {
        "event": "pre_tool",
        "matcher": "bash",
        "command": ["python", "scripts/check_command.py"],
        "timeout": 5,
        "enabled": true,
        "failure_policy": "fail",
        "inherit_to_subagents": false
      },
      {
        "event": "post_tool",
        "matcher": "edit_file",
        "command": ["python", "scripts/notify.py"],
        "timeout": 10,
        "failure_policy": "warn"
      }
    ]
  }
}
```

字段：

- `event`：事件名。
- `command`：argv 数组，首项为可执行程序。
- `matcher`：按 event、tool、mode 或 profile 进行 glob 匹配。
- `timeout`：秒数。
- `enabled`：启用开关。
- `failure_policy`：`ignore`、`warn`、`fail`。
- `inherit_to_subagents`：是否传递给 child session。

配置来源路径会进入 hook diagnostics。

## 3. 外部进程协议

Xcode 使用 `shell=False` 启动 hook，把脱敏后的 JSON 写入 stdin：

```json
{
  "event": "pre_tool",
  "tool": "bash",
  "input": "{\"command\":\"git status\"}",
  "output": "",
  "error": "",
  "metadata": {},
  "timestamp": "...",
  "session_id": "...",
  "turn_id": "...",
  "request_id": "...",
  "tool_call_id": "..."
}
```

stdout 需要是单个 JSON object，大小上限为 64 KB。`pre_tool` 允许返回：

```json
{
  "decision": "ask",
  "arguments": {"command": "git status --short"}
}
```

`decision` 取 `allow`、`ask`、`deny`；`arguments` 必须是 JSON object。外部 hook 的决策与既有决策取更严格结果，hook 形成收紧点。

## 4. 失败处理

- `ignore`：记录状态后继续。
- `warn`：记录状态并输出 warning，主流程继续。
- `fail`：抛出 `ExternalHookFailure`，当前动作进入错误路径。
- 超时、启动错误、非零退出、非法 JSON 和未知响应字段均进入 failed execution。

所有错误诊断先经过脱敏和长度限制。

## 5. 同步与后台执行

`SignalHookManager` 提供三种使用方式：

- registered callback：同步执行，适合改变当前决策或记录关键事实。
- subscribed callback：接收类型化 `HarnessEvent`。
- background callback：进入后台队列，适合外部通知和耗时观察。

`drain_background()` 等待已入队事件完成，应用关闭和受控运行可以用它完成收尾。

## 6. 审计与关联

每个 HookRecord 带 UTC timestamp、session id、turn id、request id 和 tool call id。`before_provider_request` 的 metadata 包含最终请求、工具 schema、provider 参数、composition id、prompt/request hash 和 context trace。

工具执行审计由 `observability.audit_path` 控制，详见 [security.md](security.md)。

## 7. 查看状态

```text
/hooks
```

状态输出包含 event、matcher、enabled、failure policy、subagent inheritance、source、run count、last status、last error 和 last run time。
