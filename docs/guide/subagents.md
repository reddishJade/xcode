# Subagents 子代理

`subagent` 工具为独立任务创建 session-backed child agent。child 拥有自己的 session、history、inbox、recorder、correlation 和 ToolGate，同时继承父运行时的 provider 配置、项目边界和安全规则。

## 1. One-shot 与 Continuable

| mode | 生命周期 | 适用场景 |
| --- | --- | --- |
| `one_shot` | 完成后释放 activation，session 保留 | 一次性搜索、局部修改、独立验证 |
| `continuable` | 保留 child session，可继续提交 turn | 需要后续追问或多轮协作的子任务 |

创建一个 child：

```json
{
  "description": "检查 provider 适配器",
  "prompt": "阅读相关实现，列出协议差异和风险，不修改文件。",
  "mode": "one_shot",
  "subagent_type": "research"
}
```

## 2. 批量委派

多个独立任务使用 `tasks`：

```json
{
  "tasks": [
    {"description": "检查 AI 层", "prompt": "分析 provider 错误处理", "subagent_type": "research"},
    {"description": "检查工具层", "prompt": "分析文件编辑边界", "subagent_type": "coding"}
  ],
  "max_concurrent": 2
}
```

批量任务固定为独立 `one_shot` child。单次请求最多 16 个任务，默认并发上限 4；结果按 task index 汇总，实时 update 展示每个任务的状态和当前工具。

## 3. Continuable API

```text
/subagent             通过模型创建 continuable child
subagent_continue     向 direct child 提交下一 FIFO turn
subagent_list         列出 direct child session
subagent_control      interrupt 或 release child activation
```

工具调用示例：

```json
{
  "session_id": "child-session-id",
  "prompt": "继续检查刚才发现的错误处理，给出最小修复建议。"
}
```

`subagent_continue` 只接受 direct continuable child。`subagent_control`：

```text
{"session_id": "child-session-id", "action": "interrupt"}
{"session_id": "child-session-id", "action": "release"}
```

interrupt 取消当前 child turn；release 释放进程内 activation，child 账本继续保留并可冷恢复。

## 4. Child 工具集合

child registry 默认包含核心文件、搜索、Shell、Web 和 patch 工具。通过 `tools.subagent_extra_tools` 可以追加工具，例如：

```json
{
  "tools": {
    "subagent_extra_tools": ["todowrite", "websearch"]
  }
}
```

child 收到显式 prompt 和自己的 system prompt。父 session 的完整 transcript作为父侧事实保留，child 的模型上下文从 child session 自身建立。

## 5. 权限与谱系

child 使用父 gate 的项目 root、external directories、sensitive overrides、静态权限、mode ruleset 和 grant 来源，并建立 child session id 对应的独立 correlation 与 hook manager。child 工具继续经过 ActionExtractor、PermissionEngine 和审批流程。

父 session 记录：

- `SubagentDescriptor`：child session、父 session、mode、类型、provider、composition 和工具名。
- `SubagentActivationEvent`：materialized/released activation。
- `SubagentRunEvent`：run id、batch id、task index、状态、摘要和错误。

## 6. 关闭顺序

父运行时关闭时：

1. 取消所有 active child turn。
2. 等待 child activation 在有界时间内进入 idle。
3. 逆序释放 activation。
4. 关闭 MCP、reviewer 和其他共享资源。

child 未能在期限内收敛时返回明确关闭错误，避免静默遗留运行。
