# 配置系统

Xcode 使用 JSON 运行时配置。配置先按层合并，再通过 Pydantic 模型校验；运行中的 Agent 使用校验后的快照。

## 1. 配置文件与优先级

`discover_runtime_config()` 按以下顺序读取：

```text
~/.xcode/settings.json       全局
<project>/xcode.config.json  项目
<project>/.xcode/settings.json 本地
环境变量                      最后覆盖
```

后层只覆盖自己显式出现的键，显式写入默认值也会生效。`--config PATH` 将指定文件作为项目配置来源参与解析。

常用位置：

- `xcode.config.json`：项目运行配置。
- `.xcode/settings.json`：本地覆盖。
- `~/.xcode/settings.json`：用户全局默认。
- `.xcode/mcp_config.json`：项目 MCP server 配置，单独读取。

当前仓库的 `.gitignore` 会忽略 `xcode.config.json` 与 `.xcode/`。团队需要共享配置时，按团队策略调整忽略规则，并单独管理 API key。

## 2. 顶层结构

```json
{
  "provider": { "model_profiles": {} },
  "agent": {},
  "tools": {},
  "skills": {},
  "prompt": {},
  "paths": {},
  "observability": {},
  "hooks": {},
  "request_hygiene": {},
  "security": {},
  "execution_modes": {}
}
```

未知字段进入校验错误。嵌套 profile 支持继承 `main`：字符串值可作为只改模型名的 profile，对象值覆盖继承字段。

## 3. Agent 与请求预算

```json
{
  "agent": {
    "max_steps": null,
    "rollover_message_threshold": 0,
    "rollover_token_threshold": 0,
    "automatic_rollover": true,
    "fallback_recent_messages": 10,
    "fallback_recent_tokens": 20000,
    "reserve_tokens": 16384,
    "rollover_trigger_ratio": 0.95,
    "tool_workers": 4,
    "tool_timeout_seconds": 120,
    "watchdog_repeated_tool_limit": 3
  },
  "request_hygiene": {
    "enabled": true,
    "max_tool_result_bytes": 8000,
    "max_tool_arg_length": 1000,
    "keep_head_lines": 50,
    "keep_tail_lines": 50
  }
}
```

`max_steps` 为空时，Agent 由完成、取消、provider error 和 watchdog 驱动结束。`reserve_tokens` 为输出与运行余量保留空间。自动换窗优先使用 provider profile 的 `context_window` 覆盖，否则读取当前模型注册窗口，默认在 95% 处触发，且不超过窗口减去 reserve 的硬上限。

## 4. 工具、技能与 prompt

```json
{
  "tools": {
    "shell": "auto",
    "subagent_extra_tools": ["todowrite"]
  },
  "skills": {
    "trust_project_skills": false
  },
  "prompt": {
    "modules": [
      "identity", "tool_discipline", "citations", "tools",
      "search_strategy", "environment", "cwd",
      "git_preflight", "contextual_retrieval", "notices"
    ],
    "instructions": [
      {"type": "file", "path": "TEAM_RULES.md", "priority": "critical"},
      {"type": "inline", "content": "Use the project formatter.", "priority": "high"}
    ]
  }
}
```

指令文件路径要求项目相对路径，累计注入预算为 32 KB。prompt modules 控制稳定、动态和易变 prompt 区域。

## 5. 路径、会话与观测

```json
{
  "paths": {
    "sessions_dir": ".xcode/sessions",
    "skills_dir": null
  },
  "observability": {
    "audit_path": ".xcode/audit.jsonl"
  }
}
```

相对路径以项目根目录解析。会话账本、快照、MCP 缓存和永久授权默认位于 `.xcode/`。

## 6. 安全与执行模式

```json
{
  "security": {
    "approval_policy": "on-request",
    "approval_router": "mode",
    "non_workspace_access": true,
    "auto_review_timeout_seconds": 90,
    "sandbox": {
      "mode": "workspace-write",
      "network_access": "deny"
    },
    "restricted_dirs": ["secrets"],
    "permissions": {"read": "allow", "web": "ask"},
    "tools": {"bash": "ask"}
  },
  "execution_modes": {
    "default_mode": "act",
    "plan": {"rules": []},
    "build": {"rules": []},
    "act": {"rules": []}
  }
}
```

`permissions` 先展开为工具集合，`tools` 再按具体工具覆盖。规则字段支持 `action`、`effect`、shell 的 `command`/`subcommand`/flags，以及 `resource_pattern`。决策顺序和目录边界位于 [security.md](security.md)。

## 7. 交互式编辑

```bash
xcode config
xcode config --project-root ./project
xcode config --config ./private.json
```

REPL 中使用 `/config`。浏览器当前展示常用模式、审批、sandbox 和 Shell 设置；文本或枚举写入前先验证，校验失败时保持原文件。

环境变量 `XCODE_APPROVAL_POLICY` 可以覆盖 `security.approval_policy`。provider API key 按 profile、provider 环境变量和通用 `OPENAI_API_KEY` 等顺序解析，详见 [providers.md](providers.md)。
