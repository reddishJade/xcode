# Xcode 配置参考

`xcode.config.json` 位于项目根目录。`python -m xcode.main` 和 `build_app(project_root=...)` 会自动读取它；命令行 `--config` 只用于显式指定其他配置文件。相对路径均按 `--project-root` 解析。

没有 `xcode.config.json` 时，运行时只启用 `core` 工具组。

```powershell
.\.venv\Scripts\python.exe -m xcode.main
```

---

## provider

### model_profiles

支持 `main`、`subagent`、`fallback` 三个常用 profile，也可添加自定义 profile。未配置的 profile 由 provider 工厂按默认模型配置补齐。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `transport` | string | `"chat_completions"` | 模型传输协议。当前配置类型声明支持 `chat_completions`、`responses_stateful`；provider 工厂中还装配了 Anthropic、DeepSeek、MiMo 等兼容入口。 |
| `chat_model` | string | `"deepseek-v4-flash"` | 聊天模型名。 |
| `base_url` | string | `"https://api.deepseek.com"` | OpenAI-compatible API 地址。 |
| `api_key` | string | `""` | 显式 API key；留空时按 profile 环境变量和通用环境变量查找。 |
| `thinking` | bool | `true` | 传给支持 thinking 的 provider。 |
| `reasoning_effort` | string/null | `"high"` | 传给支持 reasoning effort 的 provider。 |

示例：

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "chat_completions",
        "chat_model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": ""
      },
      "subagent": {
        "chat_model": "deepseek-v4-flash"
      },
      "fallback": {
        "chat_model": "deepseek-v4-flash"
      }
    }
  }
}
```

---

## agent

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `max_steps` | int | `20` | 单次任务最大模型/工具循环轮次。 |
| `compact_threshold` | int | `0` | 消息数达到阈值时触发压缩；0 表示关闭。 |
| `compact_token_threshold` | int | `0` | 估算 token 达到阈值时触发压缩；0 表示关闭。 |
| `max_recent_messages` | int | `10` | 压缩时保留的近期消息数。 |
| `tool_workers` | int | `4` | 只读且并发安全工具的最大并行数。 |
| `watchdog_repeated_tool_limit` | int | `3` | 连续重复同一工具输入的停止阈值。 |

压缩流程包含 stale `read_file` 裁剪、大工具输出预算裁剪、旧 `tool_result` 微压缩、transcript 落盘和摘要压缩。REPL `/compact` 可手动请求压缩。

---

## paths

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `sessions_dir` | string/null | `null` | REPL 会话 JSONL 目录；未配置时 CLI 使用 `.local/sessions`。 |
| `skills_dir` | string/null | `null` | Skill 扫描目录；独立 checkout 建议设为 `"skills"`。 |

固定本地路径：

- `.local/session_index.json`：REPL 会话索引
- `.local/session_artifacts/`：Plan artifact
- `.local/mcp_cache.json`：MCP schema cache
- `.local/mcp_config.json`：优先级高于项目根 `mcp_config.json`

---

## observability

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `audit_path` | string/null | `null` | 工具调用审计日志；未配置时不写审计 JSONL。 |

---

## tools

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled_groups` | array | `["core"]` | 启用的工具组。 |

### 工具组

| group | 状态 | 提供能力 |
| --- | --- | --- |
| `core` | 默认启用 | `read_file`、`write_file`、`edit_file`、`glob_files`、`grep_search`、`ls`、`bash` |
| `skills` | 可选 | `load_skill` |
| `subagent` | 可选 | `submit_subagent`、`check_subagent`、`cancel_subagent` |
| `worktree` | experimental | `create_worktree_task`、`remove_worktree_task` |
| `mcp` | experimental | 从 MCP server 动态生成 `mcp__server__tool`；延迟模式会额外提供 fetch/search 工具 |
| `tasks` | experimental | `create_task`、`update_task`、`list_tasks`、`get_task` |
| `mailbox` | experimental | `send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message` |
| `progress` | experimental | `save_task_progress`、`resume_task_progress` |
| `memory` | experimental | 启用压缩摘要写入 `MEMORY.md` 的 consolidation hook |
| `plugins` | experimental | 扫描 `.local/plugins/*.py` 并注册暴露的工具和 hooks |
| `daemon` | experimental | 构造 `HeartbeatDaemon` |
| `speculation` | experimental | 构造 `SpeculationPlanner` |
| `experimental` | 总开关 | 展开为全部 experimental group |

`bm25` 是 `memory` 的内部检索实现，不单独作为启用入口。

### tools.bash

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `network_commands` | string | `"ask"` | 网络命令策略：`allow`、`ask`、`deny`。 |
| `shell` | string | `"auto"` | `auto`、`pwsh`、`powershell`、`cmd`、`bash`、`zsh`、`sh`、`fish`。 |

`bash` 是 core 工具，但每条命令仍经过 risk evaluator、PermissionPolicy、HITL 和 deny rules。Shell Adapter 使用 argv 调用宿主 shell，不是安全沙箱替代品。

---

## MCP

MCP server 不写入 `xcode.config.json` 的 `tools` 段，而是放在 `.local/mcp_config.json` 或项目根 `mcp_config.json`：

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {},
      "defer_loading": true,
      "overrides": {
        "tool_name": {
          "risk": "high"
        }
      }
    }
  }
}
```

只有 `enabled_groups` 包含 `"mcp"` 或 `"experimental"` 时，Xcode 才读取 MCP 配置并包装 MCP 工具。MCP 使用 stdio JSON-RPC `Content-Length` framing，工具 schema 缓存在 `.local/mcp_cache.json`，缓存会随 command、args、env hash 变化而失效。

---

## skills

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `auto_trigger` | bool | `true` | 启用 `skills` 工具组且找到 skills 目录时，按当前任务自动匹配 SKILL.md。 |

`skills` 不是默认工具组。只有 `enabled_groups` 包含 `"skills"` 时，`auto_trigger` 才会生效。

---

## prompt

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `modules` | array | `["identity", "tool_discipline", "tools", "environment", "git_preflight", "cwd", "instructions", "notices"]` | 参与拼接的 prompt 模块。 |

可选模块包括 `search_strategy`、`contextual_retrieval`、`skills`。默认不把 Skill 内容塞进 prompt；`contextual_retrieval` 只会注入当前任务已经访问过的最近文件和工具结果摘要。

---

## daemon

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | 是否允许构造 `HeartbeatDaemon`。还必须启用 `daemon` 或 `experimental` group。 |
| `interval_seconds` | int | `30` | 心跳轮询间隔。 |

---

## 完整示例

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "chat_completions",
        "chat_model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": ""
      },
      "subagent": {
        "chat_model": "deepseek-v4-flash"
      },
      "fallback": {
        "chat_model": "deepseek-v4-flash"
      }
    }
  },
  "agent": {
    "max_steps": 20,
    "compact_threshold": 8,
    "compact_token_threshold": 12000,
    "max_recent_messages": 10,
    "tool_workers": 4,
    "watchdog_repeated_tool_limit": 3
  },
  "tools": {
    "enabled_groups": ["core", "mcp"],
    "bash": {
      "network_commands": "ask",
      "shell": "auto"
    }
  },
  "skills": {
    "auto_trigger": true
  },
  "prompt": {
    "modules": [
      "identity",
      "tool_discipline",
      "tools",
      "environment",
      "git_preflight",
      "cwd",
      "instructions",
      "notices"
    ]
  },
  "paths": {
    "sessions_dir": ".local/sessions",
    "skills_dir": "skills"
  },
  "observability": {
    "audit_path": ".local/audit.jsonl"
  },
  "daemon": {
    "enabled": false,
    "interval_seconds": 30
  }
}
```
