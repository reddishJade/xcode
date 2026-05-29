# Xcode 配置参考

全局配置通过项目根目录的 `xcode.config.json` 提供。`python -m xcode.main`
和 `build_real_app(project_root=...)` 会自动读取该文件；`--config` 仅用于显式指定其他配置文件。
配置中的相对路径均按 `--project-root` 解析。

```powershell
.\.venv\Scripts\python.exe -m xcode.main
```

项目根目录没有 `xcode.config.json` 时走全部默认值：只启用 `core` 工具组。

## provider

### model_profiles

支持 `main`、`subagent`、`fallback` 三个常用 profile，也可添加自定义 profile。未配置的 `subagent` 和 `fallback` 会继承 `main`。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `transport` | string | `"chat_completions"` | `chat_completions`、`responses_stateful`、`anthropic_messages`、`deepseek_chat`、`mimo_chat` |
| `chat_model` | string | `"deepseek-v4-flash"` | 聊天模型名（如 `deepseek-v4-flash`、`mimo-v2.5-pro`、`mimo-v2.5`） |
| `base_url` | string | `"https://api.deepseek.com"` | OpenAI-compatible API 地址（MiMo: `https://api.xiaomimimo.com/v1`） |
| `api_key` | string | `""` | 显式 API key；留空时按 `{PROFILE}_API_KEY` → `OPENAI_API_KEY` → `API_KEY` 查找环境变量或 `.env` |

## agent

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_steps` | int | `20` | 最大模型/工具循环轮次 |
| `compact_threshold` | int | `0` | 消息数达到此值时触发压缩；0 表示关闭 |
| `compact_token_threshold` | int | `0` | 估算 token 达到此值时触发压缩；0 表示关闭 |
| `max_recent_messages` | int | `10` | 压缩时保留的近期消息数 |
| `tool_workers` | int | `4` | 只读且并发安全工具的最大并行数 |
| `watchdog_repeated_tool_limit` | int | `3` | 连续重复同一工具输入的停止阈值 |

压缩流程包含 stale `read_file` 裁剪、超大工具结果头尾截断、旧 tool_result 微压缩、transcript 落盘和摘要压缩。REPL `/compact` 可手动请求压缩。

## paths

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `sessions_dir` | string/null | `null` | REPL 会话 JSONL 目录；未配置时 CLI 使用 `.local/sessions` |
| `skills_dir` | string/null | `null` | Skill 扫描目录；独立 checkout 建议显式设为 `"skills"` |

REPL 会话索引固定写入 `.local/session_index.json`。Plan artifact 写入 `.local/session_artifacts/`。MCP schema cache 写入 `.local/mcp_cache.json`。

## observability

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `audit_path` | string/null | `null` | 工具调用审计日志；未配置时不写审计 JSONL |

## tools

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled_groups` | array | `["core"]` | 启用的工具分组 |

可选 group：`core`、`validation`、`worktree`、`mcp`、`skills`、`subagent`、`tasks`。

`core` 分组包含：`read_file`、`write_file`、`edit_file`、`glob_files`、`grep_search`、`bash`。`run_validation` 通过 `"validation"` 启用。MCP 工具必须通过 `"mcp"` 启用。

### tools.bash

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `network_commands` | string | `"ask"` | 网络命令策略：`allow`、`ask`、`deny` |
| `shell` | string | `"auto"` | `auto`、`pwsh`、`powershell`、`cmd`、`bash`、`zsh`、`sh`、`fish` |

`bash` 始终是 core 工具，但每条命令仍经过 risk evaluator、PermissionPolicy、HITL 和 deny rules。Shell Adapter 使用 `shell=False` 执行宿主 shell argv；它不是安全沙箱替代品。

## MCP

MCP server 不写进 `xcode.config.json` 的 `tools` 段，而是放在 `.local/mcp_config.json` 或项目根 `mcp_config.json`：

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {}
    }
  }
}
```

只有 `enabled_groups` 包含 `"mcp"` 时，Xcode 才读取配置并包装 MCP 工具。MCP 使用 stdio JSON-RPC `Content-Length` framing，工具 schema 缓存在 `.local/mcp_cache.json`，缓存会随 command/args/env hash 变化而失效。

## skills

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `auto_trigger` | bool | `true` | 启用 `skills` 工具组且找到 skills 目录时，按当前任务自动匹配 SKILL.md |

## prompt

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `modules` | array | `["identity", "tool_discipline", "tools", "environment", "git_preflight", "cwd", "instructions", "notices"]` | 参与拼接的 prompt 模块 |

可选模块包括 `search_strategy`、`contextual_retrieval`、`skills`。默认不把 Skill 内容塞进 prompt；`contextual_retrieval` 只会注入当前任务已经访问过的最近文件和工具结果摘要。
