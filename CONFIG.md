# Xcode 配置参考

`xcode.config.json` 位于项目根目录。`python -m xcode.main` 和 `build_app()` 自动读取它；`--config` 用于显式指定其他路径。相对路径按 `--project-root` 解析。

配置发现栈（优先级从低到高）：全局 `~/.xcode/settings.json` → 项目 `xcode.config.json` → 本地 `.local/settings.json` → 环境变量 `XCODE_SANDBOX_MODE`、`XCODE_PERMISSION_MODE`、`XCODE_APPROVAL_POLICY`。

**没有配置文件时**只启用 `core` 工具组，零配置可用。

---

## provider

### model_profiles

支持 `main`、`subagent`、`fallback` 三个 profile。未配置的 profile 由 `_resolve_model_profiles` 按 main 配置补齐：字符串视为 model 名称，字典与 main 配置合并。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `transport` | string | `"openai_chat"` | `openai_chat`、`deepseek_chat`、`mimo_chat`、`chatglm_chat` |
| `chat_model` | string | `"deepseek-v4-flash"` | 聊天模型名 |
| `base_url` | string | `""` | OpenAI-compatible API 地址 |
| `api_key` | string | `""` | 显式 API key；留空按环境变量查找 |
| `thinking` | bool | `true` | 传给支持 thinking 的 provider |
| `reasoning_effort` | string/null | `"high"` | DeepSeek 等支持 effort 的 provider。值：`off`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max` |
| `clear_thinking` | bool | `false` | ChatGLM 保留式思考 |
| `tool_stream` | bool | `true` | ChatGLM 工具流式输出 |
| `response_format` | object/null | `null` | 结构化输出，如 `{"type":"json_object"}` |

#### DeepSeek

- **默认 base_url**: `https://api.deepseek.com`
- **Thinking mode**: 默认开启，`extra_body={"thinking": {"type": "enabled"}}`
- **reasoning_effort**: 默认 `"high"`，复杂 agent 请求自动设为 `"max"`
- **缓存统计**: 原生 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`

#### MiMo

- **默认 base_url**: `https://api.xiaomimimo.com/v1`
- **Thinking**: mimo-v2.5-pro/mimo-v2.5 默认开启，mimo-v2-flash 默认关闭
- **缓存统计**: 兼容字段 `prompt_tokens_details.cached_tokens`

#### ChatGLM

- **默认 base_url**: `https://open.bigmodel.cn/api/paas/v4/`
- **tool_stream**: 仅 `glm-4.6`/`glm-4.7` 支持
- **缓存统计**: 兼容字段 `prompt_tokens_details.cached_tokens`

---

## agent

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_steps` | int | `20` | 单次任务最大循环轮次 |
| `execution_mode` | string | `"act"` | 默认执行模式：`plan`、`build`、`act` |
| `compact_threshold` | int | `0` | 消息数阈值；0 关闭 |
| `compact_token_threshold` | int | `0` | token 阈值；0 关闭 |
| `max_recent_messages` | int | `10` | 压缩时保留的近期消息数 |
| `tool_workers` | int | `4` | 并发安全工具最大并行数 |
| `watchdog_repeated_tool_limit` | int | `3` | 连续重复同一工具阈值 |

---

## request_hygiene

控制发给模型的消息历史压缩策略，不影响磁盘完整历史。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `true` | 是否启用 |
| `max_tool_result_bytes` | int | `8000` | tool_result 最大字节数 |
| `max_tool_arg_length` | int | `1000` | 已完成工具调用参数字符串最大长度 |
| `keep_head_lines` | int | `50` | 压缩 tool_result 保留头部行数 |
| `keep_tail_lines` | int | `50` | 压缩 tool_result 保留尾部行数 |

实现位置：`src/xcode/agent/history.py`、`src/xcode/harness/config.py`。

---

## paths

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `sessions_dir` | string/null | `null` | REPL 会话目录；未配置时 CLI 使用 `.local/sessions` |
| `skills_dir` | string/null | `null` | 最高优先级 Skill 扫描目录；相对路径按项目根目录解析 |

固定本地路径：`.local/session_index.json`、`.local/session_artifacts/`、`.local/mcp_cache.json`、`.local/mcp_config.json`、`.local/tasks.json.d/`。

---

## observability

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `audit_path` | string/null | `null` | 审计日志路径 |

---

## hooks

`hooks.entries` 声明受信任的外部命令 hook。每个配置层的 entries 数组整体替换
低优先级数组，不按元素合并。命令必须是 argv 数组；不经过 shell，不支持进程内
Python callback。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `event` | string | 必填 | `pre_tool`、`post_tool`、`on_error`、`on_compact`、`before_agent_start`、`before_provider_request` |
| `command` | string[] | 必填 | 非空 argv 数组 |
| `matcher` | string/null | `null` | 可选事件匹配表达式 |
| `timeout` | number | `10.0` | 正数秒数 |
| `enabled` | bool | `true` | 是否启用 |
| `failure_policy` | string | `"warn"` | `ignore`、`warn`、`fail` |
| `inherit_to_subagents` | bool | `false` | 是否显式传播给 subagent；默认不传播 |

```json
{
  "hooks": {
    "entries": [
      {
        "event": "pre_tool",
        "matcher": "bash",
        "command": ["python", "hooks/check_shell.py"],
        "timeout": 5,
        "failure_policy": "fail",
        "inherit_to_subagents": false
      }
    ]
  }
}
```

---

## security

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `permission_mode` | string | `"normal"` | `strict`、`normal`、`permissive` |
| `sandbox_mode` | bool | `false` | 沙箱模式 |
| `approval_policy` | string | `"never"` | `always`、`never` |
| `network_access` | bool | `true` | 网络访问 |
| `writable_roots` | array | `[]` | 可写目录白名单 |
| `restricted_dirs` | array | `[]` | 禁止访问目录列表 |
| `rules` | array | `[]` | 静态权限规则列表（替换已移除的 deny_tools/ask_tools/allow_tools） |
| `global_default` | string/null | `null` | 无规则匹配时的默认决策：`allow`、`ask`、`deny` |
| `external_directories` | array | `[]` | 外部目录白名单，每条包含 `path`（必填）和 `access`（可选，默认 `"read"`；可选值 `read`/`write`/`read_write`） |

### rules 规则格式

```json
{"tool": "read_file", "decision": "allow"}
{"tool": "bash", "decision": "ask", "input_contains": "curl"}
{"tool": "*", "decision": "deny"}
```

规则按声明顺序匹配，最后匹配的规则生效（last-match-wins）。无规则匹配时使用 `global_default`。与全局 resolver 优先级 `non_bypassable_deny > deny > ask > allow` 配合，Boundary 和安全 evaluator 产生的 `deny` 不受静态 `allow` 规则覆盖。

### external_directories 示例

```json
{"path": "/home/user/reference", "access": "read"}
{"path": "/shared/templates", "access": "read_write"}
```

- `access=read`：仅允许读取操作
- `access=write`：仅允许写入操作
- `access=read_write`：读写均允许
- `.env`、`.env.*`、`.git`、凭据路径在所有目录中均被拒绝
- `.env.example` 读取允许，写入拒绝

---

## tools

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled_groups` | array | `["core"]` | 启用的工具组 |
| `shell` | string | `"auto"` | `auto`、`pwsh`、`powershell`、`cmd`、`bash`、`zsh`、`sh`、`fish` |

## skills

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `trust_project_skills` | bool | `false` | 是否信任并披露项目内 `.xcode/skills/` 与 `.agents/skills/`；默认仅发现用户级技能 |

无可见 skill 时不注册 `load_skill`，也不向上下文注入空 catalog。

Skill discovery 按 first-wins 处理同名技能，覆盖顺序为：
显式 `paths.skills_dir` / `build_app(skills_dir=...)` → 项目
`.xcode/skills/` → 项目 `.agents/skills/` → 用户 `~/.xcode/skills/` → 用户
`~/.agents/skills/`。项目固定目录仍受 `trust_project_skills` 控制；显式目录表示
调用方已信任。显式目录不存在时记录 warning。

`load_skill` 首次激活返回 skill root、正文及 `scripts/`、`references/`、
`assets/` 相对路径元数据，但不会主动读取或执行资源。相同 session 内重复激活
只返回简短状态；activation 状态可从会话历史恢复，并在上下文压缩时保留。

### 工具组

| group | 状态 | 工具 |
|---|---|---|
| `core` | 默认 | `read_file`、`write_file`、`edit_file`、`glob_files`、`find_files`、`grep_search`、`ls`、`bash`、`shell`、`search_tools` |
| `skills` | 可选 | `load_skill` |
| `subagent` | 可选 | `submit_subagent`、`check_subagent`、`cancel_subagent` |
| `worktree` | 可选 | `create_worktree_task`、`remove_worktree_task` |
| `tasks` | 可选 | `create_task`、`update_task`、`advance_task`、`list_tasks`、`get_task`、`resolve_blocked` |
| `mailbox` | 可选 | `send_mailbox_message`、`read_mailbox_messages`、`acknowledge_mailbox_message` |
| `progress` | 可选 | `save_task_progress`、`resume_task_progress`、`start_task_run`、`resume_task_run`、`retry_task_run`、`expire_task_runs` |
| `memory` | 可选 | 启用 `MemoryManager` 压缩摘要 consolidation |
| `daemon` | 可选 | 构造 `HeartbeatDaemon` |

`shell` 工具是 OpenAI Responses builtin 的本地执行桥，接收 `commands` 数组。`search_tools` 工具按关键字搜索已注册工具。
MCP 属于核心运行时：存在 `.local/mcp_config.json` 时自动注册动态
`mcp__{server}__{tool}` 和 `mcp_tool_search`，无配置时不增加工具。
MCP schema cache 记录配置 hash、协商协议版本和 server identity；缺少这些
协商元数据的旧缓存会自动重新发现。

---

---

## prompt

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `modules` | array | 9 个模块 | 参与拼接的 prompt 模块 |
| `instructions` | array | `[]` | 指令源列表（见下方） |

默认模块顺序：`identity`、`tool_discipline`、`tools`、`search_strategy`、`environment`、`cwd`、`git_preflight`、`contextual_retrieval`、`notices`。

分三个缓存区域：STABLE（identity/tool_discipline/tools/search_strategy）→ DYNAMIC（environment/cwd）→ VOLATILE（git_preflight/contextual_retrieval/notices）。

### prompt.instructions 格式

每个元素为包含以下字段的对象：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | 是 | `"file"` 或 `"inline"` |
| `path` | string | 仅 file | 项目相对路径，禁止绝对路径、`~`、`..` 遍历 |
| `content` | string | 仅 inline | 指令文本 |
| `priority` | string | 否 | `"critical"`、`"high"`、`"medium"`、`"low"`；默认 `"critical"` |

示例：
```json
{"type": "file", "path": "AGENTS.md", "priority": "critical"}
{"type": "inline", "content": "No external dependencies without approval.", "priority": "high"}
```

未配置 `instructions` 时自动回退到 `AGENTS.md` / `CLAUDE.md`。
配置非空时：先收集配置源，再收集回退文件。配置源与回退文件按路径去重，配置源优先。

所有指令内容按 UTF-8 字节计入预算：≤32KB 完整注入，>32KB 压缩保留关键章节。

---

## daemon

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `false` | 是否构造 `HeartbeatDaemon`（还需启用 `daemon` group） |
| `interval_seconds` | int | `30` | 心跳轮询间隔 |

---

## 缓存优化

### 统计口径

1. **原生优先**（DeepSeek）：`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
2. **兼容回退**（OpenAI/ChatGLM/MiMo）：`prompt_tokens_details.cached_tokens`
3. **命中率公式**：`hit / (hit + miss)`

### 工具稳定化

1. 工具列表按 `name` 字母排序
2. Schema 键递归排序（`sort_keys=True`）
3. SHA256 前 16 字符指纹

实现位置：`src/xcode/ai/cache.py`。

### Token ROI 原则

优化策略：稳定可缓存前缀、压缩动态历史、控制工具输出、渐进发现工具、token-aware 压缩触发、智能重复抑制。

`LayeredCompactor`（`src/xcode/harness/agent_runtime/compaction.py`）：
- stale read_file 裁剪
- 大工具输出预算裁剪
- 旧 tool_result 微压缩
- transcript 落盘
- older messages summary compact

`RepeatDetector`（`src/xcode/agent/watchdog.py`）：文件变更感知的重复检测，变更后自动清除只读调用历史。
