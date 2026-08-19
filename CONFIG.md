# Xcode 配置参考

`xcode.config.json` 位于项目根目录。`python -m xcode.main` 和 `build_app()` 自动读取它；`--config` 用于显式指定其他路径。相对路径按 `--project-root` 解析。

配置发现栈（优先级从低到高）：全局 `~/.xcode/settings.json` → 项目 `xcode.config.json` → 本地 `.local/settings.json` → 环境变量 `XCODE_APPROVAL_POLICY`。

**没有配置文件时**启用正式内置能力（`core`、`subagent`、`memory`，以及存在
可见 skill 时的 `skills`）。零配置可用。

---

## provider

### model_profiles

支持 `main`、`subagent`、`fallback` 三个 profile。未配置的 profile 由 `_resolve_model_profiles` 按 main 配置补齐：字符串视为 model 名称，字典与 main 配置合并。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `transport` | string | `"openai_chat"` | `openai_chat`、`deepseek_chat`、`mimo_chat`、`chatglm_chat` |
| `chat_model` | string | `"deepseek-v4-flash"` | 聊天模型名 |
| `base_url` | string | `"https://api.deepseek.com"` | OpenAI-compatible API 地址 |
| `api_key` | string | `""` | 显式 API key；留空按环境变量查找 |
| `context_window` | int/null | `null` | 上下文窗口覆盖（token 数）。覆盖模型注册表默认值，影响压缩触发线与 `/context` 显示。例如 1M 窗口的模型只用 256K：`"context_window": 262144` |
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

### 运行时模型切换

REPL 中可通过 `/model` 命令动态切换模型而无需重启：

```
/model                                    # 查看当前模型信息
/model <provider>/<model>[:thinking_level]  # 切换模型
/model <profile>/<model>[:thinking_level]   # 按 profile 切换
```

| 部分 | 说明 |
|---|---|
| `provider` | transport 名：`openai_chat`、`deepseek_chat`、`mimo_chat`、`chatglm_chat`；省略时使用当前 profile |
| `profile` | 配置中的 profile 名：`main`、`subagent`、`fallback` |
| `model` | 模型 ID，如 `gpt-5.4-mini`、`deepseek-v4-flash` |
| `:thinking_level` | 可选后缀，覆盖 reasoning_effort。值：`off`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max` |

示例：
```
/model openai_chat/gpt-5.4-mini:high
/model subagent/deepseek-v4-flash
/model deepseek_chat/deepseek-v4-pro:max
```

### CLI 配置管理

`xcode config` 子命令提供多 provider profile 的管理能力，无需手写 JSON：

```
xcode config list                                     # 列出所有 profile
xcode config --project-root <path> list               # 指定项目目录
xcode config add <name>                               # 交互式添加 profile（如 fallback）
xcode config delete <name>                            # 删除 profile
xcode config set <name> <field> <value>               # 修改单个字段
```

示例：
```
xcode config add fallback                             # 交互式配置 fallback provider
xcode config set fallback transport openai_chat       # 设置 transport
xcode config set main thinking false                  # 关闭 thinking
xcode config set main reasoning_effort null            # 删除 reasoning_effort（置空）
xcode config delete subagent                          # 删除 subagent profile
```

支持 set 的字段：`transport`、`chat_model`、`base_url`、`api_key`、`thinking`（true/false）、`clear_thinking`（true/false）、`tool_stream`（true/false）、`reasoning_effort`（字符串或 `null`）。

---

## agent

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `max_steps` | 正整数（可选） | 未设置 | 单次任务最大循环轮次；默认无限制 |
| `compact_threshold` | int | `0` | 消息数阈值；0 关闭 |
| `compact_token_threshold` | int | `0` | token 阈值；0 关闭 |
| `max_recent_messages` | int | `10` | 压缩时保留的近期消息数 |
| `keep_recent_tokens` | int | `20000` | 压缩时为近期原文保留的 token 预算 |
| `reserve_tokens` | int | `16384` | 为模型输出和工具交互保留的 token 预算 |
| `compact_trigger_ratio` | float | `0.7` | 相对上下文窗口的自动压缩触发比例 |
| `tool_workers` | int | `4` | 单个 parallel batch 的最大活跃工具数；小于 1 时按 1 执行 |
| `tool_timeout_seconds` | float | `120.0` | 单个工具调用超时 |
| `watchdog_repeated_tool_limit` | int | `3` | 连续重复同一工具阈值 |

---

## execution_modes

执行模式是同一 agent 上可切换的权限 profile。新会话从
`execution_modes.default_mode` 启动（默认 `act`），恢复会话时恢复已保存的模式；
REPL 可在运行时切换，切换不会丢失会话上下文。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `default_mode` | string | `act` | 新会话的默认执行模式：`plan`、`build` 或 `act`；恢复会话时以持久化的模式为准 |

| mode | 工具可见性 | 内置规则与 fallback |
|---|---|---|
| `plan` | 只读工具，以及 `write_file` / `edit_file` | 只读允许；仅允许写入或编辑 `.xcode/plans/*.md`；fallback=`deny`，不进入 HITL |
| `build` | 全部工具 | 读、写和 shell 允许；fallback=`allow` |
| `act` | 全部工具 | 只读允许，写和 shell 询问；fallback=`ask` |

每个 mode 可配置 `rules` 数组。用户规则追加到内置规则之后，匹配采用 findLast
语义（最后一条匹配规则生效），所以用户规则优先。fallback 不作为 catch-all `*`
规则存储。

```json
{
  "execution_modes": {
    "default_mode": "build",
    "build": {
      "rules": [
        {"action": "bash", "effect": "ask", "command": "git", "subcommand": "push"},
        {"action": "write_file", "effect": "deny", "resource_pattern": "secrets/**"}
      ]
    },
    "act": {
      "rules": [
        {"action": "bash", "effect": "allow", "command": "git", "subcommand_in": ["status", "diff"]}
      ]
    }
  }
}
```

规则字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `action` | string | 必填，工具名或通配符，如 `bash`、`write_file`、`*` |
| `effect` | string | 必填，`allow`、`ask`、`deny` |
| `command` | string/null | shell 主命令，可使用通配符 |
| `subcommand` | string/null | shell 精确子命令 |
| `subcommand_in` | string[]/null | shell 子命令集合，命中任一个即可 |
| `flags_any` | string[]/null | 至少包含一个指定 flag |
| `flags_all` | string[]/null | 必须包含全部指定 flag |
| `resource_pattern` | string/null | 非 shell 的目标路径通配符；shell 中作为额外资源约束 |

结构化条件缺少对应的命令信息时不匹配。短 flag 会排序归一化，例如 `-rf` 与
`-fr` 等价。复合 shell 命令目前只按第一段提取结构化字段，规则需要据此保守配置。

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
固定本地路径：`.local/session_index.json`、`.local/session_artifacts/`、`.local/mcp_cache.json`、`.local/mcp_config.json`。

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

运行时按事件顺序启动命令，以脱敏后的 `HookRecord` JSON 作为 stdin。命令可向
stdout 返回单个 JSON object；空 stdout 等价于 `{}`。`pre_tool` 仅接受
`decision`（`allow` / `deny` / `ask`）和 `arguments`（完整参数 object）。
参数变换后会重新执行工具 schema 校验和 PermissionEngine；hook 不能放宽已有
deny。使用 `/hooks` 查看每项来源、启用状态、运行次数和最近错误。

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
| `approval_policy` | string | `"never"` | `always`、`never` |
| `restricted_dirs` | array | `[]` | 禁止访问目录列表 |
| `permissions` | object | `{}` | 权限组到决策的映射；支持 `read`、`edit`、`shell`、`web`、`subagent`、`skill` |
| `tools` | object | `{}` | 具体工具名到决策的映射；覆盖同名权限组展开结果 |
| `global_default` | string/null | `null` | 无规则匹配时的默认决策：`allow`、`ask`、`deny` |
| `external_directories` | array | `[]` | 外部目录白名单，每条包含 `path`（必填）和 `access`（可选，默认 `"read"`；可选值 `read`/`write`/`read_write`） |
| `sensitive_path_overrides` | array | `[]` | 敏感路径的精确例外；不接受通配符 |

### 静态权限示例

```json
{
  "security": {
    "permissions": {
      "read": "allow",
      "edit": "ask",
      "shell": "ask"
    },
    "tools": {
      "webfetch": "deny"
    },
    "global_default": "ask"
  }
}
```

权限组先展开为具体工具，`tools` 中的具体工具配置随后覆盖。未匹配项使用
`global_default`；未设置 `global_default` 且 `approval_policy` 为 `always` 时，
默认请求确认。路径边界、mode policy、静态策略、shell 可解析性与 mode ruleset
共同参与裁决，`deny` 和 `ask` 不能被更宽松的静态配置绕过。

权限提示与 shell 效果分析不是 OS sandbox。Xcode 不隔离 agent 进程；需要真实
隔离时，应在容器或虚拟机中运行。

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
| `subagent_extra_tools` | string[] | `[]` | 额外允许 subagent 使用的主 agent 工具名；`todowrite` 默认不继承 |
| `shell` | string | `"auto"` | `auto`、`pwsh`、`powershell`、`cmd`、`bash`、`zsh`、`sh`、`fish` |

`todowrite` 是主 agent 默认可用的会话级工具。它以完整列表替换当前清单，
最多允许一个 `in_progress` 项。清单写入
session transcript 和 `RunState`，并在每轮动态上下文中重新注入，因此不会因
compaction 丢失。只有将 `"todowrite"` 加入 `subagent_extra_tools` 时，
subagent 才会共享该会话清单。

`todowrite` 输入使用 `todos` 数组。每个 todo 需要 `id`、`content`、`status`，
可选 `priority`（`high` / `medium` / `low`）。`status` 支持 `pending`、
`in_progress`、`completed`、`cancelled`。`id` 是 xcode 会话恢复和事件关联所需，
因此不同于顶层 TypeScript 参考实现，不能省略。

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

### 工具注册

稳定工具默认注册：`read_file`、`write_file`、`edit_file`、`apply_patch`、
`glob_files`、`find_files`、`list_dir`、`grep_search`、`websearch`、
`webfetch`、`question`、`bash`、`search_tools`、`subagent`、`todowrite`、
`history`、`search_memory`。发现 skill 时注册 `load_skill`；存在
`.local/mcp_config.json` 时注册 `mcp__{server}__{tool}` 动态工具。

`search_tools` 工具按关键字搜索当前已注册工具。
`websearch` 通过 Exa / Parallel MCP provider 搜索网络，默认 Exa；支持 `query`、
`numResults`、`type`（`auto`/`fast`/`deep`）、`livecrawl`（`fallback`/`preferred`）
和 `timeout`。可通过环境变量 `EXA_API_KEY` / `PARALLEL_API_KEY` 或
`OPENCODE_EXPERIMENTAL_PARALLEL` 切换/鉴权。`webfetch` 支持 `markdown`、`text`、
`html` 输出格式，自动解压 gzip/deflate，最多读取 5MB，并在截断时标记结果。
运行时不会按每轮用户问题自动检索 Memory。Agent 通过 `search_memory` 按需合并
检索项目根 `MEMORY.md` 与 `~/.xcode/memory/MEMORY.md`；resume/rebuild 才会在
独立预算内注入相关记忆。
`search_memory` 的 schema 接受必填 `query`，以及可选 `limit`（1-10）、
`scope` 和 `layer`（`all` / `project` / `user`）；工具标记为只读。
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

未配置 `instructions` 时自动回退到 `AGENTS.md`。
配置非空时：先收集配置源，再收集回退文件。配置源与回退文件按路径去重，配置源优先。

所有指令内容按 UTF-8 字节计入预算：≤32KB 完整注入，>32KB 压缩保留关键章节。

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
