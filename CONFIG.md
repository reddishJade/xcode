# Xcode 配置参考

`xcode.config.json` 位于项目根目录。`python -m xcode.main` 和 `build_app(project_root=...)` 会自动读取它；命令行 `--config` 只用于显式指定其他配置文件。相对路径均按 `--project-root` 解析。

**没有 `xcode.config.json` 时的设计原因**：运行时只启用 `core` 工具组（read_file/write_file/bash 等基础能力），这是零配置可用的最小安全集合。实验性功能（MCP/subagent/worktree 等）必须显式 opt-in，避免意外引入复杂依赖和权限风险。

```powershell
.\.venv\Scripts\python.exe -m xcode.main
```

---

## provider

### model_profiles

支持 `main`、`subagent`、`fallback` 三个常用 profile，也可添加自定义 profile。未配置的 profile 由 provider 工厂按默认模型配置补齐。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `transport` | string | `"openai_chat"` | 模型传输协议。支持 `openai_chat`、`openai_responses`、`deepseek_chat`、`mimo_chat`、`chatglm_chat`、`anthropic_messages`。 |
| `chat_model` | string | `"deepseek-v4-flash"` | 聊天模型名。 |
| `base_url` | string | `""` | OpenAI-compatible API 地址。DeepSeek 和 ChatGLM 有默认值，MiMo 需要显式配置。 |
| `api_key` | string | `""` | 显式 API key；留空时按 profile 环境变量和通用环境变量查找。 |
| `thinking` | bool | `true` | 传给支持 thinking 的 provider。 |
| `reasoning_effort` | string/null | `"high"` | 传给支持 reasoning effort 的 provider（DeepSeek）。可选值：`high`、`max`。 |
| `response_format` | object/null | `null` | 传给支持结构化输出的 provider，例如 ChatGLM 的 `{"type":"json_object"}`。 |

#### Provider 默认 base_url

| Provider | 默认 base_url | 说明 |
|----------|--------------|------|
| DeepSeek | `https://api.deepseek.com` | 官方 API |
| ChatGLM | `https://open.bigmodel.cn/api/paas/v4/` | 智谱 AI 官方 API |
| MiMo | `https://api.xiaomimimo.com/v1` | 小米 MiMo 官方 API |

#### DeepSeek 专用说明

- **Thinking Mode**：默认开启，通过 `extra_body={"thinking": {"type": "enabled"}}` 控制
- **reasoning_effort**：默认 `"high"`，原因是 agent 场景需要多步推理和工具选择，`"high"` 是质量和成本的平衡点。复杂 agent 请求（多工具并行、嵌套推理）自动设为 `"max"`，牺牲成本换取正确性。
- **reasoning_content 处理**：
  - 无工具调用时：历史 reasoning_content 自动清除（节省上下文）
  - 有工具调用时：保留当前轮次 reasoning_content，确保 API 调用成功（DeepSeek API 要求）
- **缓存统计**：
  - 优先字段：`prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`（原生）
  - 回退字段：`prompt_tokens_details.cached_tokens`（兼容）
  - 命中率公式：`hit / (hit + miss)`，而非 `hit / prompt_tokens`
  - metrics 字段：`prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`、`cached_tokens`、`cache_hit_rate`

#### MiMo 专用说明

- **Thinking Mode**：mimo-v2.5-pro/mimo-v2.5 默认开启，mimo-v2-flash 默认关闭
- **reasoning_effort**：不支持（MiMo 无此参数）
- **reasoning_content 处理**：建议保留所有历史 reasoning_content 以获得最佳表现
- **缓存统计**：
  - 使用兼容字段：`prompt_tokens_details.cached_tokens`
  - 命中率公式：`hit / (hit + miss)`
  - metrics 字段：`cached_tokens`、`cache_hit_rate`、`cache_hit_tokens`（可选）、`cache_miss_tokens`（可选）
- **默认 base_url**：`https://api.xiaomimimo.com/v1`（小米官方 API）

#### ChatGLM 专用字段

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `clear_thinking` | bool | `false` | 保留式思考开关。`false` 保留历史 reasoning_content，官方推荐 Coding/Agent 场景使用。 |
| `tool_stream` | bool | `true` | 工具流式输出。实时流式传输工具调用参数，减少延迟，仅 `glm-4.6`/`glm-4.7` 支持。 |
| `response_format` | object/null | `null` | 结构化输出格式，常用 `{"type":"json_object"}`。 |

- **轮级思考**：ChatGLM provider 支持单次调用覆盖 `thinking`，不改变 profile 默认值。
- **上下文缓存**：ChatGLM 官方接口按请求内容自动命中缓存；xcode 记录 `cached_tokens`、`cache_hit_rate`、`prompt_tokens` 等 usage 指标。
- **缓存统计**：
  - 使用兼容字段：`prompt_tokens_details.cached_tokens`
  - 命中率公式：`hit / (hit + miss)`
  - metrics 字段：`cached_tokens`、`cache_hit_rate`、`cache_hit_tokens`（可选）、`cache_miss_tokens`（可选）

示例（DeepSeek）：

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "deepseek_chat",
        "chat_model": "deepseek-v4-pro",
        "api_key": "YOUR_DEEPSEEK_API_KEY"
      },
      "subagent": {
        "transport": "deepseek_chat",
        "chat_model": "deepseek-v4-flash"
      }
    }
  }
}
```

示例（ChatGLM）：

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "chatglm_chat",
        "chat_model": "glm-4.7",
        "api_key": "YOUR_ZHIPU_API_KEY"
      }
    }
  }
}
```

示例（MiMo）：

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "mimo_chat",
        "chat_model": "mimo-v2.5-pro",
        "api_key": "YOUR_MIMO_API_KEY"
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

## request_hygiene

请求 hygiene 配置控制发给模型的消息历史压缩策略，**不影响磁盘/session 保存的完整历史**。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | 是否启用请求 hygiene。 |
| `max_tool_result_bytes` | int | `8000` | tool_result 最大字节数，超过时保留 head + tail + signal lines。 |
| `max_tool_arg_length` | int | `1000` | 已完成工具调用的参数字符串最大长度，超过时替换为占位符。 |
| `keep_head_lines` | int | `50` | 压缩 tool_result 时保留的头部行数。 |
| `keep_tail_lines` | int | `50` | 压缩 tool_result 时保留的尾部行数。 |

### Hygiene 规则

1. **超大 tool_result**：
   - 按字节/行数上限保留 head + tail
   - 中间区域提取错误/警告等 signal lines（包含 error/exception/warning/failed 关键字）
   - 添加省略标记说明压缩行数

2. **base64 payload**：
   - 检测连续 base64 字符比例 > 90%
   - 替换为 `<base64 data, {size} bytes>`

3. **超长工具参数**：
   - 仅压缩已完成（有对应 tool_result）的工具调用
   - 超长字符串参数替换为 `<truncated, {length} chars>`
   - 嵌套字典递归压缩

### 设计原因

避免超长工具输出和参数污染缓存热前缀占比，同时保留错误信息用于调试。磁盘日志仍保留完整历史，方便回放和审计。

### 实现位置

- `src/xcode/agent/history.py` - `apply_request_hygiene()` / `repair_tool_pairing()`
- `src/xcode/harness/config.py` - `RequestHygieneConfig`

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

**MCP 配置独立的设计原因**：MCP server 配置包含启动命令、环境变量和工具覆盖，属于外部集成而非核心 agent 配置。独立文件支持本地覆盖（`.local/mcp_config.json` 优先级高于项目根 `mcp_config.json`），避免团队共享配置与个人环境冲突。

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
        "transport": "deepseek_chat",
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

---

## 缓存优化

Xcode 的缓存优化遵循统一的统计口径和工具稳定化原则，确保跨 provider 一致性。

### 统计口径

#### 优先级规则

1. **原生字段优先**（DeepSeek）：`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
2. **兼容字段回退**（OpenAI/ChatGLM/MiMo）：`prompt_tokens_details.cached_tokens`
3. **命中率公式**：`hit / (hit + miss)`，而非 `hit / prompt_tokens`

**设计原因**：DeepSeek 原生 `miss` 口径不保证等于 `prompt_tokens - hit`。使用 `hit / prompt_tokens` 作为分母可能导致统计失真。

#### 实现位置

- `src/xcode/ai/cache.py` - 统一缓存提取逻辑 `extract_cache_usage()`
- `src/xcode/ai/providers/deepseek.py` - DeepSeek 原生字段优先
- `src/xcode/ai/providers/chatglm.py` - 兼容字段回退
- `src/xcode/ai/providers/mimo.py` - 兼容字段回退

#### Metrics 字段

所有 provider 的 `metrics` 字典包含以下缓存相关字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `cached_tokens` | int | 缓存命中 token 数（所有 provider） |
| `cache_hit_rate` | float | 缓存命中率 0.0-1.0（所有 provider） |
| `prompt_cache_hit_tokens` | int | 原生命中字段（仅 DeepSeek） |
| `prompt_cache_miss_tokens` | int | 原生未命中字段（仅 DeepSeek） |
| `cache_hit_tokens` | int | 可选，当使用兼容字段时分解的命中数（ChatGLM/MiMo） |
| `cache_miss_tokens` | int | 可选，当使用兼容字段时分解的未命中数（ChatGLM/MiMo） |

### 工具稳定化

#### 规范化策略

1. **工具列表排序**：按 `tool.name` 字母顺序排序
2. **Schema 键排序**：递归排序所有字典键（`json.dumps(obj, sort_keys=True)`）
3. **指纹生成**：SHA256 hash 前 16 字符

**设计原因**：确保同一组工具在不同调用间字节稳定，避免工具注册顺序或 schema 键序抖动导致缓存前缀漂移。

#### 实现位置

- `src/xcode/ai/cache.py`:
  - `canonical_tool_schema()` - 单个工具规范化
  - `canonical_tools()` - 工具列表排序
  - `tool_catalog_fingerprint()` - 指纹生成

#### 使用示例

```python
from xcode.ai.cache import canonical_tools, tool_catalog_fingerprint
from xcode.ai.types import ToolDefinition

tools = [
    ToolDefinition(name="tool_b", description="B", schema={"z": 1, "a": 2}),
    ToolDefinition(name="tool_a", description="A", schema={"type": "object"}),
]

# 规范化并排序
canonical = canonical_tools(tools)
# 结果：[tool_a, tool_b]，schema 键排序为 {"a": 2, "z": 1}

# 生成指纹
fingerprint = tool_catalog_fingerprint(tools)
# 结果：16 字符 SHA256 hex，顺序无关
```

### Token ROI 原则

缓存优化的目标不是"让缓存数字变高"，而是**提高每个 token 的 ROI**。用户付出的上下文预算应尽量转化为有效推理、代码修改和可执行结论。

#### 优化策略组合

1. **稳定可缓存前缀**：系统提示词、工具 schema、few-shot 进入 immutable prefix
2. **压缩动态历史**：长会话通过 compaction 保留目标、约束、决策、工具结果和未解决事项
3. **控制工具输出**：只在请求边界压缩超长 tool_result，磁盘日志保留完整历史
4. **渐进发现工具**：当工具过多时，用 search/describe/call 模式避免每轮携带所有工具定义
5. **Token-aware 压缩触发**：优先使用 provider 返回的真实 `prompt_tokens` 判断压缩时机
6. **智能重复抑制**：文件变更后清除只读工具历史，避免"编辑后复读"误判

#### Token-Aware 压缩触发

`src/xcode/agent/compaction.py` 提供基于真实 token 的压缩判断：

- `should_compact_token_aware()`: 优先使用 `last_prompt_tokens`（provider 返回），回退到本地估算
- `get_model_soft_threshold()`: 各模型的软阈值（上下文窗口的 ~50-80%）
- `extract_prompt_tokens_from_usage()`: 从 usage 字典提取 `prompt_tokens`

**优先级**：真实 token > 消息数阈值 > 估算 token 阈值

**设计原因**：provider 的 `prompt_tokens` 比本地 4 字符/token 估算更准确，避免接近上下文窗口才触发压缩。

#### 重复工具调用抑制增强

`src/xcode/agent/watchdog.py` 提供文件变更感知的重复检测：

- `RepeatDetector`: 带状态的重复检测器
- `is_file_mutation_tool()`: 识别文件变更工具（write_file、edit_file、bash）
- `is_file_read_tool()`: 识别只读工具（read_file、grep_search、glob_files）
- **智能清除**：文件变更后自动清除只读调用历史，避免误判

**使用示例**：

```python
detector = RepeatDetector(limit=3)
is_repeated, reason = detector.check_and_update(tool_calls)
if is_repeated:
    # 触发重复限制，reason 包含详细信息
    pass
```

#### 验证方式

- **单元测试**：
  - `src/xcode/tests/test_cache_optimization.py` - 缓存统计和工具稳定化
  - `src/xcode/tests/test_history_hygiene.py` - 历史修复和 hygiene
  - `src/xcode/tests/test_compaction.py` - Token-aware 压缩触发
  - `src/xcode/tests/test_watchdog.py` - 重复工具调用抑制
- **Metrics API**：读取 `provider.metrics` 字典查看实时缓存统计
- **真实验证**：通过多轮对话观察缓存命中率稳定性

