# Xcode 项目 API 参考

基于 `src/xcode/` 源码分析，覆盖公开 API。

---

## 目录

1. [CLI 入口](#1-cli-入口)
2. [包顶层导出](#2-包顶层导出)
3. [应用层 `xcode.harness`](#3-应用层-xcodeharness)
4. [Agent 运行时](#4-agent-运行时)
5. [可观测性](#5-可观测性)
6. [AI 层](#6-ai-层)
7. [Agent 核心循环](#7-agent-核心循环)
8. [CLI 交互层](#8-cli-交互层)
9. [编码 Agent 工具](#9-编码-agent-工具)
10. [评测系统](#10-评测系统)
11. [实验性模块](#11-实验性模块)

---

## 1. CLI 入口

`pyproject.toml` → `[project.scripts]`

| 命令 | 入口 | 说明 |
|---|---|---|
| `xcode` | `xcode.main:main` | 主 CLI：解析参数 → 配置发现 → REPL/单轮 |
| `xcode-eval` | `xcode.evals.cli:main` | 评测 CLI |

### `xcode.main` (`src/xcode/main.py`)

**CLI 参数**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `-p, --prompt` | str | 单轮模式 |
| `--project-root` | Path | 项目根目录（默认 cwd） |
| `--config` | Path | 配置文件路径 |
| `--sessions-dir` | Path | REPL 会话目录 |
| `--resume` | bool | 启动恢复选择器 |
| `--setup` | bool | 强制运行配置向导 |

---

## 2. 包顶层导出

`src/xcode/__init__.py` 仅包含包标记，无显式导出。`xcode.harness.__init__` 导出：

| 名称 | 说明 |
|---|---|
| `AgentConfig` | Agent 循环配置 |
| `CancellationToken` | 取消令牌 |
| `ExecutionEnv` | 执行环境 protocol |
| `ExecutionResult` | 执行结果 |
| `ExecutionMode` | 执行模式（plan/review/act） |
| `HookManager` | Hook 管理器 |
| `PermissionPolicy` | 静态权限策略（rules + global_default） |
| `StaticPermission` | 单条静态权限规则 |
| `PermissionResolver` | 约束优先级解析（non_bypassable_deny > deny > ask > allow） |
| `StructuredAgent` | 结构化 agent |
| `StructuredAgentEvent` | Agent 事件 |
| `SubprocessExecutionEnv` | 子进程执行环境 |
| `ToolOutput` | 工具输出（带 metadata 的 str） |
| `ToolSpec` | 工具描述符 |

---

## 3. 应用层 `xcode.harness`

### 3.1 应用装配 `app.py`

```python
@dataclass
class XcodeApp:
    agent: StructuredAgent
    registry: tuple[ToolSpec, ...]
    contextual_state: ContextualRetrievalState | None
    daemon: HeartbeatDaemon | None
    mailbox: AgentMailbox | None
    progress: bool | None
    _model_profiles: dict[str, Any] | None
    _env_files: tuple[Path, ...]
```

方法：`set_model()`、`get_model_info()`、`ask()`、`aask()`、`ask_stream()`、`aask_stream()`、`close()`

```python
def build_app(project_root, env_files=None, agent_config=None,
              skills_dir=None, audit_path=None, runtime_config=None) -> XcodeApp
```

### 3.2 配置 `config.py`

`ProviderTransport = Literal["openai_chat", "anthropic_messages", "chatglm_chat", "deepseek_chat", "mimo_chat"]`
`ExecutionMode = Literal["plan", "build", "act"]`
`PermissionMode = Literal["strict", "normal", "permissive"]`
`ApprovalPolicy = Literal["always", "never"]`

| Dataclass | 关键字段 |
|---|---|
| `AgentConfig` | `max_steps=20, execution_mode="act", compact_threshold=0, compact_token_threshold=0, max_recent_messages=10, tool_workers=4, watchdog_repeated_tool_limit=3` |
| `RequestHygieneConfig` | `enabled=True, max_tool_result_bytes=8000, max_tool_arg_length=1000, keep_head_lines=50, keep_tail_lines=50` |
| `ModelProfileRuntimeConfig` | `transport, chat_model, base_url, api_key, thinking, reasoning_effort, clear_thinking, tool_stream, response_format` |
| `ProviderRuntimeConfig` | `model_profiles: dict[str, ModelProfileRuntimeConfig]` |
| `SecurityRuntimeConfig` | `permission_mode, sandbox_mode, approval_policy, network_access, writable_roots, restricted_dirs, rules, global_default` |
| `ToolsRuntimeConfig` | `enabled_groups=("core",), shell="auto"` |
| `SkillsRuntimeConfig` | (保留) |
| `PromptRuntimeConfig` | `modules: tuple[str, ...]` |
| `PathsRuntimeConfig` | `sessions_dir, skills_dir` |
| `ObservabilityRuntimeConfig` | `audit_path` |
| `DaemonRuntimeConfig` | `enabled=False, interval_seconds=30` |
| `XcodeRuntimeConfig` | 聚合所有子配置 |

函数：`discover_runtime_config()`、`load_runtime_config()`、`resolve_config_path()`

### 3.3 执行环境 `execution_env.py`

```python
@dataclass
class ExecutionResult:
    stdout: str; stderr: str; returncode: int; timed_out: bool; cancelled: bool

class ExecutionEnv(Protocol):
    def run(self, argv, cwd, timeout, cancel_event) -> ExecutionResult: ...

class SubprocessExecutionEnv: ...
class SandboxExecutionEnv: ...
```

### 3.4 工具注册表 `skills.py`

```python
ToolInput = dict[str, Any]
ActionHandler = Callable[[ToolInput], str]
ApprovalCallback = Callable[[ToolSpec, ToolInput], HITLResult]
ToolExecutionStatus = Literal["ok", "denied", "error", "approval_required"]

@dataclass(frozen=True)
class ToolSpec:
    name: str; description: str; input_hint: str; handler: ActionHandler
    schema: dict[str, Any] | None = None
    read_only: bool = False; concurrency_safe: bool = False
    group: str = "core"; execution_mode: ToolExecutionMode | None = None
    counts_as_progress: bool | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()
    builtin: dict[str, Any] | None = None
```

### 3.5 会话管理 `session.py`

```python
@dataclass(frozen=True)
class SessionRecord: type: str; content: Any; created_at: str
@dataclass(frozen=True)
class SessionMetadata: id: str; title: str; summary: str; project_path: str; ...
@dataclass(frozen=True)
class TreeNode: id: str; title: str; fork_type: str | None; depth: int; is_current: bool; is_leaf: bool

class SessionStore:
    def append(self, record_type, content) -> None
    def clear(self) -> None
    def fork_into(self, fork_type=None) -> SessionMetadata
    def fork_clean_into(self, fork_type=None, title=None) -> SessionMetadata
    def load_records(self, path=None) -> list[SessionRecord]
    def resume(self, target: Path | str) -> None
    def switch_branch(self, target: str) -> SessionMetadataView
    def resume_latest(self) -> Path | None
    def rewind_turns(self, turns=1) -> int
    def compact_current_session(self, max_tool_result_chars=200) -> int
    def list_sessions(self, limit=10) -> list[Path]
    def list_session_infos(self, limit=10) -> list[SessionMetadataView]
    def ensure_metadata(self, first_user_text=None) -> SessionMetadata
    def current_metadata(self) -> SessionMetadata | None
    def get_tree(self) -> list[TreeNode]
```

---

## 4. Agent 运行时

`xcode.harness.agent_runtime.__all__` 导出：`CancellationToken`、`ContextualRetrievalState`、`ManagedSubagentRunner`、`RunState`、`StructuredAgent`、`StructuredAgentEvent`、`StructuredAgentResult`、`SubagentStartEvent`、`SubagentEndEvent`、`build_managed_subagent_tools`、`estimate_message_tokens`

### 4.1 StructuredAgent

```python
class StructuredAgent:
    def __init__(self, provider, registry, config=None, approval_callback=None,
                 compactor=None, compact_controller=None, gate=None, runtime=None,
                 audit_logger=None, session_id="local", ...)
    def steer(self, msg) -> None
    def follow_up(self, msg) -> None
    def request_compaction(self) -> None
    def confirm_plan(self) -> None
    def run(self, question, mode=None) -> StructuredAgentResult
    def run_stream(self, question, mode=None) -> Iterator[StructuredAgentEvent]
    async def run_async(self, question, mode=None) -> StructuredAgentResult
```

### 4.2 子 Agent

```python
class ManagedSubagentRunner:
    def submit(self, prompt, timeout, isolation, cwd_override) -> str
    def status(self, job_id) -> SubagentStatus
    def result(self, job_id) -> str
    def cancel(self, job_id) -> None
```

### 4.3 提示词系统

```python
class SystemPromptBuilder:
    def __init__(self, runtime_config: XcodeRuntimeConfig)
    def build(self, modules: tuple[str, ...]) -> str
    def register_module(self, name: str, fn: Callable) -> None

def build_runtime_context_provider(project_root, registry, prompt_builder=None, contextual_state=None, modules=None, shell_spec=None) -> RuntimeContextProvider
```

---

## 5. 可观测性

### 5.1 审计日志 `audit.py`

```python
@dataclass
class AuditRecord: session_id; tool; dynamic_decision; policy_decision; final_status; approved; redacted_input; redacted_output; ...

class JsonlAuditLogger:
    def write(self, record: AuditRecord) -> None

def redact_text(value: Any) -> str
```

### 5.2 钩子系统 `hooks.py`

```python
HookEvent = Literal["pre_tool", "post_tool", "on_error", "on_compact",
                    "before_agent_start", "before_provider_request"]

@dataclass class PreToolEvent: type, tool, input
@dataclass class PostToolEvent: type, tool, input, output
@dataclass class ErrorEvent: type, tool, input, error
@dataclass class CompactEvent: type, metadata
@dataclass class BeforeAgentStartEvent: type, question, mode, metadata
@dataclass class BeforeProviderRequestEvent: type, messages, tools, metadata

class HookManager:
    def register(self, event: HookEvent, callback: HookCallback) -> None
    def remove(self, event: HookEvent, callback: HookCallback) -> None
    def emit(self, record: HookRecord) -> None
```

### 5.3 权限系统 `permissions.py`

```python
PermissionDecision = Literal["allow", "deny", "ask"]
HITLDecision = Literal["allow", "deny"]
HITLScope = Literal["once", "session", "permanent"]

class PermissionEngine:
    def decide(...) -> PermissionEngineResult

class PermissionPolicy:
    def decide(self, tool_name, action_input) -> PermissionDecision | None
```

---

## 6. AI 层

### 6.1 类型系统 `types.py`

```python
KnownApi = Literal["openai-completions", "anthropic-messages", "deepseek-chat", "mimo-chat", "google-gemini"]
KnownProvider = Literal["anthropic", "openai", "deepseek", "mimo", "google", "azure"]
type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
type ReasoningSummary = Literal["auto", "concise", "detailed"]
Transport = Literal["sse", "websocket", "auto"]
CacheRetention = Literal["none", "short", "long"]

@dataclass Cost: input, output, cache_read, cache_write, total
@dataclass Usage: input, output, cache_read, cache_write, total_tokens, cost
@dataclass Model: id, name, api, provider, base_url, reasoning, context_window, max_tokens, cost, thinking_level_map
@dataclass ThinkingBudgets: minimal, low, medium, high, xhigh
@dataclass StreamOptions: temperature, max_tokens, signal, api_key, transport, cache_retention, session_id, reasoning, reasoning_summary, headers, metadata, timeout_ms, max_retries, max_retry_delay_ms, thinking_budgets, thinking_level, tool_choice, top_logprobs, top_p, user, response_extra_params
@dataclass ToolDefinition: name, description, parameters, builtin=None
```

函数：`dump_context()`、`load_context()`

### 6.2 模型注册 `registry.py`

```python
get_providers() -> list[str]
get_models(provider_name) -> list[Model]
get_model(provider_name, model_id) -> Model | None
resolve_model(provider_name, model_id) -> Model
```

### 6.3 Provider 系统 `providers/`

**协议**：`ModelProvider.stream(messages, tools, options) -> AsyncIterator[ProviderEvent]`

**Provider 事件**：`TextDelta`、`ReasoningDelta`、`ToolCallEvent`、`UsageUpdate`、`FinalMessage`

**Provider 类**：`OpenAIChatProvider`、`DeepSeekProvider`、`ChatGLMProvider`、`MiMoProvider`、`FauxProvider`

`build_provider_bundle(ProviderSettings) -> ProviderBundle`

---

## 7. Agent 核心循环

### 7.1 Agent `agent.py`

```python
class Agent:
    def __init__(self, tools: list[AgentTool]) -> None
    def steer(self, msg: AgentMessage) -> None
    def follow_up(self, msg: AgentMessage) -> None
    def run(self, messages, config, *, signal=None, emit=None) -> AgentLoopResult
    def run_stream(self, messages, config, *, signal=None) -> AsyncIterator[AgentEvent]
```

### 7.2 消息类型

```python
AgentMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]
```

### 7.3 事件

```python
AgentEvent = Union[AgentStartEvent, AgentEndEvent, TurnStartEvent, TurnEndEvent,
                   MessageStartEvent, MessageUpdateEvent, MessageEndEvent,
                   ToolExecutionStartEvent, ToolExecutionUpdateEvent,
                   ToolExecutionEndEvent, ThinkingUpdateEvent, CompactionEvent]
```

### 7.4 协议 `protocols.py`

```python
ToolExecutionMode = Literal["sequential", "parallel"]
ContentBlock = Union[TextContent, ImageContent, ToolCallContent, ThinkingContent]

class AgentTool(Protocol):
    name: str; label: str; description: str; parameters: dict
    execution_mode: ToolExecutionMode; examples: list
    def execute(tool_call_id, params, signal, on_update) -> AgentToolResult
```

### 7.5 工具执行 `tool_execution.py`

```python
def execute_tool_calls(current_context, assistant_message, tool_calls, config, signal, emit) -> ExecutedToolBatch
def partition_tool_calls_for_execution(current_context, tool_calls) -> list[list[ToolCallContent]]
```

---

## 8. CLI 交互层

### 8.1 REPL 命令

| 命令 | 说明 |
|---|---|
| `/help` | 显示帮助 |
| `/clear` | 新会话 |
| `/fork [type]` | 分支（explore/verify/isolate） |
| `/rewind [n]` | 回退 n 轮 |
| `/resume [last/\|id]` | 恢复会话 |
| `/sessions` | 列出会话 |
| `/tree` | 会话树 |
| `/branch [list\|tree\|id]` | 切换分支 |
| `/model [name]` | 切换模型 |
| `/effort <level>` | 推理 effort |
| `/thinking on/off` | thinking 显示 |
| `/plan` | 只读检查 |
| `/review` | 只读审查 |
| `/act [--clear]` | 执行模式 |
| `/verbose [normal\|verbose\|debug]` | 详细级别 |
| `/debug on/off` | 调试模式 |
| `/queue [on/off]` | 排队输入 |
| `/compact` | 手动压缩 |
| `/permissions [revoke\|clear]` | 权限管理 |
| `/tool NAME INPUT\|list` | 手动工具 |
| `/exit\|/quit` | 退出 |

### 8.2 辅助

`expand_file_references(text, project_root) -> tuple[str, list[FileReference]]`，匹配 `@path` 并注入内容。

`ReplCompleter`：补全命令、effort 级别、模型名、工具名、shell 路径、`@file`。

`TerminalMarkdownRenderer`：终端 markdown 渲染。

---

## 9. 编码 Agent 工具

### 9.1 注册表

```python
def build_project_scoped_registry(project_root, enabled, contextual_state, shell_spec, cancel_event, env) -> tuple[ToolSpec, ...]
```

### 9.2 工具构建函数

| 函数 | 模块 | 工具 |
|---|---|---|
| `build_file_tools()` | `file.py` | `read_file`、`write_file`、`edit_file` |
| `build_code_tools()` | `code_search.py` | `glob_files`、`find_files`、`grep_search`、`ls` |
| `build_bash_tool()` | `bash.py` | `bash` |

### 9.3 辅助

`ShellSpec`、`detect_shell(config)`、`build_shell_argv(spec, command)`、`is_path_blocked(root, path)`、`truncate_output(text, max_lines, max_bytes)`、`resolve_read_path(root, raw_path)`

---

## 10. 评测系统

### 10.1 CLI `evals.cli.py`

参数：`--suite`、`--list-suites`、`--show-suite`、`--list-benchmarks`、`--tasks`、`--benchmark`、`--benchmark-path`、`--real`、`--project-root`、`--output-dir`、`--allow-project-mutation`、`--trials`、`--limit`

### 10.2 核心类型

```python
@dataclass EvalTask: id, prompt, mode, expected_answer_contains, expected_tool_calls, disallowed_tool_calls, max_tool_errors, llm_judge_criteria, tags, metadata
@dataclass TrialResult: task_id, trial, scores, ...
@dataclass EvalReport: report_id, timestamp, suite_name, trials, ...
```

### 10.3 Runner

```python
class EvalRunner:
    def __init__(self, tasks, app_factory, output_dir, trials_per_task=1)
    def run(self) -> EvalReport
```

---

## 11. 实验性模块

| 子模块 | 说明 |
|---|---|
| `mcp` | stdio MCP client、动态工具注册 |
| `memory` | `MEMORY.md` 记忆系统、BM25 召回 |
| `plugins` | `.local/plugins/*.py` 动态加载 |
