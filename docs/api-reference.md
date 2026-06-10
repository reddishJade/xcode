# Xcode 项目 API 参考

> 本文档基于 `src/xcode` 源码分析生成，覆盖所有公开（非 `_` 开头）API。

---

## 目录

1. [CLI 入口](#1-cli-入口)
2. [包顶层导出 `xcode`](#2-包顶层导出-xcode)
3. [应用层 `xcode.harness`](#3-应用层-xcodeharness)
   - [3.1 应用装配 —— app](#31-应用装配--app)
   - [3.2 配置 —— config](#32-配置--config)
   - [3.3 执行环境 —— execution_env](#33-执行环境--execution_env)
   - [3.4 工具注册表 —— skills](#34-工具注册表--skills)
   - [3.5 Skill 加载器 —— skill_loader](#35-skill-加载器--skill_loader)
   - [3.6 会话管理 —— session](#36-会话管理--session)
4. [Agent 运行时 `xcode.harness.agent_runtime`](#4-agent-运行时-xcodeharnessagent_runtime)
   - [4.1 StructuredAgent —— structured](#41-structuredagent--structured)
   - [4.2 事件翻译 —— event_translation](#42-事件翻译--event_translation)
   - [4.3 结果 —— result](#43-结果--result)
   - [4.4 子 Agent —— subagent](#44-子-agent--subagent)
   - [4.5 提示词系统 —— prompting](#45-提示词系统--prompting)
5. [可观测性 `xcode.harness.observability`](#5-可观测性-xcodeharnessobservability)
   - [5.1 审计日志 —— audit](#51-审计日志--audit)
   - [5.2 钩子系统 —— hooks](#52-钩子系统--hooks)
   - [5.3 权限系统 —— permissions](#53-权限系统--permissions)
6. [AI 层 `xcode.ai`](#6-ai-层-xcodeai)
   - [6.1 类型系统 —— types](#61-类型系统--types)
   - [6.2 模型注册中心 —— registry](#62-模型注册中心--registry)
   - [6.3 工具参数校验 —— validation](#63-工具参数校验--validation)
   - [6.4 Provider 系统 —— providers](#64-provider-系统--providers)
7. [Agent 核心循环 `xcode.agent`](#7-agent-核心循环-xcodeagent)
   - [7.1 Agent —— agent.py](#71-agent--agentpy)
   - [7.2 循环配置 —— config](#72-循环配置--config)
   - [7.3 事件定义 —— events](#73-事件定义--events)
   - [7.4 消息类型 —— messages](#74-消息类型--messages)
   - [7.5 协议 —— protocols](#75-协议--protocols)
   - [7.6 工具执行 —— tool_execution](#76-工具执行--tool_execution)
8. [CLI 交互层 `xcode.cli`](#8-cli-交互层-xcodecli)
   - [8.1 命令系统 —— commands](#81-命令系统--commands)
   - [8.2 REPL —— repl](#82-repl--repl)
   - [8.3 REPL 命令 —— repl_commands](#83-repl-命令--repl_commands)
   - [8.4 REPL 渲染 —— repl_rendering](#84-repl-渲染--repl_rendering)
   - [8.5 REPL 会话 —— repl_sessions](#85-repl-会话--repl_sessions)
   - [8.6 REPL 设置 —— repl_settings](#86-repl-设置--repl_settings)
   - [8.7 REPL 工具 —— repl_tools](#87-repl-工具--repl_tools)
   - [8.8 设置向导 —— setup_wizard](#88-设置向导--setup_wizard)
   - [8.9 工具目录 —— tool_catalog](#89-工具目录--tool_catalog)
   - [8.10 自动补全 —— completion](#810-自动补全--completion)
   - [8.11 文件引用 —— file_refs](#811-文件引用--file_refs)
   - [8.12 Markdown 渲染 —— markdown](#812-markdown-渲染--markdown)
   - [8.13 HITL 处理 —— repl_hitl](#813-hitl-处理--repl_hitl)
9. [编码 Agent 工具 `xcode.coding_agent`](#9-编码-agent-工具-xcodecoding_agent)
   - [9.1 注册表构建 —— registry](#91-注册表构建--registry)
   - [9.2 工具工厂 —— tools](#92-工具工厂--tools)
10. [评测系统 `xcode.evals`](#10-评测系统-xcodeevals)
    - [10.1 CLI —— evals.cli](#101-cli--evalscli)
    - [10.2 Schema —— evals.schema](#102-schema--evalsschema)
    - [10.3 Runner —— evals.runner](#103-runner--evalsrunner)
11. [实验性模块 `xcode.experimental`](#11-实验性模块-xcodeexperimental)

---

## 1. CLI 入口

**定义**: `pyproject.toml` → `[project.scripts]`

| 命令 | 入口 | 说明 |
|---|---|---|
| `xcode` | `xcode.main:main` | 主 CLI：解析参数 → 配置发现 → 单轮/REPL 模式 |
| `xcode-eval` | `xcode.evals.cli:main` | 评测 CLI：运行任务套件或外部基准测试 |

### `xcode.main` (`src/xcode/main.py`)

```python
def parse_args() -> argparse.Namespace
def main() -> int
```

**CLI 参数**:

| 参数 | 类型 | 说明 |
|---|---|---|
| `-p, --prompt` | str | 单轮模式：执行一个 prompt 后退出 |
| `--project-root` | Path | 项目根目录 (默认 cwd) |
| `--config` | Path | 配置文件路径 (默认 `xcode.config.json`) |
| `--sessions-dir` | Path | REPL 会话目录 |
| `--resume` | bool | 启动时打开恢复选择器 |
| `--setup` | bool | 强制运行配置向导 |

---

## 2. 包顶层导出 `xcode`

**定义**: `src/xcode/__init__.py`, `__all__ = [...]`

| 名称 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `AgentConfig` | class | `harness.config` | Agent 循环配置（max_steps, execution_mode, compact_threshold 等） |
| `XcodeApp` | class | `harness.app` | 应用句柄：agent、registry、生命周期 |
| `StructuredAgent` | class | `harness.agent_runtime` | 结构化工具调用 agent，含权限/审计/事件流 |
| `StructuredAgentResult` | class | `harness.agent_runtime` | Agent 执行结果 |
| `ToolSpec` | class | `harness.skills` | 工具描述符（name, handler, schema, risk, group） |
| `build_app` | function | `harness.app` | 从配置装配完整 XcodeApp |

---

## 3. 应用层 `xcode.harness`

### 3.1 应用装配 —— app

**定义**: `src/xcode/harness/app.py`

```python
@dataclass
class XcodeApp:
    agent: StructuredAgent
    registry: tuple[ToolSpec, ...]
    contextual_state: ContextualRetrievalState | None
    daemon: HeartbeatDaemon | None
    mailbox: AgentMailbox | None
    progress: type[TaskProgress] | None
```

| 方法 | 签名 | 说明 |
|---|---|---|
| `set_model` | `(*, model, profile, base_url, api_key, thinking, reasoning_effort) -> str` | 动态切换模型/配置 |
| `get_model_info` | `() -> dict[str, str]` | 获取当前模型信息 |
| `ask` | `(question: str) -> str` | 同步提问，返回最终答案 |
| `aask` | `(question: str) -> Awaitable[str]` | 异步提问 |
| `ask_stream` | `(question, mode) -> Iterator[StructuredAgentEvent]` | 同步流式提问 |
| `aask_stream` | `(question, mode) -> AsyncIterator[StructuredAgentEvent]` | 异步流式提问 |
| `close` | `() -> None` | 关闭应用，清理资源 |

```python
def build_app(
    project_root: Path,
    env_files: tuple[Path, ...] | None = None,
    agent_config: Any | None = None,
    skills_dir: Path | None = None,
    audit_path: Path | None = None,
    runtime_config: XcodeRuntimeConfig | None = None,
) -> XcodeApp
```

### 3.2 配置 —— config

**定义**: `src/xcode/harness/config.py`

| 类型别名 | 值 |
|---|---|
| `ProviderTransport` | `Literal["openai_chat", "openai_responses", "anthropic_messages", "chatglm_chat", "deepseek_chat", "mimo_chat"]` |
| `ExecutionMode` | `Literal["plan", "review", "act"]` |

| DataClass | 关键字段 | 说明 |
|---|---|---|
| `AgentConfig` | `max_steps=20, execution_mode="act", compact_threshold=0, compact_token_threshold=0, max_recent_messages=10, tool_workers=4, watchdog_repeated_tool_limit=3` | Agent 循环配置 |
| `ModelProfileRuntimeConfig` | `transport, chat_model, base_url, api_key, thinking, reasoning_effort, clear_thinking, tool_stream, response_format` | 模型 profile 配置 |
| `ProviderRuntimeConfig` | `model_profiles: dict[str, ModelProfileRuntimeConfig]` (含 main/subagent/fallback) | Provider 配置组 |
| `ToolsRuntimeConfig` | `network_commands, enabled_groups, shell` | 工具组配置 |
| `SkillsRuntimeConfig` | `auto_trigger` | Skill 配置 |
| `PromptRuntimeConfig` | `modules: tuple` (identity, tool_discipline, tools, ...) | Prompt 模块选择 |
| `PathsRuntimeConfig` | `sessions_dir, skills_dir` | 路径配置 |
| `ObservabilityRuntimeConfig` | `audit_path` | 可观测性配置 |
| `DaemonRuntimeConfig` | `enabled, interval_seconds` | 守护线程配置 |
| `XcodeRuntimeConfig` | 所有子配置的聚合 | 完整运行时配置 |
| `PROFILE_MAIN = "main"` | 常量 | 主模型 profile 名称 |
| `PROFILE_SUBAGENT = "subagent"` | 常量 | 子 agent profile 名称 |
| `PROFILE_FALLBACK = "fallback"` | 常量 | fallback profile 名称 |

| 函数 | 签名 | 说明 |
|---|---|---|
| `discover_runtime_config` | `(project_root, explicit_path) -> XcodeRuntimeConfig` | 发现并加载配置 |
| `load_runtime_config` | `(path) -> XcodeRuntimeConfig` | 从文件加载配置 |
| `to_agent_config` | `(config) -> AgentConfig` | 提取 AgentConfig |
| `resolve_config_path` | `(project_root, path) -> Path \| None` | 解析路径（支持相对路径） |

### 3.3 执行环境 —— execution_env

**定义**: `src/xcode/harness/execution_env.py`

```python
@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    cancelled: bool

class ExecutionEnv(Protocol):
    def run(self, argv, cwd, timeout, cancel_event) -> ExecutionResult: ...

class SubprocessExecutionEnv: ...  # 真实子进程执行
class SandboxExecutionEnv: ...    # 测试 Mock 环境
```

### 3.4 工具注册表 —— skills

**定义**: `src/xcode/harness/skills.py`

```python
ToolInput = dict[str, Any]
ActionHandler = Callable[[ToolInput], str]
RiskEvaluator = Callable[[ToolInput], str]
ApprovalCallback = Callable[[ToolSpec, ToolInput], HITLResult]

ToolExecutionStatus = Literal["ok", "denied", "error", "approval_required"]

class ToolOutput(str):  # 带 metadata 的字符串
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_hint: str
    handler: ActionHandler
    risk: str = "low"                    # "low" | "high"
    schema: dict | None = None
    read_only: bool = False
    concurrency_safe: bool = False
    risk_evaluator: RiskEvaluator | None = None
    group: str = "core"
    execution_mode: ToolExecutionMode | None = None
    counts_as_progress: bool | None = None
    examples: list[dict] = field(default_factory=list)
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()

@dataclass(frozen=True)
class ToolExecutionResult:
    status: ToolExecutionStatus
    content: str
    metadata: dict | None = None
```

| 函数 | 签名 | 说明 |
|---|---|---|
| `resolve_project_path` | `(project_root, raw_path) -> Path` | 安全解析项目内路径 |
| `build_tool_prompt` | `(registry) -> str` | 构建工具 system prompt |
| `build_tool_guidelines` | `(registry) -> str` | 构建工具使用指南 |
| `run_tool` | `(registry, action, action_input, approval_callback, permission_policy) -> str` | 执行工具（简化版） |
| `run_tool_result` | `(registry, action, action_input, ...) -> ToolExecutionResult` | 执行工具（完整版） |
| `stringify_tool_input` | `(action_input) -> str` | 序列化工具输入 |

**常量**: `RISK_LOW = "low"`, `RISK_HIGH = "high"`, `STATUS_OK`, `STATUS_DENIED`, `STATUS_ERROR`, `STATUS_APPROVAL_REQUIRED`, `BASE_REGISTRY = ()`

### 3.5 Skill 加载器 —— skill_loader

**定义**: `src/xcode/harness/skill_loader.py`

```python
@dataclass(frozen=True)
class SkillMatch:
    name: str; score: float
    matched_use_when: tuple[str, ...]; matched_dont_use_when: tuple[str, ...]

@dataclass(frozen=True)
class SkillMetadata:
    name: str; description: str; path: Path
    use_when: tuple[str, ...]; dont_use_when: tuple[str, ...]
    risk: str; tools: tuple[str, ...]

class SkillLoader:
    def __init__(self, skills_dir: Path) -> None
    def get_descriptions(self) -> str
    def get_catalog(self, question: str | None = None) -> str
    def get_content(self, name: str) -> str

def route_skills(question: str, skills: dict[str, SkillMetadata]) -> list[SkillMatch]
def build_skill_loader_tool(loader: SkillLoader) -> ToolSpec
```

### 3.6 会话管理 —— session

**定义**: `src/xcode/harness/session.py`

```python
FORK_TYPES = frozenset(["explore", "verify", "isolate"])

@dataclass(frozen=True)
class SessionRecord:
    type: str; content: Any; created_at: str

@dataclass(frozen=True)
class SessionMetadata:
    id: str; title: str; summary: str; project_path: str
    transcript_path: str; created_at: str; updated_at: str
    parent_id: str | None; fork_type: str | None

@dataclass(frozen=True)
class SessionMetadataView:
    id: str; title: str; summary: str; updated_at: str; path: Path
    parent_id: str | None; fork_type: str | None

@dataclass(frozen=True)
class TreeNode:
    id: str; title: str; fork_type: str | None
    depth: int; is_current: bool; is_leaf: bool

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
    def update_summary(self) -> SessionMetadata | None
    def current_metadata(self) -> SessionMetadata | None
    def protocol_info(self) -> SessionProtocolInfo
    def get_tree(self) -> list[TreeNode]
```

---

## 4. Agent 运行时 `xcode.harness.agent_runtime`

**`__all__` 导出**: `CancellationToken`, `ContextualRetrievalState`, `ManagedSubagentRunner`, `PromptContext`, `StructuredAgent`, `StructuredAgentEvent`, `StructuredAgentResult`, `SystemPromptBuilder`, `build_managed_subagent_tools`, `build_runtime_context_provider`, `estimate_message_tokens`

### 4.1 StructuredAgent —— structured

**定义**: `src/xcode/harness/agent_runtime/structured.py`

```python
class StructuredAgent:
    def __init__(
        self,
        provider: ModelProvider,
        registry: tuple[ToolSpec, ...],
        config: AgentConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        compactor: StructuredCompactor | None = None,
        manual_compact_requested: Callable | None = None,
        compact_controller: CompactController | None = None,
        audit_logger: Callable | None = None,
        session_id: str = "local",
        permission_policy: PermissionPolicy | None = None,
        hook_manager: HookManager | None = None,
        runtime_context_provider: RuntimeContextProvider | None = None,
        cancellation_token: CancellationToken | None = None,
        fallback_provider: ModelProvider | None = None,
        project_root: Path | None = None,
        request_hygiene: RequestHygieneConfig | None = None,
    )

    def steer(self, msg: AgentMessage) -> None
    def follow_up(self, msg: AgentMessage) -> None
    def request_compaction(self) -> None
    def confirm_plan(self) -> None
    def run(self, question: str, mode=None) -> StructuredAgentResult
    async def run_async(self, question: str, mode=None) -> StructuredAgentResult
    async def arun(self, question: str, mode=None) -> StructuredAgentResult
    def run_stream(self, question: str, mode=None) -> Iterator[StructuredAgentEvent]
    async def arun_stream(self, question: str, mode=None) -> AsyncIterator[StructuredAgentEvent]
```

**类型别名**:

```python
StructuredCompactor = Callable[[list[dict]], list[dict]]
RuntimeContextProvider = Callable[[str], list[str]]
```

**压缩与请求裁剪**:

- `compactor` 负责状态级上下文压缩；触发后返回的新消息会替换后续模型请求上下文。
- `request_hygiene` 负责请求边界裁剪，只影响发送给 provider 的消息副本，不修改 `HistoryManager` 和 session 日志。
- 压缩触发优先使用上一轮 provider 返回的 `prompt_tokens`，没有真实 usage 时回退到配置的消息数和估算 token 阈值。

### 4.2 事件翻译 —— event_translation

**定义**: `src/xcode/harness/agent_runtime/event_translation.py`

```python
@dataclass
class StructuredAgentEvent:
    type: str      # "text_delta" | "tool_use" | "tool_result" | "reasoning" | "final" | ...
    step: int
    data: Any

@dataclass
class ToolResultBlock:
    tool_use_id: str; content: str; status: str; type: str
```

### 4.3 结果 —— result

**定义**: `src/xcode/harness/agent_runtime/result.py`

```python
@dataclass
class StructuredAgentResult:
    answer: str
    messages: list[AgentMessage]
    steps: int
    tool_calls: int
    stopped_by_limit: bool
    metrics: AgentLoopMetrics | None
    stopped_by_watchdog: bool
    watchdog_reason: str | None
    needs_follow_up: bool
```

### 4.4 子 Agent —— subagent

**定义**: `src/xcode/harness/agent_runtime/subagent.py`

**类型别名**: `SubagentStatus = Literal["running", "done", "cancelled", "failed"]`, `RunChild = Callable[[str, str, Path | None], Awaitable[str]]`

```python
@dataclass
class SubagentJob:
    id: str; prompt: str; created_at: datetime
    timeout_seconds: float; future: asyncio.Future
    isolation: bool; cwd_override: Path | None; worktree_task_id: str | None

class ManagedSubagentRunner:
    def submit(self, prompt, timeout, isolation, cwd_override) -> str  # 返回 job_id
    def status(self, job_id) -> SubagentStatus
    def result(self, job_id) -> str
    def cancel(self, job_id) -> None
    def sweep_finished(self) -> None
    async def shutdown(self) -> None

def build_managed_subagent_tools(runner) -> tuple[ToolSpec, ...]
```

**创建的子 Agent 工具**: `submit_subagent`, `check_subagent`, `cancel_subagent`

### 4.5 提示词系统 —— prompting

**定义**: `src/xcode/harness/agent_runtime/prompting.py`

**常量**: `CORE_IDENTITY` (核心身份提示词), `TOOL_DISCIPLINE` (工具纪律 XML)

```python
@dataclass
class PromptContext:
    system_prompt: str

class SystemPromptBuilder:
    def __init__(self, runtime_config: XcodeRuntimeConfig)
    def build(self, modules: tuple[str, ...]) -> str
    def register_module(self, name: str, fn: Callable) -> None

def build_runtime_context_provider(
    project_root: Path, contextual_state, compact_controller, runtime_config
) -> RuntimeContextProvider | None
```

其他导出：`CancellationToken`, `ContextualRetrievalState`, `estimate_message_tokens`

---

## 5. 可观测性 `xcode.harness.observability`

### 5.1 审计日志 —— audit

**定义**: `src/xcode/harness/observability/audit.py`

```python
@dataclass
class AuditRecord:
    session_id: str; tool: str; static_risk: str; dynamic_decision: str
    policy_decision: str | None; final_status: str; approved: bool
    redacted_input: str; redacted_output: str; timestamp: str
    approval_scope: str | None; user_decision: str | None

class JsonlAuditLogger:
    def write(self, record: AuditRecord) -> None

def redact_text(value: Any) -> str
```

### 5.2 钩子系统 —— hooks

**定义**: `src/xcode/harness/observability/hooks.py`

```python
HookEvent = Literal["pre_tool", "post_tool", "on_error", "on_compact",
                    "before_agent_start", "before_provider_request"]
HookCallback = Callable[[HookRecord], None]

@dataclass
class HookRecord:
    event: str; tool: str | None; input: str | None
    output: str | None; error: str | None; metadata: dict | None

@dataclass class PreToolEvent:    type, tool, input
@dataclass class PostToolEvent:   type, tool, input, output
@dataclass class ErrorEvent:      type, tool, input, error
@dataclass class CompactEvent:    type, metadata
@dataclass class BeforeAgentStartEvent:     type, question, mode, metadata
@dataclass class BeforeProviderRequestEvent: type, messages, tools, metadata
HarnessEvent = Union[PreToolEvent, PostToolEvent, ErrorEvent, CompactEvent,
                     BeforeAgentStartEvent, BeforeProviderRequestEvent]

class HookManager:
    def register(self, event: HookEvent, callback: HookCallback) -> None
    def remove(self, event: HookEvent, callback: HookCallback) -> None
    def subscribe(self, event: HookEvent, callback: HarnessCallback) -> None
    def unsubscribe(self, event: HookEvent, callback: HarnessCallback) -> None
    def emit(self, record: HookRecord) -> None
```

### 5.3 权限系统 —— permissions

**定义**: `src/xcode/harness/observability/permissions.py`

```python
PermissionDecision = Literal["allow", "deny", "ask"]
HITLDecision = Literal["allow", "deny"]
HITLScope = Literal["once", "session", "permanent"]

@dataclass
class HITLResult:
    decision: HITLDecision; scope: HITLScope

@dataclass
class PermissionRule:
    tool: str; decision: PermissionDecision; input_contains: str | None

@dataclass
class PermissionCheckResult:
    blocked: bool; reason: str; decision: PermissionDecision | None; metadata: dict | None

class PermissionPolicy:
    def decide(self, tool_name, action_input) -> PermissionDecision | None
    def grant(self, tool: str, decision: PermissionDecision, input_contains: str | None = None) -> None
    def revoke(self, tool: str, input_contains: str | None = None) -> None
    def clear(self) -> None

class SessionPermissionPolicy(PermissionPolicy):  # 会话级权限覆盖
class PersistentPermissionStore(PermissionPolicy):  # 文件持久化权限
class SettingsSandboxPermissionPolicy(PermissionPolicy):  # settings.json 沙箱策略
class CompositePermissionPolicy(PermissionPolicy):  # 组合策略

def check_tool_permission(
    tool_name: str, action_input: str, *, permission_policy=None,
    approval_callback=None, tool_spec=None, tool_input=None,
    high_risk_requires_approval=True,
) -> PermissionCheckResult
```

---

## 6. AI 层 `xcode.ai`

### 6.1 类型系统 —— types

**定义**: `src/xcode.ai/types.py`

| 类型别名 | 值 |
|---|---|
| `KnownApi` | `Literal["openai-completions", "anthropic-messages", "deepseek-chat", "mimo-chat", "google-gemini"]` |
| `Api` | `KnownApi \| str` |
| `KnownProvider` | `Literal["anthropic", "openai", "deepseek", "mimo", "google", "azure"]` |
| `Provider` | `KnownProvider \| str` |
| `ThinkingLevel` | `Literal["off", "minimal", "low", "medium", "high", "xhigh"]` |
| `Transport` | `Literal["sse", "websocket", "auto"]` |
| `CacheRetention` | `Literal["none", "short", "long"]` |

| DataClass | 关键字段 |
|---|---|
| `Cost` | `input, output, cache_read, cache_write, total: float` |
| `Usage` | `input, output, cache_read, cache_write, total_tokens, cost: Cost` |
| `Model[T]` | `id, name, api, provider, base_url, reasoning, context_window, max_tokens, cost, thinking_level_map` |
| `ThinkingConfig` | `enabled: bool, effort: str \| None, clear_thinking: bool` |
| `ThinkingBudgets` | `minimal, low, medium, high: int` |
| `StreamOptions` | `temperature, max_tokens, signal, api_key, transport, cache_retention, session_id, reasoning, headers, metadata, timeout_ms, max_retries, max_retry_delay_ms, on_payload, on_response, thinking_budgets, thinking_level` |
| `TextContent` | `type: "text", text: str` |
| `ImageContent` | `type: "image", source: dict \| None` |
| `ToolCallContent` | `type: "tool_call", id, name, arguments: dict \| None` |
| `ThinkingContent` | `type: "thinking", thinking, signature` |
| `ToolResultContent` | `type: "tool_result", tool_use_id, content, status` |
| `ToolDefinition` | `name: str, description: str, schema: dict[str, Any]` |

| 函数 | 说明 |
|---|---|
| `dump_context(system_prompt, messages) -> str` | 序列化上下文 |
| `load_context(data: str) -> (system_prompt, messages)` | 反序列化上下文 |

### 6.2 模型注册中心 —— registry

**定义**: `src/xcode.ai/registry.py`

| 函数 | 签名 |
|---|---|
| `get_providers` | `() -> list[str]` |
| `get_models` | `(provider_name: str) -> list[Model]` |
| `get_model` | `(provider_name, model_id) -> Model \| None` |
| `resolve_model` | `(provider_name, model_id) -> Model` |

### 6.3 工具参数校验 —— validation

**定义**: `src/xcode.ai/validation.py`

```python
class ToolValidationError(ValueError): ...
def validate_tool_call(tools: list[ToolDefinition], name: str, arguments: dict) -> dict
```

### 6.4 Provider 系统 —— providers

**定义**: `src/xcode.ai/providers/__init__.py`

**`__all__`**: `AnthropicProvider`, `ChatGLMProvider`, `DeepSeekProvider`, `FauxProvider`, `FauxResponse`, `MiMoProvider`, `OpenAIChatProvider`, `OpenAIResponsesProvider`, `ProviderBundle`, `ProviderSettings`, `PROVIDER_REGISTRY`, `build_provider_bundle`, `faux_final`, `faux_text`, `faux_thinking`, `faux_tool_call`, `faux_usage`, `register_faux_provider`

**模块**:

| 模块 | 关键内容 |
|---|---|
| `protocol.py` | `ModelProvider` 协议: `stream(messages, tools, options) -> AsyncIterator[ProviderEvent]` |
| `factory.py` | `build_provider_bundle(settings) -> ProviderBundle`, `ProviderSettings`, `ModelProfileConfig`, `load_env_file` |
| `router.py` | `RouterProvider` — 按 `RouterFn` 在多个 provider 间路由 |
| `runtime.py` | `ProviderRuntime` — 重试+限流包装器, `RetryPolicy`, `RateLimitPolicy`, `classify_api_error`, `is_transient_provider_error` |
| `openai_compat.py` | `OpenAICompatProvider` — OpenAI 兼容基类 |
| `metrics.py` | `ProviderMetricsMixin` — provider 指标 mixin |
| `codec.py` | `make_schema_strict`, `to_chat_tool`, `to_responses_tool`, `normalize_cross_provider_messages`, `to_chat_messages`, `to_responses_input`, `tool_call_from_chat` |
| `stream_codec.py` | `chat_stream_to_events`, `responses_stream_to_events`, `responses_to_events`, `parse_tool_arguments` |
| `openai.py` | `OpenAIChatProvider`, `OpenAIResponsesProvider` |
| `deepseek.py` | `DeepSeekProvider`, `DEEPSEEK_BASE_URL` |
| `chatglm.py` | `ChatGLMProvider`, `CHATGLM_BASE_URL` |
| `mimo.py` | `MiMoProvider`, `MIMO_BASE_URL` |
| `anthropic.py` | `AnthropicProvider` (stub — raises `RuntimeError`) |
| `faux.py` | `FauxProvider`, `FauxResponse`, `faux_text/faux_thinking/faux_tool_call/faux_usage/faux_final`, `register_faux_provider` |

**事件类型** (`xcode.ai.events.py`):

```
ProviderEvent = Union[TextDelta, ToolCallEvent, UsageUpdate, FinalMessage, ReasoningDelta]
```

---

## 7. Agent 核心循环 `xcode.agent`

### 7.1 Agent —— agent.py

**定义**: `src/xcode/agent/agent.py`

```python
class Agent:
    def __init__(self, tools: list[AgentTool]) -> None
    def steer(self, msg: AgentMessage) -> None        # 注入优先级消息（下一轮前消费）
    def follow_up(self, msg: AgentMessage) -> None     # 注入跟进消息（当前循环后追加）
    @property
    def last_result(self) -> AgentLoopResult | None
    def run(self, messages, config, *, signal=None, emit=None) -> AgentLoopResult
    def run_stream(self, messages, config, *, signal=None) -> AsyncIterator[AgentEvent]
```

### 7.2 循环配置 —— config

**定义**: `src/xcode/agent/config.py`

```python
@dataclass
class AgentContext:
    system_prompt: str; messages: list[AgentMessage]; tools: list[AgentTool]

@dataclass class BeforeToolCallContext:  (assistant_message, tool_call, args, context)
@dataclass class BeforeToolCallResult:   (block: bool, reason: str)
@dataclass class AfterToolCallContext:   (assistant_message, tool_call, args, result, is_error, context)
@dataclass class AfterToolCallResult:    (content, details, is_error, terminate)
@dataclass class ShouldStopAfterTurnContext: (message, tool_results, context, new_messages)
@dataclass class AgentLoopTurnUpdate:     (context, model, thinking_level)
@dataclass class AgentLoopMetrics:  (llm_calls, tool_calls, steps, ...)
@dataclass class AgentLoopResult:   (messages, steps, stopped_by_limit, stopped_by_watchdog, ...)

@dataclass
class AgentLoopConfig:
    provider: ModelProvider
    convert_to_llm: MessageConverter
    transform_context: ContextTransformer | None
    before_tool_call: BeforeToolCallHook | None
    after_tool_call: AfterToolCallHook | None
    prepare_next_turn: PrepareNextTurnHook | None
    should_stop_after_turn: ShouldStopAfterTurnHook | None
    get_steering_messages: MessageQueueGetter | None
    get_follow_up_messages: MessageQueueGetter | None
    steering_mode: QueueMode
    follow_up_mode: QueueMode
    max_steps: int
    max_step_retries: int
    retry_backoff_base: float
    max_tokens_continuation: bool | int
    max_consecutive_continuations: int
    min_continuation_tokens: int
    watchdog_repeated_tool_limit: int
    max_consecutive_idle_steps: int
    should_compact: ShouldCompactHook | None
    compact: CompactHook | None
    compact_instructions: CompactInstructions | None
    archive_writer: ArchiveWriter | None
    is_tool_productive: IsToolProductiveHook | None
    options: dict[str, Any] | None
```

**类型别名**: `CompactPriority`, `MessageConverter`, `ContextTransformer`, `BeforeToolCallHook`, `AfterToolCallHook`, `PrepareNextTurnHook`, `ShouldStopAfterTurnHook`, `MessageQueueGetter`, `ArchiveWriter`, `ShouldCompactHook`, `CompactHook`, `IsToolProductiveHook`

### 7.3 事件定义 —— events

**定义**: `src/xcode/agent/events.py`

```python
AgentEvent = Union[AgentStartEvent, AgentEndEvent, TurnStartEvent, TurnEndEvent,
                   MessageStartEvent, MessageUpdateEvent, MessageEndEvent,
                   ToolExecutionStartEvent, ToolExecutionUpdateEvent,
                   ToolExecutionEndEvent, ThinkingUpdateEvent, CompactionEvent]
```

### 7.4 消息类型 —— messages

**定义**: `src/xcode/agent/messages.py`

```python
AgentMessage = Union[SystemMessage, UserMessage, AssistantMessage,
                     ToolResultMessage, CompactionSummaryMessage, BranchSummaryMessage]

def convert_to_llm(messages: list[AgentMessage]) -> list[dict[str, Any]]
```

**常量**: `COMPACTION_SUMMARY_PREFIX`, `COMPACTION_SUMMARY_SUFFIX`, `BRANCH_SUMMARY_PREFIX`, `BRANCH_SUMMARY_SUFFIX`

### 7.5 协议 —— protocols

**定义**: `src/xcode/agent/protocols.py`

```python
QueueMode = Literal["all", "one-at-a-time"]
ToolExecutionMode = Literal["sequential", "parallel"]
ContentBlock = Union[TextContent, ImageContent, ToolCallContent, ThinkingContent]

class AgentTool(Protocol):
    name: str; label: str; description: str; parameters: dict
    execution_mode: ToolExecutionMode; examples: list
    def execute(tool_call_id, params, signal, on_update) -> AgentToolResult

class CancellationSignal(Protocol):
    reason: str
    def is_cancelled() -> bool: ...
```

### 7.6 工具执行 —— tool_execution

**定义**: `src/xcode/agent/tool_execution.py`

```python
@dataclass
class ExecutedToolBatch:
    results: list[ToolResultMessage]; terminate: bool

def execute_tool_calls(current_context, assistant_message, tool_calls,
                       config, signal, emit) -> ExecutedToolBatch

def partition_tool_calls_for_execution(current_context, tool_calls) -> list[list[ToolCallContent]]
```

---

## 8. CLI 交互层 `xcode.cli`

### 8.1 命令系统 —— commands

**定义**: `src/xcode/cli/commands.py`

```python
PromptText = str | list[tuple[str, str]]

class PromptLike(Protocol):
    def prompt(self, prompt_text: PromptText) -> str: ...

@dataclass
class ReplState:
    mode: str; verbose: bool; approved_plan: bool
    exit_pending: bool; pending_partial: str | None
    pending_inject: str | None; queue_mode: str

@dataclass
class CommandContext:
    store: SessionStore; app: XcodeApp; renderer: MarkdownRenderer
    state: ReplState; prompt_session: PromptLike
    session_policy: SessionPermissionPolicy | None
    persistent_store: PersistentPermissionStore | None

CommandHandler = Callable[[str, CommandContext], bool]

@dataclass
class CommandEntry:
    handler: CommandHandler; desc: str; args_desc: str
    accepts_args: bool; visible: bool

def command_names(registry) -> tuple[str, ...]
def generate_help_text(registry) -> str
```

### 8.2 REPL —— repl

**定义**: `src/xcode/cli/repl.py`

```python
def run_repl(app, sessions_dir, prompt_session, resume_latest,
             renderer, project_root) -> int
```

### 8.3 REPL 命令 —— repl_commands

**定义**: `src/xcode/cli/repl_commands.py`

**常量**: `COMMAND_REGISTRY`, `COMMAND_NAMES`, `HELP_TEXT`

**可用 REPL 命令**:

| 命令 | 说明 |
|---|---|
| `/help` | 显示帮助信息 |
| `/clear` | 开始新会话 |
| `/fork [type]` | 分叉会话（explore/verify/isolate） |
| `/rewind [n]` | 回退 n 轮 |
| `/resume [id/last]` | 恢复历史会话 |
| `/sessions` | 列出所有会话 |
| `/tree` | 显示会话树 |
| `/branch [id]` | 切换到分支 |
| `/model [name]` | 切换模型 |
| `/effort` | 设置推理 effort |
| `/thinking [on/off]` | 切换 thinking 显示 |
| `/plan` | 切换到 plan 模式（只读） |
| `/review` | 切换到 review 模式 |
| `/act` | 切换到 act 模式 |
| `/verbose` | 切换详细输出 |
| `/queue [mode]` | 设置队列模式（all/one-at-a-time） |
| `/compact` | 手动触发压缩 |
| `/permissions` | 管理工具权限 |
| `/tool [name input]` | 手动执行工具 |
| `/exit`, `/quit` | 退出 REPL |

### 8.4 REPL 渲染 —— repl_rendering

**定义**: `src/xcode/cli/repl_rendering.py`

**颜色常量**: `CLI_COLOR_TITLE`, `CLI_COLOR_DIM`, `CLI_COLOR_USER`, `CLI_COLOR_ASSISTANT`, `CLI_COLOR_THINKING`, `CLI_COLOR_TOOL`, `CLI_COLOR_SUCCESS`, `CLI_COLOR_ERROR`, `CLI_COLOR_WARNING`, `CLI_COLOR_INFO`, `CLI_PROMPT_MARKER_STYLE`

```python
class PromptSessionAdapter: ...
class LiveMarkdownStream: ...
class LiveReasoningPreview: ...

def reasoning_preview_lines(text, width) -> list[str]
def format_elapsed(seconds) -> str
def single_line_preview(text, width) -> str
def should_print_reasoning_summary(text, elapsed) -> bool
def answer_renderable(text) -> Table
def print_startup_banner(app, root) -> None
def input_prompt() -> PromptText
def format_thinking(value) -> str
def create_prompt_session(project_root, registry, command_names) -> PromptLike
```

### 8.5 REPL 会话 —— repl_sessions

**定义**: `src/xcode/cli/repl_sessions.py`

```python
def resume_interactively(store, prompt_session) -> None
def resume_latest(store) -> SessionMetadataView | None
def select_session(sessions, choice) -> SessionMetadataView | None
def print_sessions(sessions) -> None
def current_view(store) -> SessionMetadataView
def resumed_message(view) -> str
def print_loaded_history(store) -> None
def print_saved_conversation(store) -> None
```

### 8.6 REPL 设置 —— repl_settings

**定义**: `src/xcode/cli/repl_settings.py`

```python
class ModelControlApp(Protocol):
    def get_model_info() -> dict: ...
    def set_model(*, model, profile, base_url, api_key, thinking, reasoning_effort) -> str: ...

def handle_permissions(command, session_policy, persistent_store) -> None
def list_permissions(session_policy, persistent_store) -> None
def handle_model_command(command, app) -> None
def handle_effort_command(command, app) -> None
def handle_thinking_command(command, app) -> None
```

### 8.7 REPL 工具 —— repl_tools

**定义**: `src/xcode/cli/repl_tools.py`

```python
def run_tool_command(command, app) -> str
def run_shell_shortcut(command, app) -> str
def parse_tool_input(tool, raw_input) -> ToolInput
def cli_shorthand_key(tool) -> str
def brief_input(name, raw_input) -> str
def tool_intent(name, raw_input) -> str
def summarize_intents(intents) -> str
def event_to_dict(event) -> dict
def print_tool_call_rich(label, console) -> None
def print_tool_result_rich(data, verbose, console) -> None
def final_stop_reason(data) -> str | None
def file_reference_event(references) -> dict
```

### 8.8 设置向导 —— setup_wizard

**定义**: `src/xcode/cli/setup_wizard.py`

```python
CONFIG_FILENAME = "xcode.config.json"
PROVIDER_PRESETS: dict[str, dict]  # 包含 DeepSeek/OpenAI/Anthropic/MiMo 等预置

def has_valid_config(project_root) -> bool
def run_setup_wizard(project_root) -> None
```

### 8.9 工具目录 —— tool_catalog

**定义**: `src/xcode/cli/tool_catalog.py`

```python
def build_tool_catalog() -> dict[str, set[str]]
# 扫描所有工具构建函数，返回 {group: {tool_names}}
```

### 8.10 自动补全 —— completion

**定义**: `src/xcode/cli/completion.py`

```python
@dataclass
class CompletionItem:
    text: str; start_position: int; display_meta: str

class ReplCompleter:
    # prompt_toolkit Completer，补全 /commands、/tool names、@file、!shell
```

### 8.11 文件引用 —— file_refs

**定义**: `src/xcode/cli/file_refs.py`

```python
FILE_REF_PATTERN: re.Pattern  # 匹配 @file 引用

@dataclass
class FileReference:
    path: str; status: str; content: str; error: str | None

def expand_file_references(text, project_root) -> tuple[str, list[FileReference]]
```

### 8.12 Markdown 渲染 —— markdown

**定义**: `src/xcode/cli/markdown.py`

```python
class MarkdownRenderer(Protocol):
    def render(text: str) -> None: ...

class TerminalMarkdownRenderer: ...
```

### 8.13 HITL 处理 —— repl_hitl

**定义**: `src/xcode/cli/repl_hitl.py`

```python
class ReplHITLHandler:
    def __call__(self, tool: ToolSpec, action_input) -> HITLResult

def has_radiolist() -> bool
def radiolist_prompt(tool, action_input) -> str
```

---

## 9. 编码 Agent 工具 `xcode.coding_agent`

### 9.1 注册表构建 —— registry

**定义**: `src/xcode/coding_agent/registry.py`

```python
def build_project_scoped_registry(
    project_root, enabled, contextual_state, shell_spec, cancel_event, env
) -> tuple[ToolSpec, ...]
```

### 9.2 工具工厂 —— tools

**定义**: `src/xcode/coding_agent/tools/`

| 函数 | 模块 | 签名 | 说明 |
|---|---|---|---|
| `build_bash_tool` | `bash.py` | `(project_root, cancel_event, shell_spec, env, command_prefix, spawn_hook, on_progress) -> ToolSpec` | Bash 命令执行工具 |
| `build_code_tools` | `code_search.py` | `(project_root, grep_ops, ls_ops, find_ops, cancel_event) -> tuple[ToolSpec, ...]` | 代码搜索工具集 |
| `build_file_tools` | `file.py` | `(project_root, context_state, operations, cancel_event) -> tuple[ToolSpec, ...]` | 文件读写编辑工具集 |

**工具集 `build_code_tools` 返回的工具**:

| 工具名 | 说明 |
|---|---|
| `glob_files` | 文件名模式匹配 |
| `find_files` | 查找文件 |
| `grep_search` | 内容搜索 |
| `ls` | 目录列表 |
| `evaluate_python` | 执行 Python 表达式 |
| `reset_namespace` | 重置 Python 命名空间 |

**工具集 `build_file_tools` 返回的工具**:

| 工具名 | 说明 |
|---|---|
| `read_file` | 读取文件 |
| `write_file` | 写入文件 |
| `edit_file` | 编辑文件 |

**其他模块工具函数**:

```python
class ShellSpec: (name, command_prefix, syntax)
def detect_shell(config="auto") -> ShellSpec
def build_shell_argv(spec, command) -> list[str]
class OutputAccumulator: ...
def with_file_mutation(file_path, fn) -> T  # 线程安全文件操作
def resolve_read_path(root, raw_path) -> Path
def is_path_blocked(root, path) -> bool  # 阻止 .git/.venv/__pycache__
def truncate_output(text, max_lines, max_bytes) -> str
def display_path(root, path) -> str
def ensure_tool(tool, silent=False) -> str | None
```

---

## 10. 评测系统 `xcode.evals`

### 10.1 CLI —— evals.cli

**定义**: `src/xcode/evals/cli.py`

**CLI 参数**:

| 参数 | 说明 |
|---|---|
| `--suite` | 运行命名任务套件 |
| `--list-suites` | 列出内置套件 |
| `--show-suite` | 显示套件详情 |
| `--list-benchmarks` | 列出外部基准适配器 |
| `--tasks` | JSON/JSONL 任务文件 |
| `--benchmark` | 加载外部基准 (humaneval/swebench-lite/...) |
| `--benchmark-path` | 基准数据路径 |
| `--real` | 使用真实 provider 而非离线 mock |
| `--project-root` | 项目根目录 |
| `--output-dir` | 输出目录 |
| `--allow-project-mutation` | 允许 eval 修改项目 |
| `--trials` | 每任务试验次数 |
| `--limit` | 最大任务数 |

### 10.2 Schema —— evals.schema

**入口**: `xcode.evals.__init__` → `__all__ = ["EvalRunner", "EvalReport", "EvalTask", "GraderResult", "TrialResult"]`

### 10.3 Runner —— evals.runner

```python
class EvalRunner:
    def __init__(self, tasks, app_factory, output_dir, trials_per_task=1)
    def run(self) -> EvalReport
```

**其他模块**: `evals.tasks` (内置套件), `evals.adapters` (外部基准适配器), `evals.benchmarks` (基准加载器), `evals.sandbox` (沙箱环境)

---

## 11. 实验性模块 `xcode.experimental`

非默认启用，需 `tools.enabled_groups` 开启：

| 子模块 | 说明 |
|---|---|
| `memory` | `MemoryManager` — `MEMORY.md` 记忆系统 |
| `plugins` | `PluginManager` — 插件系统 |
| `mcp` | MCP (Model Context Protocol) 客户端集成 |
