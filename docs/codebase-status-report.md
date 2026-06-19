# Xcode 代码库现状报告

基于 `src/xcode/` 当前源码审查，审查日期：2026-06-18。

本文描述当前实现边界，不代表未来规划。结论以源码调用链为准，未以测试名称、
旧文档或接口预留作为已实现功能的证据。

## 总体结论

Xcode 已形成较完整的本地 coding agent 架构：

```text
cli → coding_agent → harness → agent → ai
```

核心 agent loop、工具执行、权限控制、上下文组装、会话恢复、文件快照和 CLI
已经可用。MCP 和 Memory 均已迁入正式 runtime package：

- Skill 和 MCP 是核心能力，应按基础范式兼容要求长期维护。
- Memory 是正式能力，但在主动召回效果得到 eval 验证前可以保持可配置启用。
- 进程内 Python Plugins 已删除，不再作为产品扩展路径。
- Subagent 和 eval 是正式的高级能力，不要求默认启用。

当前主要架构缺口集中在：

- 缺少跨工具、权限和运行时故障的统一错误 taxonomy。
- 工具调度没有资源标签、并发额度或统一频控。
- Memory 未接入每轮 Context Assembly 主动召回。
- MCP 缺少运行时管理、健康检查和多 transport 支持。
- Observability 没有 span tracing、统一时间戳和在线指标后端。
- Subagent 缺少递归、实时输出和 agent 间通信。
- Model routing 缺少成本、配额和多级 fallback 管理。

---

## L1 · Prompt / Instruction

状态：完整，存在扩展性限制。

### 核心文件

- `src/xcode/harness/agent_runtime/prompting/builder.py`
- `src/xcode/harness/agent_runtime/prompting/identity.py`
- `src/xcode/harness/agent_runtime/prompting/token_budget.py`
- `src/xcode/agent/context_collector.py`
- `src/xcode/harness/agent_runtime/git_preflight.py`
- `src/xcode/harness/agent_runtime/contextual.py`

### 已实现

- `SystemPromptBuilder` 将 system prompt 分为三个区域：
  - stable：identity、tool discipline、tools、search strategy。
  - dynamic：environment、cwd。
  - volatile：git preflight、contextual retrieval、session notices。
- stable 区按工具注册表及启用模块签名缓存。
- dynamic 区按 project root、shell 和 CWD 目录签名缓存。
- volatile 区每轮重建。
- CWD 快照忽略 `.git`、`.venv`、`__pycache__`，最多包含
  `MAX_CWD_ENTRIES` 个条目。
- 工具 prompt 缓存签名包含：
  `name`、`description`、`input_hint`、`prompt_snippet`、
  `prompt_guidelines`。
- `PromptRuntimeConfig.modules` 可控制具体 prompt 模块。
- `InstructionCollector` 支持配置文件源、inline 源以及
  `AGENTS.md`/`CLAUDE.md` 回退。
- 指令文本超过 32 KiB 时保留开头和关键 Markdown 节段，并添加截断标记。
- `PROMPT_VERSION` 已实现：根据稳定 prompt 内容和模块定义计算 SHA-256
  摘要，格式为 `prompt:<16 hex chars>`。
- `build_runtime_context_provider()` 将 prompt builder 包装为
  `Callable[[str], list[str]]`。

### 当前限制

- 没有用户自定义 region template 或完整替换 stable/dynamic/volatile
  region 的机制。
- 模块配置只能启用或禁用内置模块，不能改变内置模块顺序。
- `ContextualRetrievalState` 是近期相关文件、工具调用和结果的轻量状态，
  不是语义检索系统，也不会主动补全缺失上下文。
- contextual retrieval 在状态对象存在时会渲染基础说明；有记录时再附加
  active file、recent files、tool results 和 tool calls。

---

## L2 · Context Assembly

状态：核心能力完整。

### 核心文件

- `src/xcode/agent/context_assembly.py`
- `src/xcode/agent/context_collector.py`
- `src/xcode/agent/config.py`
- `src/xcode/harness/skills_registry.py`

### 已实现

- 六类 `ContextBlockSource`：
  `INSTRUCTION`、`SKILL`、`ACTIVE_DIFF`、`NOTES`、
  `RECENT_VALIDATION`、`TASK_STATE`。
- 两类注入目标：`SYSTEM` 和 `USER_CONTEXT`。
- 五级优先级：
  `CRITICAL(0)`、`HIGH(10)`、`MEDIUM(20)`、`LOW(30)`、
  `BACKGROUND(40)`。
- `ContextExpiry` 支持相对 turn 和 step 过期。
- `DefaultContextAssembler.assemble()` 执行：
  - 过期过滤。
  - 稳定优先级排序。
  - greedy priority-fill 预算裁剪。
  - system/user context 消息注入。
- `ContextCollectorRegistry` 按注册顺序同步执行；单个 collector 异常采用
  log + skip 策略。
- 已接入 collector：
  - `InstructionCollector`：指令文件和 inline 指令。
  - `ActiveDiffCollector`：git stat 和短 diff 摘录，最大 8 KiB。
  - `RecentValidationCollector`：最近 shell 验证错误，最大 4 KiB。
  - `TaskStateCollector`：当前 task store 状态，最大 4 KiB。
  - `NotesCollector`：`.local/notes/` 文本，最大 4 KiB。
  - `SkillIndexCollector`：技能摘要。
- Skill catalog 会转义 XML 属性、正文和非法控制字符，并限制 description
  长度。
- Skill catalog 明确要求 description 匹配任务时先调用 `load_skill`，并要求
  无明确匹配时不加载。
- `load_skill.name` schema 使用当前可见 skill names enum；无可见 skill 时不
  注册工具，也不注入空 catalog。
- activation output 包含 skill root，以及 `scripts/`、`references/` 和
  `assets/` 的相对路径元数据；资源内容不会被主动读取或执行。
- session 内跟踪已激活 skill，重复 activation 返回简短状态，避免重复注入
  正文。
- compaction 保留 activation tool-use/result 对，session resume 从历史恢复
  activation 状态。
- 项目级 skill 默认不披露；仅在 `skills.trust_project_skills=true` 时加入
  discovery，用户级 skill 保持默认发现。
- `paths.skills_dir` 和 `build_app(skills_dir=...)` 作为最高优先级显式发现
  目录；相对配置路径按项目根目录解析，不存在时记录 warning。

### 当前限制

- `expires_after_use` 未实现，并且已从数据模型删除；目前不是一个可配置但
  无效的字段。
- collector 协议为同步接口，异步 I/O 需要实现方自行处理。
- `ContextBlock` 没有跨会话持久化层。
- greedy 裁剪不会对单个 block 进行内容级缩减；放不下的完整 block 会被丢弃。

---

## L3 · Agent Loop / Harness

状态：核心循环完整，错误模型和运行期控制仍有限。

### 核心文件

- `src/xcode/agent/agent_loop.py`
- `src/xcode/agent/agent.py`
- `src/xcode/agent/config.py`
- `src/xcode/agent/results.py`
- `src/xcode/harness/agent_runtime/structured.py`
- `src/xcode/harness/agent_runtime/config.py`
- `src/xcode/harness/agent_runtime/execution_modes.py`

### 已实现

- 外层 step loop 和内层 provider continuation/retry loop。
- 默认步骤上限由 `AgentConfig.max_steps` 控制。
- provider 错误最多重试 3 次，使用指数退避。
- `max_tokens` 自动续写，并对连续低产出续写设置上限。
- steer 队列在下一步前注入；follow-up 队列在当前运行自然结束后追加。
- `CancellationSignal` 支持 `is_cancelled()` 和取消原因。
- 重复工具调用 watchdog 和连续 idle watchdog。
- `Agent.run()` 和 `Agent.run_stream()`。
- `run_stream()` 通过 `asyncio.Queue` 实时输出事件。
- `Agent.update_tools()` 可以替换实例持有的工具列表。
- `StructuredAgent` 聚合 ToolGate、HistoryManager、CancellationToken、
  ContextualRetrievalState 和 CompactController。
- `AgentLoopConfig` 支持 context、compaction、tool、provider 和 turn hooks。
- provider 未提供 stop reason 时会归一化为 `end_turn`；provider 异常会转换为
  `stop_reason="error"`。
- `TerminationReason` 统一表示 `completed`、`cancelled`、`step_limit`、
  `watchdog` 和 `provider_error`，并传递到 structured result、final event 和
  eval metrics。
- watchdog detail 和 provider/cancellation error detail 保持独立字段。

### 当前限制

- 没有覆盖工具业务错误、权限拒绝和运行时故障的统一 `ErrorKind` taxonomy。
- 旧 `stopped_by_*` 属性仍作为由 `termination_reason` 推导的只读兼容接口。
- `StructuredAgent` 在每次 `arun_stream()` 开始时创建内部 `Agent`；
  调用方不能通过稳定公开接口在正在运行的同一轮中热修改工具列表。
- loop 本身没有长时间无事件检测或心跳。仓库级别另有可选
  `HeartbeatDaemon`，但它不是 loop heartbeat。

---

## L4 · Tool Runtime

状态：执行链完整，资源治理较弱。

### 核心文件

- `src/xcode/agent/tool_execution.py`
- `src/xcode/agent/_tool_scheduling.py`
- `src/xcode/agent/_tool_validation.py`
- `src/xcode/harness/agent_runtime/tool_adapter.py`
- `src/xcode/harness/agent_runtime/tool_hooks.py`
- `src/xcode/coding_agent/tools/bash.py`
- `src/xcode/harness/execution_env.py`
- `src/xcode/coding_agent/tools/file.py`
- `src/xcode/coding_agent/tools/code_search.py`

### 已实现

- 工具调用按 `execution_mode` 分批。
- 默认执行模式为 sequential。
- 显式声明 parallel，或 `read_only + concurrency_safe` 的工具可并行。
- parallel batch 使用 `asyncio.gather(return_exceptions=True)`，单个调用失败
  不终止同批其他调用。
- 执行顺序包括参数校验、before hook、权限检查、handler、after hook。
- `ToolSpec` 包含 risk、group、read_only、concurrency_safe、
  execution_mode、counts_as_progress 等元数据。
- JSON Schema 参数校验。
- `glob_files` 和 `find_files` 共用基于 `rg --files` 的候选枚举与 Python
  fallback；两条路径统一遵循 `.gitignore`、hidden 和 blocked path 规则。
- `glob_files` 按修改时间降序和稳定路径次序返回；grep/glob 数值参数执行明确
  边界校验，ripgrep 错误保留 stderr 诊断。
- bash 支持：
  - 每次调用传入 timeout。
  - 默认 30 秒和最大 timeout schema 约束。
  - cancellation event。
  - deadline 轮询。
  - Unix 进程组或 Windows 进程树终止。
  - timed out/cancelled 结构化状态渲染。
- `ExecutionEnv` 将命令运行与 bash 工具接口分离，支持测试替身。

### 当前限制

- timeout 是 bash/ExecutionEnv 能力，不是所有 `ToolSpec` 通用字段。
- 没有统一工具频控。
- 没有按资源标签限制并发，例如 bash process、network、CPU 或特定服务额度。
- 调度模型只有 sequential/parallel，没有互斥组、读写锁或依赖图。

---

## L5 · Extension / MCP / Skill

状态：部分实现。

### 核心文件

- `src/xcode/harness/mcp/tools.py`
- `src/xcode/harness/mcp/client.py`
- `src/xcode/harness/skills.py`
- `src/xcode/harness/skills_registry.py`
- `src/xcode/coding_agent/tools/worktree.py`

### 已实现

- MCP 从 `.local/mcp_config.json` 加载本地 stdio server。
- MCP 配置支持 command、args、env、enabled、timeout、defer_loading。
- initialize 发送最新支持协议版本，校验 server 返回版本、capabilities 和
  serverInfo，并保存 instructions。
- tools 请求仅在 server 声明 `tools` capability 后发送。
- schema cache 同时记录配置 hash、协商协议版本和 server identity；缺少协商
  元数据的旧缓存会重新发现。
- deferred server 使用 bootstrap 工具和 `mcp_tool_search` 懒加载。
- MCP 工具名称清理和碰撞检测。
- MCP `isError` 转换为结构化错误。
- server stderr 脱敏和截断。
- `LazyClientRef` 复用客户端，并暴露 pending/connected/failed/disabled 状态。
- `SkillRegistry` 发现技能；`SkillIndexCollector` 注入摘要；
  `load_skill` 懒加载正文、资源路径和 session activation 状态。

### 当前限制

- MCP 仅支持本地 stdio；没有 SSE、HTTP、Streamable HTTP、OAuth、
  resources、prompts 或 server notifications。
- 没有运行时 MCP server 增删、重连和状态管理 UI。
- `LazyClientRef` 没有连接池、周期健康检查或自动故障转移。
- MCP 配置中的 `overrides` 当前明确跳过并警告。
- skill approval 依赖宿主 PermissionEngine/HITL 配置；skill 本身不提供独立
  审批界面。

---

## L6 · Permission / Sandbox / Security

状态：决策模型完整，仍存在边界实现问题。

### 核心文件

- `src/xcode/harness/observability/permissions.py`
- `src/xcode/harness/observability/permission_model.py`
- `src/xcode/harness/observability/_safety_backstop.py`
- `src/xcode/harness/agent_runtime/tool_gate.py`
- `src/xcode/harness/agent_runtime/execution_modes.py`
- `src/xcode/harness/config.py`

### 已实现

- `PermissionEngine.decide()` 是最终权限决策入口。
- 权限模型包括 Action、Target、Constraint、Verdict 和 GrantRecord。
- evaluator 包括执行模式、静态策略、结构化路径边界、安全 backstop 和
  hook constraint providers。
- resolver 支持 non-bypassable deny、deny、ask、allow 的优先级解析。
- session 和 permanent grant store。
- ask 决策可通过 HITL callback 解决。
- Structured boundary 对工作区、外部批准目录、git 路径、敏感路径和阻断目录
  进行分类。
- Safety backstop 将 shell 命令分为 deny/ask/allow 三桶，并拆分复合命令。
- ToolGate 每轮冻结 registry、权限策略和 grant store 快照。
- plan/build/act 模式同时控制工具可见性和 execution constraint。
- MCP permanent grant 降级为 session。
- 多 target action 的 session/permanent grant 降级为 once。
- `restricted_dirs` 基于结构化 path target、平台路径大小写规则和目录 containment
  判断，不再扫描序列化 input 文本。
- 已知文件系统 shell 命令会保守提取路径；启用 `restricted_dirs` 且无法可靠
  提取高风险路径时返回 ask。

### 权限调用链

```text
ExecutionModeState
  → policy_for_mode().check_call()
  → ToolGate._precheck_permission()
  → PermissionEngine.decide(execution_decision=...)
  → evaluate_policy_constraints()
  → PermissionResolver.resolve()
  → grant lookup / HITL callback
```

执行模式是统一权限解析器的输入约束，不是绕开 PermissionEngine 的第二套最终
决策系统。

### 当前限制

- shadow model 只附加 shadow verdict、diff 和 approval candidate，不改变实际
  决策。
- Safety backstop 规则硬编码在 Python 常量中，没有配置扩展接口。
- 所谓 sandbox 主要由路径边界、权限策略和可选 worktree 提供，不是 OS 级
  容器或系统调用隔离。

---

## L7 · Hooks / Observability

状态：事件链完整，追踪与指标平台缺失。

### 核心文件

- `src/xcode/harness/observability/hooks.py`
- `src/xcode/harness/observability/audit.py`
- `src/xcode/harness/agent_runtime/tool_hooks.py`
- `src/xcode/harness/agent_runtime/config.py`
- `src/xcode/harness/agent_runtime/tool_gate.py`

### 已实现

- 六类 hook：
  `pre_tool`、`post_tool`、`on_error`、`on_compact`、
  `before_agent_start`、`before_provider_request`。
- 双通道：
  - `register()` 接收 `HookRecord`。
  - `subscribe()` 接收类型化 `HarnessEvent`。
- `on_error` 在工具结果 `is_error=True` 时触发。
- `on_compact` 在 compactor 调用前触发。
- `before_agent_start` 在 `StructuredAgent.arun_stream()` 入口触发。
- `before_provider_request` 在每次 provider request 前触发，并包含：
  - messages。
  - tools。
  - prompt version。
  - prompt SHA-256。
  - system prompt byte size。
- `JsonlAuditLogger` 写入 session、工具、输入、输出和权限信息。
- `redact_text()` 覆盖常见 API key、secret 和 token 模式。

### 当前限制

- `HookRecord` 和多数类型化 hook event 不自带时间戳。
- 没有 OpenTelemetry、span/trace parent、分布式 trace 或 exporter。
- 默认装配中的空 lambda 只是保活注册，不提供实际消费逻辑。
- 运行时有 latency/token 数组，但没有在线 histogram、percentile、告警或
  指标后端。
- `on_error` 主要覆盖工具错误，不等同于统一 runtime error bus。

---

## L8 · Validation / Eval

状态：部分实现，但 pipeline 和 sandbox 已可用。

### 核心文件

- `src/xcode/evals/runner.py`
- `src/xcode/evals/cli.py`
- `src/xcode/evals/schema.py`
- `src/xcode/evals/graders.py`
- `src/xcode/evals/tracing.py`
- `src/xcode/evals/validation.py`
- `src/xcode/evals/sandbox.py`
- `src/xcode/evals/tasks.py`
- `src/xcode/evals/reporting.py`
- `src/xcode/evals/adapters/swebench.py`

### 已实现

- 多 task × 多 trial 的 `EvalRunner`。
- grader 包括：
  - 事件和最终回答检查。
  - 预期/禁止工具检查。
  - 文件存在、修改和内容证据。
  - validation command。
  - LLM-as-judge。
- `TraceRecorder` 输出 JSONL agent 事件。
- pass@k 无偏估计和 pass^k。
- trial metrics 包括模型调用、实际 input/output token、模型延迟、工具调用和
  工具错误等。
- `AgentLoopMetrics` 已转换并传入 `StructuredAgentResult.metrics`，final event
  可被 eval runner 提取。
- sandbox 已实现：
  - fixture task 复制到独立 trial 目录。
  - 默认拒绝无 fixture 的真实任务修改当前项目。
  - 只有显式 `allow_project_mutation` 才允许使用原项目目录。
- 内置 pipeline、tool-policy、coding-fixture、smoke、tool、context、multi、
  plan suites。
- SWE-bench prediction adapter 和 HumanEval/EvalPlus/MBPP loader。

### 当前限制

- task 之间没有依赖图、冲突检测或顺序约束。
- task schema 没有独立 validator/generator 工具，复杂 metadata 仍靠手工构造。
- 相同 task/trial 没有结果缓存。
- sandbox 是目录复制隔离，不是容器、VM 或系统权限沙箱。
- LLM judge 的稳定性、成本和模型版本需要外部治理。

---

## L9 · Memory / Session / Checkpoint

状态：三个概念已清晰分离，但联动不完整。

### Session

核心文件：`src/xcode/harness/session.py`

- JSONL transcript。
- metadata index。
- session resume、fork、branch tree 和 rewind。
- rewind 后按保留的用户轮次同步截断 snapshot index。
- fork session 会复制父会话 snapshot repository，保留一致的撤销历史。
- file lock 保护索引和记录写入。

### Memory

核心文件：

- `src/xcode/harness/memory/manager.py`
- `src/xcode/harness/memory/parsing.py`

已实现：

- `MEMORY.md` 结构化块。
- BM25 检索和 metadata rerank。
- 必需字段、长度和 novelty 质量门。
- 同 title 冲突合并。
- LRU 访问记录和超限淘汰。
- compaction summary consolidation hook。

限制：

- 没有注册为 ContextCollector，也没有在每轮 Context Assembly 中根据用户问题
  自动查询。
- LRU 淘汰主要按访问时间，不综合长期价值分数。
- 文件写入没有跨进程 merge 协议，依赖单写者假设。

### Checkpoint

核心文件：`src/xcode/harness/snapshot.py`

- 每个 session 使用 `.local/snapshots/<session>/` 中的独立隐藏 git 目录。
- snapshot id 是 `git write-tree` 生成的 tree hash。
- 记录 turn 的 pre/post snapshot 和 changed files。
- `/undo` 可按文件恢复，并执行冲突检查和权限审批。
- `rewind_to_turn_count()` 删除 rewind 游标之后的记录，后续 turn id 可复用。

限制：

- snapshot 不支持跨 session 导出、共享或合并。

---

## L10 · UI / HITL / Command System

状态：本地 REPL 完整，扩展能力有限。

### 核心文件

- `src/xcode/cli/repl.py`
- `src/xcode/cli/repl_hitl.py`
- `src/xcode/cli/repl_commands.py`
- `src/xcode/cli/repl_rendering.py`
- `src/xcode/cli/repl_sessions.py`
- `src/xcode/cli/repl_settings.py`
- `src/xcode/cli/repl_turn_handler.py`
- `src/xcode/cli/setup_wizard.py`
- `src/xcode/cli/completion.py`

### 已实现

- prompt_toolkit 多行 REPL、历史记录和补全。
- slash commands，包括 session、mode、model、thinking、compact、permissions、
  tool 和 undo。
- session resume、fork、branch 和 tree UI。
- plan/build/act 热切换。
- HITL allow/deny，以及 once/session/permanent scope。
- 模型、thinking 和 reasoning effort 切换。
- Markdown 和 reasoning/tool event 渲染。
- 首次配置向导。
- 单次 prompt 和非交互运行入口。

### 当前限制

- command registry 当前由代码静态定义，没有用户自定义 slash command 插件
  接口。
- 没有历史会话全文搜索 UI。
- 多 target action 只触发一次 action 级审批；为避免宽泛授权，其持久 scope
  会降级为 once。当前不存在逐 target 批量选择 UI。
- 非交互模式下需要预配置权限策略，否则 ask 类操作无法进行交互审批。

---

## L11 · Sub-agent / Multi-agent

状态：部分实现。

### 核心文件

- `src/xcode/harness/agent_runtime/subagent.py`
- `src/xcode/harness/agent_runtime/async_worker.py`
- `src/xcode/harness/assembly.py`

### 已实现

- `submit_subagent`、`check_subagent`、`cancel_subagent`。
- `ManagedSubagentRunner` 使用独立线程和 asyncio event loop。
- context 和 worktree 两种隔离方式。
- model profile 选择。
- 默认 120 秒 timeout。
- start/end lifecycle event。
- 父 agent 为 child 构造独立 StructuredAgent、ToolGate、HookManager 和
  ContextualRetrievalState。
- child registry 排除 MCP 工具。
- worktree 模式要求启用 worktree group。
- `result()` 和 `cancel()` 会在任务完成后移除对应 job；`shutdown()` 清空全部
  job。

### 当前限制

- child registry 不包含 subagent tools，因此仅支持一层 subagent。
- 没有 child → parent、child ↔ child 消息通道。
- 没有进度事件和流式输出；父 agent 只能 poll `check_subagent`。
- Branch summary 只保留有限结果摘要。
- `sweep_finished()` 没有定时自动调用；如果调用方既不 check/cancel，也不关闭
  runner，已完成 job 会继续保留。
- context isolation 共享工作目录，不能避免并发文件修改冲突。

---

## L12 · Model Routing / Cost

状态：provider 和 retry 完整，成本治理部分实现。

### 核心文件

- `src/xcode/ai/providers/router.py`
- `src/xcode/ai/providers/runtime.py`
- `src/xcode/ai/providers/factory.py`
- `src/xcode/ai/providers/_registry.py`
- `src/xcode/ai/providers/metrics.py`
- `src/xcode/ai/providers/protocol.py`
- `src/xcode/ai/cache.py`
- `src/xcode/ai/model_modes.py`

### 已实现

- `RouterProvider` 可根据 `RouterFn(messages, tools)` 选择 provider。
- 未提供 RouterFn 时使用 default provider。
- 单个 fallback provider。
- main/subagent/fallback profile 和 profile 继承。
- OpenAI、DeepSeek、ChatGLM、MiMo、OpenAI-compatible 和 faux provider。
- `PROVIDER_REGISTRY` 已静态注册所有内置 transport，不是 stub。
- `ProviderRuntime` 支持：
  - 本地最小调用间隔。
  - timeout/connection/429/500/502/503/529 临时错误识别。
  - 最多三次随机指数退避重试。
  - API 错误归一化。
- provider stream 采集实际 input/output token usage。
- cache metrics 包括 cached token、miss token 和 cache hit rate。
- 配置按全局、项目、本地来源深度合并，并支持相关环境变量覆盖。

### 当前限制

- 没有内置智能 RouterFn；默认行为仍是固定 provider。
- 没有货币成本计算、预算上限、用户配额或请求级 cost policy。
- RouterProvider 只支持一个 fallback，不支持逐级 fallback 链。
- fallback 是异常后的 provider 切换，不是按 429、容量或成本进行模型降级的
  路由策略。
- cache hit rate 存在于 provider metrics，但没有统一进入
  `StructuredAgentResult`、eval report 或在线监控。

---

## 跨模块结论

### 错误分类

运行终止已有统一 `TerminationReason`，但当前不存在覆盖全部子系统的统一错误
taxonomy。

| 类别 | 当前表示 |
| --- | --- |
| 工具参数或执行错误 | `AgentToolResult.is_error` 和文本 |
| Provider 错误 | `termination_reason="provider_error"` 和 `error_detail` |
| Permission 拒绝 | `PermissionEngineResult` 的 decision、blocked、reason、source |
| bash timeout/cancel | `ExecutionResult.timed_out/cancelled`，最终渲染为文本 |
| subagent timeout | `asyncio.wait_for` 异常 |
| compaction | `CompactEvent`，不是错误类型 |
| watchdog | `termination_reason="watchdog"` 和 `watchdog_reason` |
| step limit | `termination_reason="step_limit"` |

未来若引入统一错误 taxonomy，仍应保持工具业务错误、权限拒绝和运行时故障的
语义区别。

### Permission 入口

权限最终入口是 `PermissionEngine.decide()`。

执行模式先产生 `allow/deny` constraint，然后作为
`execution_decision` 进入统一 evaluator/resolver。不能将其描述为完全独立的
第二套权限系统。

### Hook 触发覆盖

| Hook | 当前是否触发 | 主要触发位置 |
| --- | --- | --- |
| `pre_tool` | 是 | `ToolGate.build_before_tool_hook()` |
| `post_tool` | 是 | `emit_tool_hook()` 成功路径 |
| `on_error` | 是 | `emit_tool_hook()` 工具错误路径 |
| `on_compact` | 是 | `_compact_and_emit()` |
| `before_agent_start` | 是 | `StructuredAgent.arun_stream()` |
| `before_provider_request` | 是 | Agent provider 请求前 closure |

### Session、Memory、Checkpoint

| 概念 | 存储 | 作用域 | 用途 |
| --- | --- | --- | --- |
| Session | JSONL + session index | 当前及历史会话树 | transcript、resume、fork、rewind |
| Memory | `MEMORY.md` + LRU metadata | 项目级跨会话 | 长期知识积累和 BM25 检索 |
| Checkpoint | 独立 git tree + turn index | session 内 | 文件级 pre/post snapshot 和 undo |

三个概念的职责区分清晰；rewind 与 snapshot index 已保持同一时间线。当前主要
缺口是 memory 尚未进入每轮主动召回路径。

---

## 问题分类与实施判断

代码库中发现的限制不应全部转化为实现任务。需要区分正确性问题、需求驱动的
能力扩展和当前不值得承担的复杂度。

### A · 值得近期实现

这些问题直接影响安全、状态一致性、资源稳定性或故障定位。

1. 让现有 `tool_workers` 配置真正限制并行工具调用，并为 subagent 增加独立
   并发上限。当前不设计通用资源调度器。
2. 为 hook 和最终运行结果补充时间、session/turn/request 关联字段；先完成
   本地可追踪性，不直接引入 OpenTelemetry。

这些项目已按优先级写入 `TODO.md`。

### B · 有明确需求或 eval 证据后实现

这些能力可能有价值，但应先证明使用频率、效果或维护收益。

- Memory 每轮主动召回：先建立离线 eval，证明召回质量和上下文收益，再接入
  Context Assembly。
- MCP Streamable HTTP、OAuth、resources 和 prompts：按真实 server 集成需求
  增量实现，不以覆盖完整协议为目标。
- MCP 自动重连、list-changed 和长期连接管理：持久 MCP 使用成为主路径后实现。
- Subagent 流式进度和更丰富结果：长任务体验出现实际瓶颈后实现。
- 模型成本、预算和多级 fallback：多模型付费路由实际投入使用后实现。
- 用户自定义 prompt region、slash command、eval 缓存和 task 依赖：
  出现明确调用方后实现。

Skill 和 MCP 的基础范式兼容属于产品核心方向，但具体范围需要单独审查，见
`docs/skill-mcp-capability-analysis.md`。

### C · 当前不建议实现

这些项目会显著增加状态空间，当前没有足够收益证明。

- 通用工具资源标签、依赖图、读写锁和多资源调度器。
- 完整的全系统 `ErrorKind` 层级。
- Subagent 递归和 child-to-child 通信。
- ContextBlock 跨会话持久化。
- 跨 session checkpoint 共享或合并。
- 通用插件沙箱。
- 为所有 ToolSpec 强制统一 timeout。
- 仅为架构完整性引入 OpenTelemetry。
- Skill marketplace、harness 语义向量匹配和专用脚本执行引擎。

## 正式能力方向

### Skill

- 保持默认启用和核心 group。
- 完成 Agent Skills 的 discovery → activation → context retention 基础闭环。
- 不要求 marketplace，也不要求专用脚本运行时。

### MCP

- 已迁入正式 runtime 模块并由核心装配自动发现；无 MCP 配置时自然注册为空。
- 基础兼容目标是可靠连接官方示例和常见 stdio tools server，而不是一次实现
  MCP 所有可选 feature。

### Memory

- 已迁入正式 runtime 模块，并保持独立 `memory` group。
- consolidation 可正式使用；主动 recall 需通过 eval 后再默认接入。

## 审查边界

- 本报告是源码静态审查，不代表所有路径均经过端到端运行验证。
- “已实现”表示存在明确实现和装配调用链，不表示生产成熟度或完整测试覆盖。
